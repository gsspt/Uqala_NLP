"""
score_bert_colab.py
────────────────────
Script Colab (étage 3) : applique CAMeLBERT fine-tuné sur les candidats
pré-filtrés par le pipeline LR (openiti_candidates.json).

Utilise Google Drive pour la persistance — les fichiers survivent au
rafraîchissement de session.

Structure Drive attendue :
  Mon Drive/
  └── uqala_nlp/
      ├── openiti_candidates.json   (à uploader depuis le PC)
      ├── camelbert_majnun.zip      (à uploader depuis le PC)
      └── openiti_candidates_bert.json  (sortie, sauvegardée ici)

Usage dans Colab :
  !python score_bert_colab.py
  !python score_bert_colab.py --threshold 0.5 --top 100
"""

import json, pathlib, argparse, sys, time, re
import numpy as np

# ── Filtre isnad (inliné pour éviter dépendance externe) ──────────────────────
# Sépare l'isnad (chaîne de transmission) du matn (récit) avant inférence BERT.
# BERT a été entraîné sur des matns filtrés — la présence d'une longue chaîne
# isnad biaise le score vers les négatifs (textes hadith/biographiques).

_TRANS_VERBS = {
    'أخبرنا','أخبرني','حدثنا','حدثني','أنبأنا','أنبأني',
    'روى','روينا','سمعت','سمعنا','ذكر','أعلمنا','نقل','بلغنا','بلغني',
}
_ISNAD_CONN = {'عن','من','بن','ابن','بنت'}
_RE_KUNYA   = re.compile(r'\bأب[وياُ]\b|\bأم\b|\bابن\b|\bبن\b')
_RE_MATN    = re.compile(
    r'(?<!\w)(?:فقال|قالوا|قالت|قال|فقالت|حكى|حُكي|يُحكى|'
    r'روي\s+أن|قيل\s+(?:إن|أن)|أن(?:ه|ها|هم)\s+قال)(?!\w)',
    re.UNICODE
)
_RE_DIACS   = re.compile(r'[\u064B-\u065F\u0670]')
_RE_ARTOK   = re.compile(r'[\u0600-\u06FF]+')

def _tok_score(tok):
    n = _RE_DIACS.sub('', tok).replace('أ','ا').replace('إ','ا').replace('ة','ه').replace('ى','ي')
    if n in _TRANS_VERBS or tok in _TRANS_VERBS: return 1.5
    if n in _ISNAD_CONN  or tok in _ISNAD_CONN:  return 0.8
    if _RE_KUNYA.match(tok): return 0.7
    return 0.0

def get_matn(text):
    """Retourne le matn (contenu narratif) en supprimant l'isnad initial."""
    if not text or len(text) < 20:
        return text
    toks = _RE_ARTOK.findall(text)
    if len(toks) < 4:
        return text
    candidates = [(m.start(), m.group()) for m in _RE_MATN.finditer(text)]
    if not candidates:
        return text
    best_cut = None
    best_pos  = -1
    for char_pos, marker in candidates:
        ratio = char_pos / len(text)
        if ratio > 0.7 or ratio < 0.03:
            continue
        toks_before = _RE_ARTOK.findall(text[:char_pos])
        if len(toks_before) < 2:
            continue
        d_before = sum(_tok_score(t) for t in toks_before) / len(toks_before)
        toks_after = _RE_ARTOK.findall(text[char_pos:])
        win = 8
        d_after = (sum(_tok_score(t) for t in toks_after[:win]) / win
                   if len(toks_after) >= win else 0.0)
        if d_before >= 0.25 and (d_before - d_after) > 0.05 and char_pos > best_pos:
            best_pos = char_pos
            best_cut = char_pos + len(marker)
    if best_cut is None:
        return text
    matn = text[best_cut:].strip()
    return matn if len(matn) >= 30 else text

# ── Chemins ───────────────────────────────────────────────────────────────────
# Drive doit être monté AVANT de lancer ce script, depuis une cellule notebook :
#   from google.colab import drive; drive.mount('/content/drive')
CONTENT    = pathlib.Path('/content')
DRIVE      = pathlib.Path('/content/drive/MyDrive/Thèse')

# Entrées
INPUT     = DRIVE / 'openiti_candidates.json'
MODEL_DIR = DRIVE / 'camelbert_majnun'
OUTPUT    = DRIVE / 'openiti_candidates_bert.json'

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Seuil BERT pour conserver un candidat (défaut: 0.5)')
parser.add_argument('--top', type=int, default=50,
                    help='Nombre de candidats à afficher (défaut: 50)')
parser.add_argument('--batch', type=int, default=64,
                    help='Taille de batch BERT (défaut: 64, augmenter si GPU A100)')
parser.add_argument('--maxlen', type=int, default=256,
                    help='Longueur max tokens (défaut: 256)')
args = parser.parse_args()

# ── Vérifications ─────────────────────────────────────────────────────────────
if not MODEL_DIR.exists():
    print(f"ERREUR : modèle introuvable → {MODEL_DIR}")
    print(f"  Vérifie que camelbert_majnun/ est bien dans Mon Drive/Thèse/")
    sys.exit(1)
else:
    print(f"  Modèle      : {MODEL_DIR}/")

if not INPUT.exists():
    print(f"ERREUR : candidats introuvables → {INPUT}")
    print(f"  Place openiti_candidates.json dans Mon Drive/Thèse/")
    sys.exit(1)

print(f"  Candidats   : {INPUT}")
print(f"  Sortie      : {OUTPUT}")

# ── Chargement des candidats ──────────────────────────────────────────────────
print(f"Chargement des candidats…")
raw = json.load(open(INPUT, encoding='utf-8'))
candidates = raw['candidates'] if isinstance(raw, dict) else raw
print(f"  {len(candidates)} candidats à scorer")

# ── Chargement du modèle ──────────────────────────────────────────────────────
print(f"Chargement CAMeLBERT depuis {MODEL_DIR}/…")
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device : {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()
    print(f"  Modèle chargé ({sum(p.numel() for p in model.parameters()):,} paramètres)")
except ImportError:
    print("ERREUR : pip install transformers torch")
    sys.exit(1)

# ── Inférence par batch ────────────────────────────────────────────────────────
def bert_scores(texts, batch_size=64, max_length=256):
    all_probs = []
    n = len(texts)
    n_batches = (n + batch_size - 1) // batch_size
    LOG_EVERY = max(1, n_batches // 20)   # affiche ~20 lignes au total
    t0 = time.time()

    for bi, i in enumerate(range(0, n, batch_size)):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())

        if bi % LOG_EVERY == 0 or bi == n_batches - 1:
            done = min(i + batch_size, n)
            pct  = done / n * 100
            elapsed = time.time() - t0
            eta = (elapsed / max(done, 1)) * (n - done)
            print(f"  [{pct:5.1f}%]  {done:6,}/{n:,} textes  "
                  f"({elapsed:.0f}s écoulées, ETA {eta:.0f}s)")

    return all_probs

print(f"\nFiltrage isnad avant inférence…")
matns = []
n_filtered = 0
for c in candidates:
    matn = get_matn(c['text'])
    if matn != c['text']:
        n_filtered += 1
    matns.append(matn)
print(f"  Isnad supprimé dans {n_filtered}/{len(candidates)} textes")

print(f"\nInférence BERT sur {len(candidates)} candidats…")
probs  = bert_scores(matns, batch_size=args.batch, max_length=args.maxlen)

# ── Filtrage et tri ────────────────────────────────────────────────────────────
for c, p in zip(candidates, probs):
    c['bert_score'] = round(float(p), 4)

filtered = [c for c in candidates if c['bert_score'] >= args.threshold]
filtered.sort(key=lambda x: -(x['bert_score'] * 0.4 + x['lr_score'] * 0.6))

print(f"\n{'─'*60}")
print(f"  Candidats avant BERT  : {len(candidates)}")
print(f"  Candidats ≥ {args.threshold:.0%} BERT   : {len(filtered)}")
print(f"  Taux de rejet BERT    : {(1 - len(filtered)/max(len(candidates),1))*100:.1f}%")
print()

# Distribution par seuil BERT
for t in [0.95, 0.90, 0.80, 0.70, 0.50]:
    n = sum(1 for c in filtered if c['bert_score'] >= t)
    print(f"    ≥ {t:.0%} bert : {n:5d} candidats")
print()

# Par auteur
from collections import defaultdict
by_author = defaultdict(list)
for c in filtered:
    by_author[c['author']].append(c)

print("── Par auteur ──")
rows = sorted(by_author.items(), key=lambda x: -len(x[1]))
for auth, cs in rows:
    mean_bert = sum(c['bert_score'] for c in cs) / len(cs)
    mean_lr   = sum(c['lr_score']   for c in cs) / len(cs)
    print(f"  {len(cs):4d}  bert={mean_bert:.2f}  lr={mean_lr:.2f}  {cs[0]['author_desc']}")
print()

# Top candidats
print(f"── Top {args.top} candidats ──")
for i, c in enumerate(filtered[:args.top], 1):
    text = c['text'].replace('\n', ' ')[20:220]
    print(f"\n  {i:3d}. [bert={c['bert_score']:.3f}  lr={c['lr_score']:.3f}]  {c['author_desc']}")
    print(f"       {c['work']}")
    print(f"       {text}…")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
output = {
    'meta': {
        **({} if not isinstance(raw, dict) else raw.get('meta', {})),
        'n_after_bert':    len(filtered),
        'bert_threshold':  args.threshold,
        'bert_model':      str(MODEL_DIR.name),
    },
    'candidates': filtered,
}
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n  Résultats sauvegardés dans Drive → {OUTPUT}")
print(f"  Accessible depuis : Mon Drive/Thèse/openiti_candidates_bert.json")
