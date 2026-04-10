"""
scan_openiti_colab.py
──────────────────────
Scan direct du corpus openiti_targeted/ avec CAMeLBERT fine-tuné,
sans étage LR intermédiaire. Pipeline :

  1. Parse OpenITI mARkdown → unités narratives
  2. Gates regex (RE_JUNUN + RE_DIALOGUE + F14 fenêtre contextuelle)
  3. Filtre isnad (supprime la chaîne de transmission avant inférence)
  4. Inférence CAMeLBERT par batch
  5. Sauvegarde progressive dans Google Drive

Structure Drive attendue :
  Mon Drive/Thèse/
  ├── openiti_targeted/     ← corpus (dossiers par auteur)
  ├── camelbert_majnun/     ← modèle fine-tuné
  └── openiti_bert_direct.json  ← sortie

Usage dans Colab :
  # Cellule 1 (notebook)
  from google.colab import drive; drive.mount('/content/drive')

  # Cellule 2
  import shutil
  shutil.copy('/content/drive/MyDrive/Thèse/scan_openiti_colab.py', '/content/')

  # Cellule 3
  !python scan_openiti_colab.py
  !python scan_openiti_colab.py --threshold 0.5 --batch 128
"""

import json, re, time, pathlib, sys, argparse, unicodedata
from collections import defaultdict

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Seuil BERT (défaut: 0.5)')
parser.add_argument('--batch',     type=int,   default=64,
                    help='Taille de batch BERT (défaut: 64)')
parser.add_argument('--maxlen',    type=int,   default=256,
                    help='Longueur max tokens (défaut: 256)')
parser.add_argument('--top',       type=int,   default=50,
                    help='Candidats affichés en fin (défaut: 50)')
parser.add_argument('--save-every',type=int,   default=20,
                    help='Sauvegarde Drive tous les N fichiers (défaut: 20)')
parser.add_argument('--resume',    action='store_true',
                    help='Reprendre depuis openiti_bert_progress.json')
args = parser.parse_args()

# ── Chemins ───────────────────────────────────────────────────────────────────
DRIVE    = pathlib.Path('/content/drive/MyDrive/Thèse')
CORPUS   = DRIVE / 'openiti_targeted'
MODEL    = DRIVE / 'camelbert_majnun'
OUTPUT   = DRIVE / 'openiti_bert_direct.json'
PROGRESS = DRIVE / 'openiti_bert_progress.json'

for p, label in [(CORPUS, 'openiti_targeted/'), (MODEL, 'camelbert_majnun/')]:
    if not p.exists():
        print(f"ERREUR : {label} introuvable dans {DRIVE}")
        sys.exit(1)

# ── Auteurs ciblés ────────────────────────────────────────────────────────────
TARGET_AUTHORS = {
    "0255Jahiz":               "al-Jāḥiẓ",
    "0276IbnQutaybaDinawari":  "Ibn Qutayba",
    "0328IbnCabdRabbih":       "Ibn ʿAbd Rabbih",
    "0414AbuHayyanTawhidi":    "Abū Ḥayyān al-Tawḥīdī",
    "0850ShihabDinIbshihi":    "al-Ibshīhī",
    "0412Sulami":              "al-Sulamī",
    "0465IbnHawazinQushayri":  "al-Qushayrī",
    "0310Tabari":              "al-Ṭabarī",
    "0463KhatibBaghdadi":      "al-Khaṭīb al-Baghdādī",
    "0571IbnCasakir":          "Ibn ʿAsākir",
    "0597IbnJawzi":            "Ibn al-Jawzī",
    "0626YaqutHamawi":         "Yāqūt al-Ḥamawī",
    "0681IbnKhallikan":        "Ibn Khallikān",
    "0733Nuwayri":             "al-Nuwayrī",
    "0748Dhahabi":             "al-Dhahabī",
    "0360Tabarani":            "al-Ṭabarānī",
    "0542IbnBassamShantarini": "Ibn Bassām",
    "0279Baladhuri":           "al-Balādhurī",
}

# ── Gates regex ───────────────────────────────────────────────────────────────
_JUNUN_TERMS = [
    'مجنون','المجنون','مجنونا','مجانين','المجانين','مجنونة','المجنونة',
    'معتوه','المعتوه','معتوها','معتوهة','مدله','المدله',
    'هائم','الهائم','هائما','ممسوس','ممرور','مستهتر',
    'جنونه','جنونها','جنوني','جنونا','جنون','الجنون',
    'بهلول','بهلولا','سعدون','عليان','جعيفران','ريحانة',
    'سمنون','لقيط','حيون','حيونة','خلف','رياح',
]
_DIALOGUE_MARKERS = [
    'قلت','فقلت','قلنا','سألت','فسألت','سألني','أجبت',
    'فقلتله','قيل له','قيل لي','فقلت له',
]
_SHIR_MARKERS = [
    'أنشد','أنشأ','أنشدني','أنشدنا','فأنشد','فأنشأ','ينشد',
    'الشاعر','شعره','شعرها','شعري','أبيات','قصيدة','وأنشد',
]
_MUB_MARKERS = [
    'رأيت','فرأيت','مررت','فمررت','لقيت','فلقيت',
    'وجدت','فوجدت','أبصرت','شهدت','رأيته',
]
_ISNAD_MARKERS = [
    'حدثنا','حدثني','أخبرنا','أخبرني','أنبأنا','أنبأني',
    'رويت عن','روينا عن','سمعت من','حدث عن','يرويه عن',
]

def _compile(lst):
    return re.compile('|'.join(re.escape(t) for t in lst))

RE_JUNUN    = _compile(_JUNUN_TERMS)
RE_DIALOGUE = _compile(_DIALOGUE_MARKERS)
RE_ISNAD    = _compile(_ISNAD_MARKERS)
RE_TOK      = re.compile(r'[\u0621-\u064A\u0671-\u06D3]+')

_SPEECH_RE  = re.compile(
    '|'.join(re.escape(t) for t in _DIALOGUE_MARKERS + _SHIR_MARKERS + _MUB_MARKERS)
)
WINDOW_CHARS = 80

def _junun_near_speech(text):
    for m in _SPEECH_RE.finditer(text):
        start = max(0, m.start() - WINDOW_CHARS)
        end   = min(len(text), m.end() + WINDOW_CHARS)
        if RE_JUNUN.search(text[start:end]):
            return True
    return False

def passes_gates(text):
    """Retourne True si le texte passe les 3 gates nécessaires."""
    if not RE_JUNUN.search(text):
        return False
    if not RE_DIALOGUE.search(text):
        return False
    if RE_ISNAD.search(text) and not RE_DIALOGUE.search(text):
        return False
    if not _junun_near_speech(text):
        return False
    return True

# ── Filtre isnad ──────────────────────────────────────────────────────────────
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
    if not text or len(text) < 20:
        return text
    toks = _RE_ARTOK.findall(text)
    if len(toks) < 4:
        return text
    cands = [(m.start(), m.group()) for m in _RE_MATN.finditer(text)]
    if not cands:
        return text
    best_cut, best_pos = None, -1
    for char_pos, marker in cands:
        ratio = char_pos / len(text)
        if ratio > 0.7 or ratio < 0.03:
            continue
        tb = _RE_ARTOK.findall(text[:char_pos])
        if len(tb) < 2:
            continue
        d_before = sum(_tok_score(t) for t in tb) / len(tb)
        ta = _RE_ARTOK.findall(text[char_pos:])
        win = 8
        d_after = sum(_tok_score(t) for t in ta[:win]) / win if len(ta) >= win else 0.0
        if d_before >= 0.25 and (d_before - d_after) > 0.05 and char_pos > best_pos:
            best_pos = char_pos
            best_cut = char_pos + len(marker)
    if best_cut is None:
        return text
    matn = text[best_cut:].strip()
    return matn if len(matn) >= 30 else text

# ── Parsing OpenITI mARkdown ──────────────────────────────────────────────────
RE_PAGE   = re.compile(r'PageV\d+P\d+')
RE_MS     = re.compile(r'\bms\d+\b')
RE_SPACES = re.compile(r'[ \t]+')

def count_ar(s):
    return sum(1 for c in s if unicodedata.category(c) == 'Lo' and '\u0600' <= c <= '\u06FF')

def clean_text(t):
    t = RE_PAGE.sub('', t)
    t = RE_MS.sub('', t)
    return RE_SPACES.sub(' ', t).strip()

def extract_units(filepath, min_ar=60, max_ar=3000):
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return []
    content_started = False
    units, current = [], []

    def flush():
        if current:
            text = clean_text(' '.join(current))
            n = count_ar(text)
            if min_ar <= n <= max_ar:
                units.append(text)

    for line in lines:
        line = line.rstrip('\n')
        if '#META#Header#End#' in line:
            content_started = True
            continue
        if not content_started:
            continue
        if line.startswith('~~'):
            current.append(line[2:].strip())
        elif line.startswith('# '):
            flush()
            rest = line[2:].strip()
            if line.startswith('# |') or rest.startswith('باب') or rest.startswith('فصل'):
                current = []
            else:
                current = [rest]
        else:
            flush()
            current = []
    flush()
    return units

# ── Chargement du modèle ──────────────────────────────────────────────────────
print(f"Chargement CAMeLBERT depuis {MODEL}/…")
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device : {device}")
    if device == 'cpu':
        print("  [!] CPU — active le GPU dans Runtime > Change runtime type")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    bert_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL))
    bert_model.to(device)
    bert_model.eval()
    print(f"  Modèle chargé ({sum(p.numel() for p in bert_model.parameters()):,} paramètres)")
except ImportError:
    print("ERREUR : pip install transformers torch")
    sys.exit(1)

def bert_score_batch(texts):
    all_probs = []
    for i in range(0, len(texts), args.batch):
        batch = texts[i:i+args.batch]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=args.maxlen, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = bert_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
    return all_probs

# ── Collecter les fichiers ────────────────────────────────────────────────────
all_files = []
for author_dir in sorted(CORPUS.iterdir()):
    if not author_dir.is_dir() or author_dir.name not in TARGET_AUTHORS:
        continue
    for work_dir in sorted(author_dir.iterdir()):
        if not work_dir.is_dir():
            continue
        for fp in sorted(work_dir.rglob('*-ara1')):
            all_files.append(fp)

print(f"\n{len(all_files)} fichiers à scanner | seuil BERT = {args.threshold:.0%}\n")

# ── Reprise ───────────────────────────────────────────────────────────────────
candidates, done_files = [], set()
n_total_units, n_gates_passed = 0, 0

if args.resume and PROGRESS.exists():
    state = json.load(open(PROGRESS, encoding='utf-8'))
    candidates    = state['candidates']
    done_files    = set(state['done_files'])
    n_total_units = state['n_total_units']
    n_gates_passed= state.get('n_gates_passed', 0)
    print(f"  Reprise : {len(done_files)}/{len(all_files)} fichiers "
          f"| {len(candidates)} candidats")

remaining = [fp for fp in all_files if fp.name not in done_files]

# ── Scan ──────────────────────────────────────────────────────────────────────
t0 = time.time()

for fi, fp in enumerate(remaining):
    units = extract_units(fp)
    if not units:
        done_files.add(fp.name)
        continue

    author = fp.parent.parent.name
    work   = fp.parent.name
    n_total_units += len(units)

    # Gate filtering
    passed = [(j, u) for j, u in enumerate(units) if passes_gates(u)]
    n_gates_passed += len(passed)

    # Filtre isnad + inférence BERT sur les survivants
    n_before = len(candidates)
    if passed:
        matns = [get_matn(u) for _, u in passed]
        probs = bert_score_batch(matns)

        for (j, unit), matn, prob in zip(passed, matns, probs):
            if prob >= args.threshold:
                candidates.append({
                    'author':      author,
                    'author_desc': TARGET_AUTHORS.get(author, author),
                    'work':        work,
                    'file':        fp.name,
                    'idx':         j,
                    'bert_score':  round(float(prob), 4),
                    'text':        unit[:600],
                    'matn':        matn[:400],
                })

    done_files.add(fp.name)
    n_found   = len(candidates) - n_before
    elapsed   = time.time() - t0
    done_pct  = len(done_files) / len(all_files) * 100
    rate      = len(done_files) / elapsed if elapsed > 0 else 1
    eta       = (len(remaining) - fi - 1) / rate if rate > 0 else 0

    # Affichage progression toutes les 10 files ou si nouveau candidat
    if (fi + 1) % 10 == 0 or n_found > 0:
        bar = '█' * int(done_pct / 5) + '░' * (20 - int(done_pct / 5))
        gate_rate = n_gates_passed / max(n_total_units, 1) * 100
        print(f"  [{bar}] {done_pct:5.1f}%  "
              f"unités={n_total_units:,}  gates={gate_rate:.1f}%  "
              f"candidats={len(candidates)}  "
              f"ETA {int(eta//60)}m{int(eta%60):02d}s")
        if n_found > 0:
            desc = TARGET_AUTHORS.get(author, author)
            print(f"    ↳ {desc} / {work}  (+{n_found})")
            for c in candidates[-n_found:][:2]:
                print(f"       [{c['bert_score']:.2f}] {c['matn'][:120]}…")

    # Sauvegarde progressive dans Drive
    if (fi + 1) % args.save_every == 0:
        with open(PROGRESS, 'w', encoding='utf-8') as f:
            json.dump({
                'done_files':    list(done_files),
                'candidates':    candidates,
                'n_total_units': n_total_units,
                'n_gates_passed':n_gates_passed,
                'threshold':     args.threshold,
            }, f, ensure_ascii=False)

# ── Résultats finaux ──────────────────────────────────────────────────────────
candidates.sort(key=lambda x: -x['bert_score'])

# Déduplique
seen, unique = set(), []
for c in candidates:
    key = c['text'][:80]
    if key not in seen:
        seen.add(key)
        unique.append(c)
candidates = unique

elapsed = time.time() - t0
print(f"\n{'─'*60}")
print(f"  Unités scannées  : {n_total_units:,}")
print(f"  Passent les gates : {n_gates_passed:,} ({n_gates_passed/max(n_total_units,1)*100:.1f}%)")
print(f"  Candidats BERT   : {len(candidates)}")
print(f"  Durée totale     : {int(elapsed//60)}m{int(elapsed%60):02d}s")
print()

for t in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]:
    n = sum(1 for c in candidates if c['bert_score'] >= t)
    print(f"    >= {t:.0%} : {n:4d} candidats")
print()

by_author = defaultdict(list)
for c in candidates:
    by_author[c['author']].append(c)

print("── Par auteur ──")
for auth, cs in sorted(by_author.items(), key=lambda x: -len(x[1])):
    mb = sum(c['bert_score'] for c in cs) / len(cs)
    print(f"  {len(cs):4d}  bert={mb:.2f}  {cs[0]['author_desc']}")
print()

print(f"── Top {args.top} candidats ──")
for i, c in enumerate(candidates[:args.top], 1):
    print(f"\n  {i:3d}. [bert={c['bert_score']:.3f}]  {c['author_desc']}")
    print(f"       {c['work']}")
    print(f"       {c['matn'][:200]}…")

# ── Sauvegarde finale ─────────────────────────────────────────────────────────
output = {
    'meta': {
        'threshold':       args.threshold,
        'n_units_scanned': n_total_units,
        'n_gates_passed':  n_gates_passed,
        'n_candidates':    len(candidates),
        'bert_model':      MODEL.name,
        'pipeline':        'gates_isnad_bert_direct',
    },
    'candidates': candidates,
}
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n  Résultats → {OUTPUT}")

# Nettoyage fichier de progression
if PROGRESS.exists():
    PROGRESS.unlink()
