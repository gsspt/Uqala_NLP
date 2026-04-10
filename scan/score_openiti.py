"""
score_openiti.py
─────────────────
Applique le classificateur actantiel (Régression Logistique sur 14 features)
aux textes du corpus openiti_targeted/ pour détecter le motif du maǧnūn ʿāqil.

Pipeline :
  1. Chaque fichier OpenITI est découpé en unités narratives (~akhbars)
  2. Chaque unité reçoit un vecteur de 14 features actantielles (regex, pas LLM)
  3. Le classificateur LR donne P(motif ∈ [0,1]) pour chaque unité
  4. Les candidats au-dessus du seuil sont sauvegardés et classés

Optionnel (si rank_bm25 installé) :
  Un second score BM25 est calculé en comparant chaque candidat
  aux 141 akhbars canoniques de Nīsābūrī (similarité textuelle).

Sorties :
  scan/openiti_candidates.json  — candidats avec scores, triés par LR
  scan/openiti_progress.json    — état de progression (pour --resume)
  scan/openiti_report.txt       — rapport lisible par auteur

Usage :
  python scan/score_openiti.py
  python scan/score_openiti.py --threshold 0.70 --resume
  python scan/score_openiti.py --authors 0597IbnJawzi 0414AbuHayyanTawhidi
"""

import json, re, math, time, pathlib, sys, argparse, pickle, unicodedata
from collections import Counter, defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE      = pathlib.Path(__file__).parent.parent
CORPUS    = BASE / "openiti_targeted"
CLF_PATH  = BASE / "scan" / "actantial_classifier.pkl"
BERT_DIR  = BASE / "camelbert_majnun"
ANN       = BASE / "approche_actantielle" / "actantial_annotations.json"
MODEL_J   = BASE / "approche_actantielle" / "actantial_model.json"
AKHBAR_J  = BASE / "corpus" / "akhbar.json"
OUT       = BASE / "scan" / "openiti_candidates.json"
PROGRESS  = BASE / "scan" / "openiti_progress.json"
REPORT    = BASE / "scan" / "openiti_report.txt"

# ── Chargement optionnel CAMeLBERT ────────────────────────────────────────────
def load_camelbert():
    """
    Charge le modèle CAMeLBERT fine-tuné si disponible.
    Retourne (tokenizer, model) ou (None, None).
    """
    if not BERT_DIR.exists():
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        print(f"  CAMeLBERT chargé depuis {BERT_DIR}/")
        tok   = AutoTokenizer.from_pretrained(str(BERT_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
        model.eval()
        return tok, model
    except Exception as e:
        print(f"  [!] CAMeLBERT non disponible : {e}")
        return None, None

def bert_score_batch(tokenizer, model, texts, max_length=256, batch_size=32):
    """
    Retourne un tableau de P(motif) pour chaque texte.
    Traitement par batch pour la mémoire.
    """
    import torch
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True,
            max_length=max_length, return_tensors='pt'
        )
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        all_probs.extend(probs.tolist())
    return all_probs

# ── Features (identiques à build_features.py) ─────────────────────────────────
_JUNUN_TERMS = [
    'مجنون','المجنون','مجنونا','مجانين','المجانين','مجنونة','المجنونة',
    'معتوه','المعتوه','معتوها','معتوهة','مدله','المدله',
    'هائم','الهائم','هائما','ممسوس','ممرور','مستهتر',
    'جنونه','جنونها','جنوني','جنونا','جنون','الجنون',
    'ذاهبالعقل','ذهبعقله','ذاهب','ذهب',
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
_AUTH_MARKERS = [
    'الخليفة','أميرالمؤمنين','أمير المؤمنين','الرشيد','المأمون',
    'المتوكل','المعتصم','المهدي','المنصور','الهادي','المعتضد',
    'الوزير','الوالي','القاضي','السلطان','الملك',
]
_VAL_MARKERS = [
    'فأمر','فأعطاه','فأعطى','فوهب','جائزة',
    'فضحك','فأعجبه','أعجبه','فاستحسن',
    'فبكى','فبكت','بكاء','دموعه',
    'فسكت','فصمت','فأطرق','تعجب','فتعجب',
]
_REV_MARKERS = [
    'لكن ','لكنه','لكنها','ولكن','بل ','وبل',
    'أعقل','أحكم','أصوب','أفضل','أعلم','أصدق',
    'كذب','أخطأ','غلط','فبان','فتبين',
    'فإذاهو','وإذاهو',
]
_WASF_MARKERS = [
    'ومنها','والفعل منه','والاسم','ضروب','ضروب المجانين',
    'تقول العرب','ومن أمثالهم','ومنهم','فهو معتوه','فهو مجنون',
    'يقال له ذلك','يسمى','يعرف بـ',
]
_MUB_MARKERS = [
    'رأيت','فرأيت','مررت','فمررت','لقيت','فلقيت',
    'وجدت','فوجدت','أبصرت','شهدت','رأيته',
]
# Marqueurs d'isnad — condition de rejet (chaîne de transmission, pas un récit)
_ISNAD_MARKERS = [
    'حدثنا','حدثني','أخبرنا','أخبرني','أنبأنا','أنبأني',
    'رويت عن','روينا عن','سمعت من','حدث عن','يرويه عن',
]
RE_ISNAD = re.compile('|'.join(re.escape(t) for t in _ISNAD_MARKERS))

def _compile(lst):
    return re.compile('|'.join(re.escape(t) for t in lst))

RE_JUNUN    = _compile(_JUNUN_TERMS)
RE_DIALOGUE = _compile(_DIALOGUE_MARKERS)
RE_SHIR     = _compile(_SHIR_MARKERS)
RE_AUTH     = _compile(_AUTH_MARKERS)
RE_VAL      = _compile(_VAL_MARKERS)
RE_REV      = _compile(_REV_MARKERS)
RE_WASF     = _compile(_WASF_MARKERS)
RE_MUB      = _compile(_MUB_MARKERS)
RE_TOK      = re.compile(r'[\u0621-\u064A\u0671-\u06D3]+')
JUNUN_SET   = set(_JUNUN_TERMS)

# F14 — Fenêtre contextuelle : junūn à proximité d'un acte de parole
WINDOW_CHARS = 80
_SPEECH_RE = re.compile(
    '|'.join(re.escape(t) for t in
             _DIALOGUE_MARKERS + _SHIR_MARKERS + _MUB_MARKERS)
)

def _junun_near_speech(text):
    for m in _SPEECH_RE.finditer(text):
        start = max(0, m.start() - WINDOW_CHARS)
        end   = min(len(text), m.end() + WINDOW_CHARS)
        if RE_JUNUN.search(text[start:end]):
            return 1
    return 0

FEATURE_NAMES = [
    'has_junun','has_dialogue','has_shir','has_authority',
    'has_validation','has_reversal','has_wasf','has_mubashara',
    'junun_x_dialogue','junun_x_authority',
    'shir_alone','log_length','junun_density',
    'junun_near_speech',  # F14
]

def extract_features(text):
    tokens = RE_TOK.findall(text)
    n_tok  = max(len(tokens), 1)
    f01 = int(bool(RE_JUNUN.search(text)))
    f02 = int(bool(RE_DIALOGUE.search(text)))
    f03 = int(bool(RE_SHIR.search(text)))
    f04 = int(bool(RE_AUTH.search(text)))
    f05 = int(bool(RE_VAL.search(text)))
    f06 = int(bool(RE_REV.search(text)))
    f07 = int(bool(RE_WASF.search(text)))
    f08 = int(bool(RE_MUB.search(text)))
    f09 = f01 * f02
    f10 = f01 * f04
    f11 = f03 * (1 - f02) * (1 - f08)
    f12 = math.log(n_tok + 1)
    n_jun = sum(1 for t in tokens if t in JUNUN_SET)
    f13 = n_jun / n_tok
    f14 = _junun_near_speech(text)
    return np.array([f01,f02,f03,f04,f05,f06,f07,f08,
                     f09,f10,f11,f12,f13,f14], dtype=float)

# ── Extraction OpenITI ────────────────────────────────────────────────────────
RE_PAGE   = re.compile(r'PageV\d+P\d+')
RE_MS     = re.compile(r'\bms\d+\b')
RE_SPACES = re.compile(r'[ \t]+')

def count_ar(s):
    return sum(1 for c in s
               if unicodedata.category(c) == 'Lo' and '\u0600' <= c <= '\u06FF')

def clean_text(t):
    t = RE_PAGE.sub('', t)
    t = RE_MS.sub('', t)
    return RE_SPACES.sub(' ', t).strip()

def extract_units(filepath, min_ar=60, max_ar=3000):
    """
    Découpe un fichier OpenITI mARkdown en unités narratives.
    Retourne une liste de chaînes arabes nettoyées.
    """
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

# ── BM25 optionnel ────────────────────────────────────────────────────────────
def build_bm25_index(canonical_texts):
    """Construit un index BM25 sur les 141 akhbars canoniques (si rank_bm25 dispo)."""
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [RE_TOK.findall(t) for t in canonical_texts]
        return BM25Okapi(tokenized)
    except ImportError:
        return None

def bm25_score(index, text):
    if index is None:
        return None
    toks = RE_TOK.findall(text)
    scores = index.get_scores(toks)
    return float(scores.max())

# ── Sauvegarde ────────────────────────────────────────────────────────────────
def save_progress(candidates, done_files, n_total_units, threshold):
    with open(PROGRESS, 'w', encoding='utf-8') as f:
        json.dump({'threshold': threshold, 'done_files': list(done_files),
                   'n_total_units': n_total_units, 'n_candidates': len(candidates),
                   'candidates': candidates}, f, ensure_ascii=False)

# ── Auteurs ciblés (depuis scan_targeted.py) ─────────────────────────────────
TARGET_AUTHORS = {
    "0255Jahiz":              "al-Jāḥiẓ",
    "0276IbnQutaybaDinawari": "Ibn Qutayba",
    "0328IbnCabdRabbih":      "Ibn ʿAbd Rabbih (ʿIqd al-Farīd)",
    "0414AbuHayyanTawhidi":   "Abū Ḥayyān al-Tawḥīdī",
    "0850ShihabDinIbshihi":   "al-Ibshīhī (Mustaṭraf)",
    "0412Sulami":             "al-Sulamī",
    "0465IbnHawazinQushayri": "al-Qushayrī",
    "0310Tabari":             "al-Ṭabarī",
    "0463KhatibBaghdadi":     "al-Khaṭīb al-Baghdādī",
    "0571IbnCasakir":         "Ibn ʿAsākir",
    "0597IbnJawzi":           "Ibn al-Jawzī",
    "0626YaqutHamawi":        "Yāqūt al-Ḥamawī",
    "0681IbnKhallikan":       "Ibn Khallikān",
    "0733Nuwayri":            "al-Nuwayrī",
    "0748Dhahabi":            "al-Dhahabī",
    "0360Tabarani":           "al-Ṭabarānī",
    "0542IbnBassamShantarini":"Ibn Bassām",
    "0279Baladhuri":          "al-Balādhurī",
}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Seuil LR pour inclure un candidat (défaut: 0.75)')
    parser.add_argument('--bert-threshold', type=float, default=0.5,
                        help='Seuil CAMeLBERT sur les survivants LR (défaut: 0.5)')
    parser.add_argument('--no-bert',  action='store_true',
                        help='Désactiver CAMeLBERT même si disponible')
    parser.add_argument('--resume',   action='store_true',
                        help='Reprendre depuis la progression sauvegardée')
    parser.add_argument('--authors',  nargs='+', default=None,
                        help='Restreindre à ces auteurs (ex: 0597IbnJawzi 0412Sulami)')
    parser.add_argument('--top',      type=int, default=30,
                        help='Nombre de candidats à afficher (défaut: 30)')
    args = parser.parse_args()

    # ── Charger le classificateur LR ─────────────────────────────────────────
    if not CLF_PATH.exists():
        print(f"ERREUR : classificateur introuvable → {CLF_PATH}")
        print("         Lance d'abord : python scan/build_features.py --dataset dataset_raw.json")
        sys.exit(1)

    with open(CLF_PATH, 'rb') as f:
        bundle = pickle.load(f)
    clf     = bundle['clf']
    scaler  = bundle['scaler']
    print(f"Classificateur LR chargé ({len(FEATURE_NAMES)} features)")

    # ── Charger CAMeLBERT (étage 3, optionnel) ───────────────────────────────
    bert_tok, bert_model = (None, None) if args.no_bert else load_camelbert()
    use_bert = bert_tok is not None
    if use_bert:
        print(f"  Seuil CAMeLBERT : {args.bert_threshold:.0%}")
    else:
        print(f"  CAMeLBERT : inactif (lance finetune_camelbert.py pour l'activer)")

    # ── Charger les akhbars canoniques pour BM25 ────────────────────────────
    canonical_nums = json.load(open(MODEL_J, encoding='utf-8'))
    canonical_nums = canonical_nums['scores_canonicite']['noyau_canonique']
    akh_data       = json.load(open(AKHBAR_J, encoding='utf-8'))['akhbar']

    def get_matn(a):
        segs = a.get('content', {}).get('segments', [])
        return ' '.join(s['text'] for s in segs
                        if s.get('type') != 'isnad' and s.get('text','').strip())

    canonical_texts = [get_matn(a) for a in akh_data if a['num'] in set(canonical_nums)]
    bm25 = build_bm25_index(canonical_texts)
    bm25_label = "BM25 actif" if bm25 else "BM25 indisponible (pip install rank-bm25)"
    print(f"Index BM25 : {len(canonical_texts)} akhbars canoniques  [{bm25_label}]")

    # ── Collecter les fichiers ────────────────────────────────────────────────
    authors_filter = set(args.authors) if args.authors else None
    all_files = []
    for author_dir in sorted(CORPUS.iterdir()):
        if not author_dir.is_dir():
            continue
        if authors_filter and author_dir.name not in authors_filter:
            continue
        for work_dir in sorted(author_dir.iterdir()):
            if not work_dir.is_dir():
                continue
            for fp in sorted(work_dir.rglob('*-ara1')):
                all_files.append(fp)

    if not all_files:
        print(f"ERREUR : aucun fichier trouvé dans {CORPUS}")
        print("         Vérifie la structure openiti_targeted/")
        sys.exit(1)

    print(f"\n{len(all_files)} fichiers à traiter | seuil = {args.threshold:.0%}\n")

    # ── Reprise ───────────────────────────────────────────────────────────────
    candidates, done_files, n_total = [], set(), 0
    if args.resume and PROGRESS.exists():
        state = json.load(open(PROGRESS, encoding='utf-8'))
        if abs(state['threshold'] - args.threshold) > 0.001:
            print(f"  [!] Seuil différent ({state['threshold']:.0%}) — relancer sans --resume")
            sys.exit(1)
        candidates    = state['candidates']
        done_files    = set(state['done_files'])
        n_total       = state['n_total_units']
        print(f"  Reprise : {len(done_files)}/{len(all_files)} fichiers "
              f"| {len(candidates)} candidats existants")

    remaining = [fp for fp in all_files if fp.name not in done_files]
    t0 = time.time()
    SAVE_EVERY = 15  # fichiers entre chaque sauvegarde

    # ── Scan ─────────────────────────────────────────────────────────────────
    for fi, fp in enumerate(remaining):
        units = extract_units(fp)
        if not units:
            done_files.add(fp.name)
            continue

        # Features + proba LR sur tout le fichier en une passe numpy
        X = np.array([extract_features(u) for u in units])
        X_s = scaler.transform(X)
        probs = clf.predict_proba(X_s)[:, 1]

        n_total += len(units)
        done_files.add(fp.name)

        author = fp.parent.parent.name
        work   = fp.parent.name

        # ── Étage 1+2 : gates + LR ───────────────────────────────────────────
        lr_passed = []   # (j, unit, prob, feat_vec)
        for j, (unit, prob) in enumerate(zip(units, probs)):
            if not RE_JUNUN.search(unit):
                continue
            if not RE_DIALOGUE.search(unit):
                continue
            if RE_ISNAD.search(unit) and not RE_DIALOGUE.search(unit):
                continue
            if X[j][13] == 0:   # F14 = junun_near_speech
                continue
            if prob >= args.threshold:
                lr_passed.append((j, unit, prob, X[j].tolist()))

        # ── Étage 3 : CAMeLBERT sur les survivants LR ────────────────────────
        if use_bert and lr_passed:
            bert_texts = [u for _, u, _, _ in lr_passed]
            bert_probs = bert_score_batch(bert_tok, bert_model, bert_texts)
        else:
            bert_probs = [None] * len(lr_passed)

        n_before = len(candidates)
        for (j, unit, prob, feat_vec), bp in zip(lr_passed, bert_probs):
            # Filtrage BERT si disponible
            if bp is not None and bp < args.bert_threshold:
                continue
            active = [FEATURE_NAMES[k] for k, v in enumerate(feat_vec[:8]) if v > 0]
            entry = {
                'author':      author,
                'author_desc': TARGET_AUTHORS.get(author, author),
                'work':        work,
                'file':        fp.name,
                'idx':         j,
                'lr_score':    round(float(prob), 4),
                'features':    active,
                'text':        unit[:600],
            }
            if bp is not None:
                entry['bert_score'] = round(float(bp), 4)
            if bm25:
                entry['bm25_score'] = round(bm25_score(bm25, unit), 3)
            candidates.append(entry)

        n_found = len(candidates) - n_before

        if (fi + 1) % SAVE_EVERY == 0:
            save_progress(candidates, done_files, n_total, args.threshold)

        elapsed = time.time() - t0
        rate    = (fi + 1) / elapsed if elapsed > 0 else 0
        eta_s   = (len(remaining) - fi - 1) / rate if rate > 0 else 0
        done_pct = (len(done_files)) / len(all_files) * 100

        if (fi + 1) % 10 == 0 or n_found > 0:
            bar = '█' * int(done_pct / 5) + '░' * (20 - int(done_pct / 5))
            print(f"  [{bar}] {done_pct:5.1f}%  {len(units):4d} unités"
                  f"  {n_total:6,} total  {len(candidates):4d} candidats"
                  f"  ETA {int(eta_s//60)}m{int(eta_s%60):02d}s")
            if n_found > 0:
                top_new = sorted(candidates[-n_found:], key=lambda x: -x['lr_score'])
                desc = TARGET_AUTHORS.get(author, author)
                print(f"    ↳ {desc} / {work}")
                for c in top_new[:2]:
                    preview = c['text'].replace('\n',' ')[30:160]
                    print(f"       [{c['lr_score']:.2f}] {preview}…")

    # ── Résultats finaux ──────────────────────────────────────────────────────
    save_progress(candidates, done_files, n_total, args.threshold)

    # Trier par LR score
    candidates.sort(key=lambda x: -x['lr_score'])

    # Déduplique (même texte dans différents fichiers)
    seen, unique = set(), []
    for c in candidates:
        key = c['text'][:80]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique

    print(f"\n{'─'*60}")
    print(f"  Unités scannées : {n_total:,}")
    print(f"  Candidats ≥{args.threshold:.0%} : {len(candidates)}")
    print(f"  Taux             : {len(candidates)/max(n_total,1)*100:.2f}%")
    print()

    # Distribution par seuil
    for t in [0.95, 0.90, 0.80, 0.70, 0.65]:
        n = sum(1 for c in candidates if c['lr_score'] >= t)
        print(f"    ≥ {t:.0%} : {n:4d} candidats")
    print()

    # Par auteur
    by_author = defaultdict(list)
    for c in candidates:
        by_author[c['author']].append(c)

    print("── Par auteur (candidats ≥ 0.70) ──")
    rows = [(auth, [c for c in cs if c['lr_score'] >= 0.70])
            for auth, cs in by_author.items()]
    rows.sort(key=lambda x: -len(x[1]))
    for auth, cs in rows:
        if not cs:
            continue
        desc = TARGET_AUTHORS.get(auth, auth)
        mean = sum(c['lr_score'] for c in cs) / len(cs)
        print(f"  {len(cs):4d}  (moy {mean:.2f})  {desc}")
    print()

    # Features les plus fréquentes parmi les candidats
    feat_cnt = Counter(f for c in candidates for f in c['features'])
    print("── Features actives dans les candidats ──")
    for feat, cnt in feat_cnt.most_common():
        pct = cnt / max(len(candidates), 1) * 100
        print(f"  {cnt:4d} ({pct:5.1f}%)  {feat}")
    print()

    # Top candidats
    print(f"── Top {args.top} candidats ──")
    for i, c in enumerate(candidates[:args.top], 1):
        desc = c['author_desc']
        feats = ', '.join(c['features'])
        text  = c['text'].replace('\n', ' ')[20:200]
        bm   = f"  bm25={c['bm25_score']:.1f}" if 'bm25_score' in c else ''
        bert = f"  bert={c['bert_score']:.3f}" if 'bert_score' in c else ''
        print(f"\n  {i:3d}. [lr={c['lr_score']:.3f}{bert}{bm}]  {desc}")
        print(f"       {c['work']}")
        print(f"       features : {feats}")
        print(f"       {text}…")

    # ── Sauvegarde finale ─────────────────────────────────────────────────────
    output = {
        'meta': {
            'threshold':      args.threshold,
            'n_units_scanned':n_total,
            'n_candidates':   len(candidates),
            'taux_pct':       round(len(candidates)/max(n_total,1)*100, 3),
        },
        'candidates': candidates,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Résultats → {OUT}")

    # Rapport texte
    with open(REPORT, 'w', encoding='utf-8') as rp:
        rp.write(f"Rapport score_openiti.py — seuil {args.threshold:.0%}\n")
        rp.write(f"Unités scannées : {n_total:,} | Candidats : {len(candidates)}\n\n")
        rp.write("=== Par auteur ===\n")
        for auth, cs in sorted(by_author.items(), key=lambda x: -len(x[1])):
            desc = TARGET_AUTHORS.get(auth, auth)
            rp.write(f"\n{desc} ({len(cs)} candidats)\n")
            for c in sorted(cs, key=lambda x: -x['lr_score'])[:10]:
                rp.write(f"  [{c['lr_score']:.3f}] {c['text'][:200]}\n")
        rp.write("\n=== Top 50 tous auteurs ===\n")
        for c in candidates[:50]:
            rp.write(f"\n[{c['lr_score']:.3f}] {c['author_desc']}\n")
            rp.write(f"{c['text'][:300]}\n")
    print(f"  Rapport → {REPORT}")

    # Nettoyage fichier de progression
    if PROGRESS.exists():
        PROGRESS.unlink()

if __name__ == '__main__':
    main()
