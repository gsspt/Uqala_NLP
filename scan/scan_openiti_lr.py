"""
scan_openiti_lr.py
──────────────────
Stage 1 du pipeline LR + CAMeLBERT : Scanne openiti_corpus avec Régression Logistique.

Détecte les candidats au motif maǧnūn ʿāqil en utilisant les 14 features actantielles.
Les candidats LR sont sauvegardés pour le stage 2 (BERT scoring).

Pipeline :
  1. Extrait tous les akhbars du corpus (par structure isnad)
  2. Calcule 14 features actantielles (regex) pour chaque akhbar
  3. Applique le classificateur LR entraîné
  4. Sauvegarde les candidats ≥ seuil LR

Sortie :
  scan/openiti_lr_candidates.json — candidats LR triés par score

Usage :
  python scan/scan_openiti_lr.py
  python scan/scan_openiti_lr.py --threshold 0.7 --resume
"""

import json, re, pathlib, sys, argparse, pickle, unicodedata, time
from collections import Counter, defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE      = pathlib.Path(__file__).parent.parent
CORPUS    = BASE / "openiti_corpus" / "data"
CLF_PATH  = BASE / "scan" / "actantial_classifier.pkl"
OUT       = BASE / "scan" / "openiti_lr_candidates.json"
PROGRESS  = BASE / "scan" / "openiti_lr_progress.json"

# ── Features (identiques à score_openiti.py) ──────────────────────────────────
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
_ISNAD_MARKERS = [
    'حدثنا','حدثني','أخبرنا','أخبرني','أنبأنا','أنبأني',
    'رويت عن','روينا عن','سمعت من','حدث عن','يرويه عن',
]

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
RE_ISNAD    = _compile(_ISNAD_MARKERS)
RE_TOK      = re.compile(r'[\u0621-\u064A\u0671-\u06D3]+')
WINDOW_CHARS = 80

def _junun_near_speech(text):
    """F14: junūn dans une fenêtre de 80 caractères d'un marqueur de parole"""
    if not RE_JUNUN.search(text):
        return 0
    if not RE_DIALOGUE.search(text):
        return 0

    for m_speech in RE_DIALOGUE.finditer(text):
        pos_speech = m_speech.start()
        window_start = max(0, pos_speech - WINDOW_CHARS)
        window_end = min(len(text), pos_speech + WINDOW_CHARS)
        window = text[window_start:window_end]
        if RE_JUNUN.search(window):
            return 1
    return 0

def extract_features(text):
    """Retourne vecteur de 14 features"""
    f = np.zeros(14, dtype=np.float32)

    f[0] = 1.0 if RE_JUNUN.search(text) else 0.0
    f[1] = 1.0 if RE_DIALOGUE.search(text) else 0.0
    f[2] = 1.0 if RE_SHIR.search(text) else 0.0
    f[3] = 1.0 if RE_AUTH.search(text) else 0.0
    f[4] = 1.0 if RE_VAL.search(text) else 0.0
    f[5] = 1.0 if RE_REV.search(text) else 0.0
    f[6] = 1.0 if RE_WASF.search(text) else 0.0
    f[7] = 1.0 if RE_MUB.search(text) else 0.0

    # F8-F13 : densités lexicales
    toks = RE_TOK.findall(text)
    n_toks = len(toks)
    if n_toks > 0:
        f[8] = sum(1 for t in toks if t in _JUNUN_TERMS) / n_toks
        f[9] = sum(1 for t in toks if t in _DIALOGUE_MARKERS) / n_toks
        f[10] = sum(1 for t in toks if t in _AUTH_MARKERS) / n_toks
        f[11] = sum(1 for t in toks if t in _VAL_MARKERS) / n_toks
        f[12] = sum(1 for t in toks if t in _REV_MARKERS) / n_toks

    f[13] = float(_junun_near_speech(text))

    return f

# ── OpenITI parsing ───────────────────────────────────────────────────────────
RE_PAGE    = re.compile(r'PageV\d+P\d+')
RE_MS      = re.compile(r'\bms\d+\b')
RE_SPACES  = re.compile(r'[ \t]+')

def clean_text(t):
    t = RE_PAGE.sub('', t)
    t = RE_MS.sub('', t)
    return RE_SPACES.sub(' ', t).strip()

def extract_akhbars(filepath):
    """Extrait les unités narratives d'un fichier OpenITI"""
    def count_ar(s):
        return sum(1 for c in s if unicodedata.category(c) == 'Lo' and '\u0600' <= c <= '\u06FF')

    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except:
        return []

    content_started = False
    units, current = [], []

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
            if current:
                text = clean_text(' '.join(current))
                if 80 <= count_ar(text) <= 3000:
                    units.append(text)
            rest = line[2:].strip()
            if line.startswith('# |') or rest.startswith('باب') or rest.startswith('فصل'):
                current = []
            else:
                current = [rest]
        else:
            if current:
                text = clean_text(' '.join(current))
                if 80 <= count_ar(text) <= 3000:
                    units.append(text)
            current = []

    if current:
        text = clean_text(' '.join(current))
        if 80 <= count_ar(text) <= 3000:
            units.append(text)

    return units

# ── Scan ──────────────────────────────────────────────────────────────────────
def scan(corpus_dir, threshold=0.75, resume=False):
    """Scanne le corpus avec LR seul"""

    if not CLF_PATH.exists():
        print(f"ERREUR : classificateur LR introuvable → {CLF_PATH}")
        return

    print(f"  Chargement du classificateur LR…")
    with open(CLF_PATH, 'rb') as f:
        clf_data = pickle.load(f)
    clf = clf_data['clf']
    scaler = clf_data['scaler']

    all_files = sorted(corpus_dir.rglob('*-ara1'))
    n_files = len(all_files)
    print(f"  Fichiers OpenITI : {n_files}")

    # Reprise
    all_candidates = []
    total_akhbars = 0
    done_files = set()

    if resume and PROGRESS.exists():
        state = json.load(open(PROGRESS, encoding='utf-8'))
        all_candidates = state['candidates']
        total_akhbars = state['total_akhbars']
        done_files = set(state['done_files'])
        print(f"  Reprise : {len(done_files)}/{n_files} fichiers traités")

    remaining = [fp for fp in all_files if fp.name not in done_files]
    print(f"  À scanner : {len(remaining)}")
    print(f"  Seuil LR : {threshold:.0%}\n")

    t_start = time.time()
    SAVE_EVERY = 20

    for fi, fp in enumerate(remaining):
        akhbars = extract_akhbars(fp)
        if not akhbars:
            done_files.add(fp.name)
            continue

        # Extraire features et scorer
        features = np.array([extract_features(a) for a in akhbars])
        features_scaled = scaler.transform(features)
        probs = clf.predict_proba(features_scaled)[:, 1]
        total_akhbars += len(akhbars)
        done_files.add(fp.name)

        # Garder les candidats
        n_before = len(all_candidates)
        for j, (raw, prob) in enumerate(zip(akhbars, probs)):
            if prob >= threshold:
                all_candidates.append({
                    'file':       fp.name,
                    'author':     fp.parent.parent.name,
                    'work':       fp.parent.name,
                    'idx_in_file': j,
                    'lr_score':   float(prob),
                    'text':       raw[:500],
                })
        n_found = len(all_candidates) - n_before

        # Sauvegarde intermédiaire
        if (fi + 1) % SAVE_EVERY == 0:
            json.dump({
                'threshold': threshold,
                'total_akhbars': total_akhbars,
                'done_files': list(done_files),
                'candidates': all_candidates,
            }, open(PROGRESS, 'w', encoding='utf-8'))

        # Progression
        elapsed = time.time() - t_start
        if (fi + 1) % 50 == 0 or n_found > 0:
            pct = len(done_files) / n_files * 100
            rate = (fi + 1) / elapsed if elapsed > 0 else 0
            eta_s = (len(remaining) - fi - 1) / rate if rate > 0 else 0
            eta_min = int(eta_s // 60)
            eta_sec = int(eta_s % 60)
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}%  {len(done_files)}/{n_files} fichiers"
                  f"  |  {total_akhbars:,} akhbars  |  {len(all_candidates)} candidats"
                  f"  |  ETA {eta_min}m{eta_sec:02d}s")
            if n_found > 0:
                top = sorted(all_candidates[-n_found:], key=lambda x: -x['lr_score'])
                for c in top[:2]:
                    print(f"    ↳ [{c['lr_score']:.2f}] {c['author']} / {c['work']}")

    # Finition
    json.dump({
        'threshold': threshold,
        'total_akhbars': total_akhbars,
        'done_files': list(done_files),
        'candidates': all_candidates,
    }, open(PROGRESS, 'w', encoding='utf-8'))

    all_candidates.sort(key=lambda x: -x['lr_score'])

    print(f"\n  ── Résultats LR ──")
    print(f"  Akhbars scannés : {total_akhbars:,}")
    print(f"  Candidats ≥ {threshold:.0%} : {len(all_candidates)}")

    for t in [0.95, 0.90, 0.80, 0.70, 0.60]:
        n = sum(1 for c in all_candidates if c['lr_score'] >= t)
        print(f"    ≥ {t:.0%} : {n}")

    # Top auteurs
    authors = Counter(c['author'] for c in all_candidates)
    print(f"\n  Top auteurs:")
    for auth, cnt in authors.most_common(10):
        print(f"    {cnt:4d}  {auth}")

    json.dump({
        'threshold': threshold,
        'total_akhbars': total_akhbars,
        'n_candidates': len(all_candidates),
        'candidates': all_candidates,
    }, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f"\n  Candidats LR : {OUT}")
    if PROGRESS.exists():
        PROGRESS.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if not CORPUS.exists():
        print(f"ERREUR : corpus introuvable → {CORPUS}")
        exit(1)

    scan(CORPUS, threshold=args.threshold, resume=args.resume)
