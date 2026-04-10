"""
build_features.py
──────────────────
Extrait des features actantielles et entraîne une Régression Logistique
interprétable pour détecter le motif du maǧnūn ʿāqil.

Deux modes d'entraînement :
  --dataset dataset_raw.json   ← dataset élargi (575 pos + 3450 neg)
                                   positifs : Nīsābūrī complet + chapitres ʿuqalāʾ
                                   négatifs : hamqa, chroniques, hadith, poésie
  (défaut)                     ← akhbars annotés (score ≥ 6 vs ≤ 1)

Features binaires et numériques :
  F01  has_junun        — présence d'un terme de folie ou d'un fou connu
  F02  has_dialogue     — marqueurs de structure sual_jawab (première personne)
  F03  has_shir         — marqueurs de citation poétique
  F04  has_authority    — figure d'autorité (calife, vizir, émir...)
  F05  has_validation   — réception explicite de la parole (don, larmes, rire)
  F06  has_reversal     — marqueurs de renversement ou paradoxe
  F07  has_wasf         — marqueurs de texte définitionnel (signal négatif)
  F08  has_mubashara    — formule du témoin oculaire (رأيت مجنوناً)
  F09  junun_x_dialogue — co-occurrence F01 × F02 (cœur du motif)
  F10  junun_x_auth     — co-occurrence F01 × F04 (alibi devant l'autorité)
  F11  shir_alone       — shir sans dialogue ni mubashara (→ folie d'amour)
  F12  log_length       — log(nombre de tokens arabes) — proxy de forme
  F13  junun_density    — proportion de tokens junun dans le texte
  F14  junun_near_speech — junūn dans fenêtre de 80 chars autour d'un acte de parole

Sorties :
  scan/actantial_classifier.pkl  — modèle LR + scaler sérialisés
  scan/feature_report.json       — coefficients, AUC, exemples mal classés

Usage :
  python scan/build_features.py
  python scan/build_features.py --dataset dataset_raw.json
  python scan/build_features.py --dataset dataset_raw.json --cv 10
"""

import json, re, math, pathlib, sys, argparse
from collections import Counter
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

BASE   = pathlib.Path(__file__).parent.parent
ANN    = BASE / "approche_actantielle" / "actantial_annotations.json"
MODEL  = BASE / "approche_actantielle" / "actantial_model.json"
AKHBAR = BASE / "corpus" / "akhbar.json"
LEX    = BASE / "scan" / "actantial_lexicons.json"
CLF    = BASE / "scan" / "actantial_classifier.pkl"
REP    = BASE / "scan" / "feature_report.json"

# Parser anticipé pour --dataset (utilisé avant le chargement des données)
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--dataset', default=None)
_pre_args, _ = _pre.parse_known_args()

# ── Patterns binaires (expressions régulières sur texte arabe brut) ───────────
# Ces patterns sont des CONDITIONS NÉCESSAIRES ou FORTEMENT ASSOCIÉES au motif.
# On utilise des regex compilées pour performance sur 50 000+ textes.

# F01 — Termes de folie et noms propres de fous canoniques
_JUNUN_TERMS = [
    'مجنون','المجنون','مجنونا','مجانين','المجانين','مجنونة','المجنونة',
    'معتوه','المعتوه','معتوها','معتوهة','مدله','المدله',
    'هائم','الهائم','هائما','ممسوس','ممرور','مستهتر',
    'جنونه','جنونها','جنوني','جنونا','جنون','الجنون',
    'ذاهبالعقل','ذهبعقله','ذاهب','ذهب',
    # noms propres de fous (du modèle actantiel)
    'بهلول','بهلولا','سعدون','عليان','جعيفران','ريحانة',
    'سمنون','لقيط','حيون','حيونة','خلف','رياح',
]
RE_JUNUN = re.compile('|'.join(re.escape(t) for t in _JUNUN_TERMS))

# F02 — Dialogue : alternance je/vous (première personne active)
_DIALOGUE_MARKERS = [
    'قلت','فقلت','قلنا','سألت','فسألت','سألني','أجبت',
    'فقلتله','قيل له','قيل لي','فقلت له',
]
RE_DIALOGUE = re.compile('|'.join(re.escape(t) for t in _DIALOGUE_MARKERS))

# F03 — Poésie
_SHIR_MARKERS = [
    'أنشد','أنشأ','أنشدني','أنشدنا','فأنشد','فأنشأ','ينشد',
    'الشاعر','شعره','شعرها','شعري','أبيات','قصيدة','وأنشد',
]
RE_SHIR = re.compile('|'.join(re.escape(t) for t in _SHIR_MARKERS))

# F04 — Autorité (liste restreinte, discriminative)
_AUTH_MARKERS = [
    'الخليفة','أميرالمؤمنين','أمير المؤمنين','الرشيد','المأمون',
    'المتوكل','المعتصم','المهدي','المنصور','الهادي','المعتضد',
    'الوزير','الوالي','القاضي','السلطان','الملك',
]
RE_AUTH = re.compile('|'.join(re.escape(t) for t in _AUTH_MARKERS))

# F05 — Validation explicite de la parole
_VAL_MARKERS = [
    'فأمر','فأعطاه','فأعطى','فوهب','جائزة',
    'فضحك','فأعجبه','أعجبه','فاستحسن',
    'فبكى','فبكت','بكاء','دموعه',
    'فسكت','فصمت','فأطرق','تعجب','فتعجب',
]
RE_VAL = re.compile('|'.join(re.escape(t) for t in _VAL_MARKERS))

# F06 — Renversement / paradoxe lexical
_REV_MARKERS = [
    'لكن ','لكنه','لكنها','ولكن','بل ','وبل',
    'أعقل','أحكم','أصوب','أفضل','أعلم','أصدق',
    'كذب','أخطأ','غلط','فبان','فتبين',
    'فإذاهو','وإذاهو',
]
RE_REV = re.compile('|'.join(re.escape(t) for t in _REV_MARKERS))

# F07 — Texte définitionnel / wasf (signal NÉGATIF pour le motif)
_WASF_MARKERS = [
    'ومنها','والفعل منه','والاسم','ضروب','ضروب المجانين',
    'تقول العرب','ومن أمثالهم','ومنهم','فهو معتوه','فهو مجنون',
    'يقال له ذلك','يسمى','يعرف بـ',
]
RE_WASF = re.compile('|'.join(re.escape(t) for t in _WASF_MARKERS))

# F08 — Témoin oculaire
_MUB_MARKERS = [
    'رأيت','فرأيت','مررت','فمررت','لقيت','فلقيت',
    'وجدت','فوجدت','أبصرت','شهدت','رأيته',
]
RE_MUB = re.compile('|'.join(re.escape(t) for t in _MUB_MARKERS))

# Tokenisation arabe simple
RE_TOK = re.compile(r'[\u0621-\u064A\u0671-\u06D3]+')
JUNUN_SET = set(_JUNUN_TERMS)

# F14 — Fenêtre contextuelle : junūn à proximité immédiate d'un acte de parole
# Dans un vrai akhbar du motif, le fou EST le locuteur ou l'interlocuteur direct.
# Dans un faux positif (chronique, hadith), مجنون peut apparaître loin du dialogue.
WINDOW_CHARS = 80  # ~10-15 tokens arabes

def _junun_near_speech(text):
    """
    Retourne 1 si un terme junūn apparaît dans une fenêtre de WINDOW_CHARS
    caractères autour d'un marqueur de dialogue ou de parole poétique.
    """
    speech_re = re.compile(
        '|'.join(re.escape(t) for t in
                 _DIALOGUE_MARKERS + _SHIR_MARKERS + _MUB_MARKERS)
    )
    for m in speech_re.finditer(text):
        start = max(0, m.start() - WINDOW_CHARS)
        end   = min(len(text), m.end() + WINDOW_CHARS)
        if RE_JUNUN.search(text[start:end]):
            return 1
    return 0

FEATURE_NAMES = [
    'has_junun', 'has_dialogue', 'has_shir', 'has_authority',
    'has_validation', 'has_reversal', 'has_wasf', 'has_mubashara',
    'junun_x_dialogue', 'junun_x_authority',
    'shir_alone', 'log_length', 'junun_density',
    'junun_near_speech',   # F14
]

def extract_features(text):
    """Retourne un vecteur numpy de 14 features pour un texte arabe brut."""
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
    f09 = f01 * f02           # co-occurrence junun × dialogue
    f10 = f01 * f04           # co-occurrence junun × autorité
    f11 = f03 * (1 - f02) * (1 - f08)  # shir seul (sans dialogue ni témoin)
    f12 = math.log(n_tok + 1)
    # densité de tokens junun
    n_jun = sum(1 for t in tokens if t in JUNUN_SET)
    f13 = n_jun / n_tok
    f14 = _junun_near_speech(text)

    return np.array([f01, f02, f03, f04, f05, f06, f07, f08,
                     f09, f10, f11, f12, f13, f14], dtype=float)

# ── Chargement ────────────────────────────────────────────────────────────────
def _val(obj, *keys):
    for k in keys:
        if not isinstance(obj, dict): return None
        obj = obj.get(k)
    if obj is None: return None
    v = str(obj).strip().lower()
    return None if v in {'null','none','','absent','absente'} else v

print("Chargement…")

if _pre_args.dataset:
    # ── Mode dataset_raw.json (575 pos + 3450 neg, sources contrastives) ──────
    import sys as _sys
    _sys.path.insert(0, str(BASE))
    from isnad_filter import get_matn as _get_matn

    dataset_path = pathlib.Path(_pre_args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = BASE / dataset_path
    raw = json.load(open(dataset_path, encoding='utf-8'))

    annotated = []
    for e in raw:
        matn = _get_matn(e['text_ar'])
        if len(matn) < 30:
            matn = e['text_ar']
        annotated.append({
            'r': {}, 'num': e.get('id', 0),
            'score': e['label'] * 7,   # 0 ou 7 (pour compatibilité affichage)
            'matn': matn,
            'genre': e.get('source', ''),
        })

    print(f"  Dataset élargi : {dataset_path.name}")
    all_X   = np.array([extract_features(a['matn']) for a in annotated])
    all_y   = np.array([a['score'] for a in annotated])
    all_num = [a['num'] for a in annotated]

    X_train = all_X
    y_train = (all_y >= 7).astype(int)
    nums_train = all_num
    class_weight = 'balanced'

else:
    # ── Mode défaut : akhbars annotés (score ≥ 6 vs ≤ 1) ─────────────────────
    anns  = [r for r in json.load(open(ANN,   encoding='utf-8')) if '_error' not in r]
    model = json.load(open(MODEL, encoding='utf-8'))
    akh   = json.load(open(AKHBAR, encoding='utf-8'))['akhbar']

    scores_par_num = {int(k): v
        for k, v in model['scores_canonicite']['scores_par_num'].items()}

    def _get_matn_akh(a):
        segs = a.get('content', {}).get('segments', [])
        return ' '.join(s['text'] for s in segs
                        if s.get('type') != 'isnad' and s.get('text','').strip())

    matn_par_num = {a['num']: _get_matn_akh(a) for a in akh}

    annotated = []
    for r in anns:
        num   = r['_num']
        score = scores_par_num.get(num, 0)
        matn  = matn_par_num.get(num, '')
        annotated.append({'r': r, 'num': num, 'score': score, 'matn': matn,
                          'genre': r.get('_genre', '')})

    all_X   = np.array([extract_features(a['matn']) for a in annotated])
    all_y   = np.array([a['score'] for a in annotated])
    all_num = [a['num'] for a in annotated]

    train_mask = (all_y >= 6) | (all_y <= 1)
    X_train = all_X[train_mask]
    y_train = (all_y[train_mask] >= 6).astype(int)
    nums_train = [n for n, m in zip(all_num, train_mask) if m]
    class_weight = None

# ── Extraction des features sur tout le corpus ────────────────────────────────
print(f"Extraction des features sur {len(annotated)} textes…")
print(f"  Entraînement : {y_train.sum()} positifs + {(1-y_train).sum()} négatifs")

# ── Statistiques descriptives des features ────────────────────────────────────
print("\n── Statistiques par feature (positifs vs négatifs) ──")
pos_X = X_train[y_train == 1]
neg_X = X_train[y_train == 0]
print(f"  {'Feature':<22} {'Moy POS':>8} {'Moy NEG':>8} {'Diff':>8}")
print(f"  {'-'*50}")
for i, name in enumerate(FEATURE_NAMES):
    mp = pos_X[:, i].mean()
    mn = neg_X[:, i].mean()
    print(f"  {name:<22} {mp:8.3f} {mn:8.3f} {mp-mn:+8.3f}")
print()

# ── Entraînement Régression Logistique ───────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score, classification_report
    import pickle

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # ── Validation croisée stratifiée ────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv',      type=int,   default=5)
    parser.add_argument('--dataset', default=None,
                        help='Chemin vers dataset_raw.json (mode élargi)')
    args, _ = parser.parse_known_args()

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                             class_weight=class_weight)
    cv  = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    aucs = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring='roc_auc')
    print(f"── Validation croisée ({args.cv}-fold) ──")
    print(f"  AUC : {aucs.mean():.3f} ± {aucs.std():.3f}")
    print(f"  Par fold : {' '.join(f'{a:.3f}' for a in aucs)}")
    print()

    # ── Entraînement final ────────────────────────────────────────────────────
    clf.fit(X_train_s, y_train)
    y_pred_prob = clf.predict_proba(X_train_s)[:, 1]
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    print("── Rapport de classification (train) ──")
    print(classification_report(y_train, y_pred,
          target_names=['non-motif','motif'], digits=3))

    # ── Coefficients interprétables ───────────────────────────────────────────
    print("── Coefficients (importance des features) ──")
    coefs = list(zip(FEATURE_NAMES, clf.coef_[0]))
    coefs.sort(key=lambda x: -abs(x[1]))
    for name, c in coefs:
        bar = '█' * int(abs(c) * 3) if abs(c) > 0.1 else ''
        sign = '+' if c > 0 else '-'
        print(f"  {sign}{abs(c):.3f}  {name:<22}  {bar}")
    print()

    # ── Application sur tout le corpus (613 akhbars) ─────────────────────────
    X_all_s  = scaler.transform(all_X)
    probs_all = clf.predict_proba(X_all_s)[:, 1]

    print("── Score LR vs score canonique (corrélation) ──")
    for seuil_can in range(7, -1, -1):
        mask = all_y == seuil_can
        if mask.sum() == 0:
            continue
        mean_lr = probs_all[mask].mean()
        print(f"  Score canonique {seuil_can} (n={mask.sum():3d}) "
              f"→ P(motif) LR moyen = {mean_lr:.3f}")
    print()

    # ── Cas mal classés (analyse qualitative) ────────────────────────────────
    pos_thresh = 6   # score ≥ 6 = positif canonique (ou 7 en mode dataset)
    neg_thresh = 1   # score ≤ 1 = négatif clair (ou 0 en mode dataset)

    print("── Faux négatifs (positifs mal classés, P < 0.4) ──")
    for a, prob in zip(annotated, probs_all):
        if a['score'] >= pos_thresh and prob < 0.4:
            fn = _val(a['r'], 'junun_operateur', 'fonction')
            ac = _val(a['r'], 'parole_centrale', 'type_acte')
            genre = a.get('genre', '')
            print(f"  #{str(a['num']):>6} [{genre}] P={prob:.2f} "
                  f"fn={fn} acte={ac} | {a['matn'][:80]}…")

    print()
    print("── Faux positifs (négatifs sur-scorés, P > 0.6) ──")
    for a, prob in zip(annotated, probs_all):
        if a['score'] <= neg_thresh and prob > 0.6:
            fn = _val(a['r'], 'junun_operateur', 'fonction')
            ac = _val(a['r'], 'parole_centrale', 'type_acte')
            genre = a.get('genre', '')
            print(f"  #{str(a['num']):>6} [{genre}] P={prob:.2f} "
                  f"fn={fn} acte={ac} | {a['matn'][:80]}…")

    # ── Sauvegarde modèle ─────────────────────────────────────────────────────
    with open(CLF, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler,
                     'feature_names': FEATURE_NAMES}, f)
    print(f"\n  Modèle sauvegardé → {CLF}")

    # ── Sauvegarde rapport ────────────────────────────────────────────────────
    report = {
        'cv_auc': {'mean': float(aucs.mean()), 'std': float(aucs.std()),
                   'folds': [float(a) for a in aucs]},
        'coefficients': {name: float(c) for name, c in zip(FEATURE_NAMES, clf.coef_[0])},
        'intercept': float(clf.intercept_[0]),
        'feature_stats': {
            name: {
                'mean_pos': float(pos_X[:, i].mean()),
                'mean_neg': float(neg_X[:, i].mean()),
                'diff':     float(pos_X[:, i].mean() - neg_X[:, i].mean()),
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
        'scores_par_akhbar': {
            str(a['num']): {
                'lr_prob':       float(prob),
                'canon_score':   int(a['score']),
                'genre':         a['genre'],
                'junun_fonction': _val(a['r'],'junun_operateur','fonction'),
            }
            for a, prob in zip(annotated, probs_all)
        },
    }
    with open(REP, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Rapport sauvegardé → {REP}")

except ImportError:
    print("\n  scikit-learn non disponible.")
    print("  Installe avec : pip install scikit-learn")
    print("\n  Statistiques descriptives disponibles ci-dessus.")
    print("  Pour entraîner le modèle, installe scikit-learn et relance.")

    # Sans sklearn : afficher les statistiques brutes
    print("\n── Corrélation feature / score canonique (Pearson) ──")
    for i, name in enumerate(FEATURE_NAMES):
        x = all_X[:, i]
        y = all_y.astype(float)
        if x.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        bar = '█' * int(abs(r) * 20)
        sign = '+' if r > 0 else '-'
        print(f"  {sign}{abs(r):.3f}  {name:<22}  {bar}")
