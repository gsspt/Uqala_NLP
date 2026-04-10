#!/usr/bin/env python3
"""
build_features_50.py
────────────────────
Entraîne une Régression Logistique améliorée avec 50 features lexicales.

Remplace build_features.py (14 features) par une version 50 features.
Utilise extract_features_openiti.py pour les 50 features détaillées.

Pipeline :
  1. Charge dataset_raw.json (positifs : Nisaburi 161-612, négatifs : corpus enrichis)
  2. Extrait 50 features lexicales par regex
  3. Entraîne LR avec validation croisée
  4. Sauvegarde classifier + rapport

Usage :
  python scan/build_features_50.py
  python scan/build_features_50.py --cv 10 --test_split 0.2
"""

import json, re, sys, pathlib, argparse, pickle
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

BASE         = pathlib.Path(__file__).parent.parent
DATASET      = BASE / "dataset_raw.json"
CLF_PATH     = BASE / "scan" / "lr_classifier_50features.pkl"
REPORT_PATH  = BASE / "scan" / "lr_report_50features.json"

# ══════════════════════════════════════════════════════════════════════════════
# FEATURES 50 LEXICALES (extraites de extract_features_openiti.py)
# ══════════════════════════════════════════════════════════════════════════════

# Racines et termes pour junūn (15 features)
JUNUN_TERMS = [
    'مجنون','المجنون','مجنونا','مجانين','المجانين','مجنونة','المجنونة',
    'معتوه','المعتوه','معتوها','معتوهة','مدله','المدله',
    'هائم','الهائم','هائما','ممسوس','ممرور','مستهتر',
    'جنونه','جنونها','جنوني','جنونا','جنون','الجنون',
    'ذاهبالعقل','ذهبعقله','ذاهب','ذهب',
]

FAMOUS_FOOLS = [
    'بهلول','بهلولا','سعدون','عليان','جعيفران','ريحانة',
    'سمنون','لقيط','حيون','حيونة','خلف','رياح',
]

# Racines aql (8 features)
AQL_TERMS = [
    'عاقل','العاقل','عقل','العقل','عقلاء','العقلاء','عقلاؤهم',
    'عقول','أعقل','معقول','عقلانية','حكمة','حكيم','عقلائي',
]

# Racines hikma (5 features)
HIKMA_TERMS = [
    'حكمة','حكيم','حكماء','الحكمة','الحكيم','الحكماء','حكمته',
    'حكمهم','حكيم','أحكم',
]

# Dialogue et qala (12 features)
QALA_VARIANTS = [
    'قلت','فقلت','قلنا','قالوا','قال','قالت','قالا',
    'سألت','فسألت','سألني','أجبت','أجاب',
    'قيل له','قيل لي','فقلت له',
]

FIRST_PERSON = [
    'فعلت','رأيت','مررت','لقيت','وجدت','شهدت','سمعت','حدثني',
    'أخبرني','أنبأني','حدثنا','أخبرنا','أنبأنا',
]

QUESTIONS = [
    'كيف','كيفية','ماذا','ما','متى','أين','هل','من','لماذا',
]

# Validation/réaction (8 features)
VALIDATION = {
    'laugh': ['ضحك','ضحكت','ضحكوا','فضحك','فضحكوا'],
    'gift': ['أعطى','أعطاه','وهب','جائزة','أمر','فأمر'],
    'cry': ['بكى','بكت','بكاء','دموع','دموعه'],
    'silence': ['صمت','سكت','أطرق','طرق'],
}

# Contraste (5 features)
CONTRAST = {
    'opposition': ['لكن','لكنه','لكنها','بل','وبل','ولكن'],
    'correction': ['كذب','أخطأ','غلط','فبان','فتبين','أصح'],
    'revelation': ['فإذا','وإذا','فتبين','فتبينت'],
    'surprise_formula': ['ما أجمل','ما أعظم','ما أحسن','يا إلهي'],
}

# Autorité (4 features)
AUTHORITY = [
    'الخليفة','أميرالمؤمنين','أمير المؤمنين','الرشيد','المأمون',
    'المتوكل','المعتصم','المهدي','المنصور','الهادي','المعتضد',
    'الوزير','الوالي','القاضي','السلطان','الملك',
]

# Poésie (3 features)
SHIR_MARKERS = [
    'أنشد','أنشأ','أنشدني','أنشدنا','فأنشد','فأنشأ','ينشد',
    'الشاعر','شعره','شعرها','شعري','أبيات','قصيدة','وأنشد',
]

# Spatial (3 features)
SPATIAL = ['في','على','عند','بعد','أمام','خلف','حول','داخل','خارج']

def has_junun_filtered(text):
    """Détecte junūn, filtre les faux positifs"""
    if not any(t in text for t in JUNUN_TERMS):
        return False
    # Exclure الجنة (paradis), السجن (prison), الجن (djinns)
    if any(fp in text for fp in ['الجنة ', ' الجنة', 'الجن ', 'السجن ']):
        return False
    return True

def has_aql_filtered(text):
    """Détecte aql, filtre les faux positifs"""
    return any(t in text for t in AQL_TERMS)

def count_proximity(text, list1, list2, window=80):
    """Compte co-occurrences dans une fenêtre"""
    count = 0
    for term1 in list1:
        idx = text.find(term1)
        if idx < 0:
            continue
        for term2 in list2:
            start = max(0, idx - window)
            end = min(len(text), idx + len(term1) + window)
            if term2 in text[start:end]:
                count += 1
                break
    return count

def extract_features_50(text, metadata=None):
    """
    Extrait 50 features lexicales

    Returns:
        dict: Features nommées
    """
    features = {}

    # Tokenization arabe
    tokens = re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
    n_tokens = len(tokens)

    if n_tokens == 0:
        # Retourner vecteur de zéros
        return {f'f{i:02d}': 0.0 for i in range(50)}

    # ─── JUNUN (15 features) ────────────────────────────────────────────
    features['f00_has_junun'] = float(has_junun_filtered(text))
    features['f01_junun_density'] = sum(1 for t in tokens if t in JUNUN_TERMS) / n_tokens
    features['f02_famous_fool'] = float(any(name in text for name in FAMOUS_FOOLS))

    junun_count = sum(1 for t in JUNUN_TERMS if t in text)
    features['f03_junun_count'] = min(float(junun_count), 10.0) / 10  # Normalise [0,1]

    # Variantes spécialisées
    specialized = [t for t in JUNUN_TERMS if len(t) > 4]
    features['f04_junun_specialized'] = float(any(s in text for s in specialized))

    # Position du premier junun
    first_junun = None
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0 and (first_junun is None or idx < first_junun):
            first_junun = idx
    features['f05_junun_position'] = first_junun / max(len(text), 1) if first_junun else 0.5

    # Junun au début (titre)
    features['f06_junun_in_title'] = float(any(term in text[:50] for term in JUNUN_TERMS))

    # Junun au pluriel
    features['f07_junun_plural'] = float('مجانين' in text or 'المجانين' in text)

    # Junun dans les 3 derniers tiers
    third = len(text) // 3
    features['f08_junun_in_final_third'] = float(any(t in text[2*third:] for t in JUNUN_TERMS))

    # Junun avec racine ج-ن-ن
    jinn_root = ['جنون','جنونه','جنونها','جننت','يجن','أجنّ']
    features['f09_jinn_root'] = float(any(j in text for j in jinn_root))

    # Junun répétition
    junun_rep = sum(text.count(t) for t in JUNUN_TERMS)
    features['f10_junun_repetition'] = min(float(junun_rep) / n_tokens, 1.0)

    # Junun + dérivés morphologiques
    features['f11_junun_morpho'] = float(any(m in text for m in ['مجنون','معتوه','هائم','ممسوس']))

    # Junun sans négatif
    neg_junun = any(ng in text for ng in ['لا مجنون','ليس مجنون','لم يكن مجنون'])
    features['f12_junun_positive'] = float(not neg_junun)

    # Junun context (avant/après 20 chars)
    junun_context = 0
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0:
            context = text[max(0, idx-20):idx+len(t)+20]
            # Mots clés contextuels positifs
            if any(c in context for c in ['قال','رأيت','شهدت']):
                junun_context += 1
    features['f13_junun_good_context'] = float(junun_context > 0)

    # Junun co-occurrence avec validation
    val_all = VALIDATION['laugh'] + VALIDATION['gift'] + VALIDATION['cry']
    features['f14_junun_validation_prox'] = float(count_proximity(text, JUNUN_TERMS, val_all, 100) > 0)

    # ─── AQL (8 features) ───────────────────────────────────────────────
    features['f15_has_aql'] = float(has_aql_filtered(text))
    features['f16_aql_density'] = sum(1 for t in tokens if t in AQL_TERMS) / n_tokens

    aql_count = sum(1 for t in AQL_TERMS if t in text)
    features['f17_aql_count'] = min(float(aql_count), 10.0) / 10

    # Paradoxe junun × aql
    features['f18_paradox_junun_aql'] = float(features['f00_has_junun'] and features['f15_has_aql'])

    # Proximité junun-aql
    features['f19_junun_aql_proximity'] = float(count_proximity(text, JUNUN_TERMS, AQL_TERMS, 80) > 0)

    # Ratio junun/aql
    features['f20_junun_aql_ratio'] = features['f01_junun_density'] / (features['f16_aql_density'] + 0.001)
    features['f20_junun_aql_ratio'] = min(features['f20_junun_aql_ratio'], 10.0)  # Cap à 10

    # Superlatives (أعقل, أحكم, etc.)
    superlatives = ['أعقل','أحكم','أصوب','أفضل','أعلم','أصدق']
    features['f21_superlatives'] = float(any(s in text for s in superlatives))

    # Negation aql (rare mais important)
    neg_aql = any(ng in text for ng in ['لا عاقل','ليس عاقل','بلا عقل'])
    features['f22_aql_positive'] = float(not neg_aql)

    # ─── HIKMA (5 features) ────────────────────────────────────────────
    features['f23_has_hikma'] = float(any(t in text for t in HIKMA_TERMS))
    hikma_count = sum(1 for t in HIKMA_TERMS if t in text)
    features['f24_hikma_density'] = hikma_count / n_tokens

    # Co-occurrence hikma + junun
    features['f25_hikma_junun_prox'] = float(count_proximity(text, HIKMA_TERMS, JUNUN_TERMS, 80) > 0)

    # Co-occurrence hikma + dialogue
    features['f26_hikma_qala_prox'] = float(count_proximity(text, HIKMA_TERMS, QALA_VARIANTS, 80) > 0)

    # Hikma au début
    features['f27_hikma_in_title'] = float(any(term in text[:50] for term in HIKMA_TERMS))

    # ─── DIALOGUE/QALA (12 features) ────────────────────────────────────
    features['f28_has_qala'] = float(any(q in text for q in QALA_VARIANTS))
    qala_count = sum(1 for q in QALA_VARIANTS if q in text)
    features['f29_qala_density'] = qala_count / n_tokens

    # First person (rôle de narrateur)
    features['f30_has_first_person'] = float(any(fp in text for fp in FIRST_PERSON))
    fp_count = sum(1 for fp in FIRST_PERSON if fp in text)
    features['f31_first_person_density'] = fp_count / n_tokens

    # Co-occurrence junun + qala
    features['f32_junun_near_qala'] = float(count_proximity(text, JUNUN_TERMS, QALA_VARIANTS, 80) > 0)

    # Questions
    features['f33_has_questions'] = float(any(q in text for q in QUESTIONS))
    q_count = sum(1 for q in QUESTIONS if q in text)
    features['f34_question_density'] = q_count / n_tokens

    # Question-answer pattern
    has_sual = any(s in text for s in QUESTIONS)
    has_jawab = any(j in text for j in QALA_VARIANTS)
    features['f35_question_answer'] = float(has_sual and has_jawab)

    # Dialogue structure (alternance)
    qala_count_multiple = text.count('قال') + text.count('قلت') >= 2
    features['f36_dialogue_structure'] = float(qala_count_multiple)

    # Position premier qala
    first_qala = text.find('قال')
    if first_qala < 0:
        first_qala = text.find('قلت')
    features['f37_qala_position'] = first_qala / max(len(text), 1) if first_qala >= 0 else 0.5

    # Qala dans le dernier tiers
    features['f38_qala_in_final'] = float(any(q in text[2*third:] for q in QALA_VARIANTS))

    # ─── VALIDATION/RÉACTION (8 features) ────────────────────────────────
    all_validation = VALIDATION['laugh'] + VALIDATION['gift'] + VALIDATION['cry'] + VALIDATION['silence']
    features['f39_has_validation'] = float(any(v in text for v in all_validation))

    val_count = sum(1 for v in all_validation if v in text)
    features['f40_validation_density'] = val_count / n_tokens

    # Validation types
    has_laugh = any(l in text for l in VALIDATION['laugh'])
    has_gift = any(g in text for g in VALIDATION['gift'])
    has_cry = any(c in text for c in VALIDATION['cry'])
    features['f41_validation_laugh'] = float(has_laugh)
    features['f42_validation_gift'] = float(has_gift)
    features['f43_validation_cry'] = float(has_cry)

    # Validation au final
    features['f44_validation_in_final'] = float(any(v in text[2*third:] for v in all_validation))

    # Validation multiple types
    val_types = [has_laugh, has_gift, has_cry]
    features['f45_validation_multiple'] = float(sum(val_types) > 1)

    # Validation + junun proximity
    features['f46_validation_junun_prox'] = float(count_proximity(text, all_validation, JUNUN_TERMS, 100) > 0)

    # ─── CONTRASTE (5 features) ────────────────────────────────────────
    all_contrast = CONTRAST['opposition'] + CONTRAST['correction'] + CONTRAST['revelation']
    features['f47_has_contrast'] = float(any(c in text for c in all_contrast))

    c_count = sum(1 for c in all_contrast if c in text)
    features['f48_contrast_density'] = c_count / n_tokens

    # Contraste types
    has_opp = any(o in text for o in CONTRAST['opposition'])
    has_corr = any(c in text for c in CONTRAST['correction'])
    has_rev = any(r in text for r in CONTRAST['revelation'])
    features['f49_contrast_opposition'] = float(has_opp)
    features['f50_contrast_correction'] = float(has_corr)
    features['f51_contrast_revelation'] = float(has_rev)

    # ─── AUTORITÉ (4 features) ─────────────────────────────────────────
    features['f52_has_authority'] = float(any(a in text for a in AUTHORITY))

    auth_count = sum(1 for a in AUTHORITY if a in text)
    features['f53_authority_count'] = min(float(auth_count), 5.0) / 5

    # Autorité + junun
    features['f54_authority_junun_prox'] = float(count_proximity(text, AUTHORITY, JUNUN_TERMS, 100) > 0)

    # Autorité au début
    features['f55_authority_in_title'] = float(any(a in text[:50] for a in AUTHORITY))

    # ─── POÉSIE (3 features) ───────────────────────────────────────────
    features['f56_has_shir'] = float(any(s in text for s in SHIR_MARKERS))

    shir_count = sum(1 for s in SHIR_MARKERS if s in text)
    features['f57_shir_density'] = shir_count / n_tokens

    # Poésie solo (sans dialogue ni mubashara)
    features['f58_shir_alone'] = float(features['f56_has_shir'] and not features['f28_has_qala'])

    # ─── SPATIAL (3 features) ──────────────────────────────────────────
    features['f59_has_spatial'] = float(any(s in text for s in SPATIAL))

    spatial_count = sum(1 for s in SPATIAL if s in text)
    features['f60_spatial_density'] = spatial_count / n_tokens

    # Spatial markers richness
    unique_spatial = sum(1 for s in SPATIAL if s in text)
    features['f61_spatial_variety'] = unique_spatial / len(SPATIAL)

    return features

def load_dataset(path):
    """Charge dataset_raw.json et retourne X, y"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    X = []
    y = []

    # data is a list of dicts with 'text_ar' and 'label' keys
    for item in data:
        text = item.get('text_ar', '')
        label = item.get('label', 0)

        if not text:
            continue

        feat = extract_features_50(text)
        X.append(list(feat.values()))
        y.append(int(label))

    return np.array(X), np.array(y)

def train_lr(X, y, cv_folds=5):
    """Entraîne LR avec validation croisée"""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        class_weight='balanced',  # Important pour données déséquilibrées
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    # Cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv_folds, scoring='roc_auc')
    print(f"  Cross-validation AUC (k={cv_folds}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entraîner sur tout le dataset
    clf.fit(X_scaled, y)

    # Test split pour rapport
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  Test AUC: {test_auc:.4f}")

    # Rapport
    feature_names = [f'f{i:02d}' for i in range(len(clf.coef_[0]))]
    report = {
        'n_features': int(len(feature_names)),
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'test_auc': float(test_auc),
        'n_positives': int(sum(y == 1)),
        'n_negatives': int(sum(y == 0)),
        'coefficients': {fname: float(coef) for fname, coef in zip(feature_names, clf.coef_[0])},
    }

    return clf, scaler, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build LR classifier with 50 lexical features')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"ERROR: dataset not found → {DATASET}")
        sys.exit(1)

    print(f"Loading dataset…")
    X, y = load_dataset(DATASET)
    print(f"  {X.shape[0]} texts, {X.shape[1]} features")
    print(f"  Positives: {sum(y == 1)}, Negatives: {sum(y == 0)}")

    print(f"\nTraining LR (50 features)…")
    clf, scaler, report = train_lr(X, y, cv_folds=args.cv)

    # Save
    with open(CLF_PATH, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler}, f)
    print(f"\nClassifier saved → {CLF_PATH}")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved → {REPORT_PATH}")

    # Print top coefficients
    print(f"\nTop 10 predictive features:")
    coefs = report['coefficients']
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, coef in sorted_coefs[:10]:
        print(f"  {fname:15s} = {coef:+.4f}")
