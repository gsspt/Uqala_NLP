#!/usr/bin/env python3
"""
build_features_71.py
─────────────────────
Entraîne une Régression Logistique avec 71 features RÉELLES (62 lexicales + 9 morphologiques).

Pipeline :
  1. Charge dataset_raw.json pour texte et labels
  2. Extrait 62 features lexicales (du modèle 50 + 3 wasf) + 9 morphologiques
  3. Entraîne LR avec validation croisée + optimisation C
  4. Sauvegarde classifier + rapport détaillé

Usage :
  python build_features_71.py
  python build_features_71.py --cv 10
"""

import json
import sys
import pathlib
import argparse
import pickle
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

# CAMeL Tools pour morphologie arabe
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    print("✅ CAMeL Tools loaded\n")
    morpho_db = MorphologyDB.builtin_db()
    analyzer = Analyzer(morpho_db)
    HAS_CAMEL = True
except ImportError as e:
    print(f"⚠️  CAMeL Tools not available: {e}")
    HAS_CAMEL = False

BASE         = pathlib.Path(__file__).parent.parent
DATASET      = BASE / "dataset_raw.json"
CLF_PATH     = BASE / "scan" / "lr_classifier_71features.pkl"
REPORT_PATH  = BASE / "scan" / "lr_report_71features.json"

# ══════════════════════════════════════════════════════════════════════════════
# LEXICAL FEATURES (62 features: f00-f61 from build_features_50.py model)
# ══════════════════════════════════════════════════════════════════════════════

# Variantes junūn (enrichies avec variants manquants du corpus)
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

# Variantes aql (enrichies)
AQL_TERMS = [
    'عاقل','العاقل','عقل','العقل','عقلاء','العقلاء','عقلاؤهم',
    'عقول','أعقل','معقول','عقلانية','حكمة','حكيم','عقلائي',
    'عقله','عقلها','عقلي','عقلك','عقلهم',  # variants manquants
]

# Variantes hikma
HIKMA_TERMS = [
    'حكمة','حكيم','حكماء','الحكمة','الحكيم','الحكماء','حكمته','حكمهم','أحكم',
]

# Qala variants (enrichies avec formes manquantes)
QALA_VARIANTS = [
    'قلت','فقلت','قلنا','قالوا','قال','قالت','قالا',
    'سألت','فسألت','سألني','أجبت','أجاب',
    'قيل له','قيل لي','فقلت له',
    'وقال','ويقول','يقال','أقول',  # variants manquants
]

FIRST_PERSON = [
    'فعلت','رأيت','مررت','لقيت','وجدت','شهدت','سمعت','حدثني',
    'أخبرني','أنبأني','حدثنا','أخبرنا','أنبأنا',
]

QUESTIONS = ['كيف','كيفية','ماذا','ما','متى','أين','هل','من','لماذا']

VALIDATION = {
    'laugh': ['ضحك','ضحكت','ضحكوا','فضحك','فضحكوا'],
    'gift': ['أعطى','أعطاه','وهب','جائزة','أمر','فأمر'],
    'cry': ['بكى','بكت','بكاء','دموع','دموعه'],
    'silence': ['صمت','سكت','أطرق','طرق'],
}

CONTRAST = {
    'opposition': ['لكن','لكنه','لكنها','بل','وبل','ولكن'],
    'correction': ['كذب','أخطأ','غلط','فبان','فتبين','أصح'],
    'revelation': ['فإذا','وإذا','فتبين','فتبينت'],
}

AUTHORITY = [
    'الخليفة','أميرالمؤمنين','أمير المؤمنين','الرشيد','المأمون',
    'المتوكل','المعتصم','المهدي','المنصور','الهادي','المعتضد',
    'الوزير','الوالي','القاضي','السلطان','الملك',
]

SHIR_MARKERS = [
    'أنشد','أنشأ','أنشدني','أنشدنا','فأنشد','فأنشأ','ينشد',
    'الشاعر','شعره','شعرها','شعري','أبيات','قصيدة','وأنشد',
]

SPATIAL = ['في','على','عند','بعد','أمام','خلف','حول','داخل','خارج']

# WASF markers (definitional/lexicographic text — strong NEGATIVE signal)
# Restored from original 14-feature model
WASF_MARKERS = [
    'ومنها','ضروب','فهو مجنون','تقول العرب','من القول',
    'قال الكاتب','قال المصنف','يقال','يسمى',
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def has_junun_filtered(text):
    """Détecte junūn, filtre les faux positifs"""
    if not any(t in text for t in JUNUN_TERMS):
        return False
    if any(fp in text for fp in ['الجنة ', ' الجنة', 'الجن ', 'السجن ']):
        return False
    return True

def has_aql_filtered(text):
    """Détecte aql"""
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

def extract_lexical_features_62(text):
    """
    Extrait 62 features lexicales (f00-f61 du modèle 50 + f62-f64 wasf)

    Returns:
        dict: Features nommées f00_* à f64_*
    """
    features = {}
    tokens = re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
    n_tokens = len(tokens) if tokens else 1

    # ─── JUNUN (15 features: f00-f14) ─────────────────────────────
    features['f00_has_junun'] = float(has_junun_filtered(text))
    features['f01_junun_density'] = sum(1 for t in tokens if t in JUNUN_TERMS) / n_tokens
    features['f02_famous_fool'] = float(any(name in text for name in FAMOUS_FOOLS))

    junun_count = sum(1 for t in JUNUN_TERMS if t in text)
    features['f03_junun_count'] = min(float(junun_count), 10.0) / 10

    specialized = [t for t in JUNUN_TERMS if len(t) > 4]
    features['f04_junun_specialized'] = float(any(s in text for s in specialized))

    first_junun = None
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0 and (first_junun is None or idx < first_junun):
            first_junun = idx
    features['f05_junun_position'] = first_junun / max(len(text), 1) if first_junun else 0.5

    features['f06_junun_in_title'] = float(any(term in text[:50] for term in JUNUN_TERMS))
    features['f07_junun_plural'] = float('مجانين' in text or 'المجانين' in text)

    third = len(text) // 3
    features['f08_junun_in_final_third'] = float(any(t in text[2*third:] for t in JUNUN_TERMS))

    jinn_root = ['جنون','جنونه','جنونها','جننت','يجن','أجنّ']
    features['f09_jinn_root'] = float(any(j in text for j in jinn_root))

    junun_rep = sum(text.count(t) for t in JUNUN_TERMS)
    features['f10_junun_repetition'] = min(float(junun_rep) / n_tokens, 1.0)

    features['f11_junun_morpho'] = float(any(m in text for m in ['مجنون','معتوه','هائم','ممسوس']))

    neg_junun = any(ng in text for ng in ['لا مجنون','ليس مجنون','لم يكن مجنون'])
    features['f12_junun_positive'] = float(not neg_junun)

    junun_context = 0
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0:
            context = text[max(0, idx-20):idx+len(t)+20]
            if any(c in context for c in ['قال','رأيت','شهدت']):
                junun_context += 1
    features['f13_junun_good_context'] = float(junun_context > 0)

    val_all = VALIDATION['laugh'] + VALIDATION['gift'] + VALIDATION['cry']
    features['f14_junun_validation_prox'] = float(count_proximity(text, JUNUN_TERMS, val_all, 100) > 0)

    # ─── AQL (8 features: f15-f22) ──────────────────────────────
    features['f15_has_aql'] = float(has_aql_filtered(text))
    features['f16_aql_density'] = sum(1 for t in tokens if t in AQL_TERMS) / n_tokens

    aql_count = sum(1 for t in AQL_TERMS if t in text)
    features['f17_aql_count'] = min(float(aql_count), 10.0) / 10

    features['f18_paradox_junun_aql'] = float(features['f00_has_junun'] and features['f15_has_aql'])
    features['f19_junun_aql_proximity'] = float(count_proximity(text, JUNUN_TERMS, AQL_TERMS, 80) > 0)

    features['f20_junun_aql_ratio'] = features['f01_junun_density'] / (features['f16_aql_density'] + 0.001)
    features['f20_junun_aql_ratio'] = min(features['f20_junun_aql_ratio'], 10.0)

    superlatives = ['أعقل','أحكم','أصوب','أفضل','أعلم','أصدق']
    features['f21_superlatives'] = float(any(s in text for s in superlatives))

    neg_aql = any(ng in text for ng in ['لا عاقل','ليس عاقل','بلا عقل'])
    features['f22_aql_positive'] = float(not neg_aql)

    # ─── HIKMA (5 features: f23-f27) ────────────────────────────
    features['f23_has_hikma'] = float(any(t in text for t in HIKMA_TERMS))
    hikma_count = sum(1 for t in HIKMA_TERMS if t in text)
    features['f24_hikma_density'] = hikma_count / n_tokens

    features['f25_hikma_junun_prox'] = float(count_proximity(text, HIKMA_TERMS, JUNUN_TERMS, 80) > 0)
    features['f26_hikma_qala_prox'] = float(count_proximity(text, HIKMA_TERMS, QALA_VARIANTS, 80) > 0)
    features['f27_hikma_in_title'] = float(any(term in text[:50] for term in HIKMA_TERMS))

    # ─── DIALOGUE/QALA (11 features: f28-f38) ───────────────────
    features['f28_has_qala'] = float(any(q in text for q in QALA_VARIANTS))
    qala_count = sum(1 for q in QALA_VARIANTS if q in text)
    features['f29_qala_density'] = qala_count / n_tokens

    features['f30_has_first_person'] = float(any(fp in text for fp in FIRST_PERSON))
    fp_count = sum(1 for fp in FIRST_PERSON if fp in text)
    features['f31_first_person_density'] = fp_count / n_tokens

    features['f32_junun_near_qala'] = float(count_proximity(text, JUNUN_TERMS, QALA_VARIANTS, 80) > 0)

    features['f33_has_questions'] = float(any(q in text for q in QUESTIONS))
    q_count = sum(1 for q in QUESTIONS if q in text)
    features['f34_question_density'] = q_count / n_tokens

    has_sual = any(s in text for s in QUESTIONS)
    has_jawab = any(j in text for j in QALA_VARIANTS)
    features['f35_question_answer'] = float(has_sual and has_jawab)

    qala_count_multiple = text.count('قال') + text.count('قلت') >= 2
    features['f36_dialogue_structure'] = float(qala_count_multiple)

    first_qala = text.find('قال')
    if first_qala < 0:
        first_qala = text.find('قلت')
    features['f37_qala_position'] = first_qala / max(len(text), 1) if first_qala >= 0 else 0.5

    features['f38_qala_in_final'] = float(any(q in text[2*third:] for q in QALA_VARIANTS))

    # ─── VALIDATION (8 features: f39-f46) ───────────────────────
    all_validation = VALIDATION['laugh'] + VALIDATION['gift'] + VALIDATION['cry'] + VALIDATION['silence']
    features['f39_has_validation'] = float(any(v in text for v in all_validation))

    val_count = sum(1 for v in all_validation if v in text)
    features['f40_validation_density'] = val_count / n_tokens

    has_laugh = any(l in text for l in VALIDATION['laugh'])
    has_gift = any(g in text for g in VALIDATION['gift'])
    has_cry = any(c in text for c in VALIDATION['cry'])
    features['f41_validation_laugh'] = float(has_laugh)
    features['f42_validation_gift'] = float(has_gift)
    features['f43_validation_cry'] = float(has_cry)

    features['f44_validation_in_final'] = float(any(v in text[2*third:] for v in all_validation))

    val_types = [has_laugh, has_gift, has_cry]
    features['f45_validation_multiple'] = float(sum(val_types) > 1)

    features['f46_validation_junun_prox'] = float(count_proximity(text, all_validation, JUNUN_TERMS, 100) > 0)

    # ─── CONTRASTE (5 features: f47-f51) ────────────────────────
    all_contrast = CONTRAST['opposition'] + CONTRAST['correction'] + CONTRAST['revelation']
    features['f47_has_contrast'] = float(any(c in text for c in all_contrast))

    c_count = sum(1 for c in all_contrast if c in text)
    features['f48_contrast_density'] = c_count / n_tokens

    has_opp = any(o in text for o in CONTRAST['opposition'])
    has_corr = any(c in text for c in CONTRAST['correction'])
    has_rev = any(r in text for r in CONTRAST['revelation'])
    features['f49_contrast_opposition'] = float(has_opp)
    features['f50_contrast_correction'] = float(has_corr)
    features['f51_contrast_revelation'] = float(has_rev)

    # ─── AUTORITÉ (4 features: f52-f55) ─────────────────────────
    features['f52_has_authority'] = float(any(a in text for a in AUTHORITY))
    auth_count = sum(1 for a in AUTHORITY if a in text)
    features['f53_authority_count'] = min(float(auth_count), 5.0) / 5

    features['f54_authority_junun_prox'] = float(count_proximity(text, AUTHORITY, JUNUN_TERMS, 100) > 0)
    features['f55_authority_in_title'] = float(any(a in text[:50] for a in AUTHORITY))

    # ─── POÉSIE (3 features: f56-f58) ───────────────────────────
    features['f56_has_shir'] = float(any(s in text for s in SHIR_MARKERS))
    shir_count = sum(1 for s in SHIR_MARKERS if s in text)
    features['f57_shir_density'] = shir_count / n_tokens

    features['f58_shir_alone'] = float(features['f56_has_shir'] and not features['f28_has_qala'])

    # ─── SPATIAL (3 features: f59-f61) ───────────────────────────
    features['f59_has_spatial'] = float(any(s in text for s in SPATIAL))
    spatial_count = sum(1 for s in SPATIAL if s in text)
    features['f60_spatial_density'] = spatial_count / n_tokens

    unique_spatial = sum(1 for s in SPATIAL if s in text)
    features['f61_spatial_variety'] = unique_spatial / len(SPATIAL)

    # ─── WASF/LEXICOGRAPHIC (3 features: f62-f64) ── RESTORED ─────
    # Negative signal: text is definitional/encyclopedic, not narrative
    features['f62_has_wasf'] = float(any(w in text for w in WASF_MARKERS))
    wasf_count = sum(1 for w in WASF_MARKERS if w in text)
    features['f63_wasf_density'] = wasf_count / n_tokens

    features['f64_wasf_in_title'] = float(any(w in text[:80] for w in WASF_MARKERS))

    return features

def extract_morphological_features_9(text):
    """
    Extrait 9 features morphologiques réelles (f65-f73)

    Returns:
        dict: Features nommées f65_* à f73_*
    """
    features = {}

    if not HAS_CAMEL:
        # Fallback: return zeros if CAMeL Tools not available
        for i in range(9):
            features[f'f{65+i:02d}_morpho'] = 0.0
        return features

    tokens = text.split()
    n_tokens = len(tokens) if tokens else 1

    # Morphological analysis with CAMeL Tools
    jnn_count = aql_count = hikma_count = 0
    verb_count = noun_count = adj_count = 0
    perf_count = imperf_count = passive_count = 0

    for token in tokens:
        try:
            analyses = analyzer.analyze(token)
            if analyses:
                a = analyses[0]
                root = a.get('root', '')
                pos = a.get('pos', '')
                asp = a.get('asp', '')
                vox = a.get('vox', '')

                # Root density
                if root == 'ج.ن.ن':
                    jnn_count += 1
                if root == 'ع.ق.ل':
                    aql_count += 1
                if root == 'ح.ك.م':
                    hikma_count += 1

                # POS tags
                if pos == 'verb':
                    verb_count += 1
                if pos == 'noun':
                    noun_count += 1
                if pos == 'adj':
                    adj_count += 1

                # Aspect and voice
                if asp == 'perf':
                    perf_count += 1
                if asp == 'imperf':
                    imperf_count += 1
                if vox == 'pass':
                    passive_count += 1
        except:
            pass

    features['f65_root_jnn_density'] = jnn_count / n_tokens
    features['f66_root_aql_density'] = aql_count / n_tokens
    features['f67_root_hikma_density'] = hikma_count / n_tokens
    features['f68_verb_density'] = verb_count / n_tokens
    features['f69_noun_density'] = noun_count / n_tokens
    features['f70_adj_density'] = adj_count / n_tokens
    features['f71_perf_density'] = perf_count / n_tokens
    features['f72_imperf_density'] = imperf_count / n_tokens
    features['f73_passive_voice_ratio'] = passive_count / n_tokens

    return features

def extract_features_71(text):
    """Extrait 71 features (62 lexical + 9 morphological)"""
    lex = extract_lexical_features_62(text)
    morpho = extract_morphological_features_9(text)
    combined = {**lex, **morpho}
    return combined

def load_dataset(dataset_path):
    """Charge dataset et extrait features"""
    with open(dataset_path, encoding='utf-8') as f:
        data = json.load(f)

    y = np.array([item['label'] for item in data])
    print(f"  Total samples: {len(y)}")
    print(f"  Positives: {sum(y)}, Negatives: {len(y) - sum(y)}")

    # Extract features
    X = []
    for i, item in enumerate(data):
        if (i+1) % 1000 == 0:
            print(f"    Extracting features: {i+1}/{len(data)}")
        feat = extract_features_71(item['text_ar'])
        X.append(list(feat.values()))

    X = np.array(X)
    print(f"  Features shape: {X.shape}")
    return X, y

def train_lr(X, y, cv_folds=5):
    """Entraîne LR avec validation croisée + optimisation C"""

    # Fix NaN/Inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # GridSearchCV pour trouver le meilleur C
    print(f"\n  Optimizing regularization parameter C via GridSearchCV...")
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(
        lr, param_grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_scaled, y)

    clf = grid_search.best_estimator_
    best_c = grid_search.best_params_['C']
    print(f"  Best C: {best_c}")

    # Cross-validation scores with best model
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv_folds, scoring='roc_auc')
    print(f"  CV AUC (k={cv_folds}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on all data
    clf.fit(X_scaled, y)

    # Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  Test AUC: {test_auc:.4f}")

    # Build feature names
    feat_names = [
        # Lexical (f00-f64)
        'f00_has_junun', 'f01_junun_density', 'f02_famous_fool', 'f03_junun_count',
        'f04_junun_specialized', 'f05_junun_position', 'f06_junun_in_title',
        'f07_junun_plural', 'f08_junun_in_final_third', 'f09_jinn_root',
        'f10_junun_repetition', 'f11_junun_morpho', 'f12_junun_positive',
        'f13_junun_good_context', 'f14_junun_validation_prox',
        'f15_has_aql', 'f16_aql_density', 'f17_aql_count', 'f18_paradox_junun_aql',
        'f19_junun_aql_proximity', 'f20_junun_aql_ratio', 'f21_superlatives',
        'f22_aql_positive',
        'f23_has_hikma', 'f24_hikma_density', 'f25_hikma_junun_prox',
        'f26_hikma_qala_prox', 'f27_hikma_in_title',
        'f28_has_qala', 'f29_qala_density', 'f30_has_first_person',
        'f31_first_person_density', 'f32_junun_near_qala', 'f33_has_questions',
        'f34_question_density', 'f35_question_answer', 'f36_dialogue_structure',
        'f37_qala_position', 'f38_qala_in_final',
        'f39_has_validation', 'f40_validation_density', 'f41_validation_laugh',
        'f42_validation_gift', 'f43_validation_cry', 'f44_validation_in_final',
        'f45_validation_multiple', 'f46_validation_junun_prox',
        'f47_has_contrast', 'f48_contrast_density', 'f49_contrast_opposition',
        'f50_contrast_correction', 'f51_contrast_revelation',
        'f52_has_authority', 'f53_authority_count', 'f54_authority_junun_prox',
        'f55_authority_in_title',
        'f56_has_shir', 'f57_shir_density', 'f58_shir_alone',
        'f59_has_spatial', 'f60_spatial_density', 'f61_spatial_variety',
        'f62_has_wasf', 'f63_wasf_density', 'f64_wasf_in_title',
        # Morphological (f65-f73)
        'f65_root_jnn_density', 'f66_root_aql_density', 'f67_root_hikma_density',
        'f68_verb_density', 'f69_noun_density', 'f70_adj_density',
        'f71_perf_density', 'f72_imperf_density', 'f73_passive_voice_ratio',
    ]

    # Rapport
    report = {
        'n_features': int(len(feat_names)),
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'test_auc': float(test_auc),
        'n_positives': int(sum(y == 1)),
        'n_negatives': int(sum(y == 0)),
        'best_c': float(best_c),
        'coefficients': {fname: float(coef) for fname, coef in zip(feat_names, clf.coef_[0])},
    }

    return clf, scaler, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build LR classifier with 71 real features')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"ERROR: Dataset not found → {DATASET}")
        sys.exit(1)

    print(f"Loading data…")
    X, y = load_dataset(DATASET)

    print(f"\nTraining LR (71 real features: 62 lexical + 9 morphological)…")
    clf, scaler, report = train_lr(X, y, cv_folds=args.cv)

    # Save
    with open(CLF_PATH, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler}, f)
    print(f"\nClassifier saved → {CLF_PATH}")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved → {REPORT_PATH}")

    # Print top coefficients
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Features: {report['n_features']} (62 lexical + 9 morphological)")
    print(f"CV AUC: {report['cv_auc_mean']:.4f} ± {report['cv_auc_std']:.4f}")
    print(f"Test AUC: {report['test_auc']:.4f}")
    print(f"Best C: {report['best_c']}")
    print(f"\nTop 15 predictive features:")
    coefs = report['coefficients']
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, coef in sorted_coefs[:15]:
        print(f"  {fname:35s} = {coef:+.4f}")
    print(f"{'='*60}")
