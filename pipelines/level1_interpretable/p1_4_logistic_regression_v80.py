#!/usr/bin/env python3
"""
p1_4_logistic_regression_v80.py
─────────────────────────────────────────────────────────────────
Entraîne une Régression Logistique avec 27 features v80 (empiriquement optimisées).

Changements depuis v79 (79 features → 27 features):
  SUPPRIMÉS  : Toutes les features pédagogiques (hikma, contraste, validation, autorité, wasf)
  SUPPRIMÉS  : E7_no_formal_isnad (false signal — isnads filtered in preprocessing)
  GARDÉS     : Junun core (f00-f14), morpho CAMeL (f65-f70)
  EMPIRIQUES : 6 nouvelles features basées sur analyse corpus

Architecture v80:
  - f00-f14    : JUNUN (15 features) — core concept
  - f65-f70    : MORPHO CAMeL (6 features) — working well from v79
  - E1-E6      : Empirical features (scene, witness, dialogue, invocation, spaces)

Pipeline :
  1. Charge dataset_raw.json
  2. Extrait 27 features (15 junun + 6 morpho + 6 empirique)
  3. Entraîne LR avec CV + optimisation C
  4. Sauvegarde + rapport détaillé

Usage :
  python p1_4_logistic_regression_v80.py
  python p1_4_logistic_regression_v80.py --cv 5
"""

import json
import sys
import pathlib
import argparse
import pickle
import re
import importlib.util
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

# Smart CAMeL Tools loader (works in multiple environments)
BASE = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

try:
    from uqala_nlp.preprocessing.smart_camel_loader import HAS_CAMEL, analyzer, extract_morpho_features_safe as extract_morpho_safe
except ImportError:
    # Fallback: direct import (should work in conda venv)
    HAS_CAMEL = False
    analyzer = None
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        morpho_db = MorphologyDB.builtin_db()
        analyzer = Analyzer(morpho_db)
        HAS_CAMEL = True
        print("✅ CAMeL Tools loaded\n")
    except ImportError:
        print(f"⚠️  CAMeL Tools not available (using degraded mode)")

BASE         = pathlib.Path(__file__).resolve().parent.parent.parent  # repo root
DATASET      = BASE / "data" / "raw" / "dataset_raw.json"
CLF_PATH     = BASE / "models" / "lr_classifier_v80.pkl"
REPORT_PATH  = BASE / "models" / "lr_report_v80.json"

# ══════════════════════════════════════════════════════════════════════════════
# EMPIRICAL LEXICONS (extracted from corpus analysis)
# ══════════════════════════════════════════════════════════════════════════════

# JUNUN — core concept (from v79, kept as-is)
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

# EMPIRICAL FEATURES v80 (from corpus analysis)

# E1. Scene Introduction Verbs (introduce a first-person encounter)
SCENE_INTRO_VERBS = [
    'مررت',   # 5.65% (POS) vs 0.55% (NEG) = 10.09x
    'دخلت',   # 12.17% vs 2.78% = 4.37x
    'فرأيت',  # 5.87% vs 0.65% = 8.83x
    'لقيت',   # 2.39% vs 1.13% = 2.10x
    'أتيت',   # 2.83% vs 1.47% = 1.91x
    'سقطت',   # 0.87% vs 0.47% = 1.81x
    'خرجت',   # 4.13% vs 2.86% = 1.44x
    'وجدت',   # 2.39% vs 1.78% = 1.33x
]

# E2. Witness Observation Verbs (first-person observation)
WITNESS_VERBS = [
    'رأيت',   # 22.39% (POS) vs 9.01% (NEG) = 2.48x
    'فرأيت',  # 5.87% vs 0.65% = 8.83x (also in scene intro)
    'شاهدت',  # 0.43% vs 0.10% = 3.79x
    'أبصرت',  # 0.43% vs 0.16% = 2.60x
    'عاينت',  # 0.43% vs 0.21% = 1.98x
]

# E3. Dialogue First-Person (I said)
DIALOGUE_FIRST_PERSON = [
    'قلت',    # 99.35% (POS) vs 21.12% (NEG) = 4.70x
    'فقلت',   # variant
    'قلنا',   # 2.61% vs 1.28% = 2.02x
]

# E4. Direct Address / Vocative (يا + noun)
DIRECT_ADDRESS = [
    'يا بهلول',      # 2.39% vs 0.00% = ∞
    'يا مجنون',      # 1.74% vs 0.00% = ∞
    'يا ذا',        # 3.48% vs 0.00% = ∞
    'يا هرم',       # 1.09% vs 0.00% = ∞
    'يا هذا',       # 1.30% vs 0.50% = 2.57x
    'يا سعدون',      # 0.87% vs 0.00% = ∞
]

# E5. Divine Invocation (personal, not generic الله)
DIVINE_PERSONAL = [
    'إلهي',        # 2.17% (POS) vs 0.05% (NEG) = 34.84x
    'اللهم',       # 2.61% vs 2.04% = 1.27x
    'يا رب',       # 0.87% vs 0.39% = 2.16x
    'يا إلهي',     # 0.22% vs 0.00% = ∞
]

# E6. Sacred / Liminal Spaces
SACRED_SPACES = [
    'أزقة',         # 1.09% (POS) vs 0.00% (NEG) = 108.70x
    'المقابر',      # 3.91% vs 0.08% = 44.17x
    'خرابات',       # 2.17% vs 0.03% = 60.06x
    'الخرابات',     # 1.52% vs 0.03% = 42.04x
    'قبر',         # 7.83% vs 2.25% = 3.46x
    'سوق',         # 4.13% vs 1.31% = 3.13x
    'مسجد',        # 4.78% vs 2.28% = 2.09x
]

# NOTE: Isnad filtering is handled by src/uqala_nlp/preprocessing/isnad_filter.py
# E7 (no_formal_isnad) was REMOVED — isnads are preprocessed out
# Including it was a false signal (data artifact)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def has_junun_filtered(text):
    """Detect junun, filter false positives"""
    if not any(t in text for t in JUNUN_TERMS):
        return False
    if any(fp in text for fp in ['الجنة ', ' الجنة', 'الجن ', 'السجن ']):
        return False
    return True

def extract_junun_features_15(text):
    """
    Extract 15 JUNUN features (f00-f14).
    Core concept — kept from v79.
    """
    features = {}
    tokens = re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
    n_tokens = len(tokens) if tokens else 1

    # f00: has junun
    features['f00_has_junun'] = float(has_junun_filtered(text))

    # f01: junun density
    features['f01_junun_density'] = sum(1 for t in tokens if t in JUNUN_TERMS) / n_tokens

    # f02: famous fool (strong predictor from v79)
    features['f02_famous_fool'] = float(any(name in text for name in FAMOUS_FOOLS))

    # f03: junun count
    junun_count = sum(1 for t in JUNUN_TERMS if t in text)
    features['f03_junun_count'] = min(float(junun_count), 10.0) / 10

    # f04: junun specialized
    specialized = [t for t in JUNUN_TERMS if len(t) > 4]
    features['f04_junun_specialized'] = float(any(s in text for s in specialized))

    # f05: junun position (early = narrative)
    first_junun = None
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0 and (first_junun is None or idx < first_junun):
            first_junun = idx
    features['f05_junun_position'] = first_junun / max(len(text), 1) if first_junun else 0.5

    # f06: junun in title
    features['f06_junun_in_title'] = float(any(term in text[:50] for term in JUNUN_TERMS))

    # f07: junun plural
    features['f07_junun_plural'] = float('مجانين' in text or 'المجانين' in text)

    # f08: junun in final third
    third = len(text) // 3
    features['f08_junun_in_final_third'] = float(any(t in text[2*third:] for t in JUNUN_TERMS))

    # f09: jinn root (ج.ن.ن)
    jinn_root = ['جنون','جنونه','جنونها','جننت','يجن','أجنّ']
    features['f09_jinn_root'] = float(any(j in text for j in jinn_root))

    # f10: junun repetition
    junun_rep = sum(text.count(t) for t in JUNUN_TERMS)
    features['f10_junun_repetition'] = min(float(junun_rep) / n_tokens, 1.0)

    # f11: junun morphological forms
    features['f11_junun_morpho'] = float(any(m in text for m in ['مجنون','معتوه','هائم','ممسوس']))

    # f12: junun positive (no negation)
    neg_junun = any(ng in text for ng in ['لا مجنون','ليس مجنون','لم يكن مجنون'])
    features['f12_junun_positive'] = float(not neg_junun)

    # f13: junun good context
    junun_context = 0
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0:
            context = text[max(0, idx-20):idx+len(t)+20]
            if any(c in context for c in ['قال','رأيت','شهدت']):
                junun_context += 1
    features['f13_junun_good_context'] = float(junun_context > 0)

    # f14: junun with validation proximity
    val_all = ['ضحك','أعطى','بكى']  # simplified validation
    def count_proximity(text, list1, list2, window=100):
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
    features['f14_junun_validation_prox'] = float(count_proximity(text, JUNUN_TERMS, val_all, 100) > 0)

    return features

def extract_empirical_features_6(text):
    """
    Extract 6 NEW EMPIRICAL features (E1-E6).
    Based on corpus analysis.

    E7 (no_formal_isnad) REMOVED: isnads are filtered in preprocessing,
    so this was a false signal (data artifact, not real pattern).
    """
    features = {}
    tokens = re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
    n_tokens = len(tokens) if tokens else 1

    # E1: Scene Introduction Verbs (presence early in text)
    has_scene = any(v in text for v in SCENE_INTRO_VERBS)
    features['E1_scene_intro_presence'] = float(has_scene)

    # E2: Witness Observation Verbs (density)
    witness_count = sum(text.count(v) for v in WITNESS_VERBS)
    features['E2_witness_verb_density'] = witness_count / n_tokens

    # E3: Dialogue First-Person (قلت — very distinctive 4.70x)
    dialogue_count = sum(text.count(v) for v in DIALOGUE_FIRST_PERSON)
    features['E3_dialogue_first_person_density'] = dialogue_count / n_tokens

    # E4: Direct Address (يا + noun — very strong signal, ∞ ratio)
    has_direct_address = any(addr in text for addr in DIRECT_ADDRESS)
    features['E4_direct_address_presence'] = float(has_direct_address)

    # E5: Divine Personal Invocation (إلهي — 34.84x ratio)
    divine_count = sum(text.count(d) for d in DIVINE_PERSONAL)
    features['E5_divine_personal_intensity'] = min(float(divine_count), 5.0) / 5

    # E6: Sacred / Liminal Spaces (presence, 40-100x ratios)
    has_sacred_space = any(loc in text for loc in SACRED_SPACES)
    features['E6_sacred_spaces_presence'] = float(has_sacred_space)

    return features

def extract_morphological_features_6(text):
    """
    Extract 6 morphological features (f65-f70) from v79.
    CAMeL Tools based.
    """
    features = {}

    if not HAS_CAMEL:
        for i in range(6):
            features[f'f{65+i:02d}_morpho'] = 0.0
        return features

    tokens = text.split()
    n_tokens = len(tokens) if tokens else 1

    jnn_count = aql_count = hikma_count = 0
    verb_count = noun_count = adj_count = 0

    for token in tokens:
        try:
            analyses = analyzer.analyze(token)
            if analyses:
                a = analyses[0]
                root = a.get('root', '')
                pos = a.get('pos', '')

                if root == 'ج.ن.ن':
                    jnn_count += 1
                if root == 'ع.ق.ل':
                    aql_count += 1
                if root == 'ح.ك.م':
                    hikma_count += 1

                if pos == 'verb':
                    verb_count += 1
                if pos == 'noun':
                    noun_count += 1
                if pos == 'adj':
                    adj_count += 1
        except:
            pass

    features['f65_root_jnn_density'] = jnn_count / n_tokens
    features['f66_root_aql_density'] = aql_count / n_tokens
    features['f67_root_hikma_density'] = hikma_count / n_tokens
    features['f68_verb_density'] = verb_count / n_tokens
    features['f69_noun_density'] = noun_count / n_tokens
    features['f70_adj_density'] = adj_count / n_tokens

    return features

def extract_all_features_27(text):
    """
    Extract all 27 features (15 junun + 6 morpho + 6 empirical).
    Total v80 feature set (E7_no_formal_isnad removed).
    """
    junun = extract_junun_features_15(text)
    morpho = extract_morphological_features_6(text)
    empirical = extract_empirical_features_6(text)
    return {**junun, **morpho, **empirical}

def load_dataset(dataset_path):
    """Load dataset and extract features"""
    with open(dataset_path, encoding='utf-8') as f:
        data = json.load(f)

    y = np.array([item['label'] for item in data])
    print(f"  Total samples: {len(y)}")
    print(f"  Positives: {sum(y)}, Negatives: {len(y) - sum(y)}")

    X = []
    for i, item in enumerate(data):
        if (i+1) % 1000 == 0:
            print(f"    Extracting features: {i+1}/{len(data)}")
        feat = extract_all_features_27(item['text_ar'])
        X.append(list(feat.values()))

    X = np.array(X)
    print(f"  Features shape: {X.shape}")
    return X, y

def train_lr(X, y, cv_folds=5):
    """Train LR with CV + C optimization"""

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Optimizing regularization parameter C...")
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

    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv_folds, scoring='roc_auc')
    print(f"  CV AUC (k={cv_folds}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    clf.fit(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  Test AUC: {test_auc:.4f}")

    # Feature names
    feat_names = [
        # Junun (15)
        'f00_has_junun', 'f01_junun_density', 'f02_famous_fool', 'f03_junun_count',
        'f04_junun_specialized', 'f05_junun_position', 'f06_junun_in_title',
        'f07_junun_plural', 'f08_junun_in_final_third', 'f09_jinn_root',
        'f10_junun_repetition', 'f11_junun_morpho', 'f12_junun_positive',
        'f13_junun_good_context', 'f14_junun_validation_prox',
        # Morphological (6)
        'f65_root_jnn_density', 'f66_root_aql_density', 'f67_root_hikma_density',
        'f68_verb_density', 'f69_noun_density', 'f70_adj_density',
        # Empirical (6) — E7_no_formal_isnad removed (false signal)
        'E1_scene_intro_presence', 'E2_witness_verb_density',
        'E3_dialogue_first_person_density', 'E4_direct_address_presence',
        'E5_divine_personal_intensity', 'E6_sacred_spaces_presence',
    ]

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
    parser = argparse.ArgumentParser(description='Build LR classifier with 28 features v80')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"ERROR: Dataset not found → {DATASET}")
        sys.exit(1)

    print(f"Loading data…")
    X, y = load_dataset(DATASET)

    print(f"\nTraining LR (27 features: 15 junun + 6 morpho + 6 empirical)…")
    clf, scaler, report = train_lr(X, y, cv_folds=args.cv)

    with open(CLF_PATH, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler}, f)
    print(f"\nClassifier saved → {CLF_PATH}")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved → {REPORT_PATH}")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — v80 (REVISED)")
    print(f"{'='*60}")
    print(f"Features: {report['n_features']} (15 junun + 6 morpho + 6 empirical)")
    print(f"  NOTE: E7_no_formal_isnad removed (false signal — isnads are preprocessed)")
    print(f"CV AUC: {report['cv_auc_mean']:.4f} ± {report['cv_auc_std']:.4f}")
    print(f"Test AUC: {report['test_auc']:.4f}")
    print(f"Best C: {report['best_c']}")
    print(f"\nTop 15 predictive features:")
    coefs = report['coefficients']
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, coef in sorted_coefs[:15]:
        print(f"  {fname:40s} = {coef:+.4f}")
    print(f"{'='*60}")
