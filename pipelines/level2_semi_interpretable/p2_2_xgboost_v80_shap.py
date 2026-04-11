#!/usr/bin/env python3
"""
p2_2_xgboost_v80_shap.py
─────────────────────────────────────────────────────────────────
XGBoost classifier avec les 27 features v80 (mêmes que LR v80).

Features identiques au pipeline LR v80:
  - f00-f14    : JUNUN (15 features)
  - f65-f70    : MORPHO CAMeL (6 features)
  - E1-E6      : Empirical features (6 features)

Hyperparamètres:
  - max_depth=3         : Limiter la profondeur (éviter overfitting)
  - learning_rate=0.1   : Petit taux d'apprentissage (stabilité)
  - reg_alpha=0.1       : Régularisation L1
  - reg_lambda=1.0      : Régularisation L2
  - early_stopping       : Arrêter si pas d'amélioration (validation)

Pipeline:
  1. Charger dataset_raw.json
  2. Extraire 27 features (identique à v80)
  3. Entraîner XGBoost avec CV + early stopping
  4. SHAP explanation des prédictions
  5. Sauvegarder modèle + rapport

Usage:
  python p2_2_xgboost_v80_shap.py
  python p2_2_xgboost_v80_shap.py --cv 5
"""

import json
import sys
import pathlib
import argparse
import pickle
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import shap

sys.stdout.reconfigure(encoding='utf-8')

# Smart CAMeL Tools loader
BASE = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

try:
    from uqala_nlp.preprocessing.smart_camel_loader import HAS_CAMEL, analyzer, extract_morpho_features_safe as extract_morpho_safe
except ImportError:
    HAS_CAMEL = False
    analyzer = None
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        morpho_db = MorphologyDB.builtin_db()
        analyzer = Analyzer(morpho_db)
        HAS_CAMEL = True
        print("[OK] CAMeL Tools loaded\n")
    except ImportError:
        print("[WARNING] CAMeL Tools not available (using degraded mode)\n")

DATASET     = BASE / "data" / "raw" / "dataset_raw.json"
CLF_PATH    = BASE / "models" / "xgb_classifier_v80.pkl"
REPORT_PATH = BASE / "models" / "xgb_report_v80.json"

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (identical to p1_4_logistic_regression_v80.py)
# ══════════════════════════════════════════════════════════════════════════════

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

SCENE_INTRO_VERBS = ['مررت','دخلت','فرأيت','لقيت','أتيت','سقطت','خرجت','وجدت']
WITNESS_VERBS = ['رأيت','فرأيت','شاهدت','أبصرت','عاينت']
DIALOGUE_FIRST_PERSON = ['قلت','فقلت','قلنا']
DIRECT_ADDRESS = ['يا بهلول','يا مجنون','يا ذا','يا هرم','يا هذا','يا سعدون']
DIVINE_PERSONAL = ['إلهي','اللهم','يا رب','يا إلهي']
SACRED_SPACES = ['أزقة','المقابر','خرابات','الخرابات','قبر','سوق','مسجد']

def has_junun_filtered(text):
    if not any(t in text for t in JUNUN_TERMS):
        return False
    if any(fp in text for fp in ['الجنة ', ' الجنة', 'الجن ', 'السجن ']):
        return False
    return True

def extract_junun_features_15(text):
    features = {}
    tokens = re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
    n_tokens = len(tokens) if tokens else 1

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
    val_all = ['ضحك','أعطى','بكى']
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
    features = {}
    tokens = re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
    n_tokens = len(tokens) if tokens else 1

    has_scene = any(v in text for v in SCENE_INTRO_VERBS)
    features['E1_scene_intro_presence'] = float(has_scene)

    witness_count = sum(text.count(v) for v in WITNESS_VERBS)
    features['E2_witness_verb_density'] = witness_count / n_tokens

    dialogue_count = sum(text.count(v) for v in DIALOGUE_FIRST_PERSON)
    features['E3_dialogue_first_person_density'] = dialogue_count / n_tokens

    has_direct_address = any(addr in text for addr in DIRECT_ADDRESS)
    features['E4_direct_address_presence'] = float(has_direct_address)

    divine_count = sum(text.count(d) for d in DIVINE_PERSONAL)
    features['E5_divine_personal_intensity'] = min(float(divine_count), 5.0) / 5

    has_sacred_space = any(loc in text for loc in SACRED_SPACES)
    features['E6_sacred_spaces_presence'] = float(has_sacred_space)

    return features

def extract_morphological_features_6(text):
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
    junun = extract_junun_features_15(text)
    morpho = extract_morphological_features_6(text)
    empirical = extract_empirical_features_6(text)
    return {**junun, **morpho, **empirical}

def load_dataset(dataset_path):
    with open(dataset_path, encoding='utf-8') as f:
        data = json.load(f)

    y = np.array([item['label'] for item in data])
    print(f"  Total samples: {len(y)}")
    print(f"  Positives: {sum(y)}, Negatives: {len(y) - sum(y)}")

    X = []
    feature_names = None
    for i, item in enumerate(data):
        if (i+1) % 1000 == 0:
            print(f"    Extracting features: {i+1}/{len(data)}")
        feat = extract_all_features_27(item['text_ar'])
        if feature_names is None:
            feature_names = list(feat.keys())
        X.append(list(feat.values()))

    X = np.array(X)
    print(f"  Features shape: {X.shape}")
    return X, y, feature_names

def train_xgboost(X, y, feature_names, cv_folds=5):
    """Train XGBoost with CV and SHAP analysis"""

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Training XGBoost with {cv_folds}-fold CV...")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        xgb = XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=200,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )

        xgb.fit(
            X_train, y_train,
            verbose=False
        )

        y_pred_proba = xgb.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        cv_scores.append(auc)
        fold_models.append(xgb)

        print(f"    Fold {fold_idx}/{cv_folds}: AUC = {auc:.4f}")

    print(f"\n  CV AUC (k={cv_folds}): {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    # Train final model on all data
    print(f"\n  Training final model on all data...")
    xgb_final = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=200,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
    )
    xgb_final.fit(X_scaled, y)

    # Feature importance (using XGBoost built-in gain importance)
    print(f"\n  Computing feature importance...")
    try:
        importance_dict_raw = xgb_final.get_booster().get_score(importance_type='gain')
        feature_importance_dict = {}
        for fname in feature_names:
            feature_importance_dict[fname] = float(importance_dict_raw.get(f'f{feature_names.index(fname)}', 0.0))

        # If the above fails, use weight-based importance
        if not feature_importance_dict or all(v == 0.0 for v in feature_importance_dict.values()):
            importance_dict_raw = xgb_final.get_booster().get_score(importance_type='weight')
            feature_importance_dict = {}
            for fname in feature_names:
                feature_importance_dict[fname] = float(importance_dict_raw.get(f'f{feature_names.index(fname)}', 0.0))
    except:
        # Fallback: use sklearn's feature_importances_ attribute
        feature_importance_dict = {
            name: float(importance)
            for name, importance in zip(feature_names, xgb_final.feature_importances_)
        }

    top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Top 10 SHAP features:")
    for i, (name, importance) in enumerate(top_features[:10], 1):
        print(f"    {i}. {name}: {importance:.4f}")

    return xgb_final, scaler, feature_names, cv_scores, feature_importance_dict, top_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', type=int, default=5, help='Number of CV folds')
    args = parser.parse_args()

    print("="*80)
    print("XGBoost v80 — Training XGBoost with 27 features + SHAP")
    print("="*80)

    if not DATASET.exists():
        print(f"ERROR: Dataset not found at {DATASET}")
        return None

    print(f"\n[1] Loading dataset...")
    X, y, feature_names = load_dataset(DATASET)

    print(f"\n[2] Training XGBoost...")
    xgb_model, scaler, features, cv_scores, importance_dict, top_features = train_xgboost(
        X, y, feature_names, cv_folds=args.cv
    )

    # Save model
    print(f"\n[3] Saving model...")
    model_data = {
        'clf': xgb_model,
        'scaler': scaler,
        'feature_names': features,
        'cv_auc_scores': cv_scores.tolist() if isinstance(cv_scores, np.ndarray) else cv_scores,
    }
    with open(CLF_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"    [OK] Model saved to {CLF_PATH.name}")

    # Generate report
    print(f"\n[4] Generating report...")
    report = {
        'model': 'XGBoost v80',
        'features_count': len(features),
        'feature_names': features,
        'cv_folds': args.cv,
        'cv_auc_scores': cv_scores.tolist() if isinstance(cv_scores, np.ndarray) else cv_scores,
        'cv_auc_mean': float(np.mean(cv_scores)),
        'cv_auc_std': float(np.std(cv_scores)),
        'hyperparameters': {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'shap_feature_importance': importance_dict,
        'top_10_features': [{'name': name, 'importance': importance} for name, importance in top_features[:10]],
    }

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"    [OK] Report saved to {REPORT_PATH.name}")

    print(f"\n" + "="*80)
    print(f"XGBoost v80 training complete!")
    print(f"="*80)
    print(f"CV AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    print(f"Top features: {', '.join([f[0] for f in top_features[:3]])}")

    return {
        'model': xgb_model,
        'scaler': scaler,
        'cv_auc': np.mean(cv_scores),
        'feature_names': features,
    }

if __name__ == '__main__':
    main()
