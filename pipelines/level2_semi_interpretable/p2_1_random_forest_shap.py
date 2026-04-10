#!/usr/bin/env python3
"""
p2_1_random_forest_shap.py  (ex train_xgboost_71features.py)
──────────────────────────────────────────────────────────────
Entraîne un classifier XGBoost avec 79 features (v79) pour comparaison avec LR.

Utilise les mêmes 79 features que p1_3_logistic_regression.py.

Pipeline:
  1. Charge dataset_raw.json et extrait features via p1_3_logistic_regression
  2. Entraîne XGBoost avec hyperparamètres optimisés + régularisation renforcée
  3. Validation croisée 5-fold
  4. Sauvegarde modèle + rapport + feature importance

Usage:
  python pipelines/level2_semi_interpretable/p2_1_random_forest_shap.py --cv 5
"""

import json
import sys
import pathlib
import argparse
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')

# Import feature extraction depuis p1_3_logistic_regression (79 features v79)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "level1_interpretable"))
from p1_3_logistic_regression import extract_features_71

BASE         = pathlib.Path(__file__).resolve().parent.parent.parent  # repo root
DATASET      = BASE / "data" / "raw" / "dataset_raw.json"
OUT_DIR      = BASE / "models"
OUT_DIR.mkdir(exist_ok=True)

XGB_MODEL_PATH  = OUT_DIR / "xgb_classifier_79features.pkl"
XGB_REPORT_PATH = OUT_DIR / "xgb_report_79features.json"

def load_dataset(dataset_path):
    """Charge dataset et extrait features (réutilise build_features_71)"""
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

def train_xgboost(X, y, cv_folds=5):
    """Entraîne XGBoost avec hyperparamètres optimisés pour ce dataset"""

    # Fix NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # XGBoost ne nécessite pas de scaling, mais on le fait pour cohérence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Hyperparamètres avec régularisation renforcée pour éviter l'overfit
    # (460 positifs / 3817 négatifs = ratio 1:8.3)
    # Ancien modèle : CV AUC=0.846, Test AUC=0.991 → overfit sévère
    # Corrections : max_depth 5→3, n_estimators 100→200, reg_alpha/lambda ajoutés
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=3,           # réduit pour moins de variance
        learning_rate=0.05,    # plus lent = meilleure généralisation
        n_estimators=200,      # plus d'itérations pour compenser lr faible
        subsample=0.7,         # réduit pour moins de variance
        colsample_bytree=0.7,  # idem
        reg_alpha=0.1,         # L1 — nouveau
        reg_lambda=1.0,        # L2 — renforci
        min_child_weight=5,    # évite les feuilles sur peu de données
        random_state=42,
        scale_pos_weight=sum(y == 0) / sum(y == 1),  # Équilibre les classes
        eval_metric='auc',
        tree_method='hist',
        verbosity=0
    )

    # Cross-validation
    print(f"\n  Computing cross-validation scores (k={cv_folds})...")
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    print(f"  CV AUC (k={cv_folds}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full data
    clf.fit(X_scaled, y)

    # Test split (mêmes seed et stratégie que LR pour comparaison)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  Test AUC: {test_auc:.4f}")

    # Classification report
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Build feature names (v79 — identique à p1_3_logistic_regression.py)
    feat_names = [
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
        # f28_has_qala supprimé
        'f29_qala_density', 'f30_has_first_person',
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
        'f65_root_jnn_density', 'f66_root_aql_density', 'f67_root_hikma_density',
        'f68_verb_density', 'f69_noun_density', 'f70_adj_density',
        # f71-f73 supprimés
        'f74_question_mark_presence', 'f75_question_mark_density',
        'f76_religious_intensity', 'f77_first_person_scene',
        'f78_direct_address', 'f79_physical_reaction',
        'f80_fool_location', 'f81_mystical_verb', 'f82_love_madness',
    ]

    # Rapport
    xgb_report = {
        'algorithm': 'XGBoost',
        'n_features': X.shape[1],
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'test_auc': float(test_auc),
        'n_positives': int(sum(y == 1)),
        'n_negatives': int(sum(y == 0)),
        'hyperparameters': {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 5,
            'scale_pos_weight': float(sum(y == 0) / sum(y == 1))
        },
        'feature_importance': dict(zip(
            feat_names,
            [float(x) for x in clf.feature_importances_]
        )),
        'classification_report': report,
    }

    return clf, scaler, xgb_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XGBoost classifier with 79 features (v79)')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"ERROR: Dataset not found → {DATASET}")
        sys.exit(1)

    print(f"Loading data…")
    X, y = load_dataset(DATASET)

    print(f"\nTraining XGBoost (79 features v79, régularisation renforcée)…")
    clf, scaler, report = train_xgboost(X, y, cv_folds=args.cv)

    # Save
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler}, f)
    print(f"\nXGBoost model saved → {XGB_MODEL_PATH}")

    with open(XGB_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved → {XGB_REPORT_PATH}")

    # Print top features
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Algorithm: XGBoost")
    print(f"Features: {report['n_features']}")
    print(f"CV AUC: {report['cv_auc_mean']:.4f} ± {report['cv_auc_std']:.4f}")
    print(f"Test AUC: {report['test_auc']:.4f}")
    print(f"\nTop 15 important features:")
    importance = report['feature_importance']
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for fname, score in sorted_importance[:15]:
        print(f"  {fname:35s} = {score:.4f}")
    print(f"{'='*60}")
