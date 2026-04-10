#!/usr/bin/env python3
"""
explain_predictions.py
───────────────────────
Génère des explications SHAP pour les prédictions XGBoost.

SHAP (SHapley Additive exPlanations) explique chaque prédiction en montrant
la contribution de chaque feature au score final.

Usage:
  python comparison_ensemble/explain_predictions.py --top_n 50
"""

import json
import sys
import pathlib
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import shap

sys.stdout.reconfigure(encoding='utf-8')

# Import feature extraction
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scan"))
from build_features_71 import extract_features_71

BASE = pathlib.Path(__file__).parent.parent
DATASET = BASE / "dataset_raw.json"
COMP_DIR = pathlib.Path(__file__).parent
OUT_DIR = COMP_DIR / "results"
XGB_MODEL_PATH = OUT_DIR / "xgb_classifier_71features.pkl"
SHAP_REPORT_PATH = OUT_DIR / "shap_explanations.json"

def load_data_and_model():
    """Charge dataset et modèle XGBoost"""
    with open(DATASET, encoding='utf-8') as f:
        data = json.load(f)

    X = []
    texts = []
    labels = []

    for item in data:
        feat = extract_features_71(item['text_ar'])
        X.append(list(feat.values()))
        texts.append(item['text_ar'][:400])  # Store first 400 chars
        labels.append(item['label'])

    X = np.array(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with open(XGB_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    clf = model_data['clf']
    scaler = model_data['scaler']

    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return X_scaled, texts, labels, clf

def generate_shap_explanations(X, texts, labels, clf, top_n=50):
    """
    Génère explications SHAP pour les top_n prédictions

    top_n: nombre de textes à expliquer
    """

    # Get predictions
    probs = clf.predict_proba(X)[:, 1]

    # Select top N (mix of high confidence positive and negative)
    # Top 25 highest scores, top 25 lowest scores
    top_pos_idx = np.argsort(probs)[-top_n//2:][::-1]
    top_neg_idx = np.argsort(probs)[:top_n//2:]

    selected_idx = np.concatenate([top_pos_idx, top_neg_idx])

    # Create SHAP explainer (TreeExplainer is fast for XGBoost)
    print("Creating SHAP explainer (this may take a minute)...")
    explainer = shap.TreeExplainer(clf)

    explanations = []

    # Get feature names
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
        'f65_root_jnn_density', 'f66_root_aql_density', 'f67_root_hikma_density',
        'f68_verb_density', 'f69_noun_density', 'f70_adj_density',
        'f71_perf_density', 'f72_imperf_density', 'f73_passive_voice_ratio',
    ]

    print(f"Generating SHAP values for {len(selected_idx)} samples...")

    # Compute SHAP values
    shap_values = explainer.shap_values(X[selected_idx])

    # For binary classification, shap_values is a list [class_0, class_1]
    # We use class_1 (positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    for i, idx in enumerate(selected_idx):
        if (i + 1) % 10 == 0:
            print(f"  Processing: {i+1}/{len(selected_idx)}")

        # Get the SHAP values for this sample
        sample_shap = shap_values[i]

        # Create explanation dict
        explanation = {
            'idx': int(idx),
            'text': texts[idx],
            'true_label': int(labels[idx]),
            'predicted_probability': float(probs[idx]),
            'shap_values': {}
        }

        # Get top 15 most important features (by absolute SHAP value)
        shap_importance = np.abs(sample_shap)
        top_shap_idx = np.argsort(shap_importance)[-15:][::-1]

        for shap_idx in top_shap_idx:
            feat_name = feat_names[shap_idx]
            shap_val = float(sample_shap[shap_idx])
            explanation['shap_values'][feat_name] = shap_val

        explanations.append(explanation)

    return explanations

def print_example_explanations(explanations, n_examples=3):
    """Affiche quelques exemples"""

    print("\n" + "="*70)
    print("EXEMPLE DE SHAP EXPLANATIONS")
    print("="*70)

    for i, exp in enumerate(explanations[:n_examples]):
        label_text = "✓ POSITIF (Fou Sensé)" if exp['true_label'] == 1 else "✗ NÉGATIF"
        print(f"\nExemple {i+1}: {label_text}")
        print(f"Texte: {exp['text'][:100]}...")
        print(f"Score XGBoost: {exp['predicted_probability']:.1%}")
        print(f"SHAP Contributions (top 5):")

        shap_sorted = sorted(exp['shap_values'].items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, val in shap_sorted[:5]:
            direction = "↑" if val > 0 else "↓"
            print(f"  {direction} {feat:<35} {val:+.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SHAP explanations')
    parser.add_argument('--top_n', type=int, default=50, help='Number of samples to explain')
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"ERROR: Dataset not found → {DATASET}")
        sys.exit(1)

    if not XGB_MODEL_PATH.exists():
        print(f"ERROR: XGBoost model not found → {XGB_MODEL_PATH}")
        print("Run: python comparison_ensemble/train_xgboost_71features.py --cv 10")
        sys.exit(1)

    print("Loading data and model…")
    X, texts, labels, clf = load_data_and_model()

    print(f"Generating {args.top_n} SHAP explanations…")
    explanations = generate_shap_explanations(X, texts, labels, clf, top_n=args.top_n)

    # Save
    with open(SHAP_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(explanations, f, ensure_ascii=False, indent=2)
    print(f"\nSHAP explanations saved → {SHAP_REPORT_PATH}")

    # Print examples
    print_example_explanations(explanations)

    print("\n" + "="*70)
    print("PROCHAINES ÉTAPES")
    print("="*70)
    print(f"""
1. Examinez le fichier JSON pour les cas intéressants:
   {SHAP_REPORT_PATH}

2. Pour chaque prédiction, visualisez:
   - Les features qui AUGMENTENT la confiance (SHAP > 0)
   - Les features qui DIMINUENT la confiance (SHAP < 0)

3. Sélectionnez 3-5 cas pour inclusion dans votre thèse:
   - 1-2 cas où le modèle est très confiant (score > 0.9)
   - 1-2 cas où le modèle hésite (0.4-0.6)
   - 1 cas mal classé (prédiction ≠ vérité)

4. Pour chacun, expliquez:
   "Le modèle détecte [features] qui indiquent le fou sensé.
    Dans ce texte, [features positives] dominent, donnant un score de [%]."
""")
    print("="*70)
