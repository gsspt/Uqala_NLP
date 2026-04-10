#!/usr/bin/env python3
"""
explain_predictions_simple.py
──────────────────────────────
Version simplifiée: génère explications basées sur feature importance
(évite les problèmes de compatibilité SHAP/XGBoost)

Usage:
  python comparison_ensemble/explain_predictions_simple.py --top_n 50
"""

import json
import sys
import pathlib
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scan"))
from build_features_71 import extract_features_71

BASE = pathlib.Path(__file__).parent.parent
DATASET = BASE / "dataset_raw.json"
COMP_DIR = pathlib.Path(__file__).parent
OUT_DIR = COMP_DIR / "results"
XGB_MODEL_PATH = OUT_DIR / "xgb_classifier_71features.pkl"
XGB_REPORT = OUT_DIR / "xgb_report_71features.json"
EXPLANATIONS_PATH = OUT_DIR / "shap_explanations.json"

def load_data_and_model():
    """Charge dataset et modèle"""
    with open(DATASET, encoding='utf-8') as f:
        data = json.load(f)

    X = []
    texts = []
    labels = []

    for item in data:
        feat = extract_features_71(item['text_ar'])
        X.append(list(feat.values()))
        texts.append(item['text_ar'][:400])
        labels.append(item['label'])

    X = np.array(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with open(XGB_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    with open(XGB_REPORT) as f:
        xgb_report = json.load(f)

    clf = model_data['clf']
    scaler = model_data['scaler']

    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    probs = clf.predict_proba(X_scaled)[:, 1]

    return X_scaled, texts, labels, probs, clf, xgb_report

def generate_simple_explanations(X, texts, labels, probs, clf, xgb_report, top_n=50):
    """
    Génère explications simples basées sur feature values + importance
    """

    # Get feature importance from model
    importance = xgb_report['feature_importance']
    feat_names = list(importance.keys())

    # Select top N predictions (mix high and low confidence)
    top_pos_idx = np.argsort(probs)[-top_n//2:][::-1]
    top_neg_idx = np.argsort(probs)[:top_n//2:]
    selected_idx = np.concatenate([top_pos_idx, top_neg_idx])

    explanations = []

    for i, idx in enumerate(selected_idx):
        if (i + 1) % 10 == 0:
            print(f"  Processing: {i+1}/{len(selected_idx)}")

        sample = X[idx]
        sample_probs = probs[idx]

        # Identify top contributing features (high importance × high value)
        contributions = []
        for j, feat_name in enumerate(feat_names):
            feat_value = sample[j]
            feat_importance = importance[feat_name]

            # Contribution proxy: importance * feature_value
            contrib = feat_importance * feat_value
            contributions.append((feat_name, feat_value, feat_importance, contrib))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x[3]), reverse=True)

        explanation = {
            'idx': int(idx),
            'text': texts[idx],
            'true_label': int(labels[idx]),
            'predicted_probability': float(sample_probs),
            'prediction': 1 if sample_probs >= 0.5 else 0,
            'correct': int((sample_probs >= 0.5) == labels[idx]),
            'top_contributing_features': [
                {
                    'name': name,
                    'feature_value': float(value),
                    'importance': float(imp),
                    'contribution': float(contrib)
                }
                for name, value, imp, contrib in contributions[:10]
            ]
        }

        explanations.append(explanation)

    return explanations

def print_examples(explanations, n=3):
    """Affiche exemples"""
    print("\n" + "="*70)
    print("EXEMPLES D'EXPLICATIONS")
    print("="*70)

    for i, exp in enumerate(explanations[:n]):
        label_text = "✓ POSITIF" if exp['true_label'] == 1 else "✗ NÉGATIF"
        pred_text = "✓ Correct" if exp['correct'] else "✗ ERREUR"

        print(f"\nExemple {i+1}: {label_text} | Prédiction: {pred_text}")
        print(f"Texte: {exp['text'][:80]}...")
        print(f"Score XGBoost: {exp['predicted_probability']:.1%}")
        print(f"Top 5 contributing features:")

        for j, feat in enumerate(exp['top_contributing_features'][:5], 1):
            direction = "↑" if feat['contribution'] > 0 else "↓"
            print(f"  {j}. {direction} {feat['name']:<35} " +
                  f"(value={feat['feature_value']:.3f}, importance={feat['importance']:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate simple explanations')
    parser.add_argument('--top_n', type=int, default=50, help='Number of samples')
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"ERROR: Dataset not found → {DATASET}")
        sys.exit(1)

    if not XGB_MODEL_PATH.exists():
        print(f"ERROR: XGBoost model not found → {XGB_MODEL_PATH}")
        sys.exit(1)

    print("Loading data and model…")
    X, texts, labels, probs, clf, xgb_report = load_data_and_model()

    print(f"Generating {args.top_n} simple explanations…")
    explanations = generate_simple_explanations(X, texts, labels, probs, clf, xgb_report, top_n=args.top_n)

    # Save
    with open(EXPLANATIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(explanations, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Explanations saved → {EXPLANATIONS_PATH}")

    # Print examples
    print_examples(explanations)

    print("\n" + "="*70)
    print("ANALYSE DES PRÉDICTIONS")
    print("="*70)

    correct = sum(1 for e in explanations if e['correct'])
    accuracy = correct / len(explanations) * 100

    print(f"Accuracy sur {len(explanations)} cas: {accuracy:.1f}%")
    print(f"Correctes: {correct}, Erreurs: {len(explanations) - correct}")

    # Cas d'erreurs
    errors = [e for e in explanations if not e['correct']]
    if errors:
        print(f"\n{len(errors)} erreurs trouvées. Exemples:")
        for i, err in enumerate(errors[:3], 1):
            true_label = "Positif" if err['true_label'] == 1 else "Négatif"
            pred_label = "Positif" if err['prediction'] == 1 else "Négatif"
            print(f"  {i}. Vrai: {true_label}, Prédit: {pred_label} (score: {err['predicted_probability']:.1%})")

    print("\n" + "="*70)
    print("Pour votre thèse:")
    print("  - Sélectionnez 3-5 cas représentatifs dans le JSON")
    print("  - Copiez la structure de THESIS_TEMPLATE.md section 4.5")
    print("  - Pour chaque cas, listez les top_contributing_features")
    print("="*70)
