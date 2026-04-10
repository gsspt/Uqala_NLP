#!/usr/bin/env python3
"""
visualize_importance.py
────────────────────────
Crée des visualisations pour comparer LR et XGBoost.

Génère:
  - Feature importance comparison (LR coefficients vs XGBoost importance)
  - ROC curves comparison
  - Histogrammes de scores

Usage:
  python comparison_ensemble/visualize_importance.py
"""

import json
import sys
import pathlib
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not available. Skipping visualizations.")
    HAS_MATPLOTLIB = False

sys.stdout.reconfigure(encoding='utf-8')

# Import feature extraction
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scan"))
from build_features_71 import extract_features_71

BASE = pathlib.Path(__file__).parent.parent
DATASET = BASE / "dataset_raw.json"
SCAN_DIR = BASE / "scan"
COMP_DIR = pathlib.Path(__file__).parent
OUT_DIR = COMP_DIR / "results"
VIZ_DIR = OUT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

LR_MODEL_PATH = SCAN_DIR / "lr_classifier_71features.pkl"
LR_REPORT = SCAN_DIR / "lr_report_71features.json"
XGB_MODEL_PATH = OUT_DIR / "xgb_classifier_71features.pkl"
XGB_REPORT = OUT_DIR / "xgb_report_71features.json"

def load_data_and_models():
    """Charge les données et les deux modèles"""
    with open(DATASET, encoding='utf-8') as f:
        data = json.load(f)

    X = []
    y = []

    for item in data:
        feat = extract_features_71(item['text_ar'])
        X.append(list(feat.values()))
        y.append(item['label'])

    X = np.array(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)

    # Load LR model
    with open(LR_MODEL_PATH, 'rb') as f:
        lr_data = pickle.load(f)
    lr_clf = lr_data['clf']
    lr_scaler = lr_data['scaler']
    X_lr_scaled = lr_scaler.transform(X)
    X_lr_scaled = np.nan_to_num(X_lr_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    lr_probs = lr_clf.predict_proba(X_lr_scaled)[:, 1]

    # Load XGBoost model
    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_data = pickle.load(f)
    xgb_clf = xgb_data['clf']
    xgb_scaler = xgb_data['scaler']
    X_xgb_scaled = xgb_scaler.transform(X)
    X_xgb_scaled = np.nan_to_num(X_xgb_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    xgb_probs = xgb_clf.predict_proba(X_xgb_scaled)[:, 1]

    # Load reports
    with open(LR_REPORT) as f:
        lr_report = json.load(f)
    with open(XGB_REPORT) as f:
        xgb_report = json.load(f)

    return X, y, lr_probs, xgb_probs, lr_report, xgb_report

def plot_feature_importance_comparison(lr_report, xgb_report):
    """Compare feature importance side-by-side"""

    if not HAS_MATPLOTLIB:
        print("Skipping feature importance visualization (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # LR: Top 15 by absolute coefficient
    lr_coefs = lr_report['coefficients']
    lr_sorted = sorted(lr_coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    lr_names = [x[0] for x in lr_sorted]
    lr_values = [x[1] for x in lr_sorted]

    colors_lr = ['green' if v > 0 else 'red' for v in lr_values]
    axes[0].barh(range(len(lr_names)), lr_values, color=colors_lr, alpha=0.7)
    axes[0].set_yticks(range(len(lr_names)))
    axes[0].set_yticklabels([n.replace('f', '') for n in lr_names], fontsize=9)
    axes[0].set_xlabel('Coefficient', fontsize=11)
    axes[0].set_title('Logistic Regression\nTop 15 Features (by |coefficient|)', fontsize=12, fontweight='bold')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].grid(axis='x', alpha=0.3)

    # XGBoost: Top 15 by importance
    xgb_importance = xgb_report['feature_importance']
    xgb_sorted = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    xgb_names = [x[0] for x in xgb_sorted]
    xgb_values = [x[1] for x in xgb_sorted]

    axes[1].barh(range(len(xgb_names)), xgb_values, color='steelblue', alpha=0.7)
    axes[1].set_yticks(range(len(xgb_names)))
    axes[1].set_yticklabels([n.replace('f', '') for n in xgb_names], fontsize=9)
    axes[1].set_xlabel('Feature Importance (Gain)', fontsize=11)
    axes[1].set_title('XGBoost\nTop 15 Features (by importance)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: feature_importance_comparison.png")
    plt.close()

def plot_roc_curves(y, lr_probs, xgb_probs):
    """Compare ROC curves"""

    if not HAS_MATPLOTLIB:
        print("Skipping ROC curve visualization (matplotlib not available)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # LR ROC
    fpr_lr, tpr_lr, _ = roc_curve(y, lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)
    ax.plot(fpr_lr, tpr_lr, label=f'LR (AUC = {auc_lr:.4f})', linewidth=2.5, color='orange')

    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y, xgb_probs)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.4f})', linewidth=2.5, color='steelblue')

    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: LR vs XGBoost', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: roc_curves.png")
    plt.close()

def plot_score_distributions(y, lr_probs, xgb_probs):
    """Histogrammes des scores"""

    if not HAS_MATPLOTLIB:
        print("Skipping score distribution visualization (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LR distribution
    axes[0].hist(lr_probs[y == 0], bins=30, alpha=0.6, label='Negative', color='red', edgecolor='black')
    axes[0].hist(lr_probs[y == 1], bins=30, alpha=0.6, label='Positive (Fool)', color='green', edgecolor='black')
    axes[0].set_xlabel('LR Probability Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Logistic Regression\nScore Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

    # XGBoost distribution
    axes[1].hist(xgb_probs[y == 0], bins=30, alpha=0.6, label='Negative', color='red', edgecolor='black')
    axes[1].hist(xgb_probs[y == 1], bins=30, alpha=0.6, label='Positive (Fool)', color='green', edgecolor='black')
    axes[1].set_xlabel('XGBoost Probability Score', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('XGBoost\nScore Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'score_distributions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: score_distributions.png")
    plt.close()

if __name__ == '__main__':

    if not all([LR_MODEL_PATH.exists(), XGB_MODEL_PATH.exists(),
                LR_REPORT.exists(), XGB_REPORT.exists()]):
        print("ERROR: Missing models or reports")
        print("Run these first:")
        print("  python scan/build_features_71.py --cv 10")
        print("  python comparison_ensemble/train_xgboost_71features.py --cv 10")
        sys.exit(1)

    print("Loading data and models…")
    X, y, lr_probs, xgb_probs, lr_report, xgb_report = load_data_and_models()

    print("Generating visualizations…")

    if HAS_MATPLOTLIB:
        plot_feature_importance_comparison(lr_report, xgb_report)
        plot_roc_curves(y, lr_probs, xgb_probs)
        plot_score_distributions(y, lr_probs, xgb_probs)
        print(f"\n✓ All visualizations saved to: {VIZ_DIR}")
    else:
        print("matplotlib not available. Install with: pip install matplotlib")

    print("\n" + "="*70)
    print("VISUALIZATIONS SAVED")
    print("="*70)
    print(f"""
Location: {VIZ_DIR}

Files:
  1. feature_importance_comparison.png
     → Side-by-side comparison of most important features
     → Green bars: positive coefficients (LR) / importances (XGBoost)
     → Red bars: negative coefficients

  2. roc_curves.png
     → ROC curves for both models
     → Shows trade-off between TPR and FPR

  3. score_distributions.png
     → Histograms showing how scores separate positive/negative cases
     → Higher separation = better discrimination

Use these images in your thesis section 4.3 (Results & Comparison)
""")
    print("="*70)
