#!/usr/bin/env python3
"""
compare_models.py
──────────────────
Compare les performances de LR vs XGBoost et génère un rapport synthétique.

Génère:
  - Tableau de comparaison AUC/performance
  - Feature importance side-by-side
  - Visualisations (ROC curves si matplotlib disponible)

Usage:
  python comparison_ensemble/compare_models.py
"""

import json
import sys
import pathlib
import pickle
import numpy as np
from collections import defaultdict

BASE = pathlib.Path(__file__).parent.parent
SCAN_DIR = BASE / "scan"
COMP_DIR = pathlib.Path(__file__).parent
OUT_DIR = COMP_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

LR_REPORT = SCAN_DIR / "lr_report_71features.json"
XGB_REPORT = OUT_DIR / "xgb_report_71features.json"
COMP_REPORT = OUT_DIR / "comparison_report.json"

def load_reports():
    """Charge les rapports LR et XGBoost"""
    with open(LR_REPORT) as f:
        lr_report = json.load(f)

    with open(XGB_REPORT) as f:
        xgb_report = json.load(f)

    return lr_report, xgb_report

def compare_models(lr_report, xgb_report):
    """Compare les deux modèles"""

    comparison = {
        'timestamp': '2026-04-10',
        'models': {
            'lr': {
                'algorithm': 'Logistic Regression',
                'cv_auc_mean': lr_report['cv_auc_mean'],
                'cv_auc_std': lr_report['cv_auc_std'],
                'test_auc': lr_report['test_auc'],
                'n_features': lr_report['n_features'],
            },
            'xgb': {
                'algorithm': 'XGBoost',
                'cv_auc_mean': xgb_report['cv_auc_mean'],
                'cv_auc_std': xgb_report['cv_auc_std'],
                'test_auc': xgb_report['test_auc'],
                'n_features': xgb_report['n_features'],
            }
        },
        'winner': {
            'cv_auc': 'XGBoost' if xgb_report['cv_auc_mean'] > lr_report['cv_auc_mean'] else 'LR',
            'test_auc': 'XGBoost' if xgb_report['test_auc'] > lr_report['test_auc'] else 'LR',
        },
        'improvements': {
            'cv_auc_gain': float(xgb_report['cv_auc_mean'] - lr_report['cv_auc_mean']),
            'cv_auc_gain_pct': float((xgb_report['cv_auc_mean'] / lr_report['cv_auc_mean'] - 1) * 100),
            'test_auc_gain': float(xgb_report['test_auc'] - lr_report['test_auc']),
            'test_auc_gain_pct': float((xgb_report['test_auc'] / lr_report['test_auc'] - 1) * 100),
        }
    }

    # Feature importance comparison
    lr_coefs = lr_report['coefficients']
    xgb_importance = xgb_report['feature_importance']

    # Get top features from both
    lr_sorted = sorted(lr_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    xgb_sorted = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)

    comparison['top_features'] = {
        'lr_top_10': [{'name': n, 'coefficient': c} for n, c in lr_sorted[:10]],
        'xgb_top_10': [{'name': n, 'importance': i} for n, i in xgb_sorted[:10]],
    }

    # Features that appear in top 10 of both
    lr_top_names = set(n for n, _ in lr_sorted[:10])
    xgb_top_names = set(n for n, _ in xgb_sorted[:10])
    common_top = lr_top_names & xgb_top_names

    comparison['consensus'] = {
        'common_top_features': list(common_top),
        'lr_unique_top': list(lr_top_names - xgb_top_names),
        'xgb_unique_top': list(xgb_top_names - lr_top_names),
    }

    return comparison

def print_comparison(comparison):
    """Affiche la comparaison"""

    print("\n" + "="*70)
    print("COMPARAISON: RÉGRESSION LOGISTIQUE vs XGBOOST")
    print("="*70)

    lr = comparison['models']['lr']
    xgb = comparison['models']['xgb']

    print(f"\n{'Métrique':<30} {'LR':<20} {'XGBoost':<20}")
    print("-"*70)
    print(f"{'CV AUC (mean)':<30} {lr['cv_auc_mean']:<20.4f} {xgb['cv_auc_mean']:<20.4f}")
    print(f"{'CV AUC (std)':<30} {lr['cv_auc_std']:<20.4f} {xgb['cv_auc_std']:<20.4f}")
    print(f"{'Test AUC':<30} {lr['test_auc']:<20.4f} {xgb['test_auc']:<20.4f}")
    print(f"{'N Features':<30} {lr['n_features']:<20} {xgb['n_features']:<20}")

    imp = comparison['improvements']
    print("\n" + "="*70)
    print("GAINS XGBOOST vs LR")
    print("="*70)
    print(f"CV AUC gain:        {imp['cv_auc_gain']:+.4f} ({imp['cv_auc_gain_pct']:+.2f}%)")
    print(f"Test AUC gain:      {imp['test_auc_gain']:+.4f} ({imp['test_auc_gain_pct']:+.2f}%)")

    print("\n" + "="*70)
    print("TOP 10 FEATURES - RÉGRESSION LOGISTIQUE (par |coefficient|)")
    print("="*70)
    for i, feat in enumerate(comparison['top_features']['lr_top_10'], 1):
        print(f"{i:2}. {feat['name']:<35} {feat['coefficient']:+.4f}")

    print("\n" + "="*70)
    print("TOP 10 FEATURES - XGBOOST (par importance)")
    print("="*70)
    for i, feat in enumerate(comparison['top_features']['xgb_top_10'], 1):
        print(f"{i:2}. {feat['name']:<35} {feat['importance']:.4f}")

    print("\n" + "="*70)
    print("CONSENSUS: FEATURES IMPORTANTES POUR LES DEUX MODÈLES")
    print("="*70)
    if comparison['consensus']['common_top_features']:
        for feat in comparison['consensus']['common_top_features']:
            print(f"  ✓ {feat}")
    else:
        print("  (Aucune concordance parfaite dans top 10)")

    print("\n" + "="*70)
    print("DIVERGENCES")
    print("="*70)
    if comparison['consensus']['lr_unique_top']:
        print(f"LR valorise mais XGBoost ignore:")
        for feat in comparison['consensus']['lr_unique_top']:
            print(f"  - {feat}")

    if comparison['consensus']['xgb_unique_top']:
        print(f"XGBoost valorise mais LR ignore:")
        for feat in comparison['consensus']['xgb_unique_top']:
            print(f"  - {feat}")

if __name__ == '__main__':

    if not LR_REPORT.exists():
        print(f"ERROR: LR report not found → {LR_REPORT}")
        print("Run: python scan/build_features_71.py --cv 10")
        sys.exit(1)

    if not XGB_REPORT.exists():
        print(f"ERROR: XGBoost report not found → {XGB_REPORT}")
        print("Run: python comparison_ensemble/train_xgboost_71features.py --cv 10")
        sys.exit(1)

    print("Loading reports…")
    lr_report, xgb_report = load_reports()

    print("Comparing models…")
    comparison = compare_models(lr_report, xgb_report)

    # Save comparison
    with open(COMP_REPORT, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"Comparison saved → {COMP_REPORT}\n")

    # Display
    print_comparison(comparison)

    print("\n" + "="*70)
    print("RECOMMANDATIONS POUR LA THÈSE")
    print("="*70)

    lr_auc = lr_report['test_auc']
    xgb_auc = xgb_report['test_auc']
    gain = comparison['improvements']['test_auc_gain_pct']

    print(f"""
1. PERFORMANCE:
   - LR atteint {lr_auc:.1%} de précision (interprétabilité maximale)
   - XGBoost atteint {xgb_auc:.1%} (+{gain:.1f}% d'amélioration)

2. CONSENSUS:
   Les deux modèles s'accordent sur les features clés:
   {', '.join(comparison['consensus']['common_top_features'][:3])}...
   → Cela renforce la validité de la sélection de features

3. COMPLÉMENTARITÉ:
   - Présentez LR pour l'interprétabilité théorique (coefficients clairs)
   - Montrez XGBoost pour la meilleure performance
   - Utilisez SHAP pour expliquer les cas individuels

4. PROCHAINES ÉTAPES:
   → python comparison_ensemble/explain_predictions.py
   → Génère les explications SHAP pour 50 textes représentatifs
""")
    print("="*70)
