#!/usr/bin/env python3
"""
compare_lr_xgboost.py
─────────────────────
Compare en détail les prédictions LR et XGBoost sur openITI.
Analyse:
  - Divergences (un modèle dit oui, l'autre dit non)
  - Consensus (les deux d'accord)
  - Statistiques de confiance

Usage:
  python openiti_detection/compare_lr_xgboost.py

Entrées (générées par detect_lr_xgboost.py):
  openiti_detection/results/lr_predictions.json
  openiti_detection/results/xgb_predictions.json

Sortie:
  openiti_detection/results/detailed_comparison.json
"""

import json
import sys
import pathlib
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = pathlib.Path(__file__).parent / "results"
LR_RESULTS_PATH = OUT_DIR / "lr_predictions.json"
XGB_RESULTS_PATH = OUT_DIR / "xgb_predictions.json"
COMPARISON_PATH = OUT_DIR / "detailed_comparison.json"

def main():
    # Charger les résultats
    print("Loading prediction results…")

    try:
        with open(LR_RESULTS_PATH, 'r', encoding='utf-8') as f:
            lr_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ LR results not found: {LR_RESULTS_PATH}")
        print("   Run: python openiti_detection/detect_lr_xgboost.py")
        sys.exit(1)

    try:
        with open(XGB_RESULTS_PATH, 'r', encoding='utf-8') as f:
            xgb_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ XGBoost results not found: {XGB_RESULTS_PATH}")
        print("   Run: python openiti_detection/detect_lr_xgboost.py")
        sys.exit(1)

    # Créer un index par path
    lr_by_path = {r['path']: r for r in lr_results}
    xgb_by_path = {r['path']: r for r in xgb_results}

    # Trouver les chemins communs
    common_paths = set(lr_by_path.keys()) & set(xgb_by_path.keys())
    print(f"✓ {len(common_paths)} common texts found")

    # Analyser les résultats
    consensus_both_positive = []
    consensus_both_negative = []
    divergence_lr_yes_xgb_no = []
    divergence_lr_no_xgb_yes = []

    confidence_diffs = []

    for path in common_paths:
        lr = lr_by_path[path]
        xgb = xgb_by_path[path]

        lr_pred = lr['lr_prediction']
        xgb_pred = xgb['xgb_prediction']
        lr_prob = lr['lr_probability']
        xgb_prob = xgb['xgb_probability']
        conf_diff = abs(lr_prob - xgb_prob)

        entry = {
            'path': path,
            'text_length': lr['text_length'],
            'lr_prob': lr_prob,
            'xgb_prob': xgb_prob,
            'avg_prob': (lr_prob + xgb_prob) / 2,
            'conf_diff': conf_diff
        }

        if lr_pred == 1 and xgb_pred == 1:
            consensus_both_positive.append(entry)
        elif lr_pred == 0 and xgb_pred == 0:
            consensus_both_negative.append(entry)
        elif lr_pred == 1 and xgb_pred == 0:
            divergence_lr_yes_xgb_no.append(entry)
        else:  # lr_pred == 0 and xgb_pred == 1
            divergence_lr_no_xgb_yes.append(entry)

        confidence_diffs.append(conf_diff)

    # Statistiques
    print(f"\nResults Summary:")
    print(f"  Consensus (both positive): {len(consensus_both_positive)}")
    print(f"  Consensus (both negative): {len(consensus_both_negative)}")
    print(f"  LR yes, XGBoost no: {len(divergence_lr_yes_xgb_no)}")
    print(f"  LR no, XGBoost yes: {len(divergence_lr_no_xgb_yes)}")

    print(f"\nConfidence analysis:")
    print(f"  Avg confidence difference: {np.mean(confidence_diffs):.4f}")
    print(f"  Max confidence difference: {np.max(confidence_diffs):.4f}")
    print(f"  Median confidence difference: {np.median(confidence_diffs):.4f}")

    # Trier par confiance moyenne
    consensus_both_positive.sort(key=lambda x: x['avg_prob'], reverse=True)
    divergence_lr_yes_xgb_no.sort(key=lambda x: x['lr_prob'], reverse=True)
    divergence_lr_no_xgb_yes.sort(key=lambda x: x['xgb_prob'], reverse=True)

    # Créer le rapport détaillé
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'total_texts_analyzed': len(common_paths),
        'statistics': {
            'consensus_both_positive': len(consensus_both_positive),
            'consensus_both_negative': len(consensus_both_negative),
            'divergence_lr_only': len(divergence_lr_yes_xgb_no),
            'divergence_xgb_only': len(divergence_lr_no_xgb_yes),
            'agreement_percentage': 100 * (len(consensus_both_positive) + len(consensus_both_negative)) / len(common_paths),
            'avg_confidence_difference': float(np.mean(confidence_diffs)),
            'max_confidence_difference': float(np.max(confidence_diffs)),
            'median_confidence_difference': float(np.median(confidence_diffs))
        },
        'consensus_positive': consensus_both_positive[:50],  # Top 50
        'divergence_lr_only': divergence_lr_yes_xgb_no[:20],  # Top 20
        'divergence_xgb_only': divergence_lr_no_xgb_yes[:20]   # Top 20
    }

    # Sauvegarder
    with open(COMPARISON_PATH, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Detailed comparison saved → {COMPARISON_PATH}")

    # Afficher les top divergences
    print("\n" + "="*70)
    print("TOP DIVERGENCES")
    print("="*70)

    if divergence_lr_yes_xgb_no:
        print(f"\nLR says YES, XGBoost says NO (top 5):")
        for i, d in enumerate(divergence_lr_yes_xgb_no[:5], 1):
            print(f"  {i}. {d['path']}")
            print(f"     LR: {d['lr_prob']:.3f} | XGBoost: {d['xgb_prob']:.3f} (diff: {d['conf_diff']:.3f})")

    if divergence_lr_no_xgb_yes:
        print(f"\nLR says NO, XGBoost says YES (top 5):")
        for i, d in enumerate(divergence_lr_no_xgb_yes[:5], 1):
            print(f"  {i}. {d['path']}")
            print(f"     LR: {d['lr_prob']:.3f} | XGBoost: {d['xgb_prob']:.3f} (diff: {d['conf_diff']:.3f})")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
