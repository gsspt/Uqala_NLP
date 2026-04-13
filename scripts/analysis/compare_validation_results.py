#!/usr/bin/env python3
"""
compare_validation_results.py
──────────────────────────────────────────────────────────────
Compare LR v80, XGBoost v80, and Ensemble v80 validation results
with fuzzy matching on texts.

Usage:
  python scripts/compare_validation_results.py
"""

import json
import sys
import pathlib
from difflib import SequenceMatcher

sys.stdout.reconfigure(encoding='utf-8')

BASE = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results" / "0328IbnCabdRabbih"

# Load results
def load_results(filename):
    """Load JSON results and extract positive akhbars"""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"ERROR: {filename} not found")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    positives = data.get('positives_detailed', [])
    # Create identifiers: (filename, khabar_num)
    identifiers = {
        (p['filename'], p['khabar_num']): p for p in positives
    }
    return identifiers


print("="*100)
print("COMPARING VALIDATION RESULTS: LR v80 vs XGBoost v80 vs Ensemble v80")
print("="*100)

print("\n[1] Loading results...")

lr_results = load_results("v80_validation_proper_akhbars.json")
xgb_results = load_results("xgboost_v80_validation_proper_akhbars.json")
ensemble_results = load_results("ensemble_v80_validation_results.json")

if not (lr_results and xgb_results and ensemble_results):
    print("ERROR: Could not load all results")
    sys.exit(1)

print(f"    LR v80:     {len(lr_results)} positives")
print(f"    XGBoost v80: {len(xgb_results)} positives")
print(f"    Ensemble:    {len(ensemble_results)} positives")

# Create sets of identifiers
lr_set = set(lr_results.keys())
xgb_set = set(xgb_results.keys())
ensemble_set = set(ensemble_results.keys())

# Venn diagram-style analysis
print(f"\n[2] Agreement analysis:")
print(f"    " + "-"*96)

# All three agree
all_three = lr_set & xgb_set & ensemble_set
print(f"    ✓ All three agree:        {len(all_three)} ({100*len(all_three)/len(ensemble_set):.1f}%)")

# Two agree
lr_xgb = (lr_set & xgb_set) - ensemble_set
lr_ens = (lr_set & ensemble_set) - xgb_set
xgb_ens = (xgb_set & ensemble_set) - lr_set

print(f"    ✓ LR + XGBoost (not Ens):  {len(lr_xgb)}")
print(f"    ✓ LR + Ensemble (not XGB): {len(lr_ens)}")
print(f"    ✓ XGBoost + Ensemble (not LR): {len(xgb_ens)}")

# Only one
lr_only = lr_set - xgb_set - ensemble_set
xgb_only = xgb_set - lr_set - ensemble_set
ens_only = ensemble_set - lr_set - xgb_set

print(f"    ✗ LR only:                 {len(lr_only)}")
print(f"    ✗ XGBoost only:            {len(xgb_only)}")
print(f"    ✗ Ensemble only:           {len(ens_only)}")

# Calculate agreement percentages
agreement_lr_xgb = len(lr_set & xgb_set) / max(len(lr_set | xgb_set), 1) * 100
agreement_lr_ens = len(lr_set & ensemble_set) / max(len(lr_set | ensemble_set), 1) * 100
agreement_xgb_ens = len(xgb_set & ensemble_set) / max(len(xgb_set | ensemble_set), 1) * 100

print(f"\n    Pairwise Jaccard agreement:")
print(f"    LR ∩ XGBoost / LR ∪ XGBoost:  {agreement_lr_xgb:.1f}%")
print(f"    LR ∩ Ensemble / LR ∪ Ensemble: {agreement_lr_ens:.1f}%")
print(f"    XGBoost ∩ Ensemble / XGBoost ∪ Ensemble: {agreement_xgb_ens:.1f}%")

# Fuzzy matching for disagreement cases
print(f"\n[3] Fuzzy matching for disagreement cases:")
print(f"    " + "-"*96)

def fuzzy_match(text1, text2, threshold=0.8):
    """Calculate similarity ratio between two texts"""
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()

# Cases where LR and XGBoost disagree
lr_xgb_disagree = (lr_set ^ xgb_set)  # Symmetric difference
if lr_xgb_disagree:
    print(f"\n    LR ≠ XGBoost ({len(lr_xgb_disagree)} cases):")
    fuzzy_matches = []

    for key in list(lr_xgb_disagree)[:5]:  # Show first 5
        if key in lr_results:
            lr_pred = lr_results[key]['pred_proba']
            xgb_text = None
            xgb_pred = None
            status = "LR only"
        else:
            xgb_pred = xgb_results[key]['pred_proba']
            lr_text = None
            lr_pred = None
            status = "XGBoost only"

        filename, khabar_num = key
        print(f"      • {filename} khabar {khabar_num} ({status})")
        if key in lr_results:
            print(f"        LR proba: {lr_results[key]['pred_proba']:.3f}")
        if key in xgb_results:
            print(f"        XGB proba: {xgb_results[key]['pred_proba']:.3f}")

else:
    print(f"\n    LR and XGBoost perfectly agree!")

# Cases where Ensemble disagrees with both
ens_disagree = ens_only | lr_only | xgb_only
if ens_disagree:
    print(f"\n    Ensemble vs (LR or XGBoost) ({len(ens_disagree)} cases):")
    for key in list(ens_disagree)[:5]:  # Show first 5
        filename, khabar_num = key
        status = None
        proba = None

        if key in ensemble_results:
            status = "Ensemble only"
            proba = ensemble_results[key]['pred_ensemble']
        elif key in lr_set:
            status = "LR (not Ensemble)"
            proba = lr_results[key]['pred_proba']
        else:
            status = "XGBoost (not Ensemble)"
            proba = xgb_results[key]['pred_proba']

        print(f"      • {filename} khabar {khabar_num} ({status})")
        print(f"        Proba: {proba:.3f}")

# Summary statistics
print(f"\n[4] Model statistics:")
print(f"    " + "-"*96)

lr_probas = [p['pred_proba'] for p in lr_results.values()]
xgb_probas = [p['pred_proba'] for p in xgb_results.values()]
ensemble_probas = [p['pred_ensemble'] for p in ensemble_results.values()]

import statistics

print(f"    LR v80 probabilities:      mean={statistics.mean(lr_probas):.3f}, median={statistics.median(lr_probas):.3f}, min={min(lr_probas):.3f}, max={max(lr_probas):.3f}")
print(f"    XGBoost v80 probabilities: mean={statistics.mean(xgb_probas):.3f}, median={statistics.median(xgb_probas):.3f}, min={min(xgb_probas):.3f}, max={max(xgb_probas):.3f}")
print(f"    Ensemble probabilities:    mean={statistics.mean(ensemble_probas):.3f}, median={statistics.median(ensemble_probas):.3f}, min={min(ensemble_probas):.3f}, max={max(ensemble_probas):.3f}")

# Save comparison report
print(f"\n[5] Saving comparison report...")

comparison_file = RESULTS_DIR / "comparison_lr_xgb_ensemble.json"
with open(comparison_file, 'w', encoding='utf-8') as f:
    json.dump({
        'lr_count': len(lr_results),
        'xgb_count': len(xgb_results),
        'ensemble_count': len(ensemble_results),
        'all_three_agree': len(all_three),
        'lr_xgb_only': len(lr_xgb),
        'lr_ensemble_only': len(lr_ens),
        'xgb_ensemble_only': len(xgb_ens),
        'lr_only': len(lr_only),
        'xgb_only': len(xgb_only),
        'ensemble_only': len(ens_only),
        'agreement_jaccard': {
            'lr_xgb': agreement_lr_xgb,
            'lr_ensemble': agreement_lr_ens,
            'xgb_ensemble': agreement_xgb_ens,
        },
        'model_stats': {
            'lr': {
                'mean_proba': statistics.mean(lr_probas),
                'median_proba': statistics.median(lr_probas),
                'min_proba': min(lr_probas),
                'max_proba': max(lr_probas),
            },
            'xgb': {
                'mean_proba': statistics.mean(xgb_probas),
                'median_proba': statistics.median(xgb_probas),
                'min_proba': min(xgb_probas),
                'max_proba': max(xgb_probas),
            },
            'ensemble': {
                'mean_proba': statistics.mean(ensemble_probas),
                'median_proba': statistics.median(ensemble_probas),
                'min_proba': min(ensemble_probas),
                'max_proba': max(ensemble_probas),
            },
        },
        'disagreement_samples': {
            'lr_xgb_disagree_count': len(lr_xgb_disagree),
            'ensemble_disagree_count': len(ens_disagree),
        }
    }, f, ensure_ascii=False, indent=2)

print(f"    [OK] Report saved to: {comparison_file.name}")

print(f"\n" + "="*100)
print("COMPARISON COMPLETE")
print("="*100)
