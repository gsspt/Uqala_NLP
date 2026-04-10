#!/usr/bin/env python3
"""
analyze_results.py
──────────────────
Analyse détaillée des résultats de test_quick.py

Usage:
  python openiti_detection/analyze_results.py 0328IbnCabdRabbih
"""

import json
import sys
import pathlib
import argparse
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

def analyze(author_code):
    """Analyse détaillée des résultats JSON."""

    results_dir = pathlib.Path(__file__).parent / "results" / author_code
    results_path = results_dir / "all_predictions.json"

    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        print(f"   Run: python openiti_detection/test_quick.py {author_code}")
        sys.exit(1)

    print(f"\nLoading {results_path.name}…")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"✓ Loaded {len(results)} results\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # CATEGORIES
    # ══════════════════════════════════════════════════════════════════════════════

    consensus = [r for r in results if r['consensus'] == 1]
    lr_only = [r for r in results if r['lr_pred'] == 1 and r['xgb_pred'] == 0]
    xgb_only = [r for r in results if r['xgb_pred'] == 1 and r['lr_pred'] == 0]
    lr_false_neg = [r for r in results if r['lr_pred'] == 0 and r['xgb_pred'] == 1]
    xgb_false_neg = [r for r in results if r['lr_pred'] == 1 and r['xgb_pred'] == 0]

    print("="*80)
    print("CATEGORIES")
    print("="*80)
    print(f"Total: {len(results)}")
    print(f"Consensus (both positive): {len(consensus)} ({100*len(consensus)/len(results):.1f}%)")
    print(f"LR only (LR pos, XGB neg): {len(lr_only)} ({100*len(lr_only)/len(results):.1f}%)")
    print(f"XGB only (XGB pos, LR neg): {len(xgb_only)} ({100*len(xgb_only)/len(results):.1f}%)")
    print(f"Both negative: {len(results) - len(consensus) - len(lr_only) - len(xgb_only)} ({100*(len(results) - len(consensus) - len(lr_only) - len(xgb_only))/len(results):.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════════
    # CONSENSUS - TOUS LES HITS
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print(f"CONSENSUS HITS ({len(consensus)} total)")
    print("="*80)

    consensus_sorted = sorted(consensus, key=lambda x: x['lr_prob'] + x['xgb_prob'], reverse=True)

    for i, hit in enumerate(consensus_sorted, 1):
        avg_prob = (hit['lr_prob'] + hit['xgb_prob']) / 2
        print(f"\n{i:3d}. [{avg_prob:.3f}] LR={hit['lr_prob']:.3f} XGB={hit['xgb_prob']:.3f}")
        print(f"     File: {hit['file']}")
        print(f"     Text: {hit['text'][:250]}")
        if len(hit['text']) > 250:
            print(f"           …")

    # ══════════════════════════════════════════════════════════════════════════════
    # LR ONLY - ANALYSE DÉTAILLÉE
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print(f"LR ONLY - FAUX POSITIFS? ({len(lr_only)} hits)")
    print("="*80)

    lr_only_sorted = sorted(lr_only, key=lambda x: x['lr_prob'], reverse=True)

    print("\nHigh confidence LR only (LR ≥ 0.9):")
    lr_only_high = [r for r in lr_only_sorted if r['lr_prob'] >= 0.9]
    for i, hit in enumerate(lr_only_high[:20], 1):
        print(f"\n{i:2d}. LR={hit['lr_prob']:.3f} XGB={hit['xgb_prob']:.3f}")
        print(f"    Text: {hit['text'][:200]}…")

    print(f"\nMedium confidence LR only (0.7 ≤ LR < 0.9):")
    lr_only_med = [r for r in lr_only_sorted if 0.7 <= r['lr_prob'] < 0.9]
    print(f"Count: {len(lr_only_med)}")
    for i, hit in enumerate(lr_only_med[:10], 1):
        print(f"{i:2d}. LR={hit['lr_prob']:.3f} XGB={hit['xgb_prob']:.3f} | {hit['text'][:150]}…")

    # ══════════════════════════════════════════════════════════════════════════════
    # XGB ONLY - ANALYSE DÉTAILLÉE
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print(f"XGB ONLY - MISSED BY LR? ({len(xgb_only)} hits)")
    print("="*80)

    xgb_only_sorted = sorted(xgb_only, key=lambda x: x['xgb_prob'], reverse=True)

    print("\nHigh confidence XGB only (XGB ≥ 0.95):")
    xgb_only_high = [r for r in xgb_only_sorted if r['xgb_prob'] >= 0.95]
    print(f"Count: {len(xgb_only_high)}")
    for i, hit in enumerate(xgb_only_high[:20], 1):
        print(f"\n{i:2d}. LR={hit['lr_prob']:.3f} XGB={hit['xgb_prob']:.3f}")
        print(f"    Text: {hit['text'][:200]}…")

    # ══════════════════════════════════════════════════════════════════════════════
    # SCORE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("SCORE DISTRIBUTIONS")
    print("="*80)

    import numpy as np

    all_lr = [r['lr_prob'] for r in results]
    all_xgb = [r['xgb_prob'] for r in results]

    print(f"\nLR scores:")
    print(f"  Min: {min(all_lr):.3f}")
    print(f"  Max: {max(all_lr):.3f}")
    print(f"  Mean: {np.mean(all_lr):.3f}")
    print(f"  Median: {np.median(all_lr):.3f}")
    print(f"  Std: {np.std(all_lr):.3f}")

    print(f"\nXGB scores:")
    print(f"  Min: {min(all_xgb):.3f}")
    print(f"  Max: {max(all_xgb):.3f}")
    print(f"  Mean: {np.mean(all_xgb):.3f}")
    print(f"  Median: {np.median(all_xgb):.3f}")
    print(f"  Std: {np.std(all_xgb):.3f}")

    # Bins
    print(f"\nLR score bins:")
    bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for low, high in zip(bins[:-1], bins[1:]):
        count = sum(1 for r in results if low <= r['lr_prob'] < high)
        pct = 100 * count / len(results)
        bar = "█" * int(pct / 2)
        print(f"  {low:.1f}-{high:.1f}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\nXGB score bins:")
    for low, high in zip(bins[:-1], bins[1:]):
        count = sum(1 for r in results if low <= r['xgb_prob'] < high)
        pct = 100 * count / len(results)
        bar = "█" * int(pct / 2)
        print(f"  {low:.1f}-{high:.1f}: {count:5d} ({pct:5.1f}%) {bar}")

    # ══════════════════════════════════════════════════════════════════════════════
    # SUMMARY JSON
    # ══════════════════════════════════════════════════════════════════════════════

    summary = {
        'author': author_code,
        'total': len(results),
        'consensus': len(consensus),
        'consensus_percentage': 100 * len(consensus) / len(results),
        'lr_only': len(lr_only),
        'xgb_only': len(xgb_only),
        'consensus_hits': consensus_sorted,
        'lr_only_high_confidence': [r for r in lr_only_sorted if r['lr_prob'] >= 0.9],
        'xgb_only_high_confidence': [r for r in xgb_only_sorted if r['xgb_prob'] >= 0.95]
    }

    summary_path = results_dir / "analysis_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Summary saved → {summary_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze test_quick results')
    parser.add_argument('author', default='0328IbnCabdRabbih',
                        help='Author code')
    args = parser.parse_args()

    analyze(args.author)
