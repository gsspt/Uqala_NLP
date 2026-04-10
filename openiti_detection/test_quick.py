#!/usr/bin/env python3
"""
test_quick.py
──────────────
Test RAPIDE sur une petite œuvre (1-2 fichiers)

Usage:
  python openiti_detection/test_quick.py [author_code]

Exemples:
  python openiti_detection/test_quick.py 0542IbnBassamShantarini   # 1 fichier
  python openiti_detection/test_quick.py 0681IbnKhallikan          # 1 fichier
  python openiti_detection/test_quick.py 0733Nuwayri               # 1 fichier
  python openiti_detection/test_quick.py 0279Baladhuri             # 2 fichiers
"""

import json
import sys
import pathlib
import pickle
import argparse
import numpy as np
from collections import Counter
import time

sys.stdout.reconfigure(encoding='utf-8')

# Import du code de détection principal et isnad_filter
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from openiti_detection.detect_lr_xgboost import (
    extract_features_74, count_arabic_chars, extract_akhbars_from_file
)
from isnad_filter import get_matn

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE = pathlib.Path(__file__).parent.parent
LR_MODEL_PATH = BASE / "scan" / "lr_classifier_71features.pkl"
XGB_MODEL_PATH = BASE / "comparison_ensemble" / "results" / "xgb_classifier_71features.pkl"
OPENITI_TARGETED = BASE / "openiti_targeted"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("Loading models…")
with open(LR_MODEL_PATH, 'rb') as f:
    lr_data = pickle.load(f)
with open(XGB_MODEL_PATH, 'rb') as f:
    xgb_data = pickle.load(f)

lr_model = lr_data['clf']
lr_scaler = lr_data['scaler']
xgb_model = xgb_data['clf']
xgb_scaler = xgb_data['scaler']

# ══════════════════════════════════════════════════════════════════════════════
# PROCESS SINGLE AUTHOR
# ══════════════════════════════════════════════════════════════════════════════

def process_author(author_code, threshold_lr=0.5, threshold_xgb=0.5):
    """Traite une petite œuvre rapidement."""

    author_path = OPENITI_TARGETED / author_code

    if not author_path.exists():
        print(f"❌ Author not found: {author_code}")
        print(f"Available authors: {sorted([d.name for d in OPENITI_TARGETED.iterdir() if d.is_dir()])}")
        return None

    print(f"\n{'='*70}")
    print(f"Testing: {author_code}")
    print(f"{'='*70}")

    # Charger les akhbars
    files = [f for f in author_path.rglob("*") if f.is_file() and not f.name.startswith('.')]
    print(f"Files: {len(files)}")

    all_akhbars = []
    start_load = time.time()

    print(f"Extracting akhbars…")
    for filepath in files:
        try:
            akhbars = extract_akhbars_from_file(filepath)
            for text in akhbars:
                all_akhbars.append({
                    'file': str(filepath.name),
                    'text': text,
                    'length': len(text)
                })
        except Exception as e:
            pass

    load_time = time.time() - start_load
    print(f"✓ Loaded {len(all_akhbars)} akhbars in {load_time:.1f}s")

    if len(all_akhbars) == 0:
        print("⚠️  No akhbars found!")
        return None

    # Traiter chaque akhbar
    print(f"\nProcessing {len(all_akhbars)} akhbars…")
    results = []
    start_process = time.time()

    for idx, akhbar in enumerate(all_akhbars):
        if len(all_akhbars) > 10 and (idx + 1) % max(1, len(all_akhbars) // 5) == 0:
            elapsed = time.time() - start_process
            rate = (idx + 1) / elapsed
            remaining = (len(all_akhbars) - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx + 1}/{len(all_akhbars)} ({100*(idx+1)/len(all_akhbars):.0f}%) | {remaining:.0f}s remaining")

        try:
            X = extract_features_74(akhbar['text'])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = lr_scaler.transform(X.reshape(1, -1))[0]
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            lr_prob = lr_model.predict_proba(X_scaled.reshape(1, -1))[0][1]
            xgb_prob = xgb_model.predict_proba(X_scaled.reshape(1, -1))[0][1]

            lr_pred = 1 if lr_prob >= threshold_lr else 0
            xgb_pred = 1 if xgb_prob >= threshold_xgb else 0

            results.append({
                'file': akhbar['file'],
                'text': akhbar['text'][:200],
                'lr_prob': float(lr_prob),
                'xgb_prob': float(xgb_prob),
                'lr_pred': int(lr_pred),
                'xgb_pred': int(xgb_pred),
                'consensus': int(lr_pred == 1 and xgb_pred == 1)
            })

        except Exception as e:
            pass

    process_time = time.time() - start_process
    print(f"✓ Processed in {process_time:.1f}s")

    # Sauvegarder tous les résultats en JSON
    out_dir = pathlib.Path(__file__).parent / "results" / author_code
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "all_predictions.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ All results saved → {results_path}")

    return results

# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_results(results, author_code):
    """Affiche les résultats."""

    if not results:
        return

    lr_pos = [r for r in results if r['lr_pred'] == 1]
    xgb_pos = [r for r in results if r['xgb_pred'] == 1]
    consensus = [r for r in results if r['consensus'] == 1]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total akhbars: {len(results)}")
    print(f"LR positives: {len(lr_pos)} ({100*len(lr_pos)/len(results):.1f}%)")
    print(f"XGBoost positives: {len(xgb_pos)} ({100*len(xgb_pos)/len(results):.1f}%)")
    print(f"Consensus: {len(consensus)} ({100*len(consensus)/len(results):.1f}%)")

    # Score distribution
    lr_scores = [r['lr_prob'] for r in results]
    xgb_scores = [r['xgb_prob'] for r in results]

    print(f"\nScore ranges:")
    print(f"  LR:      min={min(lr_scores):.3f}  max={max(lr_scores):.3f}  mean={np.mean(lr_scores):.3f}  median={np.median(lr_scores):.3f}")
    print(f"  XGBoost: min={min(xgb_scores):.3f}  max={max(xgb_scores):.3f}  mean={np.mean(xgb_scores):.3f}  median={np.median(xgb_scores):.3f}")

    # Consensus hits
    if consensus:
        print(f"\n{'='*70}")
        print(f"CONSENSUS HITS (both models agree)")
        print(f"{'='*70}")
        consensus_sorted = sorted(consensus, key=lambda x: x['lr_prob'] + x['xgb_prob'], reverse=True)
        for i, hit in enumerate(consensus_sorted[:10], 1):
            print(f"\n{i}. File: {hit['file']}")
            print(f"   LR: {hit['lr_prob']:.3f} | XGBoost: {hit['xgb_prob']:.3f}")
            print(f"   Text: {hit['text'][:180]}…")
    else:
        print(f"\n⚠️  No consensus hits found")

    # LR only (high confidence)
    lr_only_high = [r for r in results if r['lr_pred'] == 1 and r['xgb_pred'] == 0 and r['lr_prob'] >= 0.7]
    if lr_only_high:
        print(f"\n{'='*70}")
        print(f"LR HIGH (LR only, high confidence)")
        print(f"{'='*70}")
        lr_only_high_sorted = sorted(lr_only_high, key=lambda x: x['lr_prob'], reverse=True)
        for i, hit in enumerate(lr_only_high_sorted[:5], 1):
            print(f"\n{i}. LR: {hit['lr_prob']:.3f} | XGB: {hit['xgb_prob']:.3f}")
            print(f"   Text: {hit['text'][:180]}…")

    # XGBoost only (high confidence)
    xgb_only_high = [r for r in results if r['xgb_pred'] == 1 and r['lr_pred'] == 0 and r['xgb_prob'] >= 0.8]
    if xgb_only_high:
        print(f"\n{'='*70}")
        print(f"XGB HIGH (XGBoost only, high confidence)")
        print(f"{'='*70}")
        xgb_only_high_sorted = sorted(xgb_only_high, key=lambda x: x['xgb_prob'], reverse=True)
        for i, hit in enumerate(xgb_only_high_sorted[:5], 1):
            print(f"\n{i}. LR: {hit['lr_prob']:.3f} | XGB: {hit['xgb_prob']:.3f}")
            print(f"   Text: {hit['text'][:180]}…")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick test on single author')
    parser.add_argument('author', nargs='?', default='0542IbnBassamShantarini',
                        help='Author code (e.g., 0542IbnBassamShantarini)')
    parser.add_argument('--threshold-lr', type=float, default=0.5)
    parser.add_argument('--threshold-xgb', type=float, default=0.5)
    args = parser.parse_args()

    results = process_author(args.author, args.threshold_lr, args.threshold_xgb)

    if results:
        analyze_results(results, args.author)
        print(f"\n{'='*70}\n")
