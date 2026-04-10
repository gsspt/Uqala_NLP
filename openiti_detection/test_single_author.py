#!/usr/bin/env python3
"""
test_single_author.py
─────────────────────
Test sur une seule œuvre: Ibn al-Jawzi (0597IbnJawzi)
Meilleur candidat pour détecter les majnun aqil.

Usage:
  python openiti_detection/test_single_author.py
"""

import json
import sys
import pathlib
import pickle
import unicodedata
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

# Import du code de détection principal et isnad_filter
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from openiti_detection.detect_lr_xgboost import (
    extract_features_74, count_arabic_chars, extract_akhbars_from_file,
    normalize_arabic
)
from isnad_filter import get_matn

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE = pathlib.Path(__file__).parent.parent
LR_MODEL_PATH = BASE / "scan" / "lr_classifier_71features.pkl"
XGB_MODEL_PATH = BASE / "comparison_ensemble" / "results" / "xgb_classifier_71features.pkl"

# Ibn al-Jawzi
AUTHOR_PATH = BASE / "openiti_targeted" / "0597IbnJawzi"
OUT_DIR = pathlib.Path(__file__).parent / "results" / "test_ibnJawzi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUT_DIR / "predictions.json"
ANALYSIS_PATH = OUT_DIR / "analysis.json"

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

def process_author(author_path, threshold_lr=0.5, threshold_xgb=0.5):
    """Traite tous les akhbars d'un auteur."""

    if not author_path.exists():
        print(f"❌ Author path not found: {author_path}")
        return None

    print(f"\nProcessing: {author_path.name}")
    print(f"Files: {len(list(author_path.rglob('*')))}")

    all_results = []
    all_akhbars = []

    # Charger tous les akhbars avec progression
    files = [f for f in author_path.rglob("*") if f.is_file() and not f.name.startswith('.')]
    print(f"Loading {len(files)} files…")

    for file_idx, filepath in enumerate(files):
        if (file_idx + 1) % 20 == 0:
            print(f"  {file_idx + 1}/{len(files)} files loaded… ({len(all_akhbars)} akhbars so far)")

        try:
            akhbars = extract_akhbars_from_file(filepath)
            for text in akhbars:
                all_akhbars.append({
                    'file': str(filepath.relative_to(AUTHOR_PATH)),
                    'text': text[:500],
                    'text_full_length': len(text)
                })
        except Exception as e:
            pass

    print(f"✓ Loaded {len(all_akhbars)} akhbars from {len(files)} files")

    if len(all_akhbars) == 0:
        print("⚠️  No akhbars found!")
        return []

    # Traiter chaque akhbar
    print(f"\nProcessing {len(all_akhbars)} akhbars...")
    start_time = __import__('time').time()

    for idx, akhbar in enumerate(all_akhbars):
        if (idx + 1) % 50 == 0:
            elapsed = __import__('time').time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(all_akhbars) - idx - 1) / rate if rate > 0 else 0
            print(f"  {idx + 1}/{len(all_akhbars)} ({100*(idx+1)/len(all_akhbars):.0f}%) | {rate:.1f}/sec | ~{remaining:.0f}s remaining")

        try:
            X = extract_features_74(akhbar['text'])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = lr_scaler.transform(X.reshape(1, -1))[0]
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            continue

        # Prédictions
        try:
            lr_prob = lr_model.predict_proba(X_scaled.reshape(1, -1))[0][1]
            xgb_prob = xgb_model.predict_proba(X_scaled.reshape(1, -1))[0][1]

            lr_pred = 1 if lr_prob >= threshold_lr else 0
            xgb_pred = 1 if xgb_prob >= threshold_xgb else 0

            all_results.append({
                'file': akhbar['file'],
                'text_preview': akhbar['text'][:150],
                'text_full_length': akhbar['text_full_length'],
                'lr_prob': float(lr_prob),
                'xgb_prob': float(xgb_prob),
                'lr_pred': int(lr_pred),
                'xgb_pred': int(xgb_pred),
                'consensus': int(lr_pred == 1 and xgb_pred == 1),
                'disagreement': int(lr_pred != xgb_pred)
            })

        except Exception as e:
            pass

    return all_results

# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

results = process_author(AUTHOR_PATH, threshold_lr=0.5, threshold_xgb=0.5)

if not results:
    print("No results")
    sys.exit(1)

# Sauvegarder les résultats bruts
with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n✓ Results saved → {RESULTS_PATH}")

# Analyser
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

lr_positives = [r for r in results if r['lr_pred'] == 1]
xgb_positives = [r for r in results if r['xgb_pred'] == 1]
consensus = [r for r in results if r['consensus'] == 1]
disagreements = [r for r in results if r['disagreement'] == 1]

print(f"\nTotal akhbars: {len(results)}")
print(f"LR positives: {len(lr_positives)} ({100*len(lr_positives)/len(results):.1f}%)")
print(f"XGBoost positives: {len(xgb_positives)} ({100*len(xgb_positives)/len(results):.1f}%)")
print(f"Consensus positives: {len(consensus)} ({100*len(consensus)/len(results):.1f}%)")
print(f"Disagreements: {len(disagreements)} ({100*len(disagreements)/len(results):.1f}%)")

# Statistiques de confiance
lr_probs = [r['lr_prob'] for r in results]
xgb_probs = [r['xgb_prob'] for r in results]

print(f"\nLR score statistics:")
print(f"  Mean: {np.mean(lr_probs):.4f}")
print(f"  Median: {np.median(lr_probs):.4f}")
print(f"  Min: {np.min(lr_probs):.4f}, Max: {np.max(lr_probs):.4f}")
print(f"  Std: {np.std(lr_probs):.4f}")

print(f"\nXGBoost score statistics:")
print(f"  Mean: {np.mean(xgb_probs):.4f}")
print(f"  Median: {np.median(xgb_probs):.4f}")
print(f"  Min: {np.min(xgb_probs):.4f}, Max: {np.max(xgb_probs):.4f}")
print(f"  Std: {np.std(xgb_probs):.4f}")

# Score bins
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lr_bins = Counter([r['lr_prob'] for r in results])
xgb_bins = Counter([r['xgb_prob'] for r in results])

print(f"\nLR score distribution:")
for low, high in zip(bins[:-1], bins[1:]):
    count = sum(1 for r in results if low <= r['lr_prob'] < high)
    pct = 100 * count / len(results)
    bar = "█" * int(pct / 2)
    print(f"  {low:.1f}-{high:.1f}: {count:4d} ({pct:5.1f}%) {bar}")

print(f"\nXGBoost score distribution:")
for low, high in zip(bins[:-1], bins[1:]):
    count = sum(1 for r in results if low <= r['xgb_prob'] < high)
    pct = 100 * count / len(results)
    bar = "█" * int(pct / 2)
    print(f"  {low:.1f}-{high:.1f}: {count:4d} ({pct:5.1f}%) {bar}")

# Top candidats
print(f"\n" + "="*70)
print("TOP CANDIDATES (Consensus)")
print("="*70)

consensus_sorted = sorted(consensus, key=lambda x: x['lr_prob'] + x['xgb_prob'], reverse=True)
for i, hit in enumerate(consensus_sorted[:10], 1):
    print(f"\n{i}. {hit['file']}")
    print(f"   LR: {hit['lr_prob']:.3f} | XGBoost: {hit['xgb_prob']:.3f}")
    print(f"   Text: {hit['text_preview']}…")

# Disagreements
print(f"\n" + "="*70)
print("DISAGREEMENTS")
print("="*70)

lr_only = sorted([r for r in disagreements if r['lr_pred'] == 1 and r['xgb_pred'] == 0],
                 key=lambda x: x['lr_prob'], reverse=True)
xgb_only = sorted([r for r in disagreements if r['lr_pred'] == 0 and r['xgb_pred'] == 1],
                  key=lambda x: x['xgb_prob'], reverse=True)

print(f"\nLR says YES, XGBoost says NO ({len(lr_only)}):")
for i, hit in enumerate(lr_only[:5], 1):
    print(f"  {i}. LR {hit['lr_prob']:.3f} | XGB {hit['xgb_prob']:.3f} - {hit['file']}")

print(f"\nXGBoost says YES, LR says NO ({len(xgb_only)}):")
for i, hit in enumerate(xgb_only[:5], 1):
    print(f"  {i}. LR {hit['lr_prob']:.3f} | XGB {hit['xgb_prob']:.3f} - {hit['file']}")

# Sauvegarder l'analyse
analysis = {
    'author': 'Ibn al-Jawzi (0597)',
    'total_akhbars': len(results),
    'lr_positives': len(lr_positives),
    'xgb_positives': len(xgb_positives),
    'consensus_positives': len(consensus),
    'disagreements': len(disagreements),
    'lr_stats': {
        'mean': float(np.mean(lr_probs)),
        'median': float(np.median(lr_probs)),
        'std': float(np.std(lr_probs))
    },
    'xgb_stats': {
        'mean': float(np.mean(xgb_probs)),
        'median': float(np.median(xgb_probs)),
        'std': float(np.std(xgb_probs))
    },
    'top_consensus': consensus_sorted[:20],
    'lr_only': lr_only[:10],
    'xgb_only': xgb_only[:10]
}

with open(ANALYSIS_PATH, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)
print(f"\n✓ Analysis saved → {ANALYSIS_PATH}")
