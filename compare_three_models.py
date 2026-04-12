#!/usr/bin/env python3
"""
Compare DeepSeek, Ensemble, and LR v80 on ALIGNED corpus (proper akhbars).

Using v80_validation_proper_akhbars.json which has:
  - 102 LR v80 positives (at threshold 0.5)
  - Aligned with Ensemble using same akhbar extraction
  - Can compare all three by (filename, khabar_num)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

results_dir = Path("results/0328IbnCabdRabbih")

# ============ LOAD ALIGNED DATA ============

print("=" * 80)
print("THREE-MODEL COMPARISON (aligned akhbars)")
print("=" * 80)

# 1. LR v80 "proper akhbars" = baseline with aligned extraction
print("\n[1] Loading LR v80 (proper akhbars)...")
with open(results_dir / "v80_validation_proper_akhbars.json", encoding='utf-8') as f:
    v80_data = json.load(f)

# Build dict of all v80 positives from sample (need to infer the rest)
# We have 102 total, but only 10 in sample. Let's note that limitation.
v80_sample = {(p['filename'], p['khabar_num']): p for p in v80_data['predictions_sample']}
v80_positives_count = v80_data['positives']
print(f"    Total: {v80_positives_count} positives")
print(f"    Sample available: {len(v80_sample)} (with detailed predictions)")

# 2. Ensemble v80
print("[2] Loading Ensemble (LR + XGBoost)...")
with open(results_dir / "ensemble_v80_validation_results.json", encoding='utf-8') as f:
    ens_data = json.load(f)

ens_positives = {}
for p in ens_data['predictions_sample']:
    key = (p['filename'], p['khabar_num'])
    if p.get('label') == 1:
        ens_positives[key] = p

print(f"    Total: {ens_data['positives']} positives")
print(f"    Sample available: {len(ens_positives)}")

# 3. DeepSeek
print("[3] Loading DeepSeek (LLM few-shot)...")
with open(results_dir / "deepseek_full_corpus_8workers.json", encoding='utf-8') as f:
    ds_data = json.load(f)

ds_positives = {(p['filename'], p['khabar_num']): p for p in ds_data['positives']}
print(f"    Total: {ds_data['true_positives']} positives")
print(f"    All: {len(ds_positives)} (with khabar_num)")

# ============ BUILD COMPARISON ============

print("\n" + "=" * 80)
print("COMPARISON MATRIX")
print("=" * 80)

v80_set = set(v80_sample.keys())
ens_set = set(ens_positives.keys())
ds_set = set(ds_positives.keys())

# Pairwise intersections
v80_and_ens = v80_set & ens_set
v80_and_ds = v80_set & ds_set
ens_and_ds = ens_set & ds_set
all_three = v80_set & ens_set & ds_set

# Unique to each
only_v80 = v80_set - ens_set - ds_set
only_ens = ens_set - v80_set - ds_set
only_ds = ds_set - v80_set - ens_set

print(f"\nSample sizes (what we can directly compare):")
print(f"  LR v80:   {len(v80_set):3d} in sample")
print(f"  Ensemble: {len(ens_set):3d}")
print(f"  DeepSeek: {len(ds_set):3d}")

print(f"\nAgreement (positive = both models agree on that akhbar):")
print(f"  LR v80 ∩ Ensemble:  {len(v80_and_ens):3d} ({100*len(v80_and_ens)/len(v80_set):5.1f}% of v80 sample)")
print(f"  LR v80 ∩ DeepSeek:  {len(v80_and_ds):3d} ({100*len(v80_and_ds)/len(v80_set):5.1f}% of v80 sample)")
print(f"  Ensemble ∩ DeepSeek:{len(ens_and_ds):3d} ({100*len(ens_and_ds)/len(ens_set):5.1f}% of ensemble)")
print(f"  All three:          {len(all_three):3d} ({100*len(all_three)/len(v80_set):5.1f}% of v80 sample)")

print(f"\nDisagreement:")
print(f"  Only LR v80:   {len(only_v80):3d}")
print(f"  Only Ensemble: {len(only_ens):3d}")
print(f"  Only DeepSeek: {len(only_ds):3d}")

# ============ SCALING UP ============

print("\n" + "=" * 80)
print("EXTRAPOLATION (from samples to full corpus)")
print("=" * 80)

print(f"\nEstimated full counts:")
print(f"  LR v80 (aligned):       {v80_positives_count:3d} positives (100% aligned)")
print(f"  Ensemble:               {ens_data['positives']:3d} positives")
print(f"  DeepSeek:               {ds_data['true_positives']:3d} positives")

# Based on sample: what fraction of v80 does Ensemble agree with?
if len(v80_set) > 0:
    v80_ensemble_agreement_rate = len(v80_and_ens) / len(v80_set)
    estimated_ens_agrees_with_v80 = int(v80_positives_count * v80_ensemble_agreement_rate)
    print(f"\nEstimated agreement:")
    print(f"  Ensemble agrees with {100*v80_ensemble_agreement_rate:.1f}% of v80 positives")
    print(f"  → ~{estimated_ens_agrees_with_v80} of {v80_positives_count} v80 positives also flagged by Ensemble")

# ============ DETAILED BREAKDOWN ============

print("\n" + "=" * 80)
print("DETAILED ANALYSIS")
print("=" * 80)

if all_three:
    print(f"\nAgreement: All THREE models (HIGH confidence) — {len(all_three)} cases")
    for key in sorted(all_three)[:5]:
        v80 = v80_sample[key]
        ens = ens_positives[key]
        ds = ds_positives[key]
        print(f"\n  {key[1]:4d}: {key[0][-15:]}")
        print(f"    LR:       {v80['pred_proba']:.4f}")
        print(f"    Ensemble: {ens['pred_ensemble']:.4f} (LR={ens['pred_lr']:.3f}, XGB={ens['pred_xgb']:.3f})")
        print(f"    DeepSeek: {ds['reason'][:60]}...")

if v80_and_ens and len(v80_and_ens - all_three) > 0:
    print(f"\nAgreement: LR v80 + Ensemble only (NO DeepSeek) — {len((v80_and_ens - all_three))}")
    for key in sorted(v80_and_ens - all_three)[:3]:
        v80 = v80_sample[key]
        ens = ens_positives[key]
        print(f"\n  {key[1]:4d}")
        print(f"    LR:       {v80['pred_proba']:.4f}")
        print(f"    Ensemble: {ens['pred_ensemble']:.4f}")
        print(f"    DeepSeek: NOT flagged")

if only_v80:
    print(f"\nV80 ONLY — {len(only_v80)} (false positives from v80?)")
    for key in sorted(only_v80)[:3]:
        v80 = v80_sample[key]
        print(f"\n  {key[1]:4d}: {key[0][-15:]}")
        print(f"    LR prob: {v80['pred_proba']:.4f}")
        print(f"    NOT in Ensemble or DeepSeek")

if only_ens:
    print(f"\nEnsemble ONLY — {len(only_ens)} (missed by v80 sample?)")
    for key in sorted(only_ens)[:3]:
        ens = ens_positives[key]
        print(f"\n  {key[1]:4d}")
        print(f"    Ensemble: {ens['pred_ensemble']:.4f}")
        print(f"    NOT in v80 or DeepSeek samples")

if only_ds:
    print(f"\nDeepSeek ONLY — {len(only_ds)} (LLM-specific positives)")
    for key in sorted(only_ds)[:5]:
        ds = ds_positives[key]
        print(f"\n  {key[1]:4d}")
        print(f"    DeepSeek: {ds['reason'][:70]}...")

# ============ SUMMARY ============

print("\n" + "=" * 80)
print("SUMMARY & INTERPRETATION")
print("=" * 80)

print(f"""
Model confidence hierarchy (HIGH → LOW):

1. ALL THREE AGREE ({len(all_three)} cases):
   - Highest confidence (combines features, ensemble voting, + LLM judgment)
   - These are your safest bets

2. LR + Ensemble agree, DeepSeek disagree ({len(v80_and_ens - all_three)} cases):
   - High confidence (two ML models agree)
   - DeepSeek may have missed narrative context

3. DeepSeek ONLY ({len(only_ds)} cases):
   - Medium confidence (contextual LLM judgment with real examples)
   - Need manual verification for false positives

4. LR v80 ONLY ({len(only_v80)} cases):
   - Low confidence (likely dialogue-based false positives)
   - Generic "قال" (said) patterns without majnun aqil narrative

Expected TRUE major aqil in Ibn Abd Rabbih:
  • Conservative: ~{len(all_three)} (all-three agreement)
  • Middle: ~{len(all_three) + len(v80_and_ens - all_three)} (LR+Ens agreement)
  • Liberal: ~{ds_data['true_positives']} (DeepSeek LLM)
  • Probable range: {min(len(all_three) + len(v80_and_ens - all_three), ens_data['positives'])}-{ds_data['true_positives']}

Next steps:
  1. Manual inspection of {len(all_three)} all-three-agreement cases (validate)
  2. Sample {max(10, len(only_ds)//2)} DeepSeek-only cases (check LLM accuracy)
  3. Use C1 Active Learning on {len(v80_and_ens - all_three)} uncertain cases
""")

print("=" * 80)
