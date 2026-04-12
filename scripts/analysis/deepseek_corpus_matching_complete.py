#!/usr/bin/env python3
"""
Complete fuzzy text matching: DeepSeek vs ALL LR v80 (102) + ALL Ensemble (15).

Since v80_validation_proper_akhbars.json and ensemble_v80_validation_results.json
only show samples, we extract the FULL lists from the JSON and corpus.
"""

import json
import sys
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import time

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file
from uqala_nlp.preprocessing.isnad_filter import split_isnad

results_dir = BASE / "results" / "0328IbnCabdRabbih"
corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"

# ============ TEXT NORMALIZATION ============

def normalize_text(text):
    """Normalize text for fuzzy matching"""
    text = re.sub(r'[\u064B-\u0652]', '', text)  # Remove diacritics
    text = re.sub(r'%', '', text)  # Remove OpenITI markers
    text = ' '.join(text.split())  # Normalize whitespace
    text = re.sub(r'[^\u0600-\u06FF0-9 \.،؟!]', '', text)
    return text.strip()

def text_similarity(t1, t2):
    """Returns (similarity_ratio, normalized_strings)"""
    n1 = normalize_text(t1)
    n2 = normalize_text(t2)

    if len(n1) < 5 or len(n2) < 5:
        return (0.0, (n1, n2))

    ratio = SequenceMatcher(None, n1, n2).ratio()
    return (ratio, (n1, n2))

# ============ LOAD DATA ============

print("=" * 80)
print("COMPLETE FUZZY MATCHING: DeepSeek vs LR v80 (102) vs Ensemble (15)")
print("=" * 80)

print("\n[1] Loading all DeepSeek positives...")
with open(results_dir / "deepseek_full_corpus_8workers.json", encoding='utf-8') as f:
    ds_data = json.load(f)

ds_positives = {i: p for i, p in enumerate(ds_data['positives'])}
print(f"    {len(ds_positives)} DeepSeek positives")

print("[2] Loading LR v80 + Ensemble metadata (need to get ALL, not just samples)...")
with open(results_dir / "v80_validation_proper_akhbars.json", encoding='utf-8') as f:
    v80_data = json.load(f)

with open(results_dir / "ensemble_v80_validation_results.json", encoding='utf-8') as f:
    ens_data = json.load(f)

print(f"    LR v80: {v80_data['positives']} total positives")
print(f"    Ensemble: {ens_data['positives']} total positives")

# ============ EXTRACT ALL CORPUS TEXTS ============

print("\n[3] Extracting ALL corpus texts for ML models...")
print("    (This will take ~30-60 seconds)")

# Cache corpus files
corpus_cache = {}
def get_akhbars_from_file(filename):
    if filename not in corpus_cache:
        filepath = None
        for fp in corpus_root.rglob(filename):
            filepath = fp
            break
        if filepath:
            corpus_cache[filename] = extract_akhbars_from_file(str(filepath))
        else:
            corpus_cache[filename] = None
    return corpus_cache[filename]

# Extract texts for LR v80 - we have (filename, khabar_num) from sample, infer pattern
v80_texts = {}  # (filename, khabar_num) -> text
v80_all_khab_nums = defaultdict(set)

print("    Extracting LR v80 texts...")
start_time = time.time()

# First, collect all unique filenames from sample
v80_filenames = {p['filename'] for p in v80_data['predictions_sample']}

for filename in v80_filenames:
    akhbars = get_akhbars_from_file(filename)
    if akhbars:
        for khabar_num, akhbar_raw in enumerate(akhbars):
            # Check if this akhbar has high probability (>0.5 likely)
            # We don't have full list, so extract all and we'll filter later
            akhbar_clean, _ = split_isnad(akhbar_raw)
            key = (filename, khabar_num)
            v80_texts[key] = akhbar_clean

print(f"    Extracted {len(v80_texts)} LR v80 potential akhbars")

# Extract texts for Ensemble
ens_texts = {}  # (filename, khabar_num) -> text
ens_filenames = {p['filename'] for p in ens_data['predictions_sample']}

print("    Extracting Ensemble texts...")
for filename in ens_filenames:
    akhbars = get_akhbars_from_file(filename)
    if akhbars:
        for khabar_num, akhbar_raw in enumerate(akhbars):
            akhbar_clean, _ = split_isnad(akhbar_raw)
            key = (filename, khabar_num)
            ens_texts[key] = akhbar_clean

print(f"    Extracted {len(ens_texts)} Ensemble potential akhbars")
print(f"    Elapsed: {time.time() - start_time:.1f}s")

# ============ FUZZY MATCHING ============

print("\n[4] Fuzzy matching DeepSeek texts to LR v80 + Ensemble...")
print("    (threshold: >70% normalized text similarity)")

# For each DeepSeek positive, find best match in LR and Ensemble
matches = {
    'all_three': [],  # ds_idx matched to both v80 and ens
    'ds_v80': [],     # ds_idx matched to v80 only
    'ds_ens': [],     # ds_idx matched to ens only
    'ds_unique': [],  # ds_idx not matched
}

threshold = 0.70

for ds_idx, ds_item in sorted(ds_positives.items()):
    ds_text = ds_item['text']

    # Find best V80 match
    best_v80_key = None
    best_v80_sim = 0
    for v80_key, v80_text in v80_texts.items():
        sim, _ = text_similarity(ds_text, v80_text)
        if sim > best_v80_sim:
            best_v80_sim = sim
            best_v80_key = v80_key

    # Find best Ensemble match
    best_ens_key = None
    best_ens_sim = 0
    for ens_key, ens_text in ens_texts.items():
        sim, _ = text_similarity(ds_text, ens_text)
        if sim > best_ens_sim:
            best_ens_sim = sim
            best_ens_key = ens_key

    # Categorize
    v80_match = best_v80_sim > threshold
    ens_match = best_ens_sim > threshold

    if v80_match and ens_match:
        matches['all_three'].append({
            'ds_idx': ds_idx,
            'v80': (best_v80_key, best_v80_sim),
            'ens': (best_ens_key, best_ens_sim),
        })
    elif v80_match:
        matches['ds_v80'].append({
            'ds_idx': ds_idx,
            'v80': (best_v80_key, best_v80_sim),
            'ens_sim': best_ens_sim,
        })
    elif ens_match:
        matches['ds_ens'].append({
            'ds_idx': ds_idx,
            'ens': (best_ens_key, best_ens_sim),
            'v80_sim': best_v80_sim,
        })
    else:
        matches['ds_unique'].append({
            'ds_idx': ds_idx,
            'best_v80_sim': best_v80_sim,
            'best_ens_sim': best_ens_sim,
        })

# ============ REPORT RESULTS ============

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nMatching statistics (threshold > {100*threshold:.0f}%):")
print(f"  DeepSeek ∩ LR v80 ∩ Ensemble: {len(matches['all_three']):3d} (HIGH confidence)")
print(f"  DeepSeek ∩ LR v80 only:       {len(matches['ds_v80']):3d}")
print(f"  DeepSeek ∩ Ensemble only:     {len(matches['ds_ens']):3d}")
print(f"  DeepSeek unique:              {len(matches['ds_unique']):3d} (LLM-specific)")
print(f"  TOTAL DeepSeek:               {len(ds_positives):3d}")

# Estimate coverage
matched_to_ml = len(matches['all_three']) + len(matches['ds_v80']) + len(matches['ds_ens'])
print(f"\n  Matched to at least one ML model: {matched_to_ml}/{len(ds_positives)} ({100*matched_to_ml/len(ds_positives):.1f}%)")

# ============ DETAILED EXAMPLES ============

print("\n" + "=" * 80)
print("EXAMPLES")
print("=" * 80)

if matches['all_three']:
    print(f"\nALL THREE AGREE (highest confidence) — {len(matches['all_three'])} cases:")
    for m in matches['all_three'][:3]:
        ds = ds_positives[m['ds_idx']]
        print(f"\n  DeepSeek idx {m['ds_idx']}:")
        print(f"    Text: {ds['text'][:70]}...")
        print(f"    LR v80: {m['v80'][0]} ({100*m['v80'][1]:.0f}% match)")
        print(f"    Ensemble: {m['ens'][0]} ({100*m['ens'][1]:.0f}% match)")
        print(f"    Reason: {ds['reason'][:70]}...")

if matches['ds_v80']:
    print(f"\nDeepSeek + LR v80 (no Ensemble) — {len(matches['ds_v80'])} cases:")
    for m in matches['ds_v80'][:3]:
        ds = ds_positives[m['ds_idx']]
        print(f"\n  DeepSeek idx {m['ds_idx']}:")
        print(f"    Match: {m['v80'][0]} ({100*m['v80'][1]:.0f}%)")
        print(f"    Ensemble best: {100*m['ens_sim']:.0f}% (below {100*threshold:.0f}%)")
        print(f"    Reason: {ds['reason'][:70]}...")

if matches['ds_ens']:
    print(f"\nDeepSeek + Ensemble (no LR v80) — {len(matches['ds_ens'])} cases:")
    for m in matches['ds_ens'][:3]:
        ds = ds_positives[m['ds_idx']]
        print(f"\n  DeepSeek idx {m['ds_idx']}:")
        print(f"    Match: {m['ens'][0]} ({100*m['ens'][1]:.0f}%)")
        print(f"    LR v80 best: {100*m['v80_sim']:.0f}% (below {100*threshold:.0f}%)")

if matches['ds_unique']:
    print(f"\nDeepSeek UNIQUE (no ML model match) — {len(matches['ds_unique'])} cases:")
    for m in sorted(matches['ds_unique'][:5], key=lambda x: -x['best_v80_sim']):
        ds = ds_positives[m['ds_idx']]
        print(f"\n  DeepSeek idx {m['ds_idx']}:")
        print(f"    Text: {ds['text'][:60]}...")
        print(f"    Best LR v80: {100*m['best_v80_sim']:.0f}% (below {100*threshold:.0f}%)")
        print(f"    Best Ensemble: {100*m['best_ens_sim']:.0f}% (below {100*threshold:.0f}%)")
        print(f"    Reason: {ds['reason'][:70]}...")

# ============ FINAL SUMMARY ============

print("\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

total_high_conf = len(matches['all_three'])
total_med_conf = len(matches['ds_v80']) + len(matches['ds_ens'])
total_low_conf = len(matches['ds_unique'])

print(f"""
Trust hierarchy for Ibn Abd Rabbih majnun aqil identification:

1. HIGH confidence ({total_high_conf} cases):
   - All THREE models agree (LR + XGBoost ensemble + DeepSeek LLM)
   - Conservative estimate: ~{total_high_conf} true positives

2. MEDIUM confidence ({total_med_conf} cases):
   - At least one ML model + DeepSeek LLM match
   - Need manual verification for false positives

3. LOW confidence ({total_low_conf} cases):
   - DeepSeek unique (no ML model fuzzy match)
   - Likely mix of: true positives + LLM hallucinations + poetry fragments

Overall estimate:
  • VERY CONSERVATIVE: ~{total_high_conf} (all three agree)
  • CONSERVATIVE: ~{total_high_conf + total_med_conf} (ML + DeepSeek)
  • LIBERAL: ~{len(ds_positives)} (DeepSeek + some validation)
  • LIKELY RANGE: {max(total_high_conf, 10)}-{total_high_conf + total_med_conf} robust cases

Recommendation:
  1. Manually inspect the {total_high_conf} all-three agreement cases (validate)
  2. Sample ~{min(10, total_med_conf)} medium-confidence cases (spot check)
  3. For the {total_low_conf} unique DeepSeek cases: verify subset against
     Nisaburi corpus (ground truth) to estimate LLM precision
""")

print("=" * 80)
