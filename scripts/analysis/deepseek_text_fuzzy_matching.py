#!/usr/bin/env python3
"""
Compare DeepSeek, Ensemble, and LR v80 positives by TEXT matching.

Since akhbar numbering differs between extraction runs, we match by text content
using fuzzy matching with high threshold (>85% similarity).
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher

sys.stdout.reconfigure(encoding='utf-8')

results_dir = Path("results/0328IbnCabdRabbih")

def text_similarity(t1, t2):
    """Compute text similarity (0-1) ignoring whitespace"""
    t1_norm = ' '.join(t1.strip().split())
    t2_norm = ' '.join(t2.strip().split())
    return SequenceMatcher(None, t1_norm, t2_norm).ratio()

def find_best_match(text, candidates, threshold=0.85):
    """Find best fuzzy match in candidates, return (idx, similarity)"""
    best_sim = 0
    best_idx = -1
    for idx, cand_text in candidates:
        sim = text_similarity(text, cand_text)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    return (best_idx, best_sim) if best_sim > threshold else (-1, best_sim)

# ============ LOAD DATA ============

print("=" * 80)
print("COMPARISON: DeepSeek vs Ensemble vs LR v80 (by TEXT matching)")
print("=" * 80)

# 1. DeepSeek (has filename + khabar_num + text)
print("\n[1] Loading DeepSeek...")
with open(results_dir / "deepseek_full_corpus_8workers.json", encoding='utf-8') as f:
    ds_data = json.load(f)
ds_positives = {i: p for i, p in enumerate(ds_data['positives'])}
print(f"    {len(ds_positives)} DeepSeek positives")

# 2. Ensemble (has filename + khabar_num, need to load text from somewhere)
#    We only have sample in the JSON, so we'll need to rely on finding DS matches
print("[2] Loading Ensemble...")
with open(results_dir / "ensemble_v80_validation_results.json", encoding='utf-8') as f:
    ens_data = json.load(f)
ens_positives = []
for p in ens_data['predictions_sample']:
    if p.get('label') == 1:
        ens_positives.append({
            'filename': p['filename'],
            'khabar_num': p['khabar_num'],
            'pred_ensemble': p.get('pred_ensemble'),
            'pred_lr': p.get('pred_lr'),
            'pred_xgb': p.get('pred_xgb'),
        })
print(f"    {len(ens_positives)} Ensemble positives (from sample)")

# 3. LR v80 (has text + prob, no khabar_num)
print("[3] Loading LR v80 top positives...")
with open(results_dir / "v80_validation_results.json", encoding='utf-8') as f:
    v80_data = json.load(f)
lr_top_positives = v80_data.get('top_positives', [])
print(f"    {len(lr_top_positives)} LR v80 top positives")

# ============ MATCH ENSEMBLE TO DEEPSEEK BY TEXT ============

print("\n" + "=" * 80)
print("MATCHING: Finding Ensemble positives in DeepSeek by text (>85% match)")
print("=" * 80)

# For each ensemble positive (filename, khabar_num), try to find matching DS by
# extracting from corpus and matching text

# For now, we'll create a mapping by (filename, khabar_num) just to show what we have
ens_set = {(p['filename'], p['khabar_num']) for p in ens_positives}
ds_set = {(p['filename'], p['khabar_num']) for p in ds_positives.values()}

print(f"\nDirect matching by (filename, khabar_num):")
print(f"  DS ∩ Ens: {len(ds_set & ens_set)}")
print(f"  Only DS:  {len(ds_set - ens_set)}")
print(f"  Only Ens: {len(ens_set - ds_set)}")

# ============ MATCH LR v80 TO DEEPSEEK BY TEXT ============

print("\n" + "=" * 80)
print("MATCHING: LR v80 top positives to DeepSeek (>85% text match)")
print("=" * 80)

ds_texts = [(i, p['text']) for i, p in ds_positives.items()]
matches_lr_to_ds = []
no_match_lr = []

for lr_idx, lr_item in enumerate(lr_top_positives):
    lr_text = lr_item['text']
    best_ds_idx, similarity = find_best_match(lr_text, ds_texts, threshold=0.85)

    if best_ds_idx >= 0:
        matches_lr_to_ds.append({
            'lr_idx': lr_idx,
            'ds_idx': best_ds_idx,
            'similarity': similarity,
            'lr_prob': lr_item['prob_v80'],
            'ds_reason': ds_positives[best_ds_idx]['reason'][:80],
        })
    else:
        no_match_lr.append({
            'lr_idx': lr_idx,
            'text_sample': lr_text[:60],
            'prob': lr_item['prob_v80'],
            'best_similarity': similarity,
        })

print(f"\nFuzzy matches (>85% text similarity):")
print(f"  LR matches to DS: {len(matches_lr_to_ds)}/{len(lr_top_positives)}")
print(f"  LR with no match: {len(no_match_lr)}/{len(lr_top_positives)}")

# ============ MATCH ENSEMBLE TO LR BY TEXT ============

print("\n" + "=" * 80)
print("MATCHING: Ensemble positives to LR v80 (>85% text match)")
print("=" * 80)

# For this we need the actual texts from Ensemble positives
# Let's extract them from the corpus since we have filename + khabar_num

from src.uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file
from src.uqala_nlp.preprocessing.isnad_filter import split_isnad

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"

# Build a map of (filename, khabar_num) -> text for ensemble
print("  Extracting corpus to match ensemble akhbars...")
ens_texts_map = {}
for filepath in sorted(corpus_root.rglob("*-ara1")):
    akhbars = extract_akhbars_from_file(str(filepath))
    for khabar_num, akhbar_raw in enumerate(akhbars):
        akhbar_clean, _ = split_isnad(akhbar_raw)
        key = (filepath.name, khabar_num)
        ens_texts_map[key] = akhbar_clean

ens_text_list = [(i, key, ens_texts_map.get(key, '')) for i, key in enumerate(ens_set)]

lr_texts = [(i, p['text']) for i, p in enumerate(lr_top_positives)]
matches_ens_to_lr = []
no_match_ens = []

for ens_idx, ens_key, ens_text in ens_text_list:
    if not ens_text:
        no_match_ens.append({'ens_key': ens_key, 'reason': 'text not found'})
        continue

    best_lr_idx, similarity = find_best_match(ens_text, lr_texts, threshold=0.85)
    if best_lr_idx >= 0:
        matches_ens_to_lr.append({
            'ens_key': ens_key,
            'lr_idx': best_lr_idx,
            'similarity': similarity,
        })
    else:
        no_match_ens.append({'ens_key': ens_key, 'best_sim': similarity})

print(f"\nFuzzy matches (>85% text similarity):")
print(f"  Ens matches to LR: {len(matches_ens_to_lr)}/{len(ens_positives)}")
print(f"  Ens with no match: {len(no_match_ens)}/{len(ens_positives)}")

# ============ SUMMARY ============

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nModel sizes:")
print(f"  DeepSeek: {len(ds_positives)} positives")
print(f"  Ensemble: {len(ens_positives)} positives")
print(f"  LR v80:   {len(lr_top_positives)} top positives")

print(f"\nModel agreement (fuzzy text matching >85%):")
print(f"  LR & DS:  {len(matches_lr_to_ds)}/{min(len(lr_top_positives), len(ds_positives))}")
print(f"  Ens & LR: {len(matches_ens_to_lr)}/{min(len(ens_positives), len(lr_top_positives))}")

# Find three-way agreement
ds_from_lr = {m['ds_idx'] for m in matches_lr_to_ds}
ens_matched_to_lr = {ens_text_list[m['ens_key'][1]][1] if isinstance(m['ens_key'], tuple) else None for m in matches_ens_to_lr}

print(f"\nExamples from matches:")

if matches_lr_to_ds:
    print(f"\n[LR v80 <-> DeepSeek] Sample matches (>85%):")
    for match in matches_lr_to_ds[:3]:
        ds = ds_positives[match['ds_idx']]
        print(f"  Similarity {match['similarity']:.1%}")
        print(f"    Text: {ds['text'][:70]}...")
        print(f"    DS reason: {match['ds_reason']}")
        print(f"    LR prob: {match['lr_prob']:.4f}\n")

if matches_ens_to_lr:
    print(f"\n[Ensemble <-> LR v80] Sample matches (>85%):")
    for i, match in enumerate(matches_ens_to_lr[:3]):
        print(f"  {match['ens_key']}")
        print(f"    Similarity: {match['similarity']:.1%}\n")

print("\n" + "=" * 80)
