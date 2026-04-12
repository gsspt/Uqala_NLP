#!/usr/bin/env python3
"""
Match DeepSeek positives to LR v80 and Ensemble by TEXT.

Strategy:
1. Load DeepSeek positives (80 items with full text)
2. Extract corpus texts for LR v80 (102) and Ensemble (15) positives
3. Normalize texts (remove diacritics, extra whitespace, punctuation)
4. Use fuzzy matching with high threshold (>75% similarity)
5. Create mapping: which DeepSeek cases match which ML models?
"""

import json
import sys
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

from uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file
from uqala_nlp.preprocessing.isnad_filter import split_isnad

results_dir = BASE / "results" / "0328IbnCabdRabbih"
corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"

# ============ TEXT NORMALIZATION ============

def normalize_text(text):
    """Normalize Arabic text for fuzzy matching:
    - Remove diacritics
    - Remove OpenITI markers (%)
    - Remove extra whitespace
    - Keep essential characters
    """
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u0652]', '', text)  # Arabic diacritics
    # Remove OpenITI markers
    text = re.sub(r'%', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Keep only Arabic + numbers + basic punctuation
    text = re.sub(r'[^\u0600-\u06FF0-9 \.،؟!]', '', text)
    return text.strip()

def text_similarity(t1, t2, threshold=0.75):
    """
    Compute normalized text similarity.
    Returns (similarity_ratio, is_match)
    """
    n1 = normalize_text(t1)
    n2 = normalize_text(t2)

    # Quick check: if very different lengths, skip
    if len(n1) < 10 or len(n2) < 10:
        return (0.0, False)

    # Compute similarity
    ratio = SequenceMatcher(None, n1, n2).ratio()
    is_match = ratio > threshold

    return (ratio, is_match)

# ============ LOAD DEEPSEEK DATA ============

print("=" * 80)
print("FUZZY TEXT MATCHING: DeepSeek ↔ LR v80 + Ensemble")
print("=" * 80)

print("\n[1] Loading DeepSeek positives...")
with open(results_dir / "deepseek_full_corpus_8workers.json", encoding='utf-8') as f:
    ds_data = json.load(f)

ds_positives = {i: p for i, p in enumerate(ds_data['positives'])}
print(f"    {len(ds_positives)} DeepSeek positives loaded")

# ============ EXTRACT CORPUS FOR ML MODELS ============

print("\n[2] Loading LR v80 positive identifiers...")
with open(results_dir / "v80_validation_proper_akhbars.json", encoding='utf-8') as f:
    v80_data = json.load(f)

v80_ids = {}  # (filename, khabar_num) -> index
for p in v80_data['predictions_sample']:
    if p['pred_label'] == 1:
        key = (p['filename'], p['khabar_num'])
        v80_ids[key] = p
print(f"    {len(v80_ids)} LR v80 positives in sample (of {v80_data['positives']} total)")

print("[3] Loading Ensemble positive identifiers...")
with open(results_dir / "ensemble_v80_validation_results.json", encoding='utf-8') as f:
    ens_data = json.load(f)

ens_ids = {}
for p in ens_data['predictions_sample']:
    if p.get('label') == 1:
        key = (p['filename'], p['khabar_num'])
        ens_ids[key] = p
print(f"    {len(ens_ids)} Ensemble positives in sample (of {ens_data['positives']} total)")

# ============ EXTRACT TEXTS FROM CORPUS ============

print("\n[4] Extracting texts from corpus for ML model positives...")
print("    This may take a moment...")

corpus_cache = {}  # Cache to avoid re-extracting same file

v80_texts = {}  # (filename, khabar_num) -> text
ens_texts = {}

for filename, khabar_num in list(v80_ids.keys()) + list(ens_ids.keys()):
    if filename not in corpus_cache:
        # Find file in corpus
        filepath = None
        for fp in corpus_root.rglob(filename):
            filepath = fp
            break

        if filepath:
            akhbars = extract_akhbars_from_file(str(filepath))
            corpus_cache[filename] = akhbars
        else:
            corpus_cache[filename] = None

    akhbars = corpus_cache[filename]
    if akhbars and khabar_num < len(akhbars):
        akhbar_raw = akhbars[khabar_num]
        akhbar_clean, _ = split_isnad(akhbar_raw)

        key = (filename, khabar_num)
        if key in v80_ids:
            v80_texts[key] = akhbar_clean
        if key in ens_ids:
            ens_texts[key] = akhbar_clean

print(f"    Extracted {len(v80_texts)} LR v80 texts")
print(f"    Extracted {len(ens_texts)} Ensemble texts")

# ============ FUZZY MATCHING ============

print("\n" + "=" * 80)
print("FUZZY MATCHING (threshold > 75%)")
print("=" * 80)

matches_ds_to_v80 = []  # (ds_idx, v80_key, similarity)
matches_ds_to_ens = []  # (ds_idx, ens_key, similarity)
no_match_ds = []

for ds_idx, ds_item in ds_positives.items():
    ds_text = ds_item['text']

    # Try to match to LR v80
    best_v80_match = None
    best_v80_sim = 0

    for v80_key, v80_text in v80_texts.items():
        sim, is_match = text_similarity(ds_text, v80_text, threshold=0.75)
        if is_match and sim > best_v80_sim:
            best_v80_sim = sim
            best_v80_match = v80_key

    # Try to match to Ensemble
    best_ens_match = None
    best_ens_sim = 0

    for ens_key, ens_text in ens_texts.items():
        sim, is_match = text_similarity(ds_text, ens_text, threshold=0.75)
        if is_match and sim > best_ens_sim:
            best_ens_sim = sim
            best_ens_match = ens_key

    if best_v80_match or best_ens_match:
        if best_v80_match:
            matches_ds_to_v80.append((ds_idx, best_v80_match, best_v80_sim))
        if best_ens_match:
            matches_ds_to_ens.append((ds_idx, best_ens_match, best_ens_sim))
    else:
        no_match_ds.append(ds_idx)

print(f"\nMatching results:")
print(f"  DeepSeek → LR v80:    {len(matches_ds_to_v80)} matches")
print(f"  DeepSeek → Ensemble:  {len(matches_ds_to_ens)} matches")
print(f"  DeepSeek (no match):  {len(no_match_ds)}")

# ============ AGREEMENT ANALYSIS ============

print("\n" + "=" * 80)
print("AGREEMENT ANALYSIS")
print("=" * 80)

# Find DeepSeek cases that match both LR and Ensemble
ds_matched_both = set()
for ds_idx, _, _ in matches_ds_to_v80:
    for ds_idx2, _, _ in matches_ds_to_ens:
        if ds_idx == ds_idx2:
            ds_matched_both.add(ds_idx)

print(f"\nAgreement levels:")
print(f"  DeepSeek ∩ LR v80 ∩ Ensemble: {len(ds_matched_both)} (HIGH confidence)")
print(f"  DeepSeek ∩ LR v80 only:       {len(matches_ds_to_v80) - len(ds_matched_both)}")
print(f"  DeepSeek ∩ Ensemble only:     {len(matches_ds_to_ens) - len(ds_matched_both)}")
print(f"  DeepSeek unique:              {len(no_match_ds)} (LLM-only findings)")

# ============ DETAILED EXAMPLES ============

print("\n" + "=" * 80)
print("EXAMPLES")
print("=" * 80)

# High confidence (all three agree)
if ds_matched_both:
    print(f"\nAll THREE AGREE (highest confidence):")
    for ds_idx in sorted(list(ds_matched_both))[:3]:
        ds = ds_positives[ds_idx]
        # Find matched v80 and ens
        v80_match = next((m for m in matches_ds_to_v80 if m[0] == ds_idx), None)
        ens_match = next((m for m in matches_ds_to_ens if m[0] == ds_idx), None)

        print(f"\n  DeepSeek idx {ds_idx}:")
        print(f"    Text: {ds['text'][:80]}...")
        print(f"    LR v80: {v80_match[1]} (similarity {v80_match[2]:.1%})")
        print(f"    Ensemble: {ens_match[1]} (similarity {ens_match[2]:.1%})")
        print(f"    DeepSeek reason: {ds['reason'][:70]}...")

# DeepSeek + LR only
ds_lr_only = set(m[0] for m in matches_ds_to_v80) - ds_matched_both
if ds_lr_only:
    print(f"\nDeepSeek ∩ LR v80 ONLY (no Ensemble):")
    for ds_idx in sorted(list(ds_lr_only))[:3]:
        ds = ds_positives[ds_idx]
        v80_match = next(m for m in matches_ds_to_v80 if m[0] == ds_idx)

        print(f"\n  DeepSeek idx {ds_idx}:")
        print(f"    Similarity: {v80_match[2]:.1%}")
        print(f"    DeepSeek reason: {ds['reason'][:80]}...")

# DeepSeek + Ensemble only
ds_ens_only = set(m[0] for m in matches_ds_to_ens) - ds_matched_both
if ds_ens_only:
    print(f"\nDeepSeek ∩ Ensemble ONLY (no LR v80):")
    for ds_idx in sorted(list(ds_ens_only))[:3]:
        ds = ds_positives[ds_idx]
        ens_match = next(m for m in matches_ds_to_ens if m[0] == ds_idx)

        print(f"\n  DeepSeek idx {ds_idx}:")
        print(f"    Similarity: {ens_match[2]:.1%}")
        print(f"    DeepSeek reason: {ds['reason'][:80]}...")

# DeepSeek unique
if no_match_ds:
    print(f"\nDeepSeek UNIQUE (not matched to LR or Ensemble):")
    for ds_idx in sorted(no_match_ds)[:5]:
        ds = ds_positives[ds_idx]

        print(f"\n  DeepSeek idx {ds_idx}:")
        print(f"    Text: {ds['text'][:60]}...")
        print(f"    Reason: {ds['reason'][:80]}...")

# ============ SUMMARY ============

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

total_matched = len(matches_ds_to_v80) + len(matches_ds_to_ens)
print(f"""
DeepSeek fuzzy-matched texts:
  • {len(ds_matched_both)} match ALL THREE (LR + Ensemble + DeepSeek)
  • {len(matches_ds_to_v80) - len(ds_matched_both)} match LR v80 only
  • {len(matches_ds_to_ens) - len(ds_matched_both)} match Ensemble only
  • {len(no_match_ds)} unique to DeepSeek (no fuzzy match found)
  • Total matches: {len(set(m[0] for m in matches_ds_to_v80 + matches_ds_to_ens))}/{len(ds_positives)}

Estimated true majnun aqil in Ibn Abd Rabbih:
  • Very conservative: ~{len(ds_matched_both)} (all three agree)
  • Conservative: ~{len(set(m[0] for m in matches_ds_to_v80 + matches_ds_to_ens))} (at least one ML model + DeepSeek)
  • Liberal: ~{len(ds_positives)} (DeepSeek LLM + some validation)
  • Probable range: {len(ds_matched_both)}-{len(set(m[0] for m in matches_ds_to_v80 + matches_ds_to_ens))} high-confidence cases
""")

print("=" * 80)
