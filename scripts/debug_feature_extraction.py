#!/usr/bin/env python3
"""
debug_feature_extraction.py
──────────────────────────────────────────────────────────────
Compare feature extraction between LR and XGBoost validation scripts.

This script:
1. Loads the same akhbars
2. Extracts features using BOTH methods
3. Compares feature names, order, and values
4. Reports any differences
"""

import json
import sys
import pathlib
import numpy as np
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

BASE = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results" / "0328IbnCabdRabbih"

sys.path.insert(0, str(BASE / "src"))
from uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file
from uqala_nlp.preprocessing.isnad_filter import split_isnad
from uqala_nlp.preprocessing.smart_camel_loader import extract_morpho_features_safe as extract_morpho_safe

# ════════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION CODE (from validate_v80_on_ibn_rabbih.py)
# ════════════════════════════════════════════════════════════════════════════════

import re

JUNUN_TERMS = [
    'مجنون','المجنون','مجنونا','مجانين','المجانين','مجنونة','المجنونة',
    'معتوه','المعتوه','معتوها','معتوهة','مدله','المدله',
    'هائم','الهائم','هائما','ممسوس','ممرور','مستهتر',
    'جنونه','جنونها','جنوني','جنونا','جنون','الجنون',
    'ذاهبالعقل','ذهبعقله','ذاهب','ذهب',
]

FAMOUS_FOOLS = ['بهلول','بهلولا','سعدون','عليان','جعيفران','ريحانة','سمنون','لقيط','حيون','حيونة','خلف','رياح']

SCENE_INTRO_VERBS = ['مررت','دخلت','فرأيت','لقيت','أتيت','سقطت','خرجت','وجدت']
WITNESS_VERBS = ['رأيت','فرأيت','شاهدت','أبصرت','عاينت']
DIALOGUE_FIRST_PERSON = ['قلت','فقلت','قلنا']
DIRECT_ADDRESS = ['يا بهلول','يا مجنون','يا ذا','يا هرم','يا هذا','يا سعدون']
DIVINE_PERSONAL = ['إلهي','اللهم','يا رب','يا إلهي']
SACRED_SPACES = ['أزقة','المقابر','خرابات','الخرابات','قبر','سوق','مسجد']


def has_junun_filtered(text):
    if not any(t in text for t in JUNUN_TERMS):
        return False
    if any(fp in text for fp in ['الجنة ', ' الجنة', 'الجن ', 'السجن ']):
        return False
    return True


def extract_junun_features_15(text):
    features = {}
    tokens = re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
    n_tokens = len(tokens) if tokens else 1

    features['f00_has_junun'] = float(has_junun_filtered(text))
    features['f01_junun_density'] = sum(1 for t in tokens if t in JUNUN_TERMS) / n_tokens
    features['f02_famous_fool'] = float(any(name in text for name in FAMOUS_FOOLS))
    junun_count = sum(1 for t in JUNUN_TERMS if t in text)
    features['f03_junun_count'] = min(float(junun_count), 10.0) / 10
    specialized = [t for t in JUNUN_TERMS if len(t) > 4]
    features['f04_junun_specialized'] = float(any(s in text for s in specialized))
    first_junun = None
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0 and (first_junun is None or idx < first_junun):
            first_junun = idx
    features['f05_junun_position'] = first_junun / max(len(text), 1) if first_junun else 0.5
    features['f06_junun_in_title'] = float(any(term in text[:50] for term in JUNUN_TERMS))
    features['f07_junun_plural'] = float('مجانين' in text or 'المجانين' in text)
    third = len(text) // 3
    features['f08_junun_in_final_third'] = float(any(t in text[2*third:] for t in JUNUN_TERMS))
    jinn_root = ['جنون','جنونه','جنونها','جننت','يجن','أجنّ']
    features['f09_jinn_root'] = float(any(j in text for j in jinn_root))
    junun_rep = sum(text.count(t) for t in JUNUN_TERMS)
    features['f10_junun_repetition'] = min(float(junun_rep) / n_tokens, 1.0)
    features['f11_junun_morpho'] = float(any(m in text for m in ['مجنون','معتوه','هائم','ممسوس']))
    neg_junun = any(ng in text for ng in ['لا مجنون','ليس مجنون','لم يكن مجنون'])
    features['f12_junun_positive'] = float(not neg_junun)
    junun_context = 0
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0:
            context = text[max(0, idx-20):idx+len(t)+20]
            if any(c in context for c in ['قال','رأيت','شهدت']):
                junun_context += 1
    features['f13_junun_good_context'] = float(junun_context > 0)
    val_all = ['ضحك','أعطى','بكى']
    def count_proximity(text, list1, list2, window=100):
        count = 0
        for term1 in list1:
            idx = text.find(term1)
            if idx < 0:
                continue
            for term2 in list2:
                start = max(0, idx - window)
                end = min(len(text), idx + len(term1) + window)
                if term2 in text[start:end]:
                    count += 1
                    break
        return count
    features['f14_junun_validation_prox'] = float(count_proximity(text, JUNUN_TERMS, val_all, 100) > 0)
    return features


def extract_empirical_features_6(text):
    features = {}
    tokens = re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
    n_tokens = len(tokens) if tokens else 1

    has_scene = any(v in text for v in SCENE_INTRO_VERBS)
    features['E1_scene_intro_presence'] = float(has_scene)

    witness_count = sum(text.count(v) for v in WITNESS_VERBS)
    features['E2_witness_verb_density'] = witness_count / n_tokens

    dialogue_count = sum(text.count(v) for v in DIALOGUE_FIRST_PERSON)
    features['E3_dialogue_first_person_density'] = dialogue_count / n_tokens

    has_direct_address = any(addr in text for addr in DIRECT_ADDRESS)
    features['E4_direct_address_presence'] = float(has_direct_address)

    divine_count = sum(text.count(d) for d in DIVINE_PERSONAL)
    features['E5_divine_personal_intensity'] = min(float(divine_count), 5.0) / 5

    has_sacred_space = any(loc in text for loc in SACRED_SPACES)
    features['E6_sacred_spaces_presence'] = float(has_sacred_space)

    return features


def extract_morphological_features_6(text):
    if extract_morpho_safe:
        return extract_morpho_safe(text)
    else:
        return {
            'f65_root_jnn_density': 0.0,
            'f66_root_aql_density': 0.0,
            'f67_root_hikma_density': 0.0,
            'f68_verb_density': 0.0,
            'f69_noun_density': 0.0,
            'f70_adj_density': 0.0,
        }


def extract_all_features_27(text):
    """Extract all 27 v80 features in INSERTION ORDER (dict.values())"""
    features = {}
    features.update(extract_junun_features_15(text))
    features.update(extract_morphological_features_6(text))
    features.update(extract_empirical_features_6(text))
    return features


# ════════════════════════════════════════════════════════════════════════════════
# MAIN DEBUG
# ════════════════════════════════════════════════════════════════════════════════

print("="*100)
print("DEBUGGING FEATURE EXTRACTION")
print("="*100)

print("\n[1] Loading sample akhbars from corpus...")

corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"
text_files = sorted(corpus_root.rglob("*-ara1"))[:3]  # First 3 files

all_features = []
feature_names_set = set()

with tqdm(total=sum(len(extract_akhbars_from_file(str(f))) for f in text_files), desc="  Extracting features") as pbar:
    for filepath in text_files:
        akhbars = extract_akhbars_from_file(str(filepath))

        for khabar_num, akhbar_raw in enumerate(akhbars[:5]):  # First 5 per file
            _, akhbar = split_isnad(akhbar_raw)

            try:
                features = extract_all_features_27(akhbar)

                if len(features) == 27:
                    all_features.append({
                        'filename': filepath.name,
                        'khabar_num': khabar_num,
                        'features': features,
                        'text_preview': akhbar[:80],
                    })
                    feature_names_set.update(features.keys())

            except Exception as e:
                pass

            pbar.update(1)

print(f"\n[OK] Extracted features from {len(all_features)} akhbars\n")

# Analyze feature names and order
print("[2] Feature names and order analysis:")
print("    " + "-"*96)

feature_names_list = sorted(feature_names_set)
print(f"\n    Total unique feature names: {len(feature_names_list)}\n")

print("    Feature names (in sorted order):")
for i, fname in enumerate(feature_names_list):
    print(f"      {i+1:2d}. {fname}")

# Check insertion order consistency
print(f"\n[3] Checking insertion order consistency...")
print("    " + "-"*96)

first_example = all_features[0]['features']
first_order = list(first_example.keys())

print(f"\n    First example insertion order:")
for i, fname in enumerate(first_order):
    print(f"      {i+1:2d}. {fname}")

all_orders_same = True
for i, ex in enumerate(all_features[1:], 1):
    current_order = list(ex['features'].keys())
    if current_order != first_order:
        print(f"\n    ⚠️  Example {i} has DIFFERENT order!")
        all_orders_same = False
        print(f"      Expected: {first_order}")
        print(f"      Got:      {current_order}")
        break

if all_orders_same:
    print(f"\n    ✓ All examples have SAME insertion order (consistent!)")

# Check for NaN values
print(f"\n[4] Checking for NaN/Inf values...")
print("    " + "-"*96)

nan_count = 0
inf_count = 0
nan_features = {}
inf_features = {}

for ex in all_features:
    for fname, fvalue in ex['features'].items():
        if np.isnan(fvalue):
            nan_count += 1
            nan_features[fname] = nan_features.get(fname, 0) + 1
        if np.isinf(fvalue):
            inf_count += 1
            inf_features[fname] = inf_features.get(fname, 0) + 1

if nan_count == 0 and inf_count == 0:
    print(f"\n    ✓ No NaN or Inf values found (good!)")
else:
    print(f"\n    ⚠️  Found issues:")
    if nan_count > 0:
        print(f"      NaN values: {nan_count} occurrences")
        print(f"      Affected features: {list(nan_features.keys())}")
    if inf_count > 0:
        print(f"      Inf values: {inf_count} occurrences")
        print(f"      Affected features: {list(inf_features.keys())}")

# Check feature value ranges
print(f"\n[5] Feature value ranges (statistics)...")
print("    " + "-"*96)

feature_stats = {}
for ex in all_features:
    for fname, fvalue in ex['features'].items():
        if fname not in feature_stats:
            feature_stats[fname] = {'min': fvalue, 'max': fvalue, 'sum': 0, 'count': 0}
        feature_stats[fname]['min'] = min(feature_stats[fname]['min'], fvalue)
        feature_stats[fname]['max'] = max(feature_stats[fname]['max'], fvalue)
        feature_stats[fname]['sum'] += fvalue
        feature_stats[fname]['count'] += 1

print(f"\n    Feature ranges (min-max, mean):\n")
for fname in sorted(feature_stats.keys()):
    stats = feature_stats[fname]
    mean = stats['sum'] / stats['count']
    print(f"      {fname:30s} | min={stats['min']:.3f} max={stats['max']:.3f} mean={mean:.3f}")

# Save debug report
print(f"\n[6] Saving debug report...")

debug_file = RESULTS_DIR / "feature_extraction_debug.json"
with open(debug_file, 'w', encoding='utf-8') as f:
    json.dump({
        'total_akhbars_sampled': len(all_features),
        'feature_count': 27,
        'feature_names': sorted(feature_names_list),
        'insertion_order': list(all_features[0]['features'].keys()) if all_features else [],
        'insertion_order_consistent': all_orders_same,
        'nan_values_found': nan_count,
        'inf_values_found': inf_count,
        'feature_statistics': {
            fname: {
                'min': float(stats['min']),
                'max': float(stats['max']),
                'mean': float(stats['sum'] / stats['count']),
            }
            for fname, stats in feature_stats.items()
        },
        'sample_akhbars': [
            {
                'filename': ex['filename'],
                'khabar_num': ex['khabar_num'],
                'text_preview': ex['text_preview'],
                'features': {k: float(v) for k, v in ex['features'].items()}
            }
            for ex in all_features[:5]
        ]
    }, f, ensure_ascii=False, indent=2)

print(f"    [OK] Report saved to: {debug_file.name}\n")

print("="*100)
print("DEBUG COMPLETE")
print("="*100)
