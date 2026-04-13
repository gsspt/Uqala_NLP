#!/usr/bin/env python3
"""
validate_xgboost_v80_on_ibn_rabbih.py
──────────────────────────────────────────────────────────────
Proper validation of XGBoost v80 on Ibn Abd Rabbih using:
1. Correct akhbar extraction (not sliding window)
2. Isnad filtering (required preprocessing)
3. All 27 features with correct feature order

Usage:
  python validate_xgboost_v80_on_ibn_rabbih.py
"""

import json
import sys
import pathlib
import pickle
import importlib.util
import numpy as np
from collections import Counter
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# Setup paths
BASE = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results" / "0328IbnCabdRabbih"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import modules
sys.path.insert(0, str(BASE / "src"))
from uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file
from uqala_nlp.preprocessing.isnad_filter import split_isnad
from uqala_nlp.preprocessing.smart_camel_loader import HAS_CAMEL, extract_morpho_features_safe as extract_morpho_safe

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (same as v80 pipeline, 27 features)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_xgboost_v80_on_ibn_rabbih():
    """Validate XGBoost v80 on all Ibn Abd Rabbih akhbars with proper preprocessing"""

    # Load model
    model_path = MODELS_DIR / 'xgb_classifier_v80.pkl'
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print(f"TIP: Run: python pipelines/level2_semi_interpretable/p2_2_xgboost_v80_shap.py")
        return None

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        if isinstance(model_data, dict) and 'clf' in model_data:
            clf = model_data['clf']
            scaler = model_data.get('scaler', None)
        else:
            clf = model_data
            scaler = None
        print(f"[OK] Loaded model from {model_path.name}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

    print(f"     CAMeL Tools: {'Available' if HAS_CAMEL else 'Not available (degraded)'}\n")

    # Find all Ibn Abd Rabbih files
    corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"
    if not corpus_root.exists():
        print(f"ERROR: Corpus path not found: {corpus_root}")
        return None

    text_files = sorted(corpus_root.rglob("*-ara1"))
    print(f"[OK] Found {len(text_files)} OpenITI files\n")
    print("Extracting akhbars and making predictions...\n")

    all_predictions = []
    total_akhbars = 0
    processed_files = 0

    with tqdm(total=len(text_files), desc="  Files", position=0, leave=True) as pbar_files:
        for file_idx, filepath in enumerate(text_files, 1):
            # Extract akhbars from file
            akhbars = extract_akhbars_from_file(str(filepath))
            if not akhbars:
                pbar_files.update(1)
                continue

            processed_files += 1
            file_akhbar_count = len(akhbars)
            total_akhbars += file_akhbar_count

            # Score each akhbar with nested progress bar
            with tqdm(total=file_akhbar_count, desc=f"  {filepath.name[:40]:40s}", position=1, leave=False) as pbar_akhbars:
                for khabar_num, akhbar_raw in enumerate(akhbars):
                    # CRITICAL: Extract matn (narrative) — split_isnad returns (isnad, matn)
                    _, akhbar = split_isnad(akhbar_raw)

                    try:
                        features = extract_all_features_27(akhbar)

                        # Check we have all 27 features
                        if len(features) != 27:
                            pbar_akhbars.update(1)
                            continue

                        # IMPORTANT: use insertion order (dict.values()), NOT sorted()
                        feature_vector = np.array(list(features.values())).reshape(1, -1)
                        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

                        # Scale if available
                        if scaler:
                            feature_vector = scaler.transform(feature_vector)

                        # Predict
                        pred_proba = clf.predict_proba(feature_vector)[0, 1]
                        pred_label = clf.predict(feature_vector)[0]

                        # Store full text for positives
                        pred_entry = {
                            'filename': filepath.name,
                            'khabar_num': khabar_num,
                            'pred_proba': float(pred_proba),
                            'pred_label': int(pred_label),
                            'text_length': len(akhbar),
                        }

                        # Add full text if positive
                        if pred_label == 1:
                            pred_entry['text'] = akhbar

                        all_predictions.append(pred_entry)
                    except Exception as e:
                        pass

                    pbar_akhbars.update(1)

            pbar_files.update(1)

    if not all_predictions:
        print("\nERROR: No predictions generated")
        return None

    # Analysis
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS: XGBoost v80 ON IBN ABD RABBIH")
    print(f"{'='*80}\n")

    total = len(all_predictions)
    positives = sum(1 for p in all_predictions if p['pred_label'] == 1)
    negatives = total - positives

    print(f"Total akhbars processed: {total_akhbars}")
    print(f"Predictions made: {total}")
    print(f"  Positive: {positives} ({100*positives/total:.1f}%)")
    print(f"  Negative: {negatives} ({100*negatives/total:.1f}%)")

    probs = [p['pred_proba'] for p in all_predictions]
    print(f"\nPrediction confidence:")
    print(f"  Mean: {np.mean(probs):.3f}")
    print(f"  Median: {np.median(probs):.3f}")
    print(f"  Std: {np.std(probs):.3f}")

    # Top positives
    top_positive = sorted(
        [p for p in all_predictions if p['pred_label'] == 1],
        key=lambda x: x['pred_proba'],
        reverse=True
    )[:10]

    if top_positive:
        print(f"\nTop 10 positive predictions:")
        for i, pred in enumerate(top_positive, 1):
            print(f"  {i}. P={pred['pred_proba']:.3f} | {pred['filename']} khabar {pred['khabar_num']}")

    # Collect all positives with full text
    all_positives = sorted(
        [p for p in all_predictions if p['pred_label'] == 1],
        key=lambda x: x['pred_proba'],
        reverse=True
    )

    # Save results
    results_file = RESULTS_DIR / "xgboost_v80_validation_proper_akhbars.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'XGBoost v80',
            'corpus': 'Ibn Abd Rabbih (proper akhbar extraction)',
            'extraction_method': 'akhbar_extraction_v2_smart (semantic understanding)',
            'total_akhbars_in_corpus': total_akhbars,
            'total_predictions': total,
            'positives': positives,
            'negatives': negatives,
            'positive_rate': positives / total,
            'mean_confidence': float(np.mean(probs)),
            'positives_detailed': [
                {
                    'idx': i+1,
                    'filename': p['filename'],
                    'khabar_num': p['khabar_num'],
                    'pred_proba': p['pred_proba'],
                    'text_length': p['text_length'],
                    'text': p.get('text', '[NOT STORED]'),
                }
                for i, p in enumerate(all_positives)
            ]
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Results saved to {results_file.name}")
    print(f"    All {positives} positives with full text stored in JSON\n")

    return {
        'total': total,
        'positives': positives,
        'positive_rate': positives / total,
        'mean_confidence': np.mean(probs),
    }


if __name__ == '__main__':
    results = validate_xgboost_v80_on_ibn_rabbih()
