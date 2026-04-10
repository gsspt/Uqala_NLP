#!/usr/bin/env python3
"""
validate_v80_external_corpus.py
──────────────────────────────────────────────────────────────────
Validate v80 on external corpus (Ibn Abd Rabbih CiqdFarid).
Compare false positive rate vs A1_conservative baseline.

Usage:
  python validate_v80_external_corpus.py [--model v80|v79]
"""

import json
import sys
import pathlib
import pickle
import re
import argparse
import importlib.util
import numpy as np
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

# Setup paths
BASE = pathlib.Path(__file__).resolve().parent
OPENITI_ROOT = BASE / "openiti_corpus" / "data"
IBN_RABBIH = OPENITI_ROOT / "0328IbnCabdRabbih" / "0328IbnCabdRabbih.CiqdFarid"
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results" / "0328IbnCabdRabbih"

# Smart CAMeL Tools loader
smart_loader_path = BASE / "smart_camel_loader.py"
if smart_loader_path.exists():
    spec = importlib.util.spec_from_file_location("smart_camel_loader", smart_loader_path)
    smart_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smart_loader)
    HAS_CAMEL = smart_loader.HAS_CAMEL
    extract_morpho_safe = smart_loader.extract_morpho_features_safe
else:
    HAS_CAMEL = False
    extract_morpho_safe = None

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (same as v80 pipeline)
# ══════════════════════════════════════════════════════════════════════════════

JUNUN_TERMS = [
    'مجنون','المجنون','مجنونا','مجانين','المجانين','مجنونة','المجنونة',
    'معتوه','المعتوه','معتوها','معتوهة','مدله','المدله',
    'هائم','الهائم','هائما','ممسوس','ممرور','مستهتر',
    'جنونه','جنونها','جنوني','جنونا','جنون','الجنون',
    'ذاهبالعقل','ذهبعقله','ذاهب','ذهب',
]

FAMOUS_FOOLS = [
    'بهلول','بهلولا','سعدون','عليان','جعيفران','ريحانة',
    'سمنون','لقيط','حيون','حيونة','خلف','رياح',
]

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
    """Extract all 27 v80 features"""
    features = {}
    features.update(extract_junun_features_15(text))
    features.update(extract_morphological_features_6(text))
    features.update(extract_empirical_features_6(text))
    return features

# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND PROCESS IBN RABBIH CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def load_ibn_rabbih_texts():
    """Load all text files from Ibn Abd Rabbih CiqdFarid"""
    texts = []

    if not IBN_RABBIH.exists():
        print(f"ERROR: Ibn Rabbih corpus not found at {IBN_RABBIH}")
        return []

    # Find all text files (not .yml, not README, not .md)
    text_files = [f for f in IBN_RABBIH.glob('*')
                  if f.is_file() and not f.suffix in ['.yml', '.md'] and not f.name.startswith('README')]

    print(f"Found {len(text_files)} text files in Ibn Rabbih corpus")

    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    texts.append({
                        'filename': text_file.name,
                        'text': content,
                        'length': len(content)
                    })
        except Exception as e:
            print(f"Warning: Could not read {text_file}: {e}")

    print(f"Loaded {len(texts)} texts")
    return texts

def split_text_into_passages(text, window_size=500, step_size=250):
    """Split long text into overlapping passages"""
    passages = []
    text_clean = text.replace('\n', ' ').strip()

    for i in range(0, len(text_clean) - window_size, step_size):
        passage = text_clean[i:i + window_size]
        if len(passage.split()) > 10:  # Only keep passages with enough words
            passages.append(passage)

    # Add final passage
    if len(text_clean) > window_size:
        final = text_clean[-window_size:]
        if len(final.split()) > 10 and final != passages[-1]:
            passages.append(final)

    return passages

def validate_v80(model_path, verbose=False, sample_size=None):
    """Load v80 model and make predictions on Ibn Rabbih corpus"""

    # Load model
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return None

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Handle both dict format (with scaler) and direct model format
        if isinstance(model_data, dict) and 'clf' in model_data:
            clf = model_data['clf']
            scaler = model_data.get('scaler', None)
        else:
            clf = model_data
            scaler = None

        print(f"✅ Loaded model from {model_path.name}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

    # Load texts
    texts = load_ibn_rabbih_texts()
    if not texts:
        return None

    # Process each text into passages and extract features
    all_predictions = []
    all_texts = []

    print("\nExtracting features from passages...")
    for idx, text_info in enumerate(texts):
        print(f"  Processing {idx+1}/{len(texts)}: {text_info['filename']}")
        passages = split_text_into_passages(text_info['text'])

        # Limit to sample size if specified
        if sample_size:
            passages = passages[:sample_size]

        for passage in passages:
            try:
                features = extract_all_features_27(passage)

                # Check if we have all 27 features
                if len(features) != 27:
                    if verbose:
                        print(f"  Warning: got {len(features)} features instead of 27")
                    continue

                feature_vector = np.array([features[k] for k in sorted(features.keys())]).reshape(1, -1)

                # Scale if scaler is available
                if scaler:
                    feature_vector = scaler.transform(feature_vector)

                # Predict
                pred_proba = clf.predict_proba(feature_vector)[0, 1]
                pred_label = clf.predict(feature_vector)[0]

                all_predictions.append({
                    'filename': text_info['filename'],
                    'passage': passage[:100] + '...' if len(passage) > 100 else passage,
                    'pred_proba': float(pred_proba),
                    'pred_label': int(pred_label),
                    'text_length': len(passage)
                })
                all_texts.append(passage)
            except Exception as e:
                print(f"  Error in passage processing: {e}", file=sys.stderr)
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue

    return all_predictions

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Validate v80 on Ibn Abd Rabbih corpus')
    parser.add_argument('--model', choices=['v80', 'v79'], default='v80',
                        help='Model to test (default: v80)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--sample', type=int, default=None,
                        help='Limit passages to N per text file (default: all)')
    args = parser.parse_args()

    # Select model
    if args.model == 'v80':
        model_path = MODELS_DIR / 'lr_classifier_v80.pkl'
    else:
        model_path = MODELS_DIR / 'lr_classifier_79features.pkl'

    print(f"Testing {args.model} on Ibn Abd Rabbih external corpus")
    print(f"Model: {model_path}")
    print(f"CAMeL Tools: {'✅ Available' if HAS_CAMEL else '⚠️  Not available (degraded mode)'}")
    if args.sample:
        print(f"Sample mode: {args.sample} passages per text\n")
    else:
        print("")

    # Validate
    predictions = validate_v80(model_path, verbose=args.verbose, sample_size=args.sample)

    if not predictions:
        print("ERROR: Could not generate predictions (no passages processed)")
        return 1

    if len(predictions) == 0:
        print("ERROR: No predictions generated")
        return 1

    # Analysis
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {args.model.upper()}")
    print(f"{'='*80}\n")

    total = len(predictions)
    positives = sum(1 for p in predictions if p['pred_label'] == 1)
    negatives = total - positives

    print(f"Total predictions: {total}")
    print(f"Positive predictions: {positives} ({100*positives/total:.1f}%)")
    print(f"Negative predictions: {negatives} ({100*negatives/total:.1f}%)")

    # Probabilities
    probs = [p['pred_proba'] for p in predictions]
    print(f"\nPrediction confidence:")
    print(f"  Mean: {np.mean(probs):.3f}")
    print(f"  Median: {np.median(probs):.3f}")
    print(f"  Std: {np.std(probs):.3f}")

    # Top confident positives
    top_positive = sorted([p for p in predictions if p['pred_label'] == 1],
                          key=lambda x: x['pred_proba'], reverse=True)[:5]

    if top_positive:
        print(f"\nTop 5 confident positive predictions:")
        for i, pred in enumerate(top_positive, 1):
            print(f"  {i}. ({pred['pred_proba']:.3f}) {pred['filename']}: {pred['passage']}")

    # Compare to baseline
    print(f"\n{'='*80}")
    print(f"COMPARISON TO A1_CONSERVATIVE BASELINE")
    print(f"{'='*80}\n")
    print(f"v79 (A1_conservative): 54.6% positives on Ibn Rabbih")
    print(f"{args.model.upper()}: {100*positives/total:.1f}% positives on Ibn Rabbih")

    if positives/total < 0.546:
        improvement = (0.546 - positives/total) * 100
        print(f"✅ IMPROVEMENT: {improvement:.1f} percentage points reduction in false positives")
    else:
        worse = (positives/total - 0.546) * 100
        print(f"⚠️  WORSE: {worse:.1f} percentage points more positives")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"{args.model}_predictions_on_ibn_rabbih.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': args.model,
            'corpus': 'Ibn Abd Rabbih CiqdFarid',
            'total_predictions': total,
            'positives': positives,
            'negatives': negatives,
            'positive_rate': positives / total,
            'mean_confidence': float(np.mean(probs)),
            'predictions': predictions
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to {results_file.name}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
