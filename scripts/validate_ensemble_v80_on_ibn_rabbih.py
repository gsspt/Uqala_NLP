#!/usr/bin/env python3
"""
validate_ensemble_v80_on_ibn_rabbih.py
──────────────────────────────────────────────────────────────────
Validate the ENSEMBLE (LR v80 + XGBoost v80 + SHAP) on Ibn Abd Rabbih corpus.

Proper validation using:
1. Correct akhbar extraction (not sliding window)
2. Isnad filtering (required preprocessing)
3. Both LR and XGBoost models with 27 features
4. Ensemble fusion (average voting + agreement-based confidence)
5. Comparison to v80 baseline

Output: Terminal progress + JSON results file

Usage:
  python validate_ensemble_v80_on_ibn_rabbih.py

Requirements:
  - models/lr_classifier_v80.pkl (from p1_4_logistic_regression_v80.py)
  - models/xgb_classifier_v80.pkl (from p2_2_xgboost_v80_shap.py)
"""

import json
import sys
import pathlib
import pickle
import re
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
# FEATURE EXTRACTION (27 v80 features, identical to both models)
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
# ENSEMBLE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_ensemble_on_ibn_rabbih():
    """Validate ensemble on all Ibn Abd Rabbih akhbars with proper preprocessing"""

    print("\n" + "="*80)
    print("ENSEMBLE v80 VALIDATION ON IBN ABD RABBIH")
    print("="*80)

    # Load models
    print("\n[STEP 1] Loading models...")
    lr_model_path = MODELS_DIR / 'lr_classifier_v80.pkl'
    xgb_model_path = MODELS_DIR / 'xgb_classifier_v80.pkl'

    if not lr_model_path.exists():
        print(f"  ERROR: LR model not found at {lr_model_path}")
        return None

    if not xgb_model_path.exists():
        print(f"  ERROR: XGBoost model not found at {xgb_model_path}")
        print(f"  TIP: Run: python pipelines/level2_semi_interpretable/p2_2_xgboost_v80_shap.py")
        return None

    try:
        with open(lr_model_path, 'rb') as f:
            lr_data = pickle.load(f)
        if isinstance(lr_data, dict) and 'clf' in lr_data:
            lr_clf = lr_data['clf']
            lr_scaler = lr_data.get('scaler', None)
        else:
            lr_clf = lr_data
            lr_scaler = None
        print(f"  [OK] Loaded LR model from {lr_model_path.name}")
    except Exception as e:
        print(f"  ERROR loading LR model: {e}")
        return None

    try:
        with open(xgb_model_path, 'rb') as f:
            xgb_data = pickle.load(f)
        if isinstance(xgb_data, dict) and 'clf' in xgb_data:
            xgb_clf = xgb_data['clf']
            xgb_scaler = xgb_data.get('scaler', None)
        else:
            xgb_clf = xgb_data
            xgb_scaler = None
        print(f"  [OK] Loaded XGBoost model from {xgb_model_path.name}")
    except Exception as e:
        print(f"  ERROR loading XGBoost model: {e}")
        return None

    print(f"  [INFO] CAMeL Tools: {'Available' if HAS_CAMEL else 'Not available (degraded)'}\n")

    # Find corpus
    print("[STEP 2] Finding Ibn Abd Rabbih corpus...")
    corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"
    if not corpus_root.exists():
        print(f"  ERROR: Corpus path not found: {corpus_root}")
        return None

    text_files = sorted(corpus_root.rglob("*-ara1"))
    print(f"  [OK] Found {len(text_files)} OpenITI files\n")

    # Extract and validate
    print("[STEP 3] Extracting akhbars and making predictions...")
    print("  " + "-"*76)

    all_predictions = []
    total_akhbars = 0
    processed_files = 0
    file_stats = []

    with tqdm(total=len(text_files), desc="  Files processed", position=0, leave=True) as pbar_files:
        for file_idx, filepath in enumerate(text_files, 1):
            # Extract akhbars from file
            akhbars = extract_akhbars_from_file(str(filepath))
            if not akhbars:
                pbar_files.update(1)
                continue

            processed_files += 1
            file_akhbar_count = len(akhbars)
            total_akhbars += file_akhbar_count
            file_predictions = []

            # Score each akhbar
            with tqdm(total=len(akhbars), desc=f"  {filepath.name[:40]:40s}", position=1, leave=False) as pbar_akhbars:
                for khabar_num, akhbar_raw in enumerate(akhbars):
                    # Extract matn (narrative) — split_isnad returns (isnad, matn)
                    _, akhbar = split_isnad(akhbar_raw)

                    try:
                        features = extract_all_features_27(akhbar)

                        # Check we have all 27 features
                        if len(features) != 27:
                            pbar_akhbars.update(1)
                            continue

                        # Create feature vector (insertion order)
                        feature_vector = np.array(list(features.values())).reshape(1, -1)
                        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

                        # Scale if available
                        if lr_scaler:
                            X_lr = lr_scaler.transform(feature_vector)
                        else:
                            X_lr = feature_vector

                        if xgb_scaler:
                            X_xgb = xgb_scaler.transform(feature_vector)
                        else:
                            X_xgb = feature_vector

                        # Get predictions
                        p_lr = lr_clf.predict_proba(X_lr)[0, 1]
                        p_xgb = xgb_clf.predict_proba(X_xgb)[0, 1]

                        # Ensemble fusion
                        p_ensemble = (p_lr + p_xgb) / 2
                        agreement = 1 - abs(p_lr - p_xgb)

                        prediction = {
                            'filename': filepath.name,
                            'khabar_num': khabar_num,
                            'pred_ensemble': float(p_ensemble),
                            'pred_lr': float(p_lr),
                            'pred_xgb': float(p_xgb),
                            'agreement': float(agreement),
                            'text_length': len(akhbar),
                            'label': 1 if p_ensemble >= 0.5 else 0,
                        }
                        all_predictions.append(prediction)
                        file_predictions.append(prediction)

                    except Exception as e:
                        pass

                    pbar_akhbars.update(1)

            file_stats.append({
                'filename': filepath.name,
                'akhbars': len(akhbars),
                'predictions': len(file_predictions),
                'positives': sum(1 for p in file_predictions if p['label'] == 1),
            })

            pbar_files.update(1)

    print("  " + "-"*76)
    print(f"  Files processed: {processed_files}/{len(text_files)}")
    print(f"  Total akhbars: {total_akhbars}")
    print(f"  Total predictions: {len(all_predictions)}\n")

    if not all_predictions:
        print("  ERROR: No predictions generated")
        return None

    # Analysis
    print("[STEP 4] Computing statistics...")
    print("  " + "-"*76)

    total = len(all_predictions)
    positives = sum(1 for p in all_predictions if p['label'] == 1)
    negatives = total - positives

    print(f"\n  Classification results:")
    print(f"    Total predictions: {total}")
    print(f"    Positives (p >= 0.5): {positives} ({100*positives/total:.1f}%)")
    print(f"    Negatives (p < 0.5): {negatives} ({100*negatives/total:.1f}%)")

    probs_ensemble = np.array([p['pred_ensemble'] for p in all_predictions])
    agreements = np.array([p['agreement'] for p in all_predictions])

    print(f"\n  Ensemble confidence (0-1):")
    print(f"    Mean: {np.mean(probs_ensemble):.3f}")
    print(f"    Median: {np.median(probs_ensemble):.3f}")
    print(f"    Std: {np.std(probs_ensemble):.3f}")

    print(f"\n  Model agreement (0-1):")
    print(f"    Mean: {np.mean(agreements):.3f}")
    print(f"    Median: {np.median(agreements):.3f}")
    print(f"    Std: {np.std(agreements):.3f}")

    # Agreement distribution
    very_high = sum(1 for a in agreements if a >= 0.90)
    high = sum(1 for a in agreements if 0.70 <= a < 0.90)
    medium = sum(1 for a in agreements if 0.50 <= a < 0.70)
    low = sum(1 for a in agreements if a < 0.50)

    print(f"\n  Agreement levels:")
    print(f"    VERY HIGH (>=0.90): {very_high} ({100*very_high/total:.1f}%)")
    print(f"    HIGH (0.70-0.89):   {high} ({100*high/total:.1f}%)")
    print(f"    MEDIUM (0.50-0.69): {medium} ({100*medium/total:.1f}%)")
    print(f"    LOW (<0.50):        {low} ({100*low/total:.1f}%)")

    # Top positive predictions
    top_positive = sorted(
        [p for p in all_predictions if p['label'] == 1],
        key=lambda x: x['pred_ensemble'],
        reverse=True
    )[:10]

    if top_positive:
        print(f"\n  Top 10 positive predictions (ensemble >= 0.5):")
        for i, pred in enumerate(top_positive, 1):
            agreement_str = f"[agreement: {pred['agreement']:.3f}]"
            print(f"    {i:2d}. E={pred['pred_ensemble']:.3f} L={pred['pred_lr']:.3f} X={pred['pred_xgb']:.3f} {agreement_str}")

    # Comparison to v80 LR baseline
    print(f"\n" + "="*80)
    print("COMPARISON TO V80 BASELINE (LR alone)")
    print("="*80)

    v80_positives = sum(1 for p in all_predictions if p['pred_lr'] >= 0.5)
    v80_positive_rate = 100 * v80_positives / total

    print(f"\n  LR v80 alone:      {v80_positives} positives ({v80_positive_rate:.1f}%)")
    print(f"  Ensemble v80:      {positives} positives ({100*positives/total:.1f}%)")

    if positives < v80_positives:
        reduction = v80_positives - positives
        print(f"  Reduction:         {reduction} fewer positives ({100*reduction/v80_positives:.1f}% fewer)")
    else:
        increase = positives - v80_positives
        print(f"  Increase:          {increase} more positives ({100*increase/v80_positives:.1f}% more)")

    # Save results
    print(f"\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")

    results_file = RESULTS_DIR / "ensemble_v80_validation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'Ensemble v80 (LR + XGBoost)',
            'corpus': 'Ibn Abd Rabbih (proper akhbar extraction)',
            'total_akhbars_in_corpus': total_akhbars,
            'total_predictions': total,
            'positives': positives,
            'negatives': negatives,
            'positive_rate': positives / total,
            'mean_ensemble_confidence': float(np.mean(probs_ensemble)),
            'mean_agreement': float(np.mean(agreements)),
            'agreement_distribution': {
                'very_high': int(very_high),
                'high': int(high),
                'medium': int(medium),
                'low': int(low),
            },
            'file_statistics': file_stats,
            'predictions_sample': top_positive,
        }, f, ensure_ascii=False, indent=2)

    print(f"  [OK] Results saved to: {results_file}")
    print(f"       {results_file.name}\n")

    print("="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)

    return {
        'total': total,
        'positives': positives,
        'positive_rate': positives / total,
        'mean_confidence': np.mean(probs_ensemble),
        'mean_agreement': np.mean(agreements),
    }


if __name__ == '__main__':
    results = validate_ensemble_on_ibn_rabbih()
