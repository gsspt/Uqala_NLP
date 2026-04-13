#!/usr/bin/env python3
"""
validate_with_variable_thresholds.py
──────────────────────────────────────────────────────────────
Test LR v80 and XGBoost v80 at different probability thresholds.

This script:
1. Loads both models
2. Tests thresholds: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
3. For each threshold: counts positives and measures agreement
4. Creates a threshold optimization report

Usage:
  python scripts/validate_with_variable_thresholds.py
"""

import json
import sys
import pathlib
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

BASE = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results" / "0328IbnCabdRabbih"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "src"))
from uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file
from uqala_nlp.preprocessing.isnad_filter import split_isnad
from uqala_nlp.preprocessing.smart_camel_loader import HAS_CAMEL, extract_morpho_features_safe as extract_morpho_safe

# ════════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (same as validation scripts, 27 features)
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
    """Extract all 27 v80 features in INSERTION ORDER"""
    features = {}
    features.update(extract_junun_features_15(text))
    features.update(extract_morphological_features_6(text))
    features.update(extract_empirical_features_6(text))
    return features


# ════════════════════════════════════════════════════════════════════════════════
# THRESHOLD TESTING
# ════════════════════════════════════════════════════════════════════════════════

def test_thresholds():
    """Test LR and XGBoost at multiple thresholds"""

    # Load models
    print("="*100)
    print("THRESHOLD OPTIMIZATION: LR v80 vs XGBoost v80")
    print("="*100)

    print("\n[1] Loading models...")

    lr_path = MODELS_DIR / 'lr_classifier_v80.pkl'
    xgb_path = MODELS_DIR / 'xgb_classifier_v80.pkl'

    if not lr_path.exists() or not xgb_path.exists():
        print("ERROR: Models not found")
        return None

    try:
        with open(lr_path, 'rb') as f:
            lr_data = pickle.load(f)
        lr_clf = lr_data['clf'] if isinstance(lr_data, dict) else lr_data
        lr_scaler = lr_data.get('scaler') if isinstance(lr_data, dict) else None

        with open(xgb_path, 'rb') as f:
            xgb_data = pickle.load(f)
        xgb_clf = xgb_data['clf'] if isinstance(xgb_data, dict) else xgb_data
        xgb_scaler = xgb_data.get('scaler') if isinstance(xgb_data, dict) else None

        print(f"  [OK] Loaded LR and XGBoost models")
    except Exception as e:
        print(f"ERROR loading models: {e}")
        return None

    # Load all predictions with probabilities
    print("\n[2] Extracting akhbars and computing probabilities...")

    corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"
    text_files = sorted(corpus_root.rglob("*-ara1"))

    all_predictions = []
    total_akhbars = 0

    with tqdm(total=len(text_files), desc="  Files", position=0, leave=True) as pbar_files:
        for filepath in text_files:
            akhbars = extract_akhbars_from_file(str(filepath))
            if not akhbars:
                pbar_files.update(1)
                continue

            file_akhbar_count = len(akhbars)
            total_akhbars += file_akhbar_count

            with tqdm(total=file_akhbar_count, desc=f"  {filepath.name[:40]:40s}", position=1, leave=False) as pbar_akhbars:
                for khabar_num, akhbar_raw in enumerate(akhbars):
                    _, akhbar = split_isnad(akhbar_raw)

                    try:
                        features = extract_all_features_27(akhbar)
                        if len(features) != 27:
                            pbar_akhbars.update(1)
                            continue

                        fv = np.array(list(features.values())).reshape(1, -1)
                        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)

                        # Get probabilities
                        X_lr = lr_scaler.transform(fv) if lr_scaler else fv
                        X_xgb = xgb_scaler.transform(fv) if xgb_scaler else fv

                        p_lr = float(lr_clf.predict_proba(X_lr)[0, 1])
                        p_xgb = float(xgb_clf.predict_proba(X_xgb)[0, 1])

                        all_predictions.append({
                            'filename': filepath.name,
                            'khabar_num': khabar_num,
                            'p_lr': p_lr,
                            'p_xgb': p_xgb,
                        })
                    except Exception as e:
                        pass

                    pbar_akhbars.update(1)

            pbar_files.update(1)

    print(f"\n[OK] Processed {total_akhbars} akhbars, {len(all_predictions)} predictions\n")

    # Test thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print("[3] Testing thresholds...")
    print("    " + "-"*96 + "\n")
    print(f"    {'Threshold':>10} | {'LR Pos':>8} {'(%)':>6} | {'XGB Pos':>8} {'(%)':>6} | {'Agreement':>10} | {'Jaccard':>8}")
    print("    " + "-"*96)

    threshold_results = {}

    for threshold in thresholds:
        lr_pos = set()
        xgb_pos = set()

        for i, pred in enumerate(all_predictions):
            if pred['p_lr'] >= threshold:
                lr_pos.add(i)
            if pred['p_xgb'] >= threshold:
                xgb_pos.add(i)

        # Calculate agreement
        both = lr_pos & xgb_pos
        either = lr_pos | xgb_pos
        union = len(either) if either else 1

        jaccard = len(both) / union * 100 if union > 0 else 0
        overlap = len(both)

        lr_pct = 100 * len(lr_pos) / len(all_predictions)
        xgb_pct = 100 * len(xgb_pos) / len(all_predictions)

        threshold_results[threshold] = {
            'lr_count': len(lr_pos),
            'lr_pct': lr_pct,
            'xgb_count': len(xgb_pos),
            'xgb_pct': xgb_pct,
            'agreement_count': overlap,
            'jaccard': jaccard,
        }

        # Print
        print(
            f"    {threshold:10.1f} | {len(lr_pos):8d} {lr_pct:5.1f}% | "
            f"{len(xgb_pos):8d} {xgb_pct:5.1f}% | {overlap:10d} | {jaccard:7.1f}%"
        )

    print("\n    " + "-"*96)

    # Find best threshold for agreement
    best_jaccard_threshold = max(thresholds, key=lambda t: threshold_results[t]['jaccard'])
    best_agreement = threshold_results[best_jaccard_threshold]

    print(f"\n    Best Jaccard agreement: threshold={best_jaccard_threshold}")
    print(f"      LR: {best_agreement['lr_count']} positives ({best_agreement['lr_pct']:.1f}%)")
    print(f"      XGBoost: {best_agreement['xgb_count']} positives ({best_agreement['xgb_pct']:.1f}%)")
    print(f"      Agreement: {best_agreement['agreement_count']} ({best_agreement['jaccard']:.1f}% Jaccard)")

    # Save results
    print(f"\n[4] Saving threshold analysis report...")

    report_file = RESULTS_DIR / "threshold_optimization_analysis.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_akhbars': total_akhbars,
            'total_predictions': len(all_predictions),
            'thresholds_tested': thresholds,
            'threshold_results': {
                str(t): {
                    'lr_count': int(r['lr_count']),
                    'lr_pct': float(r['lr_pct']),
                    'xgb_count': int(r['xgb_count']),
                    'xgb_pct': float(r['xgb_pct']),
                    'agreement_count': int(r['agreement_count']),
                    'jaccard_agreement': float(r['jaccard']),
                }
                for t, r in threshold_results.items()
            },
            'best_threshold_for_agreement': float(best_jaccard_threshold),
        }, f, ensure_ascii=False, indent=2)

    print(f"    [OK] Report saved to: {report_file.name}\n")

    print("="*100)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("="*100)

    return threshold_results


if __name__ == '__main__':
    test_thresholds()
