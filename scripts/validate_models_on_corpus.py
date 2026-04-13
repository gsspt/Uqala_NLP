#!/usr/bin/env python3
"""
validate_models_on_corpus.py
──────────────────────────────────────────────────────────────
Generic validation of LR v80, XGBoost v80, and Ensemble v80 on ANY OpenITI corpus.

Usage:
  # Ibn Abd Rabbih (default)
  python scripts/validate_models_on_corpus.py

  # Ibn Mu'tazz Tabaqat al-Shu'ara'
  python scripts/validate_models_on_corpus.py \
    --corpus-path openiti_corpus/data/0296IbnMuctazz/0296IbnMuctazz.TabaqatShucara

  # Any other corpus
  python scripts/validate_models_on_corpus.py --corpus-path path/to/corpus
"""

import json
import sys
import pathlib
import pickle
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# Setup paths
BASE = pathlib.Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE / "src"))
from uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file
from uqala_nlp.preprocessing.isnad_filter import split_isnad
from uqala_nlp.preprocessing.smart_camel_loader import HAS_CAMEL, extract_morpho_features_safe as extract_morpho_safe

# ════════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (27 features, identical to both models)
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
    """Extract all 27 v80 features"""
    features = {}
    features.update(extract_junun_features_15(text))
    features.update(extract_morphological_features_6(text))
    features.update(extract_empirical_features_6(text))
    return features


# ════════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

def validate_on_corpus(corpus_path):
    """Validate LR, XGBoost, and Ensemble on given corpus"""

    MODELS_DIR = BASE / "models"
    corpus_name = corpus_path.parent.name if isinstance(corpus_path, pathlib.Path) else pathlib.Path(corpus_path).parent.name
    RESULTS_DIR = BASE / "results" / corpus_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print(f"VALIDATION: LR v80 | XGBoost v80 | Ensemble v80")
    print(f"Corpus: {corpus_name}")
    print("="*100)

    # Load models
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

    # Find corpus files
    print(f"\n[2] Finding corpus files...")
    corpus_root = BASE / corpus_path if not isinstance(corpus_path, pathlib.Path) else corpus_path
    text_files = sorted(corpus_root.rglob("*-ara1"))

    if not text_files:
        print(f"ERROR: No -ara1 files found in {corpus_root}")
        return None

    print(f"  [OK] Found {len(text_files)} files\n")

    # Extract and validate
    print("[3] Extracting akhbars and making predictions...\n")

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

                        # Get predictions
                        X_lr = lr_scaler.transform(fv) if lr_scaler else fv
                        X_xgb = xgb_scaler.transform(fv) if xgb_scaler else fv

                        p_lr = float(lr_clf.predict_proba(X_lr)[0, 1])
                        p_xgb = float(xgb_clf.predict_proba(X_xgb)[0, 1])

                        # Ensemble fusion
                        p_ensemble = (p_lr + p_xgb) / 2

                        all_predictions.append({
                            'filename': filepath.name,
                            'khabar_num': khabar_num,
                            'p_lr': p_lr,
                            'p_xgb': p_xgb,
                            'p_ensemble': p_ensemble,
                            'text': akhbar,
                        })
                    except Exception as e:
                        pass

                    pbar_akhbars.update(1)

            pbar_files.update(1)

    print(f"\n[OK] Processed {total_akhbars} akhbars, {len(all_predictions)} predictions\n")

    if not all_predictions:
        print("ERROR: No predictions generated")
        return None

    # Analysis
    print("="*100)
    print("RESULTS")
    print("="*100 + "\n")

    lr_pos = sum(1 for p in all_predictions if p['p_lr'] >= 0.5)
    xgb_pos = sum(1 for p in all_predictions if p['p_xgb'] >= 0.5)
    ens_pos = sum(1 for p in all_predictions if p['p_ensemble'] >= 0.5)

    print(f"Total akhbars: {total_akhbars}")
    print(f"Total predictions: {len(all_predictions)}\n")

    print(f"LR v80:     {lr_pos:5d} positives ({100*lr_pos/len(all_predictions):.1f}%)")
    print(f"XGBoost:    {xgb_pos:5d} positives ({100*xgb_pos/len(all_predictions):.1f}%)")
    print(f"Ensemble:   {ens_pos:5d} positives ({100*ens_pos/len(all_predictions):.1f}%)")

    # Collect positives
    lr_positives = sorted([p for p in all_predictions if p['p_lr'] >= 0.5], key=lambda x: x['p_lr'], reverse=True)
    xgb_positives = sorted([p for p in all_predictions if p['p_xgb'] >= 0.5], key=lambda x: x['p_xgb'], reverse=True)
    ens_positives = sorted([p for p in all_predictions if p['p_ensemble'] >= 0.5], key=lambda x: x['p_ensemble'], reverse=True)

    # Save results
    print(f"\n[4] Saving results...")

    for model_name, positives, pkey in [
        ('lr', lr_positives, 'p_lr'),
        ('xgboost', xgb_positives, 'p_xgb'),
        ('ensemble', ens_positives, 'p_ensemble')
    ]:
        output_file = RESULTS_DIR / f"validation_{model_name}_{corpus_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'corpus': corpus_name,
                'extraction_method': 'akhbar_extraction_v2_smart',
                'total_akhbars': total_akhbars,
                'total_predictions': len(all_predictions),
                'positives_count': len(positives),
                'positives_rate': len(positives) / len(all_predictions),
                'positives_detailed': [
                    {
                        'filename': p['filename'],
                        'khabar_num': p['khabar_num'],
                        'probability': p[pkey],
                        'text': p['text'],
                    }
                    for p in positives
                ]
            }, f, ensure_ascii=False, indent=2)
        print(f"  [OK] {output_file.name} ({len(positives)} positives)")

    # Comparison report
    comparison_file = RESULTS_DIR / f"comparison_{corpus_name}.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump({
            'corpus': corpus_name,
            'total_akhbars': total_akhbars,
            'results': {
                'lr': {'positives': lr_pos, 'rate': lr_pos / len(all_predictions)},
                'xgboost': {'positives': xgb_pos, 'rate': xgb_pos / len(all_predictions)},
                'ensemble': {'positives': ens_pos, 'rate': ens_pos / len(all_predictions)},
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"  [OK] {comparison_file.name}")

    print("\n" + "="*100)
    print("VALIDATION COMPLETE")
    print("="*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate models on OpenITI corpus')
    parser.add_argument('--corpus-path', type=str, default='openiti_corpus/data/0328IbnCabdRabbih',
                        help='Path to corpus root directory')
    args = parser.parse_args()

    validate_on_corpus(args.corpus_path)
