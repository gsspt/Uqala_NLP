#!/usr/bin/env python3
"""
p2_3_hybrid_ensemble_shap.py
─────────────────────────────────────────────────────────────────
Hybrid Ensemble Pipeline: Combine LR v80 + XGBoost v80 with SHAP explanations.

Architecture:
  1. Load pre-trained LR v80 model (p1_4_logistic_regression_v80.py output)
  2. Load pre-trained XGBoost v80 model (p2_2_xgboost_v80_shap.py output)
  3. Extract 27 features from input text
  4. Get predictions from both models
  5. Fusion: p_ensemble = (p_lr + p_xgb) / 2
  6. Confidence: agreement = 1 - |p_lr - p_xgb|
  7. SHAP explanations:
     - LR: Direct coefficient-based explanation
     - XGBoost: TreeExplainer SHAP values
     - Consensus: Features both models agree on
  8. Output: prediction, confidence, agreement, top features, SHAP plots

Features: 27 v80 features (identical to LR and XGBoost training)

Usage:
  python p2_3_hybrid_ensemble_shap.py
  python p2_3_hybrid_ensemble_shap.py --test-corpus results/test_texts.json
"""

import json
import sys
import pathlib
import argparse
import pickle
import re
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripts
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

BASE = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

try:
    from uqala_nlp.preprocessing.smart_camel_loader import HAS_CAMEL, analyzer, extract_morpho_features_safe as extract_morpho_safe
except ImportError:
    HAS_CAMEL = False
    analyzer = None
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        morpho_db = MorphologyDB.builtin_db()
        analyzer = Analyzer(morpho_db)
        HAS_CAMEL = True
    except ImportError:
        pass

MODELS_DIR = BASE / "models"
LR_MODEL_PATH = MODELS_DIR / "lr_classifier_v80.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_classifier_v80.pkl"
RESULTS_DIR = BASE / "results" / "ensemble_v80"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (identical to both LR and XGBoost)
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
    features = {}

    if not HAS_CAMEL:
        for i in range(6):
            features[f'f{65+i:02d}_morpho'] = 0.0
        return features

    tokens = text.split()
    n_tokens = len(tokens) if tokens else 1

    jnn_count = aql_count = hikma_count = 0
    verb_count = noun_count = adj_count = 0

    for token in tokens:
        try:
            analyses = analyzer.analyze(token)
            if analyses:
                a = analyses[0]
                root = a.get('root', '')
                pos = a.get('pos', '')

                if root == 'ج.ن.ن':
                    jnn_count += 1
                if root == 'ع.ق.ل':
                    aql_count += 1
                if root == 'ح.ك.م':
                    hikma_count += 1

                if pos == 'verb':
                    verb_count += 1
                if pos == 'noun':
                    noun_count += 1
                if pos == 'adj':
                    adj_count += 1
        except:
            pass

    features['f65_root_jnn_density'] = jnn_count / n_tokens
    features['f66_root_aql_density'] = aql_count / n_tokens
    features['f67_root_hikma_density'] = hikma_count / n_tokens
    features['f68_verb_density'] = verb_count / n_tokens
    features['f69_noun_density'] = noun_count / n_tokens
    features['f70_adj_density'] = adj_count / n_tokens

    return features

def extract_all_features_27(text):
    junun = extract_junun_features_15(text)
    morpho = extract_morphological_features_6(text)
    empirical = extract_empirical_features_6(text)
    return {**junun, **morpho, **empirical}

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTION & EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    """Load both LR and XGBoost models"""
    if not LR_MODEL_PATH.exists():
        print(f"ERROR: LR model not found at {LR_MODEL_PATH}")
        return None, None, None, None

    if not XGB_MODEL_PATH.exists():
        print(f"ERROR: XGBoost model not found at {XGB_MODEL_PATH}")
        return None, None, None, None

    with open(LR_MODEL_PATH, 'rb') as f:
        lr_data = pickle.load(f)
        lr_model = lr_data['clf'] if isinstance(lr_data, dict) else lr_data
        lr_scaler = lr_data.get('scaler', None) if isinstance(lr_data, dict) else None

    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_data = pickle.load(f)
        xgb_model = xgb_data['clf'] if isinstance(xgb_data, dict) else xgb_data
        xgb_scaler = xgb_data.get('scaler', None) if isinstance(xgb_data, dict) else None
        feature_names = xgb_data.get('feature_names', None) if isinstance(xgb_data, dict) else None

    print(f"[OK] Loaded LR model from {LR_MODEL_PATH.name}")
    print(f"[OK] Loaded XGBoost model from {XGB_MODEL_PATH.name}")

    return lr_model, lr_scaler, xgb_model, xgb_scaler

def predict_ensemble(text, lr_model, lr_scaler, xgb_model, xgb_scaler):
    """
    Make ensemble prediction on a single text.

    Returns dict with:
      - prediction: p_ensemble (0-1)
      - confidence: agreement between models (0-1)
      - lr_proba: LR probability
      - xgb_proba: XGBoost probability
      - features: feature vector (for SHAP)
      - feature_dict: feature dict (for analysis)
    """
    features_dict = extract_all_features_27(text)
    features_array = np.array(list(features_dict.values())).reshape(1, -1)

    # Clean NaNs
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

    # LR prediction
    if lr_scaler:
        X_lr = lr_scaler.transform(features_array)
    else:
        X_lr = features_array
    p_lr = lr_model.predict_proba(X_lr)[0, 1]

    # XGBoost prediction
    if xgb_scaler:
        X_xgb = xgb_scaler.transform(features_array)
    else:
        X_xgb = features_array
    p_xgb = xgb_model.predict_proba(X_xgb)[0, 1]

    # Ensemble fusion
    p_ensemble = (p_lr + p_xgb) / 2
    agreement = 1 - abs(p_lr - p_xgb)

    return {
        'prediction': p_ensemble,
        'confidence': agreement,
        'lr_proba': p_lr,
        'xgb_proba': p_xgb,
        'features': features_array,
        'feature_dict': features_dict,
    }

def explain_with_shap(features, lr_model, lr_scaler, xgb_model, xgb_scaler, feature_names):
    """
    Generate SHAP explanations for both models.

    Returns dict with SHAP values and feature importance.
    """
    # LR explanation: Direct coefficient extraction
    if lr_scaler:
        X_scaled = lr_scaler.transform(features)
    else:
        X_scaled = features

    lr_coefs = lr_model.coef_[0]
    lr_shap_values = X_scaled[0] * lr_coefs
    lr_importance = list(zip(feature_names, np.abs(lr_shap_values)))
    lr_importance.sort(key=lambda x: x[1], reverse=True)

    # XGBoost explanation: TreeExplainer
    if xgb_scaler:
        X_scaled = xgb_scaler.transform(features)
    else:
        X_scaled = features

    explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = explainer.shap_values(X_scaled)

    if isinstance(xgb_shap_values, list):
        xgb_shap_class1 = xgb_shap_values[1][0]
    else:
        xgb_shap_class1 = xgb_shap_values[0]

    xgb_importance = list(zip(feature_names, np.abs(xgb_shap_class1)))
    xgb_importance.sort(key=lambda x: x[1], reverse=True)

    # Find consensus (top 3 features from both)
    lr_top3 = set([name for name, _ in lr_importance[:3]])
    xgb_top3 = set([name for name, _ in xgb_importance[:3]])
    consensus = lr_top3 & xgb_top3

    return {
        'lr_importance': lr_importance,
        'xgb_importance': xgb_importance,
        'consensus': list(consensus),
        'lr_shap_values': lr_shap_values,
        'xgb_shap_values': xgb_shap_class1,
    }

def confidence_level(agreement_score):
    """Convert agreement score (0-1) to confidence label"""
    if agreement_score >= 0.90:
        return "VERY HIGH"
    elif agreement_score >= 0.70:
        return "HIGH"
    elif agreement_score >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default=None, help='Single text to analyze')
    parser.add_argument('--test-corpus', type=str, default=None, help='Path to test corpus JSON')
    args = parser.parse_args()

    print("="*80)
    print("Ensemble v80: Hybrid LR + XGBoost Pipeline with SHAP")
    print("="*80)

    # Load models
    print(f"\n[1] Loading models...")
    lr_model, lr_scaler, xgb_model, xgb_scaler = load_models()
    if lr_model is None or xgb_model is None:
        return None

    # Get feature names from XGBoost model data
    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_data = pickle.load(f)
        feature_names = xgb_data.get('feature_names', None)

    if feature_names is None:
        # Fallback: reconstruct from a sample text
        sample_features = extract_all_features_27("نص عينة")
        feature_names = list(sample_features.keys())

    print(f"    Feature names ({len(feature_names)} features): OK")

    # Test on single text if provided
    if args.text:
        print(f"\n[2] Analyzing single text...")
        print(f"    Text: {args.text[:100]}...")

        result = predict_ensemble(args.text, lr_model, lr_scaler, xgb_model, xgb_scaler)
        shap_result = explain_with_shap(
            result['features'], lr_model, lr_scaler, xgb_model, xgb_scaler, feature_names
        )

        print(f"\n[RESULT]")
        print(f"  Prediction: {result['prediction']:.3f} (ensemble)")
        print(f"  LR proba: {result['lr_proba']:.3f}")
        print(f"  XGBoost proba: {result['xgb_proba']:.3f}")
        print(f"  Agreement: {result['confidence']:.3f}")
        print(f"  Confidence: {confidence_level(result['confidence'])}")
        print(f"\n  Consensus features: {', '.join(shap_result['consensus']) if shap_result['consensus'] else 'None'}")
        print(f"\n  Top LR features:")
        for name, importance in shap_result['lr_importance'][:5]:
            print(f"    - {name}: {importance:.4f}")
        print(f"\n  Top XGBoost features:")
        for name, importance in shap_result['xgb_importance'][:5]:
            print(f"    - {name}: {importance:.4f}")

    # Test on corpus if provided
    elif args.test_corpus:
        print(f"\n[2] Analyzing test corpus...")
        with open(args.test_corpus, encoding='utf-8') as f:
            test_data = json.load(f)

        results = []
        for i, item in enumerate(test_data):
            text = item.get('text', item) if isinstance(item, dict) else item
            result = predict_ensemble(text, lr_model, lr_scaler, xgb_model, xgb_scaler)
            results.append({
                'text': text[:100],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'lr_proba': result['lr_proba'],
                'xgb_proba': result['xgb_proba'],
            })

        # Save batch results
        output_file = RESULTS_DIR / f"ensemble_predictions_{len(test_data)}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"    [OK] Results saved to {output_file.name}")

        # Summary
        positives = sum(1 for r in results if r['prediction'] >= 0.5)
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\n  Positives: {positives}/{len(results)} ({100*positives/len(results):.1f}%)")
        print(f"  Avg confidence: {avg_confidence:.3f} ({confidence_level(avg_confidence)})")

    else:
        print(f"\nUsage:")
        print(f"  python p2_3_hybrid_ensemble_shap.py --text 'النص العربي هنا'")
        print(f"  python p2_3_hybrid_ensemble_shap.py --test-corpus path/to/corpus.json")

if __name__ == '__main__':
    main()
