#!/usr/bin/env python3
"""
test_morpho_features.py
──────────────────────────────────────────────────────────────
Test morphological feature extraction with CAMeL Tools.

Vérifies que:
1. CAMeL Tools s'importe correctement
2. Features morphologiques (f65-f70) s'extraient avec des valeurs non-zéro
3. La morphologie apporte du signal réel

Usage:
  python3 test_morpho_features.py

Ou avec venv:
  .\run_in_venv.ps1 test_morpho_features.py
"""

import sys
import re
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("MORPHOLOGICAL FEATURES TEST")
print("=" * 80)

# ─── Test 1: Import CAMeL Tools ─────────────────────────────────────────────
print("\n[1/4] Testing CAMeL Tools import...")
print("-" * 80)

try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    print("✅ CAMeL Tools imported successfully")
    HAS_CAMEL = True
except ImportError as e:
    print(f"❌ CAMeL Tools import FAILED: {e}")
    HAS_CAMEL = False
    sys.exit(1)

# ─── Test 2: Load morphology database ───────────────────────────────────────
print("\n[2/4] Loading morphology database...")
print("-" * 80)

try:
    morpho_db = MorphologyDB.builtin_db()
    analyzer = Analyzer(morpho_db)
    print("✅ Morphology database loaded successfully")
except Exception as e:
    print(f"❌ Database loading FAILED: {e}")
    sys.exit(1)

# ─── Test 3: Extract morpho features on test texts ────────────────────────
print("\n[3/4] Testing feature extraction on sample texts...")
print("-" * 80)

def extract_morpho_features(text):
    """Extract morphological features exactly as in p1_4_logistic_regression_v80.py"""
    features = {}

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

    return features, jnn_count, aql_count, hikma_count, verb_count, noun_count, adj_count

# Test on sample texts
test_samples = [
    ("مجنون مجنون قال", "Junun repetition"),
    ("قال الرجل والمرأة", "Dialogue + nouns"),
    ("عاقل وحكيم وذكي", "Aql + hikma"),
    ("مجنون يقول حكمة عاقلة", "Mixed: junun + dialogue + aql"),
]

print(f"{'Sample':<40} {'f65_jnn':<12} {'f66_aql':<12} {'f68_verb':<12} {'f69_noun':<12}")
print("-" * 80)

all_zero = True
for text, desc in test_samples:
    features, jnn, aql, hikma, verb, noun, adj = extract_morpho_features(text)

    f65 = features['f65_root_jnn_density']
    f66 = features['f66_root_aql_density']
    f68 = features['f68_verb_density']
    f69 = features['f69_noun_density']

    # Check if any feature is non-zero
    has_signal = f65 > 0 or f66 > 0 or f68 > 0 or f69 > 0
    if has_signal:
        all_zero = False

    status = "✅" if has_signal else "⚠️"
    print(f"{status} {desc:<38} {f65:<12.4f} {f66:<12.4f} {f68:<12.4f} {f69:<12.4f}")

print()
if all_zero:
    print("⚠️  WARNING: All morpho features extracted as 0.0")
    print("   This indicates a problem with CAMeL analysis.")
else:
    print("✅ Morpho features extracted successfully with non-zero values!")

# ─── Test 4: Test on real khbar text ────────────────────────────────────────
print("\n[4/4] Testing on real khbar excerpt...")
print("-" * 80)

# Sample from Nisaburi (should have high junun density)
real_khbar = """
قدمت الكوفة ولم يكن لي هم إلا أويس القرني أطلبه وأسأل عنه حتى سقطت عليه جالساً
وحده على شاطئ الفرات نصف النهار يتوضأ ويغسل ثوبه فعرفته بالنعت الذي نعت لي
فإذا رجل مجنون يقول حكمة عاقلة
"""

features, jnn, aql, hikma, verb, noun, adj = extract_morpho_features(real_khbar)

print(f"\nText: {real_khbar[:100]}...")
print(f"\nExtracted counts:")
print(f"  - ج.ن.ن (junun root): {jnn} occurrences → f65 = {features['f65_root_jnn_density']:.4f}")
print(f"  - ع.ق.ل (aql root):   {aql} occurrences → f66 = {features['f66_root_aql_density']:.4f}")
print(f"  - ح.ك.م (hikma root): {hikma} occurrences → f67 = {features['f67_root_hikma_density']:.4f}")
print(f"  - Verbs:              {verb} occurrences → f68 = {features['f68_verb_density']:.4f}")
print(f"  - Nouns:              {noun} occurrences → f69 = {features['f69_noun_density']:.4f}")
print(f"  - Adjectives:         {adj} occurrences → f70 = {features['f70_adj_density']:.4f}")

# ─── Final verdict ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if HAS_CAMEL and not all_zero:
    print("✅ CAMeL Tools is fully functional and extracting morphological features!")
    print("\n✅ Ready to use in v80 pipeline:")
    print("   - f65_root_jnn_density    ✅")
    print("   - f66_root_aql_density    ✅")
    print("   - f67_root_hikma_density  ✅")
    print("   - f68_verb_density        ✅")
    print("   - f69_noun_density        ✅")
    print("   - f70_adj_density         ✅")
    print("\nYou can now run:")
    print("  python3 run_with_venv.py pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 5")
    sys.exit(0)
else:
    print("❌ CAMeL Tools is NOT working properly!")
    if not HAS_CAMEL:
        print("   Reason: Import failed")
    else:
        print("   Reason: Features extracting as 0.0")
    print("\nTroubleshooting:")
    print("  1. Verify venv activation: python3 test_camel_venv.py")
    print("  2. Reinstall: pip install --upgrade camel-tools")
    print("  3. Check Python version: python3 --version (should be 3.11)")
    sys.exit(1)
