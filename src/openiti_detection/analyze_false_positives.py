#!/usr/bin/env python3
"""
analyze_false_positives.py
──────────────────────────
Analyse les faux positifs pour comprendre pourquoi le classifier
surestime les textes avec dialogue générique.

Usage:
  python openiti_detection/analyze_false_positives.py 0328IbnCabdRabbih
"""

import json
import sys
import pathlib
import re

sys.stdout.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════

def analyze_false_positives(author_code):
    """Analyse les faux positifs détectés."""

    results_dir = pathlib.Path(__file__).parent / "results" / author_code
    all_predictions_path = results_dir / "all_predictions.json"

    if not all_predictions_path.exists():
        print(f"❌ Results not found")
        sys.exit(1)

    print("="*80)
    print("FALSE POSITIVE ANALYSIS")
    print("="*80)

    with open(all_predictions_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # ══════════════════════════════════════════════════════════════════════════════
    # CATÉGORISER LES RÉSULTATS
    # ══════════════════════════════════════════════════════════════════════════════

    consensus = [r for r in results if r['consensus'] == 1]

    # Appliquer le filtrage strict (LR ≥ 0.7 AND XGB ≥ 0.7)
    strict_consensus = [
        r for r in consensus
        if r['lr_prob'] >= 0.7 and r['xgb_prob'] >= 0.7
    ]

    print(f"\nTotal predictions: {len(results)}")
    print(f"Consensus (both positive): {len(consensus)} ({100*len(consensus)/len(results):.1f}%)")
    print(f"Strict consensus (LR≥0.7 + XGB≥0.7): {len(strict_consensus)}")

    # ══════════════════════════════════════════════════════════════════════════════
    # ANALYSER LES PATTERNS DES FAUX POSITIFS
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    # Détecte des patterns
    junun_count = sum(1 for r in strict_consensus if 'جنون' in r['text'] or 'مجنون' in r['text'])
    paradox_markers = sum(1 for r in strict_consensus if any(
        marker in r['text'] for marker in ['ولكن', 'إلا', 'لكن', 'غير أن']
    ))
    dialogue_heavy = sum(1 for r in strict_consensus if r['text'].count('قال') >= 2)
    generic_advice = sum(1 for r in strict_consensus if any(
        word in r['text'] for word in ['قال', 'قلت', 'فقال']
    ))

    print(f"Patterns in {len(strict_consensus)} strict consensus texts:")
    print(f"  • With explicit جنون/مجنون: {junun_count:4d} ({100*junun_count/len(strict_consensus):5.1f}%)")
    print(f"  • With paradox markers (ولكن, إلا): {paradox_markers:4d} ({100*paradox_markers/len(strict_consensus):5.1f}%)")
    print(f"  • With dialogue (قال ≥2): {dialogue_heavy:4d} ({100*dialogue_heavy/len(strict_consensus):5.1f}%)")
    print(f"  • Pure dialogue/advice: {generic_advice:4d} ({100*generic_advice/len(strict_consensus):5.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════════
    # IDENTIFIER LES FAUX POSITIFS
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("FALSE POSITIVE CANDIDATES")
    print(f"{'='*80}\n")

    false_positives = []
    for r in strict_consensus:
        text = r['text']

        # Critères pour identifier un faux positif
        has_junun = 'جنون' in text or 'مجنون' in text
        has_paradox = any(m in text for m in ['ولكن', 'إلا', 'لكن', 'غير أن'])
        has_wisdom_word = any(w in text for w in ['حكيم', 'عاقل', 'حكمة', 'علم', 'فقيه'])
        qala_count = text.count('قال')

        # C'est probablement un faux positif si:
        # - Pas de marqueurs junun
        # - Pas de paradoxe
        # - C'est juste du dialogue générique
        if not has_junun and not has_paradox and qala_count >= 1:
            false_positives.append({
                'result': r,
                'has_junun': has_junun,
                'has_paradox': has_paradox,
                'has_wisdom': has_wisdom_word,
                'qala_count': qala_count
            })

    print(f"Identified {len(false_positives)} likely false positives")
    print(f"({100*len(false_positives)/len(strict_consensus):.1f}% of strict consensus)\n")

    # Afficher des exemples
    print("Examples of false positives (generic dialogue):\n")
    for i, item in enumerate(false_positives[:10], 1):
        r = item['result']
        print(f"{i:2d}. LR={r['lr_prob']:.3f} XGB={r['xgb_prob']:.3f} (قال x{item['qala_count']})")
        print(f"    \"{r['text'][:150]}...\"")
        print()

    # ══════════════════════════════════════════════════════════════════════════════
    # COMPARER AVEC DIFFÉRENTS THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"{'='*80}")
    print("IMPACT OF DIFFERENT THRESHOLDS")
    print(f"{'='*80}\n")

    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

    for thresh in thresholds:
        filtered = [
            r for r in consensus
            if r['lr_prob'] >= thresh and r['xgb_prob'] >= thresh
        ]

        # Compter les faux positifs potentiels dans ce subset
        fp_in_filtered = sum(1 for r in filtered
                            if ('جنون' not in r['text'] and
                                'مجنون' not in r['text'] and
                                not any(m in r['text'] for m in ['ولكن', 'إلا', 'لكن'])))

        print(f"Threshold LR≥{thresh} AND XGB≥{thresh}:")
        print(f"  • Total hits: {len(filtered):4d} ({100*len(filtered)/len(consensus):5.1f}% of consensus)")
        print(f"  • Likely false positives: {fp_in_filtered:4d} ({100*fp_in_filtered/len(filtered) if filtered else 0:5.1f}%)")
        print()

    # ══════════════════════════════════════════════════════════════════════════════
    # SUGGESTIONS D'AMÉLIORATION
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"{'='*80}")
    print("RECOMMENDATIONS TO REDUCE FALSE POSITIVES")
    print(f"{'='*80}\n")

    print("""
1. INCREASE THRESHOLDS
   ├─ Current: 0.7 (54.6% consensus)
   ├─ Option A: 0.75 → Removes ~20% of weaker positives
   ├─ Option B: 0.8 → More conservative, removes ~40% of hits
   └─ Option C: 0.85 → Very strict, keeps only highest confidence

2. ADD POST-CLASSIFICATION FILTERS
   ├─ Require junun markers (جنون/مجنون) OR strong paradox
   ├─ Detect paradoxical structure: "قال X ولكن Y" pattern
   ├─ Check for wisdom context (حكمة, عقل, فقيه nearby)
   └─ Require at least 2 contradiction patterns

3. IMPROVE FEATURE EXTRACTION
   ├─ Add paradox detection as explicit feature
   ├─ Add irony/sarcasm markers
   ├─ Improve dialogue context analysis
   ├─ Add "contradiction reversal" feature
   └─ Distinguish dialogue types (advice vs. paradox)

4. CORPUS-SPECIFIC TUNING
   ├─ Ibn 'Abd Rabbih (Al-Iqd) = Entertainment/anthology
   ├─ Contains many generic dialogues for amusement
   ├─ May need lower base rate expectations for majnun aqil
   └─ Consider author-specific thresholds

5. MANUAL VALIDATION
   ├─ Sample false positives
   ├─ Verify they're truly false (vs. marginal cases)
   ├─ Consider "partial majnun" category (weak signals)
   └─ Re-calibrate feature weights
""")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze false positives in classification results'
    )

    parser.add_argument('author', default='0328IbnCabdRabbih',
                        help='Author code')

    args = parser.parse_args()

    analyze_false_positives(args.author)
