#!/usr/bin/env python3
"""
strict_analysis.py
──────────────────
Analyse stricte des résultats avec détection des fous canoniques
et critères de majnun aqil.

Usage:
  python openiti_detection/strict_analysis.py 0328IbnCabdRabbih
"""

import json
import sys
import pathlib
import argparse
import re
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL FOOL NAMES (fous sages connus)
# ══════════════════════════════════════════════════════════════════════════════

CANONICAL_FOOLS = {
    'bahlul': ['بهلول', 'باهلول'],
    'sa\'doun': ['سعدون', 'ساعدون'],
    'samnun': ['سمنون', 'سمنن'],
    'khalaf': ['خلاف', 'خالف'],
    'ja\'ifran': ['جعيفران', 'جعفران'],
    'riyah': ['رياح', 'ريَّاح'],
    'ligit': ['لقيط'],
    'haywun': ['حيون', 'حي ون'],
    'alyan': ['عليان'],
    'sahun': ['سحون'],
}

# Regex pattern for canonical fool names
canonical_pattern = '|'.join(['|'.join(names) for names in CANONICAL_FOOLS.values()])

# ══════════════════════════════════════════════════════════════════════════════
# MAJNUN INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

JUNUN_WORDS = [
    'جنون', 'مجنون', 'جنّ', 'مجنن', 'أجن', 'جننت',
    'معتوه', 'هائم', 'ممسوس', 'سعيد', 'مولع'
]

AKIN_CONCEPTS = [
    'حكمة', 'عقل', 'فهم', 'درايه', 'علم',  # wisdom/intelligence
    'سخرية', 'هزء', 'لعب', 'تهكم',  # mockery
]

# ══════════════════════════════════════════════════════════════════════════════
# PARADOX/CONTRADICTION INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

PARADOX_PATTERNS = [
    r'قال.*(?:ولكن|إلا|غير|لكن)',  # contradictory statements
    r'(?:الجهل|السفه|الحماقة).*(?:يدل|يشير|يدّل).*(?:الحكمة|العقل|الفهم)',  # foolishness shows wisdom
    r'(?:قال|فقال).*(?:ضحك|ضحكوا|يضحك)',  # statement followed by laughter
    r'هزء.*صدق',  # mockery revealing truth
    r'جاهل.*حكيم|حكيم.*جاهل',  # contrasting fool and sage
]

# ══════════════════════════════════════════════════════════════════════════════

def detect_canonical_fool(text):
    """Détecte les noms de fous canoniques dans le texte."""
    for fool_name, variants in CANONICAL_FOOLS.items():
        for variant in variants:
            if variant in text:
                return fool_name
    return None

def count_junun_markers(text):
    """Compte les marqueurs de جنون."""
    count = 0
    for word in JUNUN_WORDS:
        count += text.count(word)
    return count

def has_paradox(text):
    """Détecte si le texte contient une contradiction/paradoxe."""
    for pattern in PARADOX_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def is_dialogue(text):
    """Vérifie si le texte contient du dialogue (قال)."""
    return 'قال' in text

def analyze_author(author_code, threshold_lr=0.7, threshold_xgb=0.7):
    """Analyse détaillée avec critères stricts."""

    results_dir = pathlib.Path(__file__).parent / "results" / author_code
    results_path = results_dir / "all_predictions.json"

    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        sys.exit(1)

    print(f"Loading {results_path.name}…\n")
    with open(results_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # ══════════════════════════════════════════════════════════════════════════════
    # CATÉGORISER LES RÉSULTATS
    # ══════════════════════════════════════════════════════════════════════════════

    consensus = [r for r in all_results if r['consensus'] == 1]
    lr_only = [r for r in all_results if r['lr_pred'] == 1 and r['xgb_pred'] == 0]
    xgb_only = [r for r in all_results if r['xgb_pred'] == 1 and r['lr_pred'] == 0]

    print("="*80)
    print("SAMPLE ANALYSIS")
    print("="*80)
    print(f"Analyzing: {author_code}")
    print(f"Total predictions: {len(all_results)}")
    print(f"Consensus (both positive): {len(consensus)} ({100*len(consensus)/len(all_results):.1f}%)")
    print(f"LR only: {len(lr_only)}")
    print(f"XGB only: {len(xgb_only)}")

    # ══════════════════════════════════════════════════════════════════════════════
    # ANALYSE STRICTE (SEUIL ≥ 0.7)
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"STRICT ANALYSIS (LR ≥ {threshold_lr} AND XGB ≥ {threshold_xgb})")
    print(f"{'='*80}\n")

    strict_consensus = [
        r for r in consensus
        if r['lr_prob'] >= threshold_lr and r['xgb_prob'] >= threshold_xgb
    ]

    print(f"Strict consensus hits: {len(strict_consensus)}")

    # Analyser les hits stricts
    true_majnun = []
    has_canonical = []
    has_paradox_count = 0
    dialogue_only = []

    for r in strict_consensus:
        text = r['text']

        canonical = detect_canonical_fool(text)
        junun_count = count_junun_markers(text)
        has_par = has_paradox(text)
        is_dial = is_dialogue(text)

        if canonical:
            has_canonical.append({
                'result': r,
                'fool': canonical,
                'junun_markers': junun_count
            })
        elif junun_count > 0 and (has_par or junun_count >= 2):
            true_majnun.append({
                'result': r,
                'junun_markers': junun_count,
                'has_paradox': has_par
            })
        elif junun_count > 0:
            true_majnun.append({
                'result': r,
                'junun_markers': junun_count,
                'has_paradox': has_par
            })
        elif has_par and is_dial:
            true_majnun.append({
                'result': r,
                'junun_markers': junun_count,
                'has_paradox': has_par
            })
        elif is_dial:
            dialogue_only.append(r)

    print(f"\n{len(has_canonical)} with CANONICAL FOOL names:")
    for item in has_canonical:
        r = item['result']
        print(f"  • {item['fool'].upper()}: LR={r['lr_prob']:.3f} XGB={r['xgb_prob']:.3f}")
        print(f"    {r['text'][:100]}…\n")

    print(f"\n{len(true_majnun)} with JUNUN MARKERS or PARADOX:")
    for i, item in enumerate(true_majnun[:10], 1):
        r = item['result']
        markers = f" ({item['junun_markers']} junun markers)" if item['junun_markers'] > 0 else ""
        paradox = " (has paradox)" if item['has_paradox'] else ""
        print(f"  {i}. LR={r['lr_prob']:.3f} XGB={r['xgb_prob']:.3f}{markers}{paradox}")
        print(f"     {r['text'][:100]}…\n")

    print(f"\n{len(dialogue_only)} with DIALOGUE ONLY (no junun/paradox):")
    for r in dialogue_only[:5]:
        print(f"  • LR={r['lr_prob']:.3f} XGB={r['xgb_prob']:.3f}")
        print(f"    {r['text'][:100]}…\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # RÉSUMÉ JSON
    # ══════════════════════════════════════════════════════════════════════════════

    summary = {
        'total_analyzed': len(all_results),
        'true_majnun_count': len(true_majnun),
        'true_majnun_percentage': 100 * len(true_majnun) / len(all_results),
        'false_positives_count': len(dialogue_only),
        'canonical_fools_count': len(has_canonical),
        'canonical_fools_percentage': 100 * len(has_canonical) / len(all_results),
        'canonical_fools_breakdown': Counter([item['fool'] for item in has_canonical]),
        'true_majnun_examples': [item['result'] for item in true_majnun[:10]],
        'canonical_fool_examples': [item['result'] for item in has_canonical],
    }

    summary_path = results_dir / "strict_analysis.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Convert Counter to dict for JSON serialization
        summary['canonical_fools_breakdown'] = dict(summary['canonical_fools_breakdown'])
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Strict analysis saved → {summary_path}")
    print(f"{'='*80}\n")

    print(f"Summary:")
    print(f"  True majnun aqil: {len(true_majnun)} ({100*len(true_majnun)/len(all_results):.2f}%)")
    print(f"  False positives: {len(dialogue_only)} ({100*len(dialogue_only)/len(all_results):.2f}%)")
    print(f"  Canonical fools: {len(has_canonical)} ({100*len(has_canonical)/len(all_results):.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strict analysis of detection results')
    parser.add_argument('author', default='0328IbnCabdRabbih',
                        help='Author code')
    parser.add_argument('--threshold-lr', type=float, default=0.7,
                        help='LR probability threshold (default: 0.7)')
    parser.add_argument('--threshold-xgb', type=float, default=0.7,
                        help='XGB probability threshold (default: 0.7)')
    args = parser.parse_args()

    analyze_author(args.author, args.threshold_lr, args.threshold_xgb)
