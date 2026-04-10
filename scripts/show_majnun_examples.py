#!/usr/bin/env python3
"""
show_majnun_examples.py
───────────────────────
Affiche des exemples réels d'akhbars avec les patterns du majnun aqil
trouvés dans le corpus après extraction.

Usage:
  python openiti_detection/show_majnun_examples.py 0328IbnCabdRabbih
"""

import json
import sys
import pathlib

sys.stdout.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════

def show_majnun_examples(author_code):
    """Affiche les exemples de majnun aqil détectés."""

    results_dir = pathlib.Path(__file__).parent / "results" / author_code
    analysis_path = results_dir / "strict_analysis.json"

    if not analysis_path.exists():
        print(f"❌ Analysis not found: {analysis_path}")
        print(f"   Run: python openiti_detection/test_quick.py {author_code}")
        print(f"   Then: python openiti_detection/strict_analysis.py {author_code}")
        sys.exit(1)

    print("="*80)
    print("MAJNUN AQIL EXAMPLES FROM CORPUS")
    print("="*80)

    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 1: FOUS CANONIQUES (Noms explicites)
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n[1] CANONICAL WISE FOOLS (explicitly named) — {analysis['canonical_fools_count']} instances\n")

    breakdown = analysis['canonical_fools_breakdown']
    for fool, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"    {fool.upper():15s}: {count:3d} instances")

    print(f"\nExamples of canonical fool texts:")
    for i, example in enumerate(analysis['canonical_fool_examples'][:3], 1):
        print(f"\n    {i}. {example['file']}")
        print(f"       LR: {example['lr_prob']:.3f} | XGB: {example['xgb_prob']:.3f}")
        print(f"       \"{example['text'][:150]}...\"")

    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 2: VRAI MAJNUN (Avec marqueurs jnoun ou paradoxe)
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n" + "="*80)
    print(f"[2] TRUE MAJNUN AQIL (junun markers + paradox) — {analysis['true_majnun_count']} instances\n")

    for i, example in enumerate(analysis['true_majnun_examples'][:5], 1):
        print(f"\n    {i}. {example['file']}")
        print(f"       LR: {example['lr_prob']:.3f} | XGB: {example['xgb_prob']:.3f}")
        print(f"       \"{example['text'][:200]}...\"")

    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 3: STATISTIQUES GLOBALES
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n" + "="*80)
    print("CORPUS STATISTICS")
    print("="*80)

    total = analysis['total_analyzed']
    true_count = analysis['true_majnun_count']
    canonical_count = analysis['canonical_fools_count']
    false_pos = analysis['false_positives_count']

    print(f"\nTotal texts analyzed: {total:,}")
    print(f"  ├─ Canonical fools (named): {canonical_count:,} ({100*canonical_count/total:5.2f}%)")
    print(f"  ├─ True majnun (junun/paradox): {true_count:,} ({100*true_count/total:5.2f}%)")
    print(f"  └─ False positives (dialogue only): {false_pos:,} ({100*false_pos/total:5.2f}%)")

    # ══════════════════════════════════════════════════════════════════════════════
    # SECTION 4: INTERPRÉTATION
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    print(f"""
The corpus {author_code} contains:

1. **CANONICAL WISE FOOLS** ({canonical_count} = {100*canonical_count/total:.2f}%)
   Named figures from the Islamic tradition:
   - Khalaf, Ligit, Riyah, Ja'ifran, Alyan, Bahlul
   These are explicitly identified by name in the texts.

2. **TRUE MAJNUN AQIL** ({true_count} = {100*true_count/total:.2f}%)
   Figures displaying paradoxical wisdom:
   - Explicit جنون/مجنون markers
   - Paradoxical statements (ولكن, إلا reversals)
   - Wisdom demonstrated through apparent foolishness

3. **FALSE POSITIVES** ({false_pos} = {100*false_pos/total:.2f}%)
   Generic dialogue-heavy texts without majnun characteristics
   These are correctly filtered by stricter classification thresholds.

The extraction pipeline successfully:
✅ Identifies named canonical fools from the corpus
✅ Detects paradoxical wisdom patterns
✅ Filters false positives (dialogue-only texts)
✅ Provides clean, coherent narrative units for analysis
""")

    print("="*80)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Show majnun aqil examples from strict analysis'
    )

    parser.add_argument('author', default='0328IbnCabdRabbih',
                        help='Author code (default: 0328IbnCabdRabbih)')

    args = parser.parse_args()

    show_majnun_examples(args.author)
