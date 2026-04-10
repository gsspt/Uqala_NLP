#!/usr/bin/env python3
"""
post_filter.py
──────────────
Filtrage POST-CLASSIFICATION pour réduire les faux positifs.

Le problème: 80% des "positifs" sont juste du dialogue générique
La solution: Vérifier VRAIMENT les caractéristiques du majnun aqil

Stratégies de filtrage:
1. Paradox detection: "قال X ولكن/إلا/لكن Y"
2. Junun markers: جنون, مجنون, معتوه, هائم
3. Wisdom context: حكمة, عقل, فقيه près de junun
4. Irony/sarcasm: Phrases qui se contredisent
5. Contradiction reversal: Idées opposées en succession

Usage:
  python openiti_detection/post_filter.py 0328IbnCabdRabbih
"""

import json
import sys
import pathlib
import re

sys.stdout.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
# FILTRES POST-CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class MajnunAqilFilter:
    """Vérifie les véritables caractéristiques du majnun aqil."""

    # Marqueurs de junun explicites
    JUNUN_MARKERS = [
        'جنون', 'مجنون', 'معتوه', 'هائم', 'ممسوس',
        'جنّ', 'أجنّ', 'جننت', 'يجن', 'جننه'
    ]

    # Fous canoniques
    CANONICAL_FOOLS = {
        'bahlul': ['بهلول', 'باهلول'],
        'sa\'doun': ['سعدون', 'ساعدون'],
        'samnun': ['سمنون', 'سمنن'],
        'khalaf': ['خلاف', 'خالف'],
        'ja\'ifran': ['جعيفران', 'جعفران'],
        'riyah': ['رياح', 'ريَّاح'],
        'ligit': ['لقيط'],
    }

    # Marqueurs de paradoxe/contradiction
    PARADOX_MARKERS = ['ولكن', 'إلا', 'لكن', 'غير أن', 'لكن', 'على أن']

    # Marqueurs de sagesse
    WISDOM_WORDS = [
        'حكمة', 'حكيم', 'حكم', 'عقل', 'عاقل', 'عقيل',
        'فقيه', 'فقه', 'دراية', 'علم', 'عالم',
        'رأي', 'صواب', 'حق', 'حقيقة', 'شهادة'
    ]

    # Marqueurs d'ironie/sarcasme (contexte)
    IRONY_PATTERNS = [
        r'قال.*(?:ضحك|ضحكوا|يضحكون)',  # Dit quelque chose → rire
        r'(?:أحمق|جاهل|سفيه).*(?:حكم|حكيم|عقل)',  # Fool shows wisdom
        r'قال.*(?:لطيف|ذكي|بارع).*لكن',  # Says something clever but contradicts
    ]

    # Marqueurs de validation (reconnaisance de la sagesse paradoxale)
    VALIDATION_MARKERS = [
        'ضحك', 'هز رأسه', 'استحسن', 'استحسنوا',
        'صدقت', 'أنت أصدق', 'أنت أحق', 'قال: صدقت',
        'قال: نعم', 'قال: أصبت'
    ]

    @classmethod
    def has_canonical_fool(cls, text):
        """Détecte un fou canonique nommé."""
        for fool, variants in cls.CANONICAL_FOOLS.items():
            for variant in variants:
                if variant in text:
                    return True, fool
        return False, None

    @classmethod
    def has_junun_markers(cls, text):
        """Détecte les marqueurs explicites de junun."""
        for marker in cls.JUNUN_MARKERS:
            if marker in text:
                return True
        return False

    @classmethod
    def has_paradox(cls, text):
        """Détecte une structure paradoxale."""
        # Pattern: "قال X" suivi de "ولكن/إلا/لكن Y"
        for marker in cls.PARADOX_MARKERS:
            # Chercher qala suivi par le marqueur de paradoxe
            if 'قال' in text and marker in text:
                # Vérifier que le paradoxe est proche du dialogue
                qala_pos = text.find('قال')
                marker_pos = text.find(marker)
                if qala_pos < marker_pos and marker_pos - qala_pos < 200:
                    return True
        return False

    @classmethod
    def has_wisdom_context(cls, text):
        """Vérifie que junun/paradoxe est dans un contexte de sagesse."""
        # Chercher si on parle de sagesse/intelligence
        wisdom_count = sum(1 for w in cls.WISDOM_WORDS if w in text)
        junun_present = cls.has_junun_markers(text)

        if junun_present and wisdom_count >= 2:
            return True
        if cls.has_paradox(text) and wisdom_count >= 1:
            return True
        return False

    @classmethod
    def is_ironic(cls, text):
        """Détecte l'ironie/sarcasme dans le texte."""
        for pattern in cls.IRONY_PATTERNS:
            if re.search(pattern, text):
                return True

        # Vérifier une contradiction simple: opposites rapprochés
        opposites = [
            ('أحمق', 'حكيم'), ('جاهل', 'عالم'), ('سفيه', 'عاقل'),
            ('ضعيف', 'قوي'), ('خاسر', 'رابح'), ('ذليل', 'عزيز')
        ]
        for neg, pos in opposites:
            if neg in text and pos in text:
                return True

        return False

    @classmethod
    def has_validation(cls, text):
        """Détecte la validation/reconnaissance de la sagesse paradoxale."""
        for marker in cls.VALIDATION_MARKERS:
            if marker in text:
                return True
        return False

    @classmethod
    def score(cls, text):
        """Calcule un score de confiance pour majnun aqil (0-1)."""
        score = 0.0

        # Canonical fool → confiance maximum
        has_fool, fool_name = cls.has_canonical_fool(text)
        if has_fool:
            return 1.0, "canonical_fool"

        # Points pour différentes caractéristiques
        if cls.has_junun_markers(text):
            score += 0.4

        if cls.has_paradox(text):
            score += 0.3

        if cls.is_ironic(text):
            score += 0.2

        if cls.has_wisdom_context(text):
            score += 0.1

        if cls.has_validation(text):
            score += 0.1

        if score > 0:
            category = "true_majnun"
        else:
            category = "false_positive"

        return min(score, 1.0), category


def apply_post_filter(author_code):
    """Applique le filtrage post-classification."""

    results_dir = pathlib.Path(__file__).parent / "results" / author_code
    all_predictions_path = results_dir / "all_predictions.json"

    if not all_predictions_path.exists():
        print(f"❌ Results not found")
        sys.exit(1)

    print("="*80)
    print("POST-CLASSIFICATION FILTERING")
    print("="*80)

    with open(all_predictions_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # ══════════════════════════════════════════════════════════════════════════════
    # APPLIQUER LE FILTRE
    # ══════════════════════════════════════════════════════════════════════════════

    consensus = [r for r in results if r['consensus'] == 1]
    strict_consensus = [
        r for r in consensus
        if r['lr_prob'] >= 0.7 and r['xgb_prob'] >= 0.7
    ]

    print(f"\nProcessing {len(strict_consensus)} strict consensus predictions...\n")

    filtered_results = {
        'canonical_fools': [],
        'true_majnun_aqil': [],
        'false_positives': [],
        'rejected': []
    }

    for r in strict_consensus:
        text = r['text']
        post_score, category = MajnunAqilFilter.score(text)

        r['post_filter_score'] = post_score
        r['post_filter_category'] = category

        if category == "canonical_fool":
            filtered_results['canonical_fools'].append(r)
        elif post_score >= 0.5:
            filtered_results['true_majnun_aqil'].append(r)
        elif post_score > 0:
            filtered_results['false_positives'].append(r)
        else:
            filtered_results['rejected'].append(r)

    # ══════════════════════════════════════════════════════════════════════════════
    # RÉSULTATS
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"{'='*80}")
    print("FILTERING RESULTS")
    print(f"{'='*80}\n")

    total = len(strict_consensus)
    print(f"Before filtering (strict consensus): {total}")
    print(f"\nAfter post-classification filtering:")
    print(f"  ✅ Canonical fools: {len(filtered_results['canonical_fools']):4d} ({100*len(filtered_results['canonical_fools'])/total:5.1f}%)")
    print(f"  ✅ True majnun aqil: {len(filtered_results['true_majnun_aqil']):4d} ({100*len(filtered_results['true_majnun_aqil'])/total:5.1f}%)")
    print(f"  ⚠️  Weak signals: {len(filtered_results['false_positives']):4d} ({100*len(filtered_results['false_positives'])/total:5.1f}%)")
    print(f"  ❌ Rejected (pure dialogue): {len(filtered_results['rejected']):4d} ({100*len(filtered_results['rejected'])/total:5.1f}%)")

    print(f"\nReliable positives (canonical + true majnun):")
    reliable = len(filtered_results['canonical_fools']) + len(filtered_results['true_majnun_aqil'])
    print(f"  {reliable:4d} out of {total} ({100*reliable/total:5.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════════
    # AFFICHER DES EXEMPLES
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print("EXAMPLES AFTER FILTERING")
    print(f"{'='*80}\n")

    print("CANONICAL FOOLS (high confidence):")
    for r in filtered_results['canonical_fools'][:3]:
        print(f"  LR={r['lr_prob']:.3f} XGB={r['xgb_prob']:.3f}")
        print(f"  \"{r['text'][:100]}...\"")
        print()

    print("TRUE MAJNUN AQIL (paradox + wisdom):")
    for r in filtered_results['true_majnun_aqil'][:3]:
        print(f"  LR={r['lr_prob']:.3f} XGB={r['xgb_prob']:.3f} PostFilter={r['post_filter_score']:.2f}")
        print(f"  \"{r['text'][:100]}...\"")
        print()

    print("REJECTED (pure dialogue, no paradox):")
    for r in filtered_results['rejected'][:3]:
        print(f"  LR={r['lr_prob']:.3f} XGB={r['xgb_prob']:.3f}")
        print(f"  \"{r['text'][:100]}...\"")
        print()

    # ══════════════════════════════════════════════════════════════════════════════
    # SAUVEGARDER
    # ══════════════════════════════════════════════════════════════════════════════

    output_path = results_dir / "post_filtered_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)

    print(f"{'='*80}")
    print(f"✅ Results saved → {output_path}")
    print(f"{'='*80}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Apply post-classification filtering to reduce false positives'
    )

    parser.add_argument('author', default='0328IbnCabdRabbih',
                        help='Author code')

    args = parser.parse_args()

    apply_post_filter(args.author)
