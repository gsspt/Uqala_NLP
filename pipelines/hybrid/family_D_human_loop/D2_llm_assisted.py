#!/usr/bin/env python3
"""
llm_analysis.py
───────────────
Extrait les akhbars et les présente pour analyse qualitative par LLM.

Usage:
  python openiti_detection/llm_analysis.py 0328IbnCabdRabbih [count]

Exemple:
  python openiti_detection/llm_analysis.py 0328IbnCabdRabbih 50
"""

import json
import sys
import pathlib
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from openiti_detection.detect_lr_xgboost import extract_akhbars_from_file, count_arabic_chars

# ══════════════════════════════════════════════════════════════════════════════

BASE = pathlib.Path(__file__).parent.parent
OPENITI_TARGETED = BASE / "openiti_targeted"

# ══════════════════════════════════════════════════════════════════════════════

def analyze_akhbars_for_llm(author_code, sample_size=50):
    """Extrait et formate les akhbars pour analyse LLM."""

    author_path = OPENITI_TARGETED / author_code

    if not author_path.exists():
        print(f"❌ Author not found: {author_code}")
        sys.exit(1)

    # Extraire tous les akhbars
    print(f"Extracting akhbars from {author_code}...\n")

    all_akhbars = []
    files = [f for f in author_path.rglob("*") if f.is_file() and not f.name.startswith('.')]

    for filepath in files:
        try:
            akhbars = extract_akhbars_from_file(filepath)
            all_akhbars.extend(akhbars)
        except Exception as e:
            pass

    print(f"Total akhbars extracted: {len(all_akhbars)}\n")

    if not all_akhbars:
        print("No akhbars found!")
        sys.exit(1)

    # Prendre un échantillon
    sample = all_akhbars[:sample_size]

    # ══════════════════════════════════════════════════════════════════════════════
    # ANALYSER LES AKHBARS POUR COHÉRENCE
    # ══════════════════════════════════════════════════════════════════════════════

    print("="*80)
    print(f"LLM ANALYSIS: {sample_size} AKHBARS FROM {author_code}")
    print("="*80)

    # Caractéristiques à analyser
    categories = {
        'biographical': [],      # Biographies, informations personnelles
        'dialogue': [],          # Dialogues entre personnages
        'narrative': [],         # Récits d'événements
        'wisdom': [],            # Dictons, sagesse, conseils
        'poetry': [],            # Vers ou textes poétiques
        'historical': [],        # Événements historiques
        'anecdotal': [],         # Anecdotes amusantes
        'mixed': [],             # Multiple genres
    }

    structural_features = {
        'has_names': 0,          # Contient des noms propres
        'has_dates': 0,          # Contient des dates/années
        'has_quotes': 0,         # Contient des citations (قال)
        'has_commands': 0,       # Contient des ordres/conseils
        'has_questions': 0,      # Contient des questions
        'is_coherent': 0,        # Semble cohérent narrativement
        'is_complete': 0,        # Semble être une unité complète
        'is_fragmented': 0,      # Semble fragmenté
    }

    for i, akhbar in enumerate(sample, 1):
        ar_count = count_arabic_chars(akhbar)

        # Analyser les caractéristiques structurelles
        has_names = len([w for w in akhbar.split() if 'بن' in w or w[0].isupper()]) > 0
        has_dates = any(year in akhbar for year in ['سنة', 'هـ', 'م'])
        has_quotes = 'قال' in akhbar or 'قالت' in akhbar
        has_commands = any(cmd in akhbar for cmd in ['افعل', 'لا تفعل', 'يجب', 'يلزم'])
        has_questions = '؟' in akhbar

        # Analyser la cohérence
        sentences = [s.strip() for s in akhbar.split('|') if s.strip()]
        is_coherent = len(sentences) >= 1 and ar_count >= 100
        is_complete = ar_count >= 150 and (has_quotes or len(sentences) >= 2)
        is_fragmented = ar_count < 80 or (has_quotes and ar_count < 100)

        if has_names:
            structural_features['has_names'] += 1
        if has_dates:
            structural_features['has_dates'] += 1
        if has_quotes:
            structural_features['has_quotes'] += 1
        if has_commands:
            structural_features['has_commands'] += 1
        if has_questions:
            structural_features['has_questions'] += 1
        if is_coherent:
            structural_features['is_coherent'] += 1
        if is_complete:
            structural_features['is_complete'] += 1
        if is_fragmented:
            structural_features['is_fragmented'] += 1

        # Catégoriser
        if 'ولد' in akhbar or 'نشأ' in akhbar or 'مات' in akhbar:
            categories['biographical'].append(i)
        elif has_quotes and akhbar.count('قال') >= 2:
            categories['dialogue'].append(i)
        elif 'قال' in akhbar or 'رأى' in akhbar or 'رجل' in akhbar:
            categories['narrative'].append(i)
        elif any(w in akhbar for w in ['الحكمة', 'الحكيم', 'العاقل', 'الفقيه']):
            categories['wisdom'].append(i)
        elif '%' in akhbar or 'شعر' in akhbar or 'بيت' in akhbar:
            categories['poetry'].append(i)
        elif any(w in akhbar for w in ['حرب', 'غزا', 'قتال', 'معركة', 'سنة']):
            categories['historical'].append(i)
        elif 'قال' in akhbar and ('ضحك' in akhbar or 'لطيف' in akhbar):
            categories['anecdotal'].append(i)
        else:
            categories['mixed'].append(i)

    # ══════════════════════════════════════════════════════════════════════════════
    # RÉSULTATS
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n[STRUCTURAL ANALYSIS]\n")
    print(f"Features présentes dans les {sample_size} akhbars:")
    print(f"  • With names (noms propres):           {structural_features['has_names']:3d} ({100*structural_features['has_names']/sample_size:5.1f}%)")
    print(f"  • With dates/years (dates):            {structural_features['has_dates']:3d} ({100*structural_features['has_dates']/sample_size:5.1f}%)")
    print(f"  • With quotes (dialogue - قال):        {structural_features['has_quotes']:3d} ({100*structural_features['has_quotes']/sample_size:5.1f}%)")
    print(f"  • With commands/advice (ordres):       {structural_features['has_commands']:3d} ({100*structural_features['has_commands']/sample_size:5.1f}%)")
    print(f"  • With questions (questions):          {structural_features['has_questions']:3d} ({100*structural_features['has_questions']/sample_size:5.1f}%)")

    print(f"\nQualité narrative:")
    print(f"  • Narratively coherent (cohérent):     {structural_features['is_coherent']:3d} ({100*structural_features['is_coherent']/sample_size:5.1f}%)")
    print(f"  • Complete units (complet):            {structural_features['is_complete']:3d} ({100*structural_features['is_complete']/sample_size:5.1f}%)")
    print(f"  • Appear fragmented (fragmenté):       {structural_features['is_fragmented']:3d} ({100*structural_features['is_fragmented']/sample_size:5.1f}%)")

    print(f"\n[CONTENT CATEGORIES]\n")
    print(f"Genre distribution des akhbars:")
    for genre, indices in categories.items():
        if indices:
            print(f"  • {genre:20s}: {len(indices):3d} ({100*len(indices)/sample_size:5.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════════
    # AFFICHER DES EXEMPLES DÉTAILLÉS POUR ANALYSE LLM
    # ══════════════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("DETAILED EXAMPLES FOR LLM ANALYSIS")
    print("="*80)

    examples_to_show = [
        ("BIOGRAPHICAL", categories['biographical'][:2]),
        ("DIALOGUE", categories['dialogue'][:2]),
        ("NARRATIVE", categories['narrative'][:2]),
        ("WISDOM", categories['wisdom'][:2]),
        ("ANECDOTAL", categories['anecdotal'][:2]),
    ]

    for genre, indices in examples_to_show:
        if not indices:
            continue

        print(f"\n{genre}:")
        print("─" * 80)

        for idx in indices:
            akhbar = sample[idx - 1]
            ar_count = count_arabic_chars(akhbar)

            print(f"\n[{idx:2d}] (Arabic chars: {ar_count})")
            print(f"─────────────────────────────────────────────────────────────────────────────")
            print(akhbar)
            print()

    # ══════════════════════════════════════════════════════════════════════════════
    # SAUVEGARDER POUR ANALYSE
    # ══════════════════════════════════════════════════════════════════════════════

    output_data = {
        'author': author_code,
        'total_extracted': len(all_akhbars),
        'sample_size': sample_size,
        'structural_features': structural_features,
        'content_categories': {k: len(v) for k, v in categories.items()},
        'sample_akhbars': sample[:20],  # Les 20 premiers pour inspection
    }

    output_path = pathlib.Path(__file__).parent / "results" / author_code / "llm_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("="*80)
    print(f"\n✅ Analysis saved → {output_path}\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze extracted akhbars for LLM review')
    parser.add_argument('author', default='0328IbnCabdRabbih',
                        help='Author code')
    parser.add_argument('count', nargs='?', type=int, default=50,
                        help='Number of akhbars to analyze (default: 50)')
    args = parser.parse_args()

    analyze_akhbars_for_llm(args.author, args.count)
