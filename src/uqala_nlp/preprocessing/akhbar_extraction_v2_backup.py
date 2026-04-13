#!/usr/bin/env python3
"""
akhbar_extraction_v2_smart.py
─────────────────────────────────────────────────────────────────

Extraction améliorée des akhbars avec compréhension SÉMANTIQUE
des citations et délimitations logiques.

PROBLÈME RÉSOLU:
  • Captures les citations complètes (pas juste "قال X :")
  • Distingue les vraies sections (# |) des citations (# )
  • Produit des akhbars cohérents et non tronqués

STRATÉGIE: Hybrid (Heuristiques sémantiques + structure)
"""

import unicodedata
import pathlib
import re
from typing import List, Tuple


def count_arabic_chars(text: str) -> int:
    """Count Arabic characters"""
    return sum(1 for c in text if unicodedata.category(c) == 'Lo' and '\u0600' <= c <= '\u06FF')


def is_citation_continuation(last_line: str, current_line: str) -> bool:
    """
    Heuristiques pour déterminer si une ligne # est une CITATION
    (qui continue l'akhbar) ou une VRAIE SECTION (qui le termine).

    Args:
        last_line: Dernière ligne accumulée (texte de la ligne précédente)
        current_line: Ligne # actuelle (sans le #)

    Returns:
        True si c'est une citation (inclure), False si c'est section (break)
    """

    current_content = current_line[2:].strip() if len(current_line) > 2 else ""

    # ────────────────────────────────────────────────────────
    # HEURISTIQUE 1: Commence par ( = citation entre parenthèses
    # ────────────────────────────────────────────────────────
    if current_content.startswith('('):
        # Presque toujours une citation au Prophète ou sage
        # Exemple: ( عدل ساعة ... )
        return True

    # ────────────────────────────────────────────────────────
    # HEURISTIQUE 2: Commence par % = poésie citée
    # ────────────────────────────────────────────────────────
    if current_content.startswith('%'):
        # C'est une verse poétique citée
        return True

    # ────────────────────────────────────────────────────────
    # HEURISTIQUE 3: Verbe "qāla" en fin de ligne précédente?
    # ────────────────────────────────────────────────────────
    if last_line and any(verb in last_line for verb in ['قال', 'حدث', 'أخبر', 'روى']):
        # La ligne # suivante est probablement la citation
        # Exemple:
        #   "وقال النبي صلى الله عليه وسلم :"
        #   "# ( عدل ساعة في حكومة... )"
        return True

    # ────────────────────────────────────────────────────────
    # HEURISTIQUE 4: Contient [ ... ] = référence Coran
    # ────────────────────────────────────────────────────────
    if '[' in current_content and ']' in current_content:
        # Probablement une citation avec référence Coran
        return True

    # ────────────────────────────────────────────────────────
    # HEURISTIQUE 5: Est-ce un titre? (# | ou formule titre)
    # ────────────────────────────────────────────────────────
    if current_line.startswith('# |'):
        # C'est un titre → pas une citation
        return False

    # Par défaut: c'est une section/break
    return False


def extract_akhbars_from_file_v2(
    filepath: str,
    min_len: int = 80,
    max_len: int = 3000,
    allow_partial: bool = False,
) -> List[str]:
    """
    Extract akhbars with SEMANTIC understanding of citations.

    This version understands that lines starting with "# " can be:
      a) CITATIONS (continue the akhbar) → include them
      b) SECTIONS (end the akhbar) → break here

    Args:
        filepath: Path to OpenITI file
        min_len: Minimum Arabic characters (default 80)
        max_len: Maximum Arabic characters (default 3000)
        allow_partial: If True, keep incomplete akhbars

    Returns:
        List of complete, coherent akhbars
    """

    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    akhbars = []
    current_lines = []
    content_started = False

    for i, line in enumerate(lines):
        line = line.rstrip('\n\r')

        # ────────────────────────────────────────────────────────
        # SKIP METADATA
        # ────────────────────────────────────────────────────────
        if not content_started:
            if '#META#Header#End#' in line:
                content_started = True
            continue

        # ────────────────────────────────────────────────────────
        # VRAI TITRE (# |) → FIN D'AKHBAR
        # ────────────────────────────────────────────────────────
        if line.startswith('# |'):
            # Titre de section → toujours un break
            if current_lines:
                text = ' '.join(current_lines)
                n_ar = count_arabic_chars(text)
                if min_len <= n_ar <= max_len:
                    akhbars.append(text)
                elif allow_partial:
                    akhbars.append(text)

            current_lines = []
            continue

        # ────────────────────────────────────────────────────────
        # POTENTIELLE CITATION (# sans |)
        # ────────────────────────────────────────────────────────
        if line.startswith('# '):
            content = line[2:].strip()

            # Déterminer si c'est citation ou section
            last_line = current_lines[-1] if current_lines else ""
            is_citation = is_citation_continuation(last_line, line)

            if is_citation:
                # C'est une CITATION → inclure dans l'akhbar
                if content:
                    current_lines.append(content)

            else:
                # C'est une vraie SECTION → break
                if current_lines:
                    text = ' '.join(current_lines)
                    n_ar = count_arabic_chars(text)
                    if min_len <= n_ar <= max_len:
                        akhbars.append(text)
                    elif allow_partial:
                        akhbars.append(text)

                current_lines = []

            continue

        # ────────────────────────────────────────────────────────
        # CONTENU NORMAL (~~ ou autres)
        # ────────────────────────────────────────────────────────
        if line.startswith('~~'):
            content = line[2:].strip()
            if content:
                current_lines.append(content)

        # Autres lignes vides/spéciales → ignorer
        elif line.strip() and not line.startswith('#'):
            # Parfois du texte sans ~~ → inclure aussi
            if current_lines:
                current_lines.append(line.strip())

    # ────────────────────────────────────────────────────────
    # FIN: Sauvegarder le dernier akhbar
    # ────────────────────────────────────────────────────────
    if current_lines:
        text = ' '.join(current_lines)
        n_ar = count_arabic_chars(text)
        if min_len <= n_ar <= max_len:
            akhbars.append(text)
        elif allow_partial:
            akhbars.append(text)

    return akhbars


# ════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY: Alias for old code
# ════════════════════════════════════════════════════════════════════════════

def extract_akhbars_from_file(filepath: str, min_len: int = 80, max_len: int = 3000, allow_partial: bool = False) -> List[str]:
    """Backward compatibility wrapper — calls the v2 smart method"""
    return extract_akhbars_from_file_v2(filepath, min_len, max_len, allow_partial)


# ════════════════════════════════════════════════════════════════════════════
# TESTING
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    filepath = (
        "openiti_corpus/data/0328IbnCabdRabbih/"
        "0328IbnCabdRabbih.CiqdFarid/"
        "0328IbnCabdRabbih.CiqdFarid.JK009200-ara1"
    )

    print("Comparing extraction methods:\n")

    # Old method
    from uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file

    old_akhbars = extract_akhbars_from_file(filepath)
    print(f"OLD METHOD (v1): {len(old_akhbars)} akhbars")
    print(f"  Total chars: {sum(len(a) for a in old_akhbars):,}")

    # New method
    new_akhbars = extract_akhbars_from_file_v2(filepath)
    print(f"\nNEW METHOD (v2): {len(new_akhbars)} akhbars")
    print(f"  Total chars: {sum(len(a) for a in new_akhbars):,}")

    # Find differences
    print(f"\nDifference: {len(new_akhbars) - len(old_akhbars):+d} akhbars")
    print(
        f"  "
        f"{abs(sum(len(a) for a in new_akhbars) - sum(len(a) for a in old_akhbars)):+,} "
        f"chars"
    )

    # Show some examples where they differ
    print("\n" + "=" * 80)
    print("EXAMPLES: Cases where new method is more complete")
    print("=" * 80 + "\n")

    # Find akhbars that end with "قال X :"
    truncated_count = 0
    for i, akhbar in enumerate(old_akhbars[:100]):
        if (
            akhbar.rstrip().endswith(':')
            and any(v in akhbar for v in ['قال', 'حدث', 'أخبر'])
        ):
            truncated_count += 1

            if truncated_count <= 3:
                print(f"Akhbar #{i} (TRUNCATED in v1):")
                print(f"  Length: {len(akhbar)}")
                print(f"  Ends with: ...{akhbar[-80:]}")
                print()

    print(f"Total truncated akhbars found: {truncated_count}")
