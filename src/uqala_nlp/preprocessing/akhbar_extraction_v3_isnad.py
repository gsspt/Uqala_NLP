#!/usr/bin/env python3
"""
akhbar_extraction_v3_isnad.py
────────────────────────────────────────────────────────────────

STRATÉGIE PHILOLOGIQUE CORRECTE:
1. Nettoyer le texte de TOUTES les métadonnées OpenITI
2. Chercher les ISNADS (chaînes de transmission) comme délimiteurs naturels
3. Chaque nouvel isnad = nouvel akhbar
4. Valider que chaque akhbar a une matn substantielle

PRINCIPE:
  En arabe classique, chaque akhbar commence par une chaîne de transmission (isnad).
  Les métadonnées OpenITI (# |, ~~, PageXXX, etc.) ne reflètent que la structure du manuscrit,
  pas la structure narrative.

Usage:
  from akhbar_extraction_v3_isnad import extract_akhbars_from_file_v3
  akhbars = extract_akhbars_from_file_v3('openiti_corpus/data/.../file.txt')
"""

import re
import unicodedata
from typing import List, Tuple

# ════════════════════════════════════════════════════════════════════════════════
# PATTERNS D'ISNAD (verbes de transmission)
# ════════════════════════════════════════════════════════════════════════════════

ISNAD_VERBS = {
    # حدّث (raconter/transmettre)
    'حدثنا', 'حدثني', 'حدثه', 'حدثهم', 'حدثها', 'حدثك', 'حدثت',
    # أخبر (informer)
    'أخبرنا', 'أخبرني', 'أخبره', 'أخبرهم', 'أخبرها', 'أخبرك',
    # أنبأ (annoncer)
    'أنبأنا', 'أنبأني', 'أنبأه', 'أنبأت', 'أنبأهم',
    # روى (rapporter)
    'روى', 'روينا', 'رواه', 'روت', 'رويت', 'رووا',
    # سمع (entendre/écouter)
    'سمعت', 'سمعنا', 'سمعه', 'سمعهم', 'سمعتُ', 'سمعوا',
    # ذكر (mentionner)
    'ذكر', 'ذكره', 'ذكرنا', 'ذكرت', 'ذكروا',
    # Autres
    'أعلمنا', 'أعلمني', 'نقل', 'نقله', 'قرأت', 'قرأنا',
    'وصل', 'بلغنا', 'بلغني', 'بلغه',
}

# Construire un regex pour trouver les isnads
ISNAD_REGEX = r'\b(?:' + '|'.join(re.escape(v) for v in ISNAD_VERBS) + r')\b'


def count_arabic_chars(text: str) -> int:
    """Compter les caractères arabes"""
    return sum(1 for c in text if unicodedata.category(c) == 'Lo' and '\u0600' <= c <= '\u06FF')


def clean_openiti_metadata(text: str) -> str:
    """
    Nettoyer le texte de TOUTES les métadonnées OpenITI:
    - Balises # | (titres)
    - Balises # (sections/citations)
    - Balises ~~ (contenu marqué)
    - Références de pages [ ... ]
    - Numéros de pages PageXXX
    - Marques de manuscrit msXXXX
    - Balises Coran ^ ... ^
    """

    # Enlever les marques de début de ligne
    text = re.sub(r'^# \|', '', text, flags=re.MULTILINE)  # Titres
    text = re.sub(r'^# ', '', text, flags=re.MULTILINE)     # Sections
    text = re.sub(r'^~~', '', text, flags=re.MULTILINE)     # Contenu

    # Enlever les références de page
    text = re.sub(r'\[\s*[^\]]*?\s*\]', '', text)  # [Page V01P023]
    text = re.sub(r'PageV?\d+P?\d+', '', text)     # PageV01P023
    text = re.sub(r'Page\d+', '', text)            # Page123

    # Enlever les marques de manuscrit
    text = re.sub(r'ms\d+', '', text)  # ms0001, ms1234

    # Enlever les balises Coran
    text = re.sub(r'\^\s*\(\s*', '(', text)  # ^( → (
    text = re.sub(r'\s*\)\s*\^', ')', text)  # )^ → )

    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_akhbars_from_file_v3(
    filepath: str,
    min_len: int = 80,
    max_len: int = 5000,
) -> List[str]:
    """
    Extraire les akhbars en segmentant par ISNADS.

    Algorithme:
    1. Charger le fichier
    2. Trouver #META#Header#End#
    3. Nettoyer toutes les métadonnées OpenITI
    4. Chercher tous les isnads (حدثنا، أخبرني، etc.)
    5. Segmenter : chaque isnad = début d'un nouvel akhbar
    6. Valider : chaque akhbar doit avoir 80-5000 caractères arabes

    Args:
        filepath: Chemin vers le fichier OpenITI
        min_len: Minimum de caractères arabes (défaut 80)
        max_len: Maximum de caractères arabes (défaut 5000)

    Returns:
        List[str]: Akhbars segmentés et validés
    """

    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Erreur lecture {filepath}: {e}")
        return []

    # ────────────────────────────────────────────────────────
    # ÉTAPE 1: Extrait le contenu (après métadonnées)
    # ────────────────────────────────────────────────────────
    content_started = False
    content_lines = []

    for line in lines:
        if '#META#Header#End#' in line:
            content_started = True
            continue
        if content_started:
            content_lines.append(line)

    if not content_lines:
        return []

    raw_text = ''.join(content_lines)

    # ────────────────────────────────────────────────────────
    # ÉTAPE 2: Nettoyer toutes les métadonnées OpenITI
    # ────────────────────────────────────────────────────────
    clean_text = clean_openiti_metadata(raw_text)

    # ────────────────────────────────────────────────────────
    # ÉTAPE 3: Trouver tous les isnads
    # ────────────────────────────────────────────────────────
    isnad_matches = list(re.finditer(ISNAD_REGEX, clean_text))

    if not isnad_matches:
        # Aucun isnad trouvé → tout le texte = un akhbar
        n_ar = count_arabic_chars(clean_text)
        if min_len <= n_ar <= max_len:
            return [clean_text]
        return []

    # ────────────────────────────────────────────────────────
    # ÉTAPE 4: Segmenter par isnads
    # ────────────────────────────────────────────────────────
    akhbars = []

    for i, match in enumerate(isnad_matches):
        isnad_start = match.start()

        # Trouver la fin de cet akhbar (= début du prochain isnad)
        if i + 1 < len(isnad_matches):
            akhbar_end = isnad_matches[i + 1].start()
        else:
            # Dernier isnad → jusqu'à la fin du texte
            akhbar_end = len(clean_text)

        # Extraire le segment
        segment = clean_text[isnad_start:akhbar_end].strip()

        if not segment:
            continue

        # Valider
        n_ar = count_arabic_chars(segment)
        if min_len <= n_ar <= max_len:
            akhbars.append(segment)

    return akhbars


# ════════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ════════════════════════════════════════════════════════════════════════════════

def extract_akhbars_from_file(
    filepath: str,
    min_len: int = 80,
    max_len: int = 3000,
    allow_partial: bool = False,
) -> List[str]:
    """
    Wrapper pour compatibilité backward avec le code existant.
    Appelle maintenant la v3 (isnad-based).
    """
    return extract_akhbars_from_file_v3(filepath, min_len, max(max_len, 5000))


# ════════════════════════════════════════════════════════════════════════════════
# TESTING
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    filepath = (
        "openiti_corpus/data/0328IbnCabdRabbih/"
        "0328IbnCabdRabbih.CiqdFarid/"
        "0328IbnCabdRabbih.CiqdFarid.JK009200-ara1"
    )

    print("AKHBAR EXTRACTION v3 (ISNAD-BASED)")
    print("=" * 100 + "\n")

    akhbars_v3 = extract_akhbars_from_file_v3(filepath)
    print(f"Total akhbars (v3 isnad): {len(akhbars_v3)}")
    print(f"Total caractères: {sum(len(a) for a in akhbars_v3):,}")
    print(f"Moyenne par akhbar: {sum(len(a) for a in akhbars_v3) / len(akhbars_v3) if akhbars_v3 else 0:.0f} chars\n")

    print("=" * 100)
    print("PREMIERS 5 AKHBARS")
    print("=" * 100 + "\n")

    for i, akh in enumerate(akhbars_v3[:5]):
        print(f"[{i+1}] Longueur: {len(akh)} chars")
        # Find first isnad verb
        match = re.search(ISNAD_REGEX, akh)
        if match:
            print(f"     Commence par: ...{akh[:80]}...")
        print()
