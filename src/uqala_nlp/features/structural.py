"""
Features de structure narrative temporelle.

Ce module capture l'ORDRE d'apparition des éléments clés dans le texte,
en partant de l'hypothèse que la structure narrative du maǧnūn ʿāqil
est aussi importante que la présence des éléments eux-mêmes.

Structure canonique du motif :
    JUNŪN → DIALOGUE → PAROLE_PARADOXALE → VALIDATION

Référence méthodologique :
    Propp, V. (1928). Morphologie du conte.
    Adaptation au khabar arabe : voir docs/features_catalog.md §12

STATUT : À IMPLÉMENTER
    - Définir les patterns de détection par position : À FAIRE
    - Valider sur corpus positif (452 akhbars) : À FAIRE
    - Comparer avec features lexicales : À FAIRE
"""

from __future__ import annotations
import re


# Patterns de détection (à raffiner empiriquement)
RE_JUNUN = re.compile(r'(?:مجنون|جنون|معتوه|مدله|خبل|هائم|ممسوس|بهلول|سعدون)', re.UNICODE)
RE_DIALOGUE = re.compile(r'(?:قال|فقال|قلت|سألت|سألوه|أجاب)', re.UNICODE)
RE_VALIDATION = re.compile(r'(?:فضحك|فبكى|أهدى|أعطاه|صدقت|فتعجب|فأعجبه)', re.UNICODE)
RE_AUTHORITY = re.compile(r'(?:الخليفة|الأمير|الوزير|القاضي|الملك)', re.UNICODE)


def detect_element_positions(text: str) -> dict[str, list[int]]:
    """
    Détecte les positions (en caractères) de chaque élément narratif dans le texte.

    Returns:
        Dict avec positions de chaque type d'élément.

    TODO: Implémenter.
    """
    raise NotImplementedError("detect_element_positions() n'est pas encore implémentée.")


def extract_structural_features(text: str) -> dict[str, float]:
    """
    Extrait les features de structure narrative temporelle.

    Features produites (toutes binaires ou continues) :
        f_order_canonical    : JUNŪN → DIALOGUE → VALIDATION (ordre canonique)
        f_junun_before_dial  : JUNŪN apparaît avant DIALOGUE
        f_validation_final   : VALIDATION dans le dernier tiers du texte
        f_authority_junun_prox : AUTHORITY et JUNŪN dans < 100 chars
        f_revelation_pattern : DIALOGUE → JUNŪN (folie révélée progressivement)
        f_distance_jd        : Distance en mots entre JUNŪN et DIALOGUE (normalisé)
        f_inversion_pattern  : VALIDATION avant JUNŪN (anti-pattern → FP probable)

    Returns:
        Dict de features structurelles (float)

    TODO: Implémenter en s'appuyant sur detect_element_positions().
    """
    raise NotImplementedError("extract_structural_features() n'est pas encore implémentée.")
