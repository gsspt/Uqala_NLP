"""
Pipeline 1.1 — Règles booléennes (interprétabilité maximale).

Principe : Pattern matching sur lexique + structure syntaxique.
Toute décision est traçable : chaque règle peut être lue comme
une hypothèse philologique falsifiable.

Avantages :
    - Décision entièrement traçable et explicable
    - Zéro dépendance ML, aucun entraînement requis
    - Modifiable instantanément (ajuster un seuil = tester une hypothèse)

Limites :
    - Recall faible (~10-15%) : rate tous les maǧnūn implicites
    - Précision médiocre (~30%) : capte غزل (ġazal) avec "جنون الحب"

Performance attendue : Precision ~30%, Recall ~15%, F1 ~0.20

STATUT : À IMPLÉMENTER
    Priorité : BASSE (utile comme baseline, pas comme pipeline final)

Référence : Discussion Claude 8 avr. — Pipeline 1.1
"""

from __future__ import annotations
import re

# ──────────────────────────────────────────────────────────────
# Lexiques (à enrichir depuis docs/features_catalog.md)
# ──────────────────────────────────────────────────────────────
JUNUN_TERMS = ['جنون', 'مجنون', 'معتوه', 'مدله', 'خبل', 'هائم']
AQL_TERMS = ['عقل', 'حكمة', 'رشد', 'لب', 'فطنة']
DIALOGUE_MARKERS = ['قال', 'فقال', 'قلت', 'سألت', 'أجاب']
VALIDATION_MARKERS = ['فضحك', 'فبكى', 'أهدى', 'أعطاه', 'صدقت']
FAMOUS_FOOLS = ['بهلول', 'سعدون', 'عليان', 'جعيفران', 'خلاف', 'رياح', 'لقيط']

# ──────────────────────────────────────────────────────────────
# Règles
# ──────────────────────────────────────────────────────────────

def rule_has_junun(text: str) -> bool:
    """R1 : Le texte contient au moins un marqueur de folie."""
    return any(term in text for term in JUNUN_TERMS)


def rule_has_aql(text: str) -> bool:
    """R2 : Le texte contient au moins un marqueur de sagesse/raison."""
    return any(term in text for term in AQL_TERMS)


def rule_has_dialogue(text: str) -> bool:
    """R3 : Le texte contient au moins un marqueur de dialogue."""
    return any(m in text for m in DIALOGUE_MARKERS)


def rule_has_validation(text: str) -> bool:
    """R4 : Le texte contient une réaction de validation (rire, don)."""
    return any(m in text for m in VALIDATION_MARKERS)


def rule_is_famous_fool(text: str) -> bool:
    """R5 : Le texte nomme un fou canonique (détection quasi-automatique)."""
    return any(name in text for name in FAMOUS_FOOLS)


def rule_not_pure_poetry(text: str) -> bool:
    """R6 : Le texte n'est pas de la poésie pure (< 70% de vers)."""
    lines = text.split('\n')
    verse_lines = sum(1 for l in lines if l.strip().startswith('%'))
    return (verse_lines / max(len(lines), 1)) < 0.7


def rule_adequate_length(text: str) -> bool:
    """R7 : Le texte a une longueur compatible avec un khabar narratif."""
    n_words = len(text.split())
    return 50 < n_words < 600


# ──────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────

def score(text: str) -> int:
    """
    Score booléen simple (0-7).
    Seuil empirique recommandé : >= 4 pour classifier positif.

    TODO: Calibrer ce seuil sur le corpus annoté.
    """
    raise NotImplementedError(
        "score() n'est pas encore calibrée. "
        "Tester différents seuils sur le corpus positif + négatif."
    )


def classify(text: str, threshold: int = 4) -> bool:
    """
    Classification binaire par règles booléennes.

    TODO: Implémenter après calibration du score.
    """
    raise NotImplementedError("classify() dépend de score() — voir ci-dessus.")
