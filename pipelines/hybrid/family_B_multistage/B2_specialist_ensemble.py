"""
Pipeline Hybride B2 — Ensemble de spécialistes.

Principe : Entraîner 3 modèles spécialisés sur des sous-types de khabars,
puis combiner leurs prédictions par vote pondéré.

    Spécialiste 1 : Khabars COURTS (< 150 mots)
        Features : lexicales uniquement (rapides)
        Modèle : RandomForest

    Spécialiste 2 : Khabars LONGS (>= 150 mots, prose)
        Features : lexicales + structurelles
        Modèle : XGBoost

    Spécialiste 3 : Khabars AVEC POÉSIE (poetry_ratio > 0.2)
        Features : rhétoriques + actantielles
        Modèle : SVM

Motivation philologique :
    Un khabar court de 80 mots (maxime paradoxale) et un khabar long
    de 300 mots (récit avec dialogue) sont des sous-genres différents
    qui nécessitent des critères de détection distincts.

Performance attendue : Precision ~80%, Recall ~72%, F1 ~0.76

STATUT : À IMPLÉMENTER
    Prérequis : features/actantial.py et features/structural.py implémentés

Référence : Discussion Claude 8 avr. — Pipeline B2 (Specialist Ensemble)
"""

from __future__ import annotations


def route_to_specialist(khabar: dict) -> str:
    """
    Détermine quel spécialiste utiliser pour ce khabar.

    Returns: 'short', 'long', ou 'poetry'
    """
    raise NotImplementedError()


def run(corpus: list[dict], specialists: dict) -> list[dict]:
    """
    Classifie chaque khabar avec le spécialiste approprié.

    TODO: Implémenter la logique de routage + vote pondéré.
    """
    raise NotImplementedError()
