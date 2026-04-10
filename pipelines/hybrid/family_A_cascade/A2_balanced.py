"""
Pipeline Hybride A2 — Cascade équilibrée (précision/rappel balancés).

Principe : Cascade en 3 étapes avec seuils intermédiaires.
    Étape 1 : Pré-filtrage structurel (garde ~20% du corpus)
    Étape 2 : Scoring ML avec seuil = 0.65 (balanced)
    Étape 3 : Post-filtrage souple (score_filter >= 0.25)

Différence avec A1 (conservative) :
    - Seuil ML plus bas (0.65 vs 0.85) → plus de candidats
    - Post-filtrage plus souple → récupère les maǧnūn implicites
    - Valide ~500 candidats au lieu de ~200

Performance attendue :
    Precision ~75%, Recall ~70%, F1 ~0.72
    (vs A1 : Precision ~90%, Recall ~40%)

Recommandation : Pipeline par défaut pour l'exploration initiale.

STATUT : À IMPLÉMENTER
    Prérequis : Modèles entraînés (models/lr_classifier_71features.pkl)

Référence : Discussion Claude 8 avr. — Pipeline A2 (Balanced Trade-off)
"""

from __future__ import annotations


def run(corpus: list[dict], lr_model, xgb_model,
        ml_threshold: float = 0.65,
        filter_threshold: float = 0.25,
        top_n: int = 500) -> list[dict]:
    """
    Exécute le pipeline A2 sur un corpus de khabars.

    TODO: Adapter depuis A1_conservative.py en ajustant les seuils.
    """
    raise NotImplementedError(
        "Adapter depuis A1_conservative.py (detect_lr_xgboost.py). "
        "Modifier THRESHOLD_LR et THRESHOLD_XGB à 0.65, "
        "et post_filter.score() threshold à 0.25."
    )
