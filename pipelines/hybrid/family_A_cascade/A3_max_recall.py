"""
Pipeline Hybride A3 — Cascade à rappel maximal.

Principe : Pré-filtrage minimal + seuil ML très bas → filet large.
    Étape 1 : Pré-filtrage par genre uniquement (retire dīwāns, fiqh, naḥw)
    Étape 2 : Scoring ML avec seuil = 0.35 → recall élevé, precision faible
    Étape 3 : Validation collaborative par 2 annotateurs sur top-1000

Objectif : Ne rater aucun maǧnūn ʿāqil au prix de beaucoup de faux positifs.
Utile pour : Construire un corpus exhaustif, puis nettoyer manuellement.

Performance attendue :
    Precision ~45%, Recall ~88%, F1 ~0.59
    (vs A2 : Precision ~75%, Recall ~70%)

Cas d'usage recommandé :
    - Première exploration d'un nouveau corpus
    - Quand la couverture prime sur la précision

STATUT : À IMPLÉMENTER
Référence : Discussion Claude 8 avr. — Pipeline A3 (Maximum Recall)
"""

from __future__ import annotations


def run(corpus: list[dict], lr_model, xgb_model,
        ml_threshold: float = 0.35,
        max_candidates: int = 1000) -> list[dict]:
    """
    TODO: Adapter depuis A1_conservative.py avec seuils abaissés.
    """
    raise NotImplementedError()
