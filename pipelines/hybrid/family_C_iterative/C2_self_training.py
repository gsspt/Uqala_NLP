"""
Pipeline Hybride C2 — Self-training (pseudo-labels).

Principe : Utiliser les prédictions très confiantes du modèle
comme nouvelles données d'entraînement (pseudo-labels).

    Cycle 1 : Entraîner sur corpus annoté (1,224 textes)
    Cycle 2 : Prédire sur non-annotés → garder prob > 0.95 comme positifs,
              prob < 0.05 comme négatifs → ~5,000 pseudo-labels
    Cycle 3 : Ré-entraîner sur corpus élargi → meilleures prédictions
    Répéter 5 fois.

Avantage : Exploite les 200,000 textes non-annotés sans effort humain.
Risque : Propagation d'erreurs si biais initial (effet "boule de neige").

Performance attendue : Precision ~72%, Recall ~68%, F1 ~0.70
    (inférieur à C1 car moins de contrôle sur la qualité)

Mitigation du risque :
    - Seuil de confiance élevé (0.95) pour les pseudo-labels
    - Validation croisée à chaque cycle
    - Arrêt si performance baisse

STATUT : À IMPLÉMENTER
    Recommandé après C1 pour exploiter les textes non-annotés.

Référence : Discussion Claude 8 avr. — Pipeline C2 (Bootstrapping)
"""

from __future__ import annotations


def run_cycle(clf, X_train, y_train, X_unlabeled,
              pos_threshold: float = 0.95,
              neg_threshold: float = 0.05) -> tuple:
    """
    Ajoute les pseudo-labels confiants au corpus d'entraînement.

    Returns:
        (X_train_augmented, y_train_augmented, n_pseudo_labels_added)

    TODO: Implémenter.
    """
    raise NotImplementedError()
