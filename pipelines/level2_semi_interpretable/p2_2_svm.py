"""
Pipeline 2.2 — SVM à noyau RBF + analyse des vecteurs de support.

Principe : Frontière de décision non-linéaire dans l'espace des features.
L'interprétabilité est post-hoc : on identifie les exemples "limites"
(vecteurs de support) qui définissent la frontière du motif.

Intérêt philologique :
    Les vecteurs de support sont les khabars les plus informatifs.
    Ce sont les cas prototypiques du maǧnūn ʿāqil → corpus de référence.

Avantages :
    - Précision ~68-73%
    - Identifie les "prototypes" du motif (vecteurs de support)
    - Robuste aux features corrélées (noyau RBF)

Limites :
    - Pas de règles explicites (frontière géométrique)
    - Hyperparamètres (gamma, C) sans interprétation directe
    - Lent sur gros corpus

Performance attendue : Precision ~70%, Recall ~65%, F1 ~0.67

STATUT : À IMPLÉMENTER
    Étape 1 : GridSearchCV sur C ∈ [0.1, 1, 10, 100] et gamma ∈ ['scale', 'auto', 0.01, 0.1]
    Étape 2 : Analyser les vecteurs de support (exemples limites)
    Étape 3 : Comparer avec LR (Pipeline 1.3) sur même corpus

Référence : Discussion Claude 8 avr. — Pipeline 2.2
"""

from __future__ import annotations


def train(X_train, y_train, C: float = 1.0, gamma: str = 'scale'):
    """
    Entraîne un SVM à noyau RBF avec calibration des probabilités.

    TODO: Implémenter avec CalibratedClassifierCV pour obtenir des probabilités.
    """
    raise NotImplementedError()


def get_support_vectors(clf, corpus: list, indices: list) -> list[dict]:
    """
    Retourne les khabars correspondant aux vecteurs de support.
    Ces exemples définissent empiriquement les frontières du motif.

    TODO: Implémenter.
    """
    raise NotImplementedError()
