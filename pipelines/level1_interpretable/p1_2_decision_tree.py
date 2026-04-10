"""
Pipeline 1.2 — Arbre de décision entraîné sur features manuelles.

Principe : Extraire 20-30 features linguistiques interprétables,
puis entraîner un arbre de décision peu profond (max_depth=5).

La force de cet approche est que les règles apprisespeuvent être lues
comme des hypothèses philologiques explicites :

    |--- poetry_ratio <= 0.3
    |   |--- paradox_pairs > 0
    |   |   |--- has_isnad == True
    |   |   |   |--- class: POSITIF

Avantages :
    - Décision explicable : "positif car poetry_ratio=0.2 ET paradox_pairs=1"
    - Permet d'identifier les features les plus discriminantes
    - Modifiable (ajuster max_depth pour plus/moins de complexité)

Limites :
    - Précision ~50-60% max (surapprentissage sur lexique)
    - Nécessite corpus négatif équilibré

Performance attendue : Precision ~55%, Recall ~50%, F1 ~0.52

STATUT : À IMPLÉMENTER
    Étape 1 : Extraire features sur corpus positif + négatif
    Étape 2 : GridSearchCV sur max_depth (3, 5, 7, 10)
    Étape 3 : Visualiser l'arbre (export_text)
    Étape 4 : Valider philologiquement les règles apprises

Référence : Discussion Claude 8 avr. — Pipeline 1.2
"""

from __future__ import annotations


def train(X_train, y_train, max_depth: int = 5, min_samples_leaf: int = 20):
    """
    Entraîne un arbre de décision sur les features manuelles.

    Args:
        X_train: Matrice de features (n_samples, n_features)
        y_train: Labels binaires (1 = maǧnūn ʿāqil, 0 = négatif)
        max_depth: Profondeur maximale de l'arbre
        min_samples_leaf: Minimum d'exemples par feuille

    Returns:
        Arbre de décision entraîné + texte des règles apprises

    TODO: Implémenter.
    """
    raise NotImplementedError(
        "Nécessite corpus négatif constitué. "
        "Voir data/negatives/README.md pour les instructions."
    )


def print_rules(clf, feature_names: list[str]) -> str:
    """
    Affiche les règles de l'arbre sous forme humainement lisible.

    TODO: Utiliser sklearn.tree.export_text.
    """
    raise NotImplementedError()
