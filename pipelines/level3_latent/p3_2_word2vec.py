"""
Pipeline 3.2 — Word2Vec + classification sur embeddings moyennés.

Principe : Entraîner Word2Vec sur le corpus OpenITI (~200,000 textes)
pour apprendre des représentations distributionnelles du vocabulaire arabe
classique. Chaque khabar est représenté par la moyenne des embeddings
de ses mots (vecteur de 200 dimensions).

Avantage principal : Capture la similarité sémantique, pas juste lexicale.
    "معتوه" ≈ "مجنون" même sans co-occurrence directe.

Avantages :
    - Capture similarité sémantique (pas juste lexicale)
    - Précision ~65-72%
    - Représentation dense (200 dims vs 5000 dims TF-IDF)

Limites :
    - Embeddings = boîte noire (impossible de savoir pourquoi "جنون" ≈ "عشق")
    - La moyenne perd l'ordre syntaxique
    - Nécessite corpus non-annoté pour l'entraînement

Performance attendue : Precision ~68%, Recall ~65%, F1 ~0.66

STATUT : À IMPLÉMENTER
    Étape 1 : Entraîner Word2Vec sur OpenITI complet (~6h)
    Étape 2 : Sauvegarder modèle dans models/word2vec_openiti.bin
    Étape 3 : Représenter chaque khabar par moyenne embeddings
    Étape 4 : Entraîner LogisticRegression sur embeddings

Référence : Discussion Claude 8 avr. — Pipeline 3.2
"""

from __future__ import annotations


def train_word2vec(corpus_texts: list[str], vector_size: int = 200,
                   window: int = 5, min_count: int = 5, epochs: int = 10):
    """
    Entraîne Word2Vec sur le corpus OpenITI complet.

    Args:
        corpus_texts: Liste de textes arabes (un texte = un khabar)
        vector_size: Dimension des embeddings
        window: Fenêtre contextuelle
        min_count: Fréquence minimale pour inclure un mot
        epochs: Nombre de passages sur le corpus

    TODO: Utiliser gensim.models.Word2Vec.
          Sauvegarder dans models/word2vec_openiti.bin.
    """
    raise NotImplementedError()


def khabar_to_vector(text: str, w2v_model) -> 'np.ndarray':
    """
    Représente un khabar par la moyenne de ses embeddings de mots.

    TODO: Implémenter. Gérer les mots hors vocabulaire (retourner zeros).
    """
    raise NotImplementedError()
