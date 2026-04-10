"""
Pipeline 3.1 — TF-IDF + k-NN (similarité lexicale).

Principe : Représenter chaque khabar par ses n-grammes TF-IDF pondérés,
puis classer par proximité cosinus avec les 5 voisins les plus proches.

Intérêt : Décision justifiable par analogie :
    "Ce texte est positif car il ressemble aux khabars #45, #102, #234"

Avantages :
    - Précision ~60-70%
    - Décision justifiable par analogie (lazy learning)
    - Pas d'entraînement (applicable immédiatement)

Limites :
    - Sensible au lexique de surface (rate les paraphrases)
    - TF-IDF insensible à l'ordre des mots
    - Mémoire O(n × vocab_size)

Performance attendue : Precision ~63%, Recall ~58%, F1 ~0.60

STATUT : À IMPLÉMENTER
    Étape 1 : Tokeniser avec Farasa/CAMeL Tools (racinisation)
    Étape 2 : TfidfVectorizer(ngram_range=(1,3), max_features=5000)
    Étape 3 : KNeighborsClassifier(n_neighbors=5, metric='cosine')
    Étape 4 : Pour chaque prédiction, afficher les 5 voisins

Référence : Discussion Claude 8 avr. — Pipeline 3.1
"""

from __future__ import annotations


def build_tfidf_index(corpus: list[str]):
    """
    Construit l'index TF-IDF sur le corpus.

    TODO: Utiliser sklearn.feature_extraction.text.TfidfVectorizer
          avec tokenizer arabe (Farasa ou split simple).
    """
    raise NotImplementedError()


def classify_with_explanation(text: str, tfidf_index, corpus: list[dict],
                               k: int = 5) -> dict:
    """
    Classe un texte et retourne ses k voisins les plus proches.

    Returns:
        {
            'label': int,
            'confidence': float,
            'neighbors': [{'khabar': dict, 'similarity': float}]
        }

    TODO: Implémenter.
    """
    raise NotImplementedError()
