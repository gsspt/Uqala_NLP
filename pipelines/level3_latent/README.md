# Niveau 3 — Représentations latentes

Modèles capturant la sémantique profonde, mais dont les décisions
ne sont plus directement justifiables par des règles explicites.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `p3_1_tfidf_knn.py` | TF-IDF + k plus proches voisins | À implémenter |
| `p3_2_word2vec.py` | Word2Vec + classification | À implémenter |
| `p3_3_camelbert.py` | CAMeLBERT fine-tuning | À implémenter (GPU requis) |

## Comparaison des approches

### TF-IDF + k-NN (P3.1)
L'interprétabilité reste partielle : "Ce texte est positif car il ressemble
aux khabars #45, #102, #234." C'est une justification par analogie, pas par règles.

### Word2Vec (P3.2)
Entraîner Word2Vec sur OpenITI complet (~200,000 textes) permettrait de
découvrir des équivalences sémantiques non anticipées (معتوه ≈ مجنون ≈ أحمق
dans certains contextes narratifs).

Modèle à entraîner : `models/word2vec_openiti.bin` (~400 MB, non versionné)

### CAMeLBERT (P3.3)
Modèle pré-entraîné recommandé : `CAMeL-Lab/bert-base-arabic-camelbert-mix`

⚠️ **Note épistémologique** : Ce pipeline est le MOINS adapté pour une thèse
de philologie (boîte noire totale). À utiliser uniquement comme borne supérieure
de performance dans la comparaison des pipelines.

## Prérequis communs
- `pip install gensim` (Word2Vec)
- `pip install transformers torch` (CAMeLBERT)
- GPU recommandé pour P3.3 (entraînement ~2-4h)
