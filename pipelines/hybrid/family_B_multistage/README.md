# Famille B — Multi-étages spécialisés

Pipelines en deux étapes où chaque étape est optimisée pour un objectif différent.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `B1_two_stage_cascade.py` | Filtre rapide (LR) → classifier précis (CAMeLBERT) | À implémenter |
| `B2_specialist_ensemble.py` | Ensemble de 3 experts selon le type de texte | À implémenter |

## B1 — Two-Stage Cascade

**Motivation économique** : CAMeLBERT est 100x plus lent que LR.
Appliquer LR en première étape (seuil bas = 0.30) pour réduire le corpus
de 200,000 → 20,000 textes avant d'appliquer CAMeLBERT.

```
200,000 textes → LR (seuil 0.30) → 20,000 candidats → CAMeLBERT → 500 positifs
Temps total : ~2h (vs ~20h pour CAMeLBERT seul)
```

## B2 — Specialist Ensemble

**Motivation philologique** : Les sous-genres du maǧnūn ʿāqil ont
des signatures différentes :
- Khabar court (< 150 mots) : maxime paradoxale → features lexicales suffisent
- Khabar long (> 150 mots, prose) : récit dialogué → features structurelles importantes
- Khabar avec poésie : citation poétique dans récit → features rhétoriques

Chaque spécialiste est optimisé pour son sous-type.
