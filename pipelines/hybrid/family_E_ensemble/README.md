# Famille E — Fusion multi-modèles

Pipelines combinant plusieurs modèles pour maximiser la performance.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `E1_stacking.py` | Stacking avec méta-modèle calibré | Partiel |
| `E2_mixture_experts.py` | Mixture of Experts avec réseau de routage | À implémenter |

## E1 — Stacking

Architecture en deux niveaux :
- **Niveau 1** : 5 modèles de base (LR, RF, XGBoost, SVM, k-NN) entraînés
  sur les mêmes features
- **Niveau 2** : Méta-modèle (LR calibrée) apprend à combiner les prédictions

Le méta-modèle apprend :
- "Quand LR et RF s'accordent → très fiable"
- "Quand SVM diverge → ignorer SVM sur ce type de texte"

**Performance attendue** : F1 ~0.78 (meilleur que chaque modèle seul)

## E2 — Mixture of Experts

Différence avec B2 : E2 route selon la **difficulté** du cas (pas le type de texte).
- Cas faciles (prob > 0.8 ou < 0.2) → Expert rapide (LR)
- Cas ambigus (prob 0.3-0.7) → Expert précis (CAMeLBERT)
- Cas avec poésie → Expert rhétorique
