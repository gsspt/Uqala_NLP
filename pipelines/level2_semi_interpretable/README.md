# Niveau 2 — Interprétabilité moyenne

Modèles plus performants, dont les décisions individuelles sont approximables
via des méthodes post-hoc (SHAP, vecteurs de support).

| Fichier | Description | Statut |
|---------|-------------|--------|
| `p2_1_random_forest_shap.py` | XGBoost + analyse SHAP | ✅ Entraîné |
| `p2_2_svm.py` | SVM noyau RBF + vecteurs de support | À implémenter |

## P2.1 — XGBoost + SHAP

**Performance** : AUC CV = 0.846, AUC test = 0.991 (⚠️ surapprentissage)

**Attention** : L'AUC test = 0.991 indique un surapprentissage sévère.
Utiliser la validation croisée (AUC = 0.846) pour les comparaisons.

**Modèle sauvegardé** : `models/xgb_classifier_71features.pkl`

Le fichier `p2_1_random_forest_shap.py` contient aussi les outils
d'explication SHAP — voir `family_D/D1_explainable_loop.py` pour
l'intégration dans un workflow de validation humaine.

## P2.2 — SVM

Intérêt particulier : Les vecteurs de support identifient les khabars
qui définissent empiriquement les frontières du motif maǧnūn ʿāqil.
Ce sont les exemples "prototypiques" — utiles comme corpus de référence.
