# Modèles entraînés

Ce dossier contient les artefacts ML versionnés (modèles légers < 1 MB)
et les rapports d'évaluation.

## Modèles versionnés

| Fichier | Algorithme | Features | AUC (CV) | Taille |
|---------|-----------|----------|----------|--------|
| `lr_classifier_71features.pkl` | Logistic Regression | 71 | 0.838 | 3.3 KB |
| `lr_classifier_50features.pkl` | Logistic Regression | 50 | — | 2.9 KB |
| `xgb_classifier_71features.pkl` | XGBoost | 71 | 0.846 | 220 KB |

## Rapports

| Fichier | Contenu |
|---------|---------|
| `lr_report_71features.json` | Métriques LR (AUC, F1, matrice confusion, coefficients) |
| `lr_report_50features.json` | Métriques LR legacy |
| `xgb_report_71features.json` | Métriques XGBoost |
| `comparison_report.json` | Comparaison LR vs XGBoost |
| `shap_explanations.json` | Valeurs SHAP sur corpus de test |

## Lexiques

| Fichier | Contenu |
|---------|---------|
| `actantial_lexicons.json` | Lexiques enrichis (ج-ن-ن 36 formes, ع-ق-ل 25 formes...) |

## Modèles non-versionnés (trop volumineux)

Créer avec les pipelines correspondants :
```
models/
├── word2vec_openiti.bin          Pipeline P3.2 (~400 MB)
├── camelbert_finetuned/          Pipeline P3.3 (~400 MB)
└── active_learning/
    ├── cycle_01_model.pkl        Pipeline C1 (généré automatiquement)
    ├── cycle_02_model.pkl
    └── ...
```

## Note sur le surapprentissage XGBoost

Le modèle `xgb_classifier_71features.pkl` présente un surapprentissage sévère :
- AUC validation croisée : **0.846**
- AUC test set : **0.991** ← ne pas utiliser pour comparer avec d'autres pipelines

Utiliser AUC CV = 0.846 dans toutes les comparaisons.
Pour la généralisation inter-corpus, préférer `lr_classifier_71features.pkl`.
