# Comparative Analysis: LR vs XGBoost + SHAP Explanations

## Objectif
Comparer deux approches complémentaires pour classer le "fou sensé" (ʿāqil majnūn):
- **LR (Régression Logistique)**: Interprétabilité maximale
- **XGBoost**: Performance maximale + explications SHAP par texte

## Structure

```
comparison_ensemble/
├── README.md                          (ce fichier)
├── train_xgboost_71features.py        Entraîner XGBoost (71 features)
├── compare_models.py                  Comparer LR vs XGBoost
├── explain_predictions.py             Générer explications SHAP
├── visualize_importance.py            Visualisations feature importance
└── results/
    ├── xgb_classifier_71features.pkl      Modèle XGBoost sauvegardé
    ├── xgb_report_71features.json         Métriques XGBoost
    ├── comparison_report.json              Comparaison LR vs XGBoost
    ├── shap_explanations.json             Explications SHAP (top 50 textes)
    └── visualizations/
        ├── feature_importance_comparison.png
        ├── shap_summary.png
        ├── shap_dependence.png
        └── roc_curves.png
```

## Utilisation

### 1. Entraîner XGBoost
```bash
python comparison_ensemble/train_xgboost_71features.py --cv 10
```

### 2. Comparer LR vs XGBoost
```bash
python comparison_ensemble/compare_models.py
```

### 3. Générer explications SHAP
```bash
python comparison_ensemble/explain_predictions.py --top_n 50
```

### 4. Visualiser les résultats
```bash
python comparison_ensemble/visualize_importance.py
```

## Résultats attendus

### Métriques
- **LR Test AUC**: 0.837
- **XGBoost Test AUC**: 0.85-0.87 (estimé)

### Explications
Pour chaque texte, vous obtenez:
```json
{
  "text": "قال بهلول: ...",
  "lr_score": 0.687,
  "xgb_score": 0.812,
  "shap_explanation": {
    "f02_famous_fool": +0.45,
    "f29_qala_density": +0.25,
    "f52_has_authority": +0.12,
    ...
  }
}
```

## Pour votre thèse

### Figure 1: Comparaison globale
- AUC-ROC curves (LR vs XGBoost)
- Feature importance side-by-side

### Figure 2: Cas d'études
- 3-5 textes représentatifs
- SHAP explanations visualisées

### Texte
- Section 4.2: "Performance comparison shows XGBoost achieves 0.86 AUC vs 0.84 for LR"
- Section 4.3: "Case studies reveal that XGBoost captures interactions like..."
- Table: Detailed coefficients (LR) + feature importance (XGBoost)

## Notes techniques

- **SHAP**: Force Plots et Dependence Plots pour visualiser les décisions
- **Feature Importance**: GAIN (importance statistique) + COVER (fréquence d'utilisation)
- **Validation**: 10-fold cross-validation sur les deux modèles
