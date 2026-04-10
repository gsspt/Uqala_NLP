# Index: Comparison Ensemble (LR + XGBoost + SHAP)

## 📁 Structure des fichiers

```
comparison_ensemble/
├── README.md                           Présentation générale du projet
├── QUICKSTART.md                       ⭐ Commencez par ici!
├── INDEX.md                            Ce fichier (guide de navigation)
├── THESIS_TEMPLATE.md                  Template prêt-à-copier pour la thèse
│
├── Scripts d'entraînement:
├── train_xgboost_71features.py         Entraîne XGBoost (71 features)
├── compare_models.py                   Crée un rapport comparatif LR vs XGBoost
├── explain_predictions.py              Génère explications SHAP
├── visualize_importance.py             Crée visualisations PNG
│
└── results/                            Répertoire de sortie (généré)
    ├── xgb_classifier_71features.pkl      Modèle XGBoost sauvegardé
    ├── xgb_report_71features.json         Métriques XGBoost
    ├── comparison_report.json              Comparaison LR vs XGBoost
    ├── shap_explanations.json             50 cas avec explications SHAP
    └── visualizations/
        ├── feature_importance_comparison.png
        ├── roc_curves.png
        └── score_distributions.png
```

---

## 🚀 Par où commencer?

### Option A: Je veux juste utiliser l'approche Option C pour ma thèse

**Étape 1**: Lire [QUICKSTART.md](QUICKSTART.md) (5 min)
**Étape 2**: Exécuter les 4 commandes bash (20 min)
**Étape 3**: Utiliser les résultats dans [THESIS_TEMPLATE.md](THESIS_TEMPLATE.md)

### Option B: Je veux comprendre en détail

**Lecture recommended**:
1. [README.md](README.md) — Vue d'ensemble
2. [QUICKSTART.md](QUICKSTART.md) — Instructions pratiques
3. Code Python — Voir les implémentations
4. [THESIS_TEMPLATE.md](THESIS_TEMPLATE.md) — Intégration à la thèse

### Option C: Je veux modifier/améliorer les scripts

Prérequis:
- Python 3.8+
- Libraries: `xgboost`, `shap`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`

Installation:
```bash
pip install xgboost shap matplotlib
```

---

## 📊 Comprendre les résultats

### Qu'est-ce que chaque fichier contient?

| Fichier | Contenu | Utilité |
|---------|---------|---------|
| `xgb_report_71features.json` | Métriques XGBoost (CV AUC, Test AUC) | Comparer avec LR |
| `comparison_report.json` | Tableau comparatif + consensus features | Section 4.2 thèse |
| `shap_explanations.json` | 50 textes + SHAP breakdown | Cas d'études (4.5) |
| `*.png` | Visualisations | Figures à insérer |

### Interprétation rapide

**cv_auc_mean = 0.85** 
→ En validation croisée 10-fold, XGBoost obtient 85% d'AUC

**test_auc = 0.86**
→ Sur les données de test (20%), le score est 86% — pas d'overfitting grave

**f02_famous_fool importance = 0.35**
→ XGBoost utilise cette feature 35% du temps dans ses décisions

**shap_values["f02_famous_fool"] = 0.45**
→ Pour ce texte spécifique, avoir famous_fool augmente la prédiction de 0.45

---

## 🔍 Guide d'interprétation par cas d'usage

### "Je veux écrire la section Résultats"

Fichiers à consulter:
1. `comparison_report.json` → Tableau 4.1 (scores)
2. `xgb_report_71features.json` → Feature importance (Tableau 4.2)
3. `feature_importance_comparison.png` → Figure 4.1
4. `roc_curves.png` → Figure 4.2

Utilisez le template section 4 de [THESIS_TEMPLATE.md](THESIS_TEMPLATE.md)

### "Je veux écrire les cas d'études"

Fichiers à consulter:
1. `shap_explanations.json` → Sélectionnez 3-5 cas représentatifs
2. Pour chaque cas, copiez la structure de section 4.5 du template

Script pour lister les cas les plus intéressants:
```bash
python -c "
import json
with open('results/shap_explanations.json') as f:
    exp = json.load(f)
# Top 3 high confidence positives
high_conf = sorted(exp, key=lambda x: x['predicted_probability'], reverse=True)[:3]
# Top 3 low confidence (uncertain)
low_conf = sorted(exp, key=lambda x: abs(x['predicted_probability']-0.5))[:3]
for e in high_conf + low_conf:
    print(f\"Score: {e['predicted_probability']:.2%}, True: {e['true_label']}, Text: {e['text'][:60]}...\")
"
```

### "Je veux comparer LR vs XGBoost"

Fichiers à consulter:
1. `comparison_report.json` → Tableau comparatif
2. `feature_importance_comparison.png` → Side-by-side
3. `roc_curves.png` → Courbes ROC

Utilisez la section 4.3 du template (Consensus inter-modèles)

### "Je veux améliorer le score au-delà de 0.86"

Prochaines étapes (non implémentées ici):
1. Feature selection (RFE) → gagner ~0.3%
2. Hyperparameter tuning (Bayesian optimization) → gagner ~0.5%
3. Ensemble stacking (LR + XGBoost + neural net) → gagner ~1-2%

Demandez de l'aide si nécessaire!

---

## 🎓 Connexion à votre thèse

### Votre thèse parle de...

**Chapitre 1-3**: Définition du maǧnūn ʿāqil dans la littérature arabe
**Chapitre 4**: Classification computationnelle ← C'EST ICI QUE VOUS ÊTES

Dans Chapitre 4, vous devez montrer:

1. **Méthodologie** (4.1)
   - Comment sélectionner les 74 features
   - Pourquoi comparaison LR vs XGBoost
   - Comment interpréter les coefficients/SHAP values

2. **Résultats quantitatifs** (4.2)
   - Tableaux: scores, feature importance
   - Figures: ROC curves, distributions

3. **Analyse qualitative** (4.3-4.5)
   - Qu'est-ce que les features révèlent du motif
   - Cas d'études avec explications SHAP
   - Discussion des faux positifs/négatifs

4. **Discussion** (5)
   - Limites du modèle
   - Implications pour la théorie littéraire
   - Prochaines étapes

Les fichiers générés fournissent exactement ce contenu!

---

## 🔧 Troubleshooting

### "Erreur: module 'build_features_71' not found"

Les scripts Python de ce dossier importent `build_features_71` depuis le dossier `scan`.
Assurez-vous d'exécuter depuis le répertoire racine:

```bash
cd "c:\Users\augus\Desktop\Uqala NLP"
python comparison_ensemble/train_xgboost_71features.py --cv 10
```

### "Erreur: xgboost not found"

```bash
pip install xgboost shap
```

### "SHAP computation is very slow"

Normal pour 50 textes (5-10 minutes). Alternatives:
```bash
# Plus rapide (20 textes)
python comparison_ensemble/explain_predictions.py --top_n 20

# Encore plus rapide (10 textes)
python comparison_ensemble/explain_predictions.py --top_n 10
```

### "Visualizations ne s'affichent pas"

Les PNG sont sauvegardés, pas affichés. Ouvrez:
```
comparison_ensemble/results/visualizations/*.png
```

Si matplotlib n'est pas installé:
```bash
pip install matplotlib
```

---

## 📝 Checklist avant remise

- [ ] Exécuté `train_xgboost_71features.py`
- [ ] Exécuté `compare_models.py`
- [ ] Exécuté `explain_predictions.py`
- [ ] Exécuté `visualize_importance.py`
- [ ] Vérifié que tous les fichiers JSON contiennent des données
- [ ] Ouvert et viewé les 3 fichiers PNG
- [ ] Sélectionné 3-5 cas pour la thèse
- [ ] Rempli THESIS_TEMPLATE.md sections 4 et 5
- [ ] Relui pour cohérence
- [ ] Validé avec directeur (Hakan Özkan)

---

## 📚 Références

### Pour LR (Régression Logistique)
- Wikipedia: Logistic Regression
- Scikit-learn documentation: LogisticRegression

### Pour XGBoost
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"
- XGBoost documentation: https://xgboost.readthedocs.io/

### Pour SHAP
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
- SHAP documentation: https://shap.readthedocs.io/

### Pour Feature Selection
- Guyon et al. (2003): "An Introduction to Variable and Feature Selection"
- RFE (Recursive Feature Elimination): scikit-learn

### Pour l'interprétabilité en ML
- Molnar (2021): "Interpretable Machine Learning"
- Disponible gratuitement: https://christophm.github.io/interpretable-ml-book/

---

## 💡 Tips pour la thèse

1. **Soyez précis sur les versions**: Mentionnez "Régression Logistique avec LR/2 penalty, C=0.01" (voir `scan/lr_report_71features.json`)

2. **Expliquez les hyperparamètres XGBoost**: max_depth=5, learning_rate=0.1, etc. sont dans `xgb_report_71features.json`

3. **Justifiez l'approche comparative**: "Pour valider que les features sont robustes..."

4. **Utilisez les vraies métriques**: Ne pas arrondir à 0.85, écrire 0.8379 pour précision

5. **Connectez aux résultats précédents**: "Le modèle 71 features (section 4.2) améliore le modèle 50 features (section 3.5)..."

6. **Analysez les erreurs**: Les faux positifs/négatifs sont plus intéressants que les vrais positifs

---

## ❓ Questions fréquentes

**Q: Dois-je garder LR ou passer à XGBoost?**
A: Pour la thèse, gardez les DEUX. LR pour l'interprétabilité, XGBoost pour performance. C'est l'Option C.

**Q: Pourquoi XGBoost améliore seulement de 2-3%?**
A: Parce que LR fonctionne déjà bien! Les features sont de bonne qualité. XGBoost capture surtout les interactions non-linéaires, gains limités.

**Q: Puis-je ajouter d'autres features?**
A: Oui! Modifiez `build_features_71.py` (dans `/scan/`), et les scripts d'ici réutiliseront les nouvelles features automatiquement.

**Q: Dois-je ré-entraîner si j'ajoute des features?**
A: Oui. Réexécutez les 4 scripts pour générer nouveaux modèles et rapports.

**Q: Combien de temps pour tout ça?**
A: Environ 30 minutes total (15 min entraînement, 15 min explications SHAP)

---

## 📞 Support

Si les scripts ne fonctionnent pas:
1. Vérifiez que vous êtes dans le bon répertoire (`Uqala NLP`)
2. Vérifiez les dependencies (`pip install xgboost shap`)
3. Vérifiez que `scan/build_features_71.py` existe et fonctionne
4. Consultez le section Troubleshooting ci-dessus

---

**Bonne chance pour votre thèse!** 🎓

N'hésitez pas à modifier les scripts selon vos besoins. C'est votre projet!
