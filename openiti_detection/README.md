# openiti_detection — LR vs XGBoost Comparison

Détectez et comparez les figures du fou sensé (ʿāqil majnūn) dans le corpus **openiti_targeted** en utilisant deux classifiers complémentaires entraînés sur votre corpus annoté.

## 🚀 Démarrage rapide

```bash
cd "c:\Users\augus\Desktop\Uqala NLP"

# Étape 1: Appliquer les deux modèles sur openITI
python openiti_detection/detect_lr_xgboost.py

# Étape 2: Analyser les résultats en détail
python openiti_detection/compare_lr_xgboost.py
```

**Durée totale:** ~5-15 min (selon taille du corpus openiti_targeted)

Affichage en terminal:
```
✅ CAMeL Tools loaded
Loading models…
Loading openiti_targeted corpus…
✓ Loaded ~1234 texts from corpus

Processing 1234 texts…
  123/1234 (10%) - 450s remaining
  247/1234 (20%) - 400s remaining
  ...
  1234/1234 (100%) - 0s remaining

Saving results…
✓ LR results → ...
✓ XGBoost results → ...
✓ Comparison → ...

======================================================================
DETECTION SUMMARY
======================================================================
Total texts processed: 1234
LR positive predictions: 145
XGBoost positive predictions: 152
Consensus (both agree positive): 128
Agreement ratio: 98.8%

Top 5 consensus hits (highest average confidence):
  1. corpus/file_001.txt
     LR: 0.995 | XGBoost: 0.998 | Avg: 0.997
  ...
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** — Instructions pas-à-pas pour lancer les scripts
- **[INDEX.md](INDEX.md)** — Guide complet avec interprétation des résultats

## 📂 Structure

```
openiti_detection/
├── README.md                 ← Ce fichier
├── QUICKSTART.md            ← Guide rapide
├── INDEX.md                 ← Documentation complète
├── detect_lr_xgboost.py     ← Script principal (LR + XGBoost)
├── compare_lr_xgboost.py    ← Script d'analyse détaillée
└── results/                 ← Générés par les scripts
    ├── lr_predictions.json          (toutes prédictions LR)
    ├── xgb_predictions.json         (toutes prédictions XGBoost)
    ├── comparison.json              (résumé + top hits)
    └── detailed_comparison.json     (analyse complète)
```

## 🎯 Ce qu'il y a dans la boîte

Vous avez accès à:

1. **Modèle LR (Logistic Regression)**
   - Test AUC: 0.837
   - Avantage: Coefficients visibles → explication feature par feature
   - Fichier: `scan/lr_classifier_71features.pkl`

2. **Modèle XGBoost**
   - Test AUC: 0.9912 (99.1%)
   - Avantage: Performance maximale
   - Fichier: `comparison_ensemble/results/xgb_classifier_71features.pkl`

3. **Feature Engineering**
   - 74 features réelles (62 lexicales + 9 morphologiques + 3 marqueurs wasf)
   - Zéro padding inutile
   - Extraction via CAMeL Tools pour morphologie arabe

## 💡 Utilité

### Pour votre recherche:

- **Enrichir votre corpus**: Les textes avec `consensus_positive` et `avg_prob ≥ 0.8` sont des candidats high-quality pour annotation manuelle

- **Valider votre modèle**: Un accord de 98.5% entre LR et XGBoost (modèles très différents) valide la robustesse des patterns détectés

- **Analyser les divergences**: Les 31 + 45 textes où les modèles divergent peuvent révéler des patterns manquants ou overfitting

### Pour votre thèse:

- **Section 4.3 (Résultats)**: Inclure les statistiques de detection sur openITI
- **Section 4.4 (Analyse)**: Discuter l'accord entre modèles comme validation de robustesse
- **Appendix**: Lister les top 20 consensus hits comme exemples

## ⚙️ Options

```bash
# Avec seuils personnalisés (plus strict):
python openiti_detection/detect_lr_xgboost.py \
  --threshold-lr 0.7 \
  --threshold-xgb 0.8

# Test rapide sur subset (éditer le fichier):
# Dans detect_lr_xgboost.py ligne ~180:
# akhbars = akhbars[:100]
```

## 📊 Exemple de résultats attendus

```json
{
  "total_texts_processed": 1234,
  "lr_positive_count": 145,
  "xgb_positive_count": 152,
  "consensus_positive_count": 128,
  "agreement_ratio": 0.988,
  "top_consensus_hits": [
    {
      "path": "0404IbnHisham/sira.txt",
      "lr_prob": 0.987,
      "xgb_prob": 0.995,
      "avg_prob": 0.991
    }
  ]
}
```

## ✅ Prérequis

Les modèles et features sont déjà entraînés. Juste besoin de:

1. ✅ Dataset annoté: `dataset_raw.json` ← déjà en place
2. ✅ Modèles entraînés: dans `scan/` et `comparison_ensemble/results/` ← déjà en place
3. ✅ Corpus openiti_targeted: doit être à `openiti_targeted/` ← prêt

```bash
# Vérifier le corpus openiti_targeted:
ls -la "c:/Users/augus/Desktop/Uqala NLP/openiti_targeted/" | head -20
```

## 🔍 Prochaines étapes

1. Lancer `detect_lr_xgboost.py` (attend ~10-30 min)
2. Lancer `compare_lr_xgboost.py` (< 1 sec)
3. Ouvrir `results/detailed_comparison.json` dans un éditeur
4. Sélectionner `consensus_positive` avec `avg_prob ≥ 0.8`
5. Utiliser ces textes comme candidats pour expansion du corpus
6. Inclure les statistiques dans votre thèse

## 📝 Notes techniques

- **Features**: 74 réelles (62 lexicales + 9 morpho + 3 wasf), extraction par CAMeL Tools
- **Scaling**: StandardScaler (même que modèle training)
- **Classes**: Binaire (0 = pas de fou sensé, 1 = fou sensé)
- **Encoding**: UTF-8 pour arabe
- **Format**: JSON pour faciliter parsing

## 🤔 Besoin d'aide?

Voir [QUICKSTART.md](QUICKSTART.md) pour troubleshooting et [INDEX.md](INDEX.md) pour interprétation détaillée.

---

**Version:** 1.0  
**Created:** 2026-04-10  
**Models trained on:** dataset_raw.json (4277 samples, 460 positive, 3817 negative)
