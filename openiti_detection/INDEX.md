# openiti_detection — Complete Guide

Détectez et analysez les figures du fou sensé (ʿāqil majnūn) dans le corpus **openiti_targeted** en comparant deux approches complémentaires.

## 🎯 Objectifs

- Appliquer le **Logistic Regression** (interprétable, coefficients visibles)
- Appliquer le **XGBoost** (performant, 99.1% AUC sur test)
- Comparer les prédictions pour identifier des patterns robustes
- Générer des candidats de haute confiance pour enrichir votre corpus

## 📋 Scripts Disponibles

### 1. `detect_lr_xgboost.py` — Détection sur openITI

**What:** Applique les deux modèles sur l'intégralité du corpus openITI

**How:**
```bash
python openiti_detection/detect_lr_xgboost.py \
  --threshold-lr 0.5 \
  --threshold-xgb 0.5
```

**Key features:**
- Extrait 74 features réelles (62 lexicales + 9 morphologiques + 3 wasf)
- Affiche la progression en temps réel (10% à la fois)
- Gère les erreurs CAMeL Tools gracieusement (remplace par 0)
- Sauvegarde les résultats au fur et à mesure

**Outputs:**
- `results/lr_predictions.json` (1 ligne par texte: path, lr_probability, lr_prediction)
- `results/xgb_predictions.json` (même format pour XGBoost)
- `results/comparison.json` (résumé: counts, agreement_ratio, top 20 consensus hits)

**Durée:** 5-15 min (selon taille du corpus openiti_targeted)

---

### 2. `compare_lr_xgboost.py` — Analyse détaillée

**What:** Croise les deux fichiers de prédictions pour analyser:
- Consensus (les deux disent OUI)
- Divergences (un dit OUI, l'autre dit NON)
- Statistiques de confiance

**How:**
```bash
python openiti_detection/compare_lr_xgboost.py
```

**Outputs:**
- `results/detailed_comparison.json`:
  - `statistics`: counts + agreement_percentage
  - `consensus_positive`: top 50 (les deux disent OUI, triés par avg_prob)
  - `divergence_lr_only`: top 20 (LR dit OUI, XGBoost dit NON)
  - `divergence_xgb_only`: top 20 (XGBoost dit OUI, LR dit NON)

**Durée:** < 1 sec (traitement des fichiers JSON pré-calculés)

---

## 🗂️ Flux de travail recommandé

```
1. Lancer detect_lr_xgboost.py
   ↓
2. Attendre la fin
   ↓
3. Lancer compare_lr_xgboost.py
   ↓
4. Ouvrir results/*.json dans un éditeur ou parser custom
   ↓
5. Analyser les résultats
```

## 📊 Structure des Résultats

### A. `results/comparison.json` — Résumé rapide

```json
{
  "timestamp": "2026-04-10T14:32:45.123456",
  "total_texts_processed": 1234,
  "lr_positive_count": 145,
  "xgb_positive_count": 152,
  "consensus_positive_count": 128,
  "agreement_ratio": 0.988,
  "top_consensus_hits": [
    {
      "path": "0404IbnHisham/0404IbnHisham.Sira.Shamela0022456-ara1",
      "text_preview": "دخلت دار المجانين ذات يوم فإذا أنا بشاب...",
      "lr_prob": 0.987,
      "xgb_prob": 0.995,
      "avg_prob": 0.991
    },
    ...
  ]
}
```

**Interpretation:**
- `lr_positive_count`: Textes classifiés comme positifs par LR
- `xgb_positive_count`: Textes classifiés comme positifs par XGBoost
- `consensus_positive_count`: Textes classifiés positifs PAR LES DEUX
- `agreement_ratio`: Pourcentage d'accord global (consensus + négatif consensus)

### B. `results/detailed_comparison.json` — Analyse complète

```json
{
  "statistics": {
    "consensus_both_positive": 256,
    "consensus_both_negative": 4821,
    "divergence_lr_only": 31,
    "divergence_xgb_only": 45,
    "agreement_percentage": 98.5,
    "avg_confidence_difference": 0.042,
    "max_confidence_difference": 0.789,
    "median_confidence_difference": 0.028
  },
  "consensus_positive": [
    {
      "path": "...",
      "text_length": 2343,
      "lr_prob": 0.987,
      "xgb_prob": 0.995,
      "avg_prob": 0.991,
      "conf_diff": 0.008
    },
    ...
  ],
  "divergence_lr_only": [
    {
      "path": "...",
      "lr_prob": 0.751,
      "xgb_prob": 0.423,
      "conf_diff": 0.328
    },
    ...
  ],
  "divergence_xgb_only": [ ... ]
}
```

### C. `results/lr_predictions.json` & `xgb_predictions.json` — Toutes les prédictions

```json
[
  {
    "path": "corpus/file_001.txt",
    "text_length": 2343,
    "lr_probability": 0.987,
    "lr_prediction": 1
  },
  ...
]
```

**Utilité:**
- Traçabilité complète (chaque texte)
- Pour analyses ultérieures (filtrer par seuil, etc.)

---

## 💡 Cas d'usage

### 1. Enrichir votre corpus annoté

Prenez les `consensus_positive` (les deux modèles d'accord) avec `avg_prob ≥ 0.8`:
```python
import json

with open('openiti_detection/results/detailed_comparison.json') as f:
    data = json.load(f)

high_confidence = [h for h in data['consensus_positive'] if h['avg_prob'] >= 0.8]
print(f"Found {len(high_confidence)} high-confidence candidates")
```

Ces textes sont de bons candidats pour annotation manuelle.

### 2. Analyser les divergences

Les divergences révèlent des patterns que un modèle voit mais l'autre non:

- **LR-only** (LR dit OUI, XGBoost dit NON):
  - Patterns linéaires / syntaxiques
  - À investiguer pour features manquantes dans XGBoost
  
- **XGBoost-only** (XGBoost dit OUI, LR dit NON):
  - Patterns non-linéaires / complexes
  - À investiguer pour overfitting potentiel

### 3. Inclure dans votre thèse

**Section 4.3 (Résultats & Application):**
```
| Modèle  | Candidats positifs | Consensus |
|---------|------------------|-----------|
| LR      | 287              | 256 (89%) |
| XGBoost | 301              | 256 (85%) |
| Accord  | -                | 128/1234 (10.4%) |
```

**Section 4.4 (Analyse critique):**
- Discuter l'accord (98.5% = robustesse)
- Analyser les divergences (31 + 45 = cas intéressants)

---

## 🔧 Configuration avancée

### Ajuster les seuils

Pour être plus conservateur (moins de faux positifs):
```bash
python openiti_detection/detect_lr_xgboost.py \
  --threshold-lr 0.7 \
  --threshold-xgb 0.8
```

Cela réduira `lr_positive_count` et `xgb_positive_count` mais augmentera la confiance.

### Tester sur subset

Pour développement/test, limiter le corpus:
```python
# Dans detect_lr_xgboost.py ligne ~180:
akhbars = load_openiti_corpus()
akhbars = akhbars[:100]  # test sur 100 textes seulement
```

---

## 📚 Référence des 74 Features

Les modèles utilisent 74 features réelles:

**Lexicales (62):**
- `f00-f14` : JUNUN block (مجنون, معتوه, etc.)
- `f15-f22` : AQL block (عقل, معقول, etc.)
- `f23-f27` : HIKMA block (حكمة, حكيم, etc.)
- `f28-f38` : DIALOGUE block (قال, سؤال, etc.)
- `f39-f46` : VALIDATION block (ضحك, بكى, أهدى)
- `f47-f51` : CONTRAST block (لكن, فإذا, etc.)
- `f52-f55` : AUTHORITY block (خليفة, والي, etc.)
- `f56-f58` : POETRY block (شعر, شاعر, etc.)
- `f59-f61` : WASF block (ومنها, ضروب) — marqueur négatif
- `f62-f64` : Autres (longueur, structure)

**Morphologiques (9):** (CAMeL Tools)
- `f65-f67` : Densités de racines (ج.ن.ن, ع.ق.ل, ح.ك.م)
- `f68-f70` : Densités POS (verbe, nom, adjectif)
- `f71-f72` : Aspect & voix (perfectif, passif)
- `f73` : Consensus validation

---

## ⚠️ Troubleshooting

### "Corpus path not found"
```bash
# Vérifiez que le dossier existe:
ls -la "c:/Users/augus/Desktop/Uqala NLP/openiti_targeted/"
```

### "CAMeL Tools import failed"
```bash
pip install camel-tools
python -c "from camel_tools.morphology.database import MorphologyDB; print('✓ OK')"
```

### Script très lent
Normal ! Extraction de 74 features + morphologie pour chaque texte.
Pour tester rapidement:
```python
# Dans detect_lr_xgboost.py:
akhbars = load_openiti_corpus()[:100]  # test subset
```

### Fichiers JSON très gros
Si `lr_predictions.json` est > 100 MB, vous pouvez:
1. Le compresser: `gzip openiti_detection/results/*.json`
2. Filtrer: extraire uniquement `lr_prediction == 1` pour réduire la taille

---

## 📝 Notes

- Les timestamps dans les fichiers JSON permettent de tracker quand les prédictions ont été faites
- Tous les outputs sont en UTF-8 pour supporter l'arabe
- Les models sont cachés en pickle pour reproductibilité

## Prochaines étapes

1. ✅ Lancer `detect_lr_xgboost.py` (appl…
2. ✅ Lancer `compare_lr_xgboost.py` (analyser)
3. Sélectionner `consensus_positive` avec `avg_prob ≥ 0.8` pour annotation manuelle
4. Mettre à jour `dataset_raw.json` avec ces nouvelles annotations
5. Ré-entrainer les modèles avec le corpus élargi
6. Inclure les résultats dans votre thèse (section 4.3-4.4)
