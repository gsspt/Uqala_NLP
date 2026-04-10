# openiti_detection — Détecter le fou sensé dans openITI

Comparez les approches **LR (interprétable)** et **XGBoost (performante)** pour détecter la figure du fou sensé (ʿāqil majnūn) dans le corpus openITI.

## Setup Rapide

```bash
cd "c:\Users\augus\Desktop\Uqala NLP"
```

## Étape 1 : Appliquer les deux modèles

Lance les deux classifiers (LR et XGBoost) sur le corpus **openiti_targeted**.

```bash
python openiti_detection/detect_lr_xgboost.py --threshold-lr 0.5 --threshold-xgb 0.5
```

**Options :**
- `--threshold-lr` : Seuil de décision pour LR (défaut: 0.5)
- `--threshold-xgb` : Seuil de décision pour XGBoost (défaut: 0.5)

**Durée estimée :** Dépend de la taille du corpus openITI (5-30 min selon votre machine)

**Sorties :**
- `openiti_detection/results/lr_predictions.json` — Toutes les prédictions LR
- `openiti_detection/results/xgb_predictions.json` — Toutes les prédictions XGBoost
- `openiti_detection/results/comparison.json` — Résumé + top 20 consensus hits

## Étape 2 : Analyser les divergences

Comparez en détail où les modèles sont d'accord ou en désaccord.

```bash
python openiti_detection/compare_lr_xgboost.py
```

**Sorties :**
- `openiti_detection/results/detailed_comparison.json` — Analyse détaillée:
  - Consensus (les deux disent OUI)
  - Divergences (LR dit oui, XGBoost dit non)
  - Divergences (XGBoost dit oui, LR dit non)
  - Statistiques de confiance

## Résultats

Après les deux étapes, explorez les fichiers JSON dans `openiti_detection/results/`:

### 1. `comparison.json` — Vue rapide
```json
{
  "total_texts_processed": 1234,
  "lr_positive_count": 145,
  "xgb_positive_count": 152,
  "consensus_positive_count": 128,  ← Les deux models d'accord
  "agreement_ratio": 0.988,
  "top_consensus_hits": [
    {
      "path": "corpus/text1.txt",
      "lr_prob": 0.987,
      "xgb_prob": 0.995,
      "avg_prob": 0.991
    },
    ...
  ]
}
```

### 2. `detailed_comparison.json` — Analyse approfondie
```json
{
  "statistics": {
    "consensus_both_positive": 256,
    "consensus_both_negative": 4821,
    "divergence_lr_only": 31,
    "divergence_xgb_only": 45,
    "agreement_percentage": 98.5,
    "avg_confidence_difference": 0.042
  },
  "consensus_positive": [ ... ],  ← Top 50 (les deux disent OUI)
  "divergence_lr_only": [ ... ],   ← LR dit OUI, XGBoost dit NON
  "divergence_xgb_only": [ ... ]   ← XGBoost dit OUI, LR dit NON
}
```

### 3. `lr_predictions.json` et `xgb_predictions.json`
Contiennent TOUTES les prédictions pour chaque texte, utilisables pour analyse ultérieure.

## Interprétation des résultats

### Consensus positif (les deux disent OUI)
C'est le plus fiable. Ces textes:
- Ont des patterns forts de "fou sensé" reconnus par les deux modèles
- Sont de bons candidats pour l'inclusion dans votre corpus annoté

### Divergences (LR dit OUI, XGBoost dit NON)
Textes où LR détecte un pattern que XGBoost n'a pas capturé:
- Peut indiquer des patterns syntaxiques subtils (force de LR)
- Peut indiquer des faux positifs (si LR surestime)
- Bon pour investigation manuelle

### Divergences (XGBoost dit OUI, LR dit NON)
Textes où XGBoost détecte un pattern que LR n'a pas capturé:
- Peut indiquer des patterns complexes/non-linéaires (force de XGBoost)
- Peut indiquer une certaine overfitting de XGBoost
- Bon pour investigation manuelle

## Pour votre thèse

Utilisez ces résultats pour:

1. **Affiner votre corpus annoté** : Sélectionnez les texts du consensus positif comme nouveaux candidats
2. **Analyser les divergences** : Étudiez les cas où les modèles ne s'accordent pas pour comprendre les patterns manquants
3. **Illustrer la comparaison LR/XGBoost** : Incluez des statistiques de agreement dans votre section "Résultats"

## Troubleshooting

### "Corpus path not found"
Vérifiez que le dossier `openiti_targeted/` existe et contient des fichiers texte.

### Script lent
C'est normal — extraction de 74 features + morphologie par CAMeL Tools pour chaque texte prend du temps.
Pour tester rapidement, éditez `detect_lr_xgboost.py` et limitez le corpus:
```python
akhbars = akhbars[:100]  # test sur 100 textes seulement
```

### Erreur CAMeL Tools
Si vous recevez une erreur lors de l'import de CAMeL Tools:
```bash
pip install camel-tools
```

## Fichier Structure

```
openiti_detection/
├── detect_lr_xgboost.py          ← Script principal (LR + XGBoost)
├── compare_lr_xgboost.py         ← Script d'analyse détaillée
├── QUICKSTART.md                 ← Ce fichier
└── results/
    ├── lr_predictions.json       ← Toutes prédictions LR
    ├── xgb_predictions.json      ← Toutes prédictions XGBoost
    ├── comparison.json           ← Résumé + top hits
    └── detailed_comparison.json  ← Analyse approfondie
```
