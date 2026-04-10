# Famille C — Pipelines itératifs

Pipelines qui améliorent le modèle par cycles d'apprentissage,
en exploitant le feedback humain ou les prédictions confiantes.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `C1_active_learning.py` | Active learning (cas incertains → annotation) | **PRIORITÉ HAUTE** |
| `C2_self_training.py` | Self-training (pseudo-labels confiants) | À implémenter |

## C1 — Active Learning : le pipeline recommandé pour la thèse

### Principe

Au lieu d'annoter 10,000 khabars aléatoirement (épuisant, redondant),
l'algorithme sélectionne les 50 exemples sur lesquels il est le **moins certain**.
Ces cas ambigus sont les plus informatifs pour améliorer le modèle.

### Planning (10 cycles × 2h = 20h)

```
Semaine 1 : Préparation
  ├── Constituer corpus négatif (data/negatives/)
  ├── Entraîner baseline (Pipeline A1)
  └── Itération 0 : évaluation initiale

Semaines 2-3 : Cycles 1-10 (2h/jour)
  ├── Cycle 1 : Annoter 50 cas → réentraîner
  ├── Cycle 2 : Annoter 50 cas → réentraîner
  └── ...

Semaine 4 : Analyse finale
  ├── Courbe d'apprentissage
  ├── Identification des sous-types ambigus
  └── Corpus final de ~430 khabars validés
```

### Gain attendu

```
Annotation aléatoire (500 textes) → F1 ≈ 0.70
Active learning (500 textes ciblés) → F1 ≈ 0.85
Gain : +15% F1 pour même effort
```

### Contribution à la thèse

Les cas incertains révèlent les **frontières conceptuelles** du motif :
- Quels khabars sont ambigus ? → Sous-catégories du maǧnūn ʿāqil
- Sur quoi le modèle hésite-t-il ? → Hypothèses falsifiables sur le genre

## C2 — Self-training

Complémentaire à C1 : exploite les 200,000 textes non-annotés en ajoutant
les prédictions très confiantes (prob > 0.95) comme pseudo-labels.
À utiliser après 5 cycles de C1 pour étendre le corpus d'entraînement.
