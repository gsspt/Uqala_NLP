# Pipelines de détection du maǧnūn ʿāqil

Ce dossier contient l'ensemble des pipelines de détection, organisés du plus
au moins interprétable. Chaque pipeline répond à une question épistémologique
différente sur la détectabilité computationnelle du genre littéraire.

## Taxonomie des pipelines

```
pipelines/
├── level1_interpretable/       Interprétabilité maximale
├── level2_semi_interpretable/  Interprétabilité moyenne
├── level3_latent/              Représentations latentes (boîte noire partielle)
├── level4_llm/                 LLM — interprétabilité nulle
└── hybrid/                     Pipelines hybrides (combinaisons)
    ├── family_A_cascade/       Cascade règles → ML → règles
    ├── family_B_multistage/    Multi-étages spécialisés
    ├── family_C_iterative/     Apprentissage itératif
    ├── family_D_human_loop/    Humain dans la boucle
    ├── family_E_ensemble/      Fusion multi-modèles
    └── family_F_transfer/      Transfert de connaissances
```

## Tableau comparatif

| Pipeline | Précision | Rappel | F1 | Interprétabilité | Statut |
|----------|-----------|--------|----|-----------------|--------|
| **P1.1** Règles booléennes | ~30% | ~15% | 0.20 | ★★★★★ | À implémenter |
| **P1.2** Arbre de décision | ~55% | ~50% | 0.52 | ★★★★★ | À implémenter |
| **P1.3** Régression logistique | ~75% | ~68% | 0.71 | ★★★★☆ | ✅ **En production** |
| **P2.1** Random Forest + SHAP | ~78% | ~72% | 0.75 | ★★★★☆ | ✅ Entraîné |
| **P2.2** SVM noyau RBF | ~70% | ~65% | 0.67 | ★★★☆☆ | À implémenter |
| **P3.1** TF-IDF + k-NN | ~63% | ~58% | 0.60 | ★★☆☆☆ | À implémenter |
| **P3.2** Word2Vec | ~68% | ~65% | 0.66 | ★★☆☆☆ | À implémenter |
| **P3.3** CAMeLBERT | ~78% | ~72% | 0.75 | ★☆☆☆☆ | À implémenter (GPU) |
| **P4.1** Few-shot GPT-4 | ~85% | ~75% | 0.80 | ☆☆☆☆☆ | Partiel |
| **P4.2** DeepSeek annotation | ~80% | ~70% | 0.75 | ☆☆☆☆☆ | À implémenter |
| **A1** Cascade conservatrice | ~90% | ~40% | 0.56 | ★★★★☆ | ✅ **En production** |
| **A2** Cascade équilibrée | ~75% | ~70% | 0.72 | ★★★★☆ | À implémenter |
| **A3** Cascade rappel max | ~45% | ~88% | 0.59 | ★★★☆☆ | À implémenter |
| **B1** Two-stage cascade | ~82% | ~65% | 0.73 | ★★★☆☆ | À implémenter |
| **B2** Specialist ensemble | ~80% | ~72% | 0.76 | ★★★☆☆ | À implémenter |
| **C1** Active learning | ~85% | ~72% | 0.78 | ★★★★☆ | **PRIORITÉ HAUTE** |
| **C2** Self-training | ~72% | ~68% | 0.70 | ★★★☆☆ | À implémenter |
| **D1** Human-in-the-loop | ~95% | ~60% | 0.74 | ★★★★★ | Partiel |
| **D2** LLM-assisted | ~88% | ~65% | 0.75 | ★★★☆☆ | Partiel |
| **E1** Stacking | ~83% | ~74% | 0.78 | ★★☆☆☆ | Partiel |
| **E2** Mixture of experts | ~79% | ~71% | 0.75 | ★★★☆☆ | À implémenter |
| **F1** Distillation | ~73% | ~68% | 0.70 | ★★☆☆☆ | À implémenter |
| **F2** Cross-lingual | ~70% | ~65% | 0.67 | ★★☆☆☆ | À implémenter |

*Performances attendues (estimations théoriques — à valider empiriquement)*

## Recommandation pour la thèse

**Pipeline optimal** : `C1_active_learning` + features actantielles + SHAP

Séquence recommandée :
1. **A1** (production) → corpus de candidats initiaux
2. **C1** (active learning) → amélioration ciblée du modèle
3. **D1** (human-in-the-loop) → validation finale avec traçabilité SHAP

## Choix épistémologique

> « L'objectif n'est pas que l'algorithme sache ce qu'est un maǧnūn ʿāqil,
> mais qu'il me fasse gagner 80% du temps de lecture en pré-filtrant le corpus,
> tout en gardant le contrôle interprétatif. »
>
> — Discussion avec Claude, 8 avril 2026
