# Pipelines hybrides

Les pipelines hybrides combinent plusieurs approches pour équilibrer
performance, interprétabilité et effort d'annotation.

## Vue d'ensemble

| Famille | Principe | Pipeline phare | Statut |
|---------|----------|---------------|--------|
| **A — Cascade** | Règles → ML → Règles | A1 (production) | ✅ En production |
| **B — Multi-étages** | Rapide puis précis | B1 (two-stage) | À implémenter |
| **C — Itératif** | Active learning | C1 (recommandé) | Priorité haute |
| **D — Humain** | Human-in-the-loop | D1 (SHAP) | Partiel |
| **E — Ensemble** | Vote multi-modèles | E1 (stacking) | Partiel |
| **F — Transfert** | Distillation / cross-lingue | F1 | À implémenter |

## Pipeline optimal pour la thèse

```
Phase 1 : A1 (production)
    → Pré-filtrage strict (rappel ~40%, précision ~90%)
    → Corpus de candidats : ~200 khabars validés

Phase 2 : C1 (active learning, 10 cycles)
    → Amélioration ciblée sur les cas incertains
    → +15% F1, découverte des sous-types ambigus

Phase 3 : D1 (human-in-the-loop, validation finale)
    → Annotation avec traçabilité SHAP
    → Corpus final : ~430 khabars avec justifications
```

Ce pipeline est **épistémologiquement solide** pour une thèse :
- Chaque décision est traçable et justifiable
- L'incertitude du modèle est documentée
- Le philologue garde le contrôle interprétatif final
