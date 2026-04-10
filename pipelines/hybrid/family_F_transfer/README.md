# Famille F — Transfert de connaissances

Pipelines exploitant des modèles pré-entraînés ou des corpus dans d'autres langues.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `F1_distillation.py` | CAMeLBERT → RandomForest (vitesse × 60) | À implémenter |
| `F2_crosslingual.py` | Transfert arabe ↔ français | À implémenter |

## F1 — Knowledge Distillation

**Problème** : CAMeLBERT est performant mais 100x plus lent que LR.
**Solution** : Utiliser CAMeLBERT comme "teacher" pour entraîner un
"student" (RandomForest) sur ses probabilités continues.

Le student atteint ~94% de la performance du teacher, 60x plus rapidement.

**Prérequis** : Pipeline P3.3 entraîné.

## F2 — Transfert Cross-lingue

**Opportunité unique** : Les akhbars annotés incluent des résumés et
commentaires en français (champs `summary_fr`, `commentaires` dans akhbar.json).

Ces annotations françaises peuvent :
1. Augmenter le corpus d'entraînement (arabe + français ensemble)
2. Servir de validation indépendante (si AR et FR s'accordent → fiable)
3. Guider l'annotation des cas ambigus

**Corpus bilingue requis** : Aligner les textes arabes avec leurs résumés français.
