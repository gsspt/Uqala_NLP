# Famille A — Cascade règles → ML → règles

Pipelines en cascade : pré-filtrage strict → scoring ML → post-filtrage expert.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `A1_conservative.py` | Cascade haute précision (seuil 0.85) | ✅ **En production** |
| `A1_post_filter.py` | Post-filtrage par règles expertes | ✅ En production |
| `A1_strict_analysis.py` | Analyse stricte avec détection fous canoniques | ✅ En production |
| `A2_balanced.py` | Cascade équilibrée (seuil 0.65) | À implémenter |
| `A3_max_recall.py` | Cascade rappel maximal (seuil 0.35) | À implémenter |

## A1 — Pipeline de production actuel

Résultats sur Ibn ʿAbd Rabbih (Al-ʿIqd al-Farīd) :

| Catégorie | Avant post-filtrage | Après post-filtrage |
|-----------|--------------------|--------------------|
| Fous canoniques | 100 | 98 |
| Vrai maǧnūn ʿāqil | 874 | 16 |
| Faux positifs | 3,408 (82%) | 0 (0%) |

**Note critique** : Le post-filtrage est trop restrictif. 874 → 16 vrais positifs
suggère que les maǧnūn implicites (57% du corpus de référence) sont systématiquement
rejetés. Voir `A2_balanced.py` pour une version plus souple.

## Différences entre A1, A2, A3

```
             Précision  Rappel   Cas d'usage
A1 (0.85)    ~90%       ~40%     Corpus de référence, analyse qualitative
A2 (0.65)    ~75%       ~70%     Exploration initiale (recommandé)
A3 (0.35)    ~45%       ~88%     Recensement exhaustif + nettoyage manuel
```
