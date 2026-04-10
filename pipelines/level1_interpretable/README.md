# Niveau 1 — Interprétabilité maximale

Pipelines dont chaque décision est traçable et philologiquement justifiable.
Aucune boîte noire : toutes les règles peuvent être lues comme des hypothèses
sur le motif maǧnūn ʿāqil.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `p1_1_rules.py` | Règles booléennes (regex + seuils manuels) | À implémenter |
| `p1_2_decision_tree.py` | Arbre de décision (max_depth=5) | À implémenter |
| `p1_3_logistic_regression.py` | Régression logistique — **pipeline de référence** | ✅ En production |
| `p1_3_logistic_regression_v50.py` | Version legacy (50 features) | Archive |

## Pipeline de référence : Régression logistique (P1.3)

Le modèle actuel est une régression logistique avec 71 features
(62 lexicales + 9 morphologiques), entraînée sur 1,221 textes annotés.

**Coefficients clés** (valeurs appriseset interprétation) :
- `famous_fool` : +0.663 — La présence d'un nom de fou canonique est le signal le plus fort
- `verb_density` : +0.420 — La prose narrative (beaucoup de verbes) est caractéristique
- `qala_density` : +0.331 — La structure dialogique est centrale au motif
- `has_authority` : -0.266 — ⚠️ Signal négatif (voir note ci-dessous)
- `wasf_density` : -0.168 — Les textes définitionnels ne sont pas du maǧnūn ʿāqil

> **Note sur `has_authority`** : Ce coefficient négatif est contre-intuitif
> (la scène canonique implique souvent un calife). Il suggère que la feature
> actuelle capte aussi les chroniques politiques (faux positifs).
> À corriger : créer `authority_with_junun` (positif) vs `authority_alone` (négatif).

## Corpus d'entraînement
- 460 positifs (Kitāb ʿUqalāʾ al-Maǧānīn, Nīsābūrī, akhbars 161-612)
- 761 négatifs (composition à documenter — voir `data/negatives/README.md`)
