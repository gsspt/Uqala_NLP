# Uqala NLP — Détection du maǧnūn ʿāqil dans les textes arabes classiques

Projet de philologie numérique pour détecter le motif littéraire du **sage-fou**
(*maǧnūn ʿāqil*) dans le corpus OpenITI (~200,000 textes arabes classiques).

**Auteur** : Augustin Pot — IREMAM / Aix-Marseille Université  
**Directeur** : Hakan Özkan  
**Version** : 0.2.0

---

## Le motif maǧnūn ʿāqil

Le *maǧnūn ʿāqil* (sage-fou / fou-sage) est un motif narratif récurrent dans
la littérature d'adab classique : un personnage apparemment fou révèle, par sa
transgression des normes, une sagesse que les "sages" ne peuvent pas exprimer.

**Structure canonique** :
```
JUNŪN (présentation du fou) → DIALOGUE (rencontre/question)
→ PAROLE PARADOXALE → VALIDATION (rire, don, admiration)
```

**Corpus de référence** : *Kitāb ʿUqalāʾ al-Maǧānīn* (Nīsābūrī, Xe s.)
— 452 akhbars annotés utilisés comme positifs d'entraînement.

---

## Installation

```bash
git clone https://github.com/gsspt/uqala_nlp.git
cd uqala_nlp
pip install -e ".[dev]"
```

Pour les fonctionnalités LLM :
```bash
pip install -e ".[llm]"
cp .env.example .env
# Remplir OPENAI_API_KEY ou DEEPSEEK_API_KEY dans .env
```

---

## Pipeline de production (état actuel)

```bash
# Appliquer le modèle LR+XGBoost sur un auteur OpenITI
python pipelines/hybrid/family_A_cascade/A1_conservative.py \
    --author 0328IbnCabdRabbih \
    --threshold 0.70

# Résultats dans results/0328IbnCabdRabbih/
```

**Résultats sur Ibn ʿAbd Rabbih** (*Al-ʿIqd al-Farīd*, 10,113 akhbars) :

| Catégorie | Nombre | Proportion |
|-----------|--------|------------|
| Fous canoniques (Khalaf, Riyāh…) | 100 | 0.99% |
| Maǧnūn ʿāqil (après post-filtrage) | 16 | 0.16% |
| Modèle : LR AUC = 0.838 | XGB AUC (CV) = 0.846 | |

---

## Architecture des pipelines

L'objectif est de tester **tous les pipelines possibles** du plus au moins
interprétable, pour identifier la meilleure approche pour la thèse.

```
pipelines/
├── level1_interpretable/       ← Règles booléennes, arbre décision, LR (✅ en prod.)
├── level2_semi_interpretable/  ← Random Forest + SHAP (✅ entraîné)
├── level3_latent/              ← TF-IDF, Word2Vec, CAMeLBERT (à implémenter)
├── level4_llm/                 ← GPT-4, DeepSeek annotation (partiel)
└── hybrid/
    ├── family_A_cascade/       ← Règles → ML → Règles (✅ en prod.)
    ├── family_B_multistage/    ← Two-stage, specialist ensemble
    ├── family_C_iterative/     ← Active learning (⭐ PRIORITÉ HAUTE)
    ├── family_D_human_loop/    ← Human-in-the-loop + SHAP (partiel)
    ├── family_E_ensemble/      ← Stacking, mixture of experts
    └── family_F_transfer/      ← Distillation, cross-lingual
```

Voir `pipelines/README.md` pour le tableau comparatif complet.

---

## Features

Le modèle extrait **74 features** organisées en 10 catégories :

| Catégorie | Features | Signal |
|-----------|----------|--------|
| Folie (ج-ن-ن) | 15 | Présence lexicale + noms propres |
| Raison (ع-ق-ل) | 8 | Co-occurrence paradoxale |
| Sagesse (ح-ك-م) | 5 | Contexte de renversement |
| Dialogue | 11 | Structure dialogique (قال density) |
| Validation | 8 | Rire, don, admiration |
| Contraste | 5 | ولكن, إلا, لكن |
| Autorité | 4 | Scène calife/vizir |
| Poésie | 3 | Poésie-dans-récit (neutre/positif) |
| Wasf | 3 | Définitionnel (négatif) |
| Morphologie | 9 | Racines, POS, aspect (CAMeL Tools) |

**Features actantielles** (en cours) : voir `src/uqala_nlp/features/actantial.py`  
**Découverte inductive** : voir `scripts/discover_features.py`

---

## Structure du projet

```
Uqala_NLP/
├── src/uqala_nlp/          Package Python principal
│   ├── config.py           Configuration centralisée
│   ├── preprocessing/      Nettoyage + filtrage isnād
│   ├── features/           Extraction de features (lexical, morpho, actantiel)
│   └── utils/              Utilitaires arabes
├── pipelines/              Tous les pipelines (voir ci-dessus)
├── scripts/                Scripts CLI
├── data/                   Corpus annotés
├── models/                 Modèles entraînés
├── results/                Résultats d'analyse
├── docs/                   Documentation
├── tests/                  Tests pytest
└── notebooks/              Exploration Jupyter
```

---

## Choix épistémologique

> « L'objectif n'est pas que l'algorithme *sache* ce qu'est un maǧnūn ʿāqil,
> mais qu'il me fasse gagner 80% du temps de lecture en pré-filtrant le corpus,
> tout en gardant le contrôle interprétatif. »

Le projet vise un **concordancier augmenté**, pas un classificateur autonome.
Chaque décision du modèle doit être traçable, justifiable, et corrigeable
par le philologue.

---

## Documentation

- `docs/workflow.md` — Workflow complet (5 phases)
- `docs/features_catalog.md` — Catalogue des 153 features possibles
- `docs/guide_morphologie.md` — Guide CAMeL Tools
- `pipelines/README.md` — Comparaison des pipelines
