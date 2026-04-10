# Données

## Structure

```
data/
├── raw/
│   └── dataset_raw.json              Corpus d'entraînement annoté (1,221 textes)
├── annotated/
│   └── Kitab_Uqala_al_Majanin_annotated.json   Corpus de référence annoté
└── negatives/
    ├── README.md                     Instructions de constitution
    ├── ghazal/                       200 akhbars de poésie amoureuse (à constituer)
    ├── manaaqib/                     200 akhbars de manāqib/faḍāʾil (à constituer)
    └── adab_neutral/                 212 akhbars d'adab neutre (à constituer)
```

## dataset_raw.json

**Format** : JSON, 1,221 entrées

```json
{
  "id": "pos_001",
  "label": 1,
  "text_ar": "...",
  "source": "Nisaburi_161",
  "genre": "akhbar"
}
```

| Catégorie | Nombre | Source |
|-----------|--------|--------|
| Positifs (label=1) | 460 | Kitāb ʿUqalāʾ al-Maǧānīn (Nīsābūrī), akhbars 161-612 |
| Négatifs (label=0) | 761 | À documenter — voir `negatives/README.md` |

## Kitab_Uqala_al_Majanin_annotated.json

Corpus source du Kitāb ʿUqalāʾ al-Maǧānīn avec annotations enrichies
(résumés français, commentaires sur la forme, mécanisme narratif, rôle de la folie).

Ce fichier est la source primaire pour :
- L'analyse inductives des features (voir `scripts/discover_features.py`)
- L'annotation actantielle (voir `pipelines/level4_llm/p4_2_deepseek_annotation.py`)
- La validation du modèle actantiel (voir `src/uqala_nlp/features/actantial.py`)

## Données non versionnées

Les fichiers suivants ne sont pas versionnés (trop volumineux ou non-distribués) :
- `openiti_corpus/` : Corpus OpenITI complet (~200,000 textes arabes)
- `models/word2vec_openiti.bin` : Modèle Word2Vec entraîné sur OpenITI (~400 MB)
- `models/camelbert_finetuned/` : CAMeLBERT fine-tuné

Voir `.gitignore` pour la liste complète.
