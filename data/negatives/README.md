# Corpus négatif — Instructions de constitution

Le corpus négatif est indispensable pour l'entraînement des modèles ML.
Il doit représenter les genres les plus susceptibles de générer des faux positifs.

## Composition cible (612 textes)

| Sous-corpus | Nombre | Sources OpenITI suggérées |
|-------------|--------|--------------------------|
| `ghazal/` | 200 | Dīwāns préislamiques (ʿAmr ibn Kulthūm, Labīd, Imruʾ al-Qays) |
| `manaaqib/` | 200 | Ṭabaqāt al-ʿulamāʾ, Manāqib, Faḍāʾil |
| `adab_neutral/` | 212 | ʿUyūn al-aḫbār (Ibn Qutayba), Al-Aġānī (non-maǧnūn), nawādir |

## Critères de sélection

### ghazal/ — Poésie amoureuse
- Textes avec "جنون الحب" → faux positifs classiques du modèle actuel
- Extraire les akhbars sur Majnūn Laylā (≠ maǧnūn ʿāqil)
- **Critère d'exclusion** : Tout texte où la folie est métaphorique (amour)

### manaaqib/ — Hagiographies et mérites
- Textes avec مجنون comme sujet d'anecdote (sans renversement paradoxal)
- Chroniques politiques avec mentions de ḫulafāʾ (faux positifs "autorité")
- **Critère d'exclusion** : Tout texte avec paradoxe folie/sagesse

### adab_neutral/ — Adab non-maǧnūn
- Nawādir humoristiques sans thème de folie
- Ḥikam et maximes (sans structure narrative)
- Anecdotes de ẓurafāʾ (élégants) et ʿuqalāʾ (sages) sans folie
- **Critère d'exclusion** : Tout akhbar pouvant être classé maǧnūn ʿāqil

## Format attendu

Chaque fichier dans les sous-dossiers doit être au format JSON :
```json
[
  {
    "id": "neg_ghazal_001",
    "text_ar": "...",
    "label": 0,
    "source": "Dīwān Labīd",
    "genre": "ghazal"
  }
]
```

## État actuel

- `ghazal/` : ⬜ À constituer
- `manaaqib/` : ⬜ À constituer
- `adab_neutral/` : ⬜ À constituer

Le corpus négatif actuel (`data/raw/dataset_raw.json` — 761 négatifs)
est disponible mais sa composition exacte n'est pas documentée.
