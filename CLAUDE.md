# CLAUDE.md — Uqala NLP Project

> Ce fichier est lu automatiquement par Claude Code à chaque démarrage de session.
> Il maintient la continuité entre environnements (web, VSCode, smartphone).

---

## Projet : Détection du maǧnūn ʿāqil

**Objectif** : Identifier automatiquement les récits de « fou sage » (majnun aqil) dans la littérature arabe classique, à partir du corpus de Nisaburi (*Kitāb ʿUqalāʾ al-Maǧānīn*).

**Chercheur** : Augustin (gsspt)
**Dépôt** : `https://github.com/gsspt/Uqala_NLP`
**Branche active** : `claude/repo-structure-review-07E9d`

---

## Ritual de session

```bash
# DÉBUT de chaque session
git pull origin claude/repo-structure-review-07E9d

# FIN de chaque session
git add -p          # staging sélectif
git commit -m "..."
git push -u origin claude/repo-structure-review-07E9d
```

**GitHub PAT** : configuré dans `~/.claude/settings.json` via hook `SessionStart`.
Si le push échoue : vérifier que le remote utilise bien le token.
```bash
git remote get-url origin
# Doit être : https://gsspt:ghp_...@github.com/gsspt/Uqala_NLP.git
```

---

## 🔄 Workflow OpenITI → Pipeline modèles

**IMPORTANT**: Tout texte extrait du corpus OpenITI DOIT passer par ce workflow:

### 1️⃣ Extraction des akhbars (unités narratives)
```bash
from src.uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file

# Pour UN fichier OpenITI:
akhbars = extract_akhbars_from_file('openiti_corpus/data/0328IbnCabdRabbih/.../file.txt')
# Retourne: list[str] — chaque str est un khabar cohérent, 80-3000 caractères arabes

# Pour TOUT le corpus:
from src.uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_corpus
akhbars, total = extract_akhbars_from_corpus('openiti_corpus/data/')
```

**Format OpenITI**:
- Métadonnées : lignes `#META# ...`
- Contenu marqué par : `#META#Header#End#`
- Paragraphes : lignes commençant par `~~`
- Sections/akhbars séparés par : lignes commençant par `# ` (pas `# |`)
- Titres : `# |` (ignorés)

**Filtre qualité**:
- Min: 80 caractères arabes (élimine fragments, métadonnées)
- Max: 3000 caractères arabes (évite textes énormes)

### 2️⃣ Filtrage des isnads
```bash
from src.uqala_nlp.preprocessing.isnad_filter import remove_isnad

# Chaque akhbar DOIT être purifié:
akhbar_clean = remove_isnad(akhbar)
```

**Pourquoi** : Les isnads (chaînes d'autorité) apparaissent dans TOUS les types de textes (positifs ET négatifs), donc ce ne sont pas un signal pertinent. La fonction `isnad_filter.py` les supprime avant extraction de features.

### 3️⃣ Passage au pipeline modèles
```bash
# Exemple : Score un akhbar avec v80
from pipelines.level1_interpretable.p1_4_logistic_regression_v80 import extract_all_features_27
import pickle

akhbar = remove_isnad(akhbar)  # ← TOUJOURS filtrer les isnads d'abord!
features = extract_all_features_27(akhbar)
# Utiliser les features pour scoring
```

**Résumé du workflow**:
```
OpenITI file
    ↓ extract_akhbars_from_file()
Akhbars (liste)
    ↓ remove_isnad() [pour chaque akhbar]
Akhbars nettoyés
    ↓ extract_all_features_27() [pour v80]
Features vectors
    ↓ predict()
Scores (0.0-1.0)
```

---

## État actuel du projet (mise à jour : 2026-04-11)

### Données
| Fichier | Contenu |
|---|---|
| `data/raw/dataset_raw.json` | 4 277 textes : 460 positifs (label=1) + 3 817 négatifs (label=0) |
| `data/annotated/Kitab_Uqala_al_Majanin_annotated.json` | 612 akhbars annotés (référence philologique) |

**Sources des négatifs** : cuyun (500), adhkiya_ch1-29 (400), tabari_tarikh (400), dhahabi_tarikh (400), khatib_baghdad (400), hamqa (300), jahiz_hayawan (300), baladhuri_ansab (300), poetry_diwan (200), + diwan poétiques.
**Sources des positifs** : nisaburi (444), adhkiya_ch30 (16).

### Modèles entraînés (v79)
| Modèle | CV AUC | Test AUC | Fichier |
|---|---|---|---|
| Logistic Regression | 0.861 ± 0.044 | 0.890 | `models/lr_classifier_79features.pkl` |
| XGBoost | 0.872 ± 0.045 | 0.948 | `models/xgb_classifier_79features.pkl` |

Les `.pkl` sont gitignorés — **re-entraîner si absents** :
```bash
python3 pipelines/level1_interpretable/p1_3_logistic_regression.py --cv 5
python3 pipelines/level2_semi_interpretable/p2_1_random_forest_shap.py --cv 5
```

### Features v79 (79 features)
**Pipeline principal** : `pipelines/level1_interpretable/p1_3_logistic_regression.py`

Catégories :
- Junun (f00-f14) : termes de folie, noms célèbres (بهلول, سعدون...)
- Aql/Hikma (f15-f27) : sagesse, paradoxe junun×aql
- Dialogue (f29-f38) : densité qala, 1ère personne, questions *(f28 supprimé)*
- Validation (f39-f46) : rires, dons, pleurs
- Contraste (f47-f51) : révélation, opposition
- Autorité (f52-f55) : signal négatif (خليفة, وزير)
- Poésie (f56-f58), Spatial (f59-f61), Wasf (f62-f64)
- Morpho CAMeL (f65-f70) : racines ج.ن.ن / ع.ق.ل *(f71-f73 supprimés)*
- **Nouvelles v79 (f74-f82)** :
  - `f74` ؟ présence (94.7% des positifs, 0% des négatifs) ← signal clé
  - `f75` densité ؟
  - `f76` intensité religieuse (إلهي/اللهم/يا رب)
  - `f77` narration scénique 1ère pers. (كنت/فرأيت)
  - `f78` interpellation directe (يا بهلول)
  - `f79` réactions physiques (شهق/تعجب/ويلي)
  - `f80` lieux de la folie (دار المرضى/مقابر)
  - `f81` verbes mystiques (عرف/أدرك)
  - `f82` folie amoureuse (لليلى/هيام/عشق — 12 termes)

**Top 5 features LR** : f02_famous_fool (+0.638), f65_root_jnn (+0.353), f51_contrast_revelation (+0.284), f76_religious (+0.266), f69_noun_density (+0.266)

---

## Architecture des pipelines

```
pipelines/
├── level1_interpretable/
│   └── p1_3_logistic_regression.py   ← PIPELINE PRINCIPAL (79 features)
├── level2_semi_interpretable/
│   └── p2_1_random_forest_shap.py    ← XGBoost (importe de level1)
├── hybrid/
│   └── family_A_cascade/
│       └── A1_conservative.py        ← Pipeline de production (LR+XGB+post-filtre)
└── [14 autres pipelines — stubs NotImplementedError]
```

---

## Problèmes connus

### Faux positifs sur corpus externe
A1_conservative produit 54.6% de positifs sur Ibn Abd Rabbih (corpus externe).
Après post-filtrage : 114/10 113. Cause : le modèle confond "dialogue générique قال" avec le signal majnun aqil.
**Solution planifiée** : C1 Active Learning (`pipelines/hybrid/family_C_iterative/C1_active_learning.py`).

### XGBoost overfit (réduit mais persistant)
Gap CV↔Test : 7.7 pts (était 14.5 pts avant v79). Régularisation renforcée (max_depth=3, reg_alpha=0.1).

### f77/f78/f79 — importance = 0
Ces 3 features sont zeroed par la régularisation (redondantes avec f30 et f02). À supprimer dans v80.

---

## Prochaines étapes (par priorité)

### Court terme
1. **Supprimer f77-f79** dans v80 (redondantes, importance=0)
2. **Affiner f76** : remplacer `الله` générique par إلهي/اللهم/يا رب uniquement
3. **Implémenter C1 Active Learning** — solution structurelle aux faux positifs

### Moyen terme
4. **Features structurales** (`src/uqala_nlp/features/structural.py`) — ordre narratif Propp, 7 features
5. **Features actantielles** (`src/uqala_nlp/features/actantial.py`) — modèle greimassien, nécessite annotation LLM
6. **DeepSeek annotation** (`pipelines/level4_llm/p4_2_deepseek_annotation.py`) — débloque les features actantielles

### Long terme
7. Familles B, D (two-stage, human-in-loop)
8. Tests d'intégration
9. CI/CD GitHub Actions

---

## Dépendances

```bash
pip install numpy scikit-learn xgboost scipy tqdm matplotlib shap
# CAMeL Tools (optionnel — features morphologiques f65-f70)
pip install camel-tools
```

CAMeL Tools absent → features morphologiques tombent à 0 (fallback automatique, modèle fonctionne quand même).

---

## Structure du repo

```
Uqala_NLP/
├── CLAUDE.md                          ← ce fichier
├── src/uqala_nlp/                     ← librairie core
│   ├── preprocessing/isnad_filter.py  ← filtrage isnad (complet)
│   ├── features/structural.py         ← stub (NotImplementedError)
│   ├── features/actantial.py          ← stub (NotImplementedError)
│   └── utils/arabic.py                ← normalisation, tokenisation
├── pipelines/                         ← 21 pipelines (3 complets, 14 stubs)
├── data/raw/dataset_raw.json          ← 4 277 textes (3.7 GB)
├── data/annotated/                    ← corpus de référence Nisaburi
├── models/                            ← modèles + rapports JSON
├── results/0328IbnCabdRabbih/         ← résultats production (Ibn Abd Rabbih)
├── docs/                              ← documentation philologique
└── scripts/scan_corpus.py             ← scanner OpenITI (production)
```

---

## Conventions

- **Commits** : `type(scope): message` — ex: `feat(features): add question mark feature`
- **Nommage features** : `f{nn}_{nom_descriptif}` — numérotation continue
- **Rapports modèles** : JSON dans `models/` — jamais de `.pkl` dans git
- **Langue** : code en anglais, commentaires en français, commits en anglais
