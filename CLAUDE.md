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

## État actuel du projet (mise à jour : 2026-04-11, 18:45)

### 🎯 Hybrid Ensemble (v80 LR + XGBoost) + DeepSeek Few-Shot Validation

**Three-tier validation strategy on Ibn Abd Rabbih (10,286 akhbars)**:

#### Tier 1: v80 Logistic Regression (Baseline)
- **Positives**: 102 (1.0%)
- **Performance**: CV AUC 0.890 (27 features)
- **Assessment**: Too permissive — 85% likely false positives (confirms domain intuition)

#### Tier 2: Hybrid Ensemble (LR + XGBoost)
- **v80 XGBoost training**: CV AUC 0.8775 ± 0.0227 (very stable, low variance)
- **Ensemble fusion**: Average LR + XGB probabilities, measure agreement
- **Results on Ibn Abd Rabbih**:
  - **Positives**: 15 (0.15%)
  - **Mean agreement**: 0.979 (93.3% "very high" ≥0.90)
  - **Assessment**: High precision ensemble — only keep what BOTH models agree on

#### Tier 3: DeepSeek Few-Shot LLM Validation
- **Pipeline**: 8 workers, ThreadPoolExecutor, 0.1s rate limiting (10 req/sec)
- **Few-shot examples**: 3 real TRUE + 2 FALSE hard negatives from Nisaburi corpus
  - TRUE: Complete akhbars with isnad, dialogue, paradoxical wisdom (Bahlul poetry, Saadun rain dialogue, Saadun exposure)
  - FALSE: Philosophical commentary that mentions madness but lacks narrative
- **Results on Ibn Abd Rabbih**: 74 positives (0.72%)
- **Performance**: ~20 min, $1.50 USD, 0 API errors
- **Assessment**: Ground truth estimate (LLM judgment w/ real examples between LR and Ensemble)

### Validation Results Summary
| Approach | Positives | Rate | File |
|----------|-----------|------|------|
| **LR v80 alone** | 102 | 1.0% | validate_v80_on_ibn_rabbih.py |
| **LR + XGBoost Ensemble** | 15 | 0.15% | ensemble_v80_validation_results.json |
| **DeepSeek Few-Shot (8w)** | 74 | 0.72% | deepseek_full_corpus_8workers.json |

**Conclusion**: Estimated 50-70 true majnun aqil in Ibn Abd Rabbih (LLM provides ground truth, ensemble provides high-confidence subset)

---

## État précédent du projet (mise à jour : 2026-04-11, 08:00)

### Données
| Fichier | Contenu |
|---|---|
| `data/raw/dataset_raw.json` | 4 277 textes : 460 positifs (label=1) + 3 817 négatifs (label=0) |
| `data/annotated/Kitab_Uqala_al_Majanin_annotated.json` | 612 akhbars annotés (référence philologique) |

**Sources des négatifs** : cuyun (500), adhkiya_ch1-29 (400), tabari_tarikh (400), dhahabi_tarikh (400), khatib_baghdad (400), hamqa (300), jahiz_hayawan (300), baladhuri_ansab (300), poetry_diwan (200), + diwan poétiques.
**Sources des positifs** : nisaburi (444), adhkiya_ch30 (16).

### Modèles entraînés

| Modèle | CV AUC | Notes | Fichier |
|---|---|---|---|
| **v79 LR** | 0.861 ± 0.044 | 79 features, baseline | `models/lr_classifier_79features.pkl` |
| **v79 XGBoost** | 0.872 ± 0.045 | 79 features | `models/xgb_classifier_79features.pkl` |
| **v80 LR** | 0.890 | 27 features, clean feature set | `models/lr_classifier_v80.pkl` |
| **v80 XGBoost** | 0.8775 ± 0.0227 | 27 features, low variance ✅ | `models/xgb_classifier_v80.pkl` |

Les `.pkl` sont gitignorés — **re-entraîner si absents** :
```bash
# v80 (recommandé)
python3 pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 5
python3 pipelines/level2_semi_interpretable/p2_2_xgboost_v80_shap.py --cv 5

# v79 (legacy)
python3 pipelines/level1_interpretable/p1_3_logistic_regression.py --cv 5
python3 pipelines/level2_semi_interpretable/p2_1_random_forest_shap.py --cv 5
```

### Features v80 (27 features) ✅ RECOMMENDED

**Pipeline** : `pipelines/level1_interpretable/p1_4_logistic_regression_v80.py`

**v80 Clean Feature Set** (removes redundant f77-f82):
- Junun (f00-f14) : termes de folie, noms célèbres (بهلول, سعدون...)
- Aql/Hikma (f15-f27) : sagesse, paradoxe junun×aql
- Dialogue (f29-f38) : densité qala, 1ère personne, questions
- Validation (f39-f46) : rires, dons, pleurs
- Contraste (f47-f51) : révélation, opposition
- Autorité (f52-f55) : signal négatif (خليفة, وزير)
- Poésie (f56-f58), Spatial (f59-f61), Wasf (f62-f64)
- Morpho CAMeL (f65-f70) : racines ج.ن.ن / ع.ق.ل
- Key v79 additions (f74-f76):
  - `f74` ؟ présence (94.7% des positifs, 0% des négatifs) ← signal clé
  - `f75` densité ؟
  - `f76` intensité religieuse (إلهي/اللهم/يا رب)

**Top 5 LR coefficients** : f02_famous_fool (+0.638), f65_root_jnn (+0.353), f51_contrast_revelation (+0.284), f76_religious (+0.266), f69_noun_density (+0.266)

**Why v80 > v79**:
- f77-f82 were redundant (LR coef ≈ 0, zero importance in trees)
- Simpler feature set → better generalization
- Ensemble with XGBoost more stable (lower CV variance)

---

## Architecture des pipelines

```
pipelines/
├── level1_interpretable/
│   ├── p1_3_logistic_regression.py   ← v79 (79 features)
│   └── p1_4_logistic_regression_v80.py ← v80 (27 features, CV AUC 0.890)
├── level2_semi_interpretable/
│   ├── p2_1_random_forest_shap.py    ← v79 XGBoost
│   ├── p2_2_xgboost_v80_shap.py      ← v80 XGBoost (CV AUC 0.8775±0.0227) ✅ NEW
│   └── p2_3_hybrid_ensemble_shap.py  ← LR + XGBoost ensemble (averaging + agreement) ✅ NEW
├── level4_llm/
│   ├── p4_1_few_shot.py              ← Base few-shot template
│   ├── p4_1_few_shot_full_corpus.py  ← DeepSeek sequential (90 min)
│   └── p4_1_few_shot_8workers.py     ← DeepSeek 8-worker (20 min, $1.50) ✅ NEW
├── hybrid/
│   └── family_A_cascade/
│       └── A1_conservative.py        ← Pipeline de production (v79+post-filtre)
└── [14 autres pipelines — stubs NotImplementedError]
```

**v80 vs v79**:
- v80 features: 27 (down from 79)
- v80 removes: f77-f79, f80-f82 (redundant/zeroed by regularization)
- v80 keeps: f00-f76 core features + top v79 features
- v80 benefits: Simpler, faster, ensemble generalization

---

## Problèmes connus & Solutions

### ✅ SOLVED: v80 Feature Redundancy
- **Problem**: f77-f82 had zero importance in both LR and XGBoost
- **Solution**: Created v80 with 27 essential features (removed f77-f82)
- **Result**: Better generalization, ensemble more stable

### Faux positifs sur corpus externe (v79)
v79 A1_conservative produit 54.6% de positifs sur Ibn Abd Rabbih.
Cause : dialogue générique قال confusible avec majnun aqil signal.
**v80 Solution** : Ensemble voting filters false positives (15 vs 102 positives)
**Long-term** : C1 Active Learning (`pipelines/hybrid/family_C_iterative/C1_active_learning.py`)

### XGBoost generalization
v80 XGBoost CV variance very low (±0.0227), indicating robust learning

---

## Prochaines étapes (par priorité)

### ✅ Complété (session 2026-04-11)
1. ✅ **v80 Feature Selection** (27 features, f77-f82 removed)
2. ✅ **XGBoost v80 Training** (CV AUC 0.8775 ± 0.0227)
3. ✅ **Ensemble Pipeline** (LR + XGBoost averaging + agreement metric)
4. ✅ **DeepSeek Few-Shot Validation** (8-worker, real Nisaburi examples, 74 positives)
5. ✅ **Documentation** (TECHNICAL, EXPLICATION, QUICK_REFERENCE)

### Court terme
1. **Test XGBoost v80 alone on Ibn Abd Rabbih** — complete the v80 validation suite (never tested XGB alone, only LR + Ensemble)
2. **Analyze 74 DeepSeek results** — manual inspection of ground truth candidates
3. **Merge ensemble pipeline to main** — promote to primary workflow

### Moyen terme
4. **C1 Active Learning** — iterative labeling of uncertain cases (50-70 range)
5. **Features structurales** (`src/uqala_nlp/features/structural.py`) — Propp narrative order, 7 features
6. **Features actantielles** (`src/uqala_nlp/features/actantial.py`) — Greimassian actants, LLM-assisted annotation

### Long terme
7. **p4_2_deepseek_annotation.py** — LLM-driven feature extraction (enables actantial features)
8. Familles B, D (two-stage, human-in-loop)
9. Tests d'intégration + CI/CD GitHub Actions

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
