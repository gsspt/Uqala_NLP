# Uqala NLP - Documentation Index

Bienvenue! Ce projet détecte les figures du **fou sensé (ʿāqil majnūn)** dans les textes arabes classiques en utilisant le machine learning.

## 🚀 Démarrage rapide

**Nouveau sur le projet?** Commencez ici:

1. **[README.md](README.md)** — Vue d'ensemble, résultats clés, structure du repo
2. **[COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)** — Guide technique complet (LIRE D'ABORD!)
3. **[WORKFLOW.md](WORKFLOW.md)** — Flux complet et historique de développement

## 📚 Documentation par sujet

### 🎯 Pour Comprendre le Projet
- **[README.md](README.md)** — Overview, quick start, key results
- **[COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)** — Complete technical guide with all details

### 🔬 Classification et Modèles
- **[COMPREHENSIVE_GUIDE.md - Section: Model Performance](COMPREHENSIVE_GUIDE.md#model-performance)** — Performance metrics for LR and XGBoost
- **[ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md)** — Analyse des 82% faux positifs et 4 solutions
- **[VERIFICATION_COMPLETE.md](VERIFICATION_COMPLETE.md)** — Vérification du pipeline

### 🔧 Features et Linguistique  
- **[COMPREHENSIVE_GUIDE.md - Section: Feature Engineering](COMPREHENSIVE_GUIDE.md#feature-engineering)** — 71 features detailed
- **[features_catalog_majnun_aqil.md](features_catalog_majnun_aqil.md)** — Catalogue de 153 features avec justifications
- **[guide_morphologie.md](guide_morphologie.md)** — Guide morphologie arabe et CAMeL Tools

### 🛣️ Améliorations et Roadmap
- **[COMPREHENSIVE_GUIDE.md - Section: Improvement Roadmap](COMPREHENSIVE_GUIDE.md#improvement-roadmap)** — Phases 1-4 planned improvements
- **[ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md)** — 4 strategies for improvement

### 📖 Référence et Historique
- **[WORKFLOW.md](WORKFLOW.md)** — Complete workflow documentation
- **[archived/](archived/)** — Historical documentation (one-time reference docs)

## 📂 Structure du projet

```
src/              # All production code
├── openiti_detection/    # Main detection pipeline
├── scan/                 # Model training & features
├── comparison_ensemble/  # XGBoost models
└── isnad_filter.py      # Utility

data/             # Training datasets
├── dataset_raw.json
└── Kitab_Uqala_al_Majanin_annotated.json (NEW)

models/           # Additional trained models directory

results/          # Output directory

openiti_corpus/   # OpenITI corpus (28GB, not in git)

docs/             # Documentation
├── README.md
├── COMPREHENSIVE_GUIDE.md
├── INDEX.md (this file)
├── WORKFLOW.md
├── ANALYSIS_FALSE_POSITIVES.md
├── VERIFICATION_COMPLETE.md
├── features_catalog_majnun_aqil.md
├── guide_morphologie.md
└── archived/
```

## 🎯 Résultats clés

### Ibn ʿAbd Rabbih (Al-Iqd al-Farid)
- **10,113 textes** analysés
- **100 fous canoniques** (Khalaf, Bahlul, etc.) — 0.99% du corpus
- **874 candidats majnun aqil** (critères stricts)
- **114 positifs fiables** après post-filtering (2.5%)
- **Faux positifs réduits:** 82% → 0% (via post-filter)

### Performance des classifiers
| Model | AUC | F1 | Notes |
|-------|-----|-----|-------|
| LR (71 features) | 0.804 | 0.75 | ⚠️ Under improvement |
| XGBoost (71 features) | 0.991 | 0.98 | ✅ Excellent |
| LR (50 features) | 0.849 | 0.83 | Better baseline |
| **Consensus** | — | **0.98** | Very reliable |

## ⚙️ Installation

```bash
# Clone
git clone https://github.com/gsspt/Uqala_NLP.git
cd Uqala_NLP

# Virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install
pip install -r requirements.txt
```

## 🔍 Comment explorer la documentation

**Si tu veux:**

1. **Comprendre rapidement le projet** 
   → Lire [README.md](README.md) (5 min) + [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md) sections 1-3 (10 min)

2. **Comprendre les modèles et performance**
   → [COMPREHENSIVE_GUIDE.md - Model Performance](COMPREHENSIVE_GUIDE.md#model-performance)
   → [ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md)

3. **Comprendre les features et comment ça marche**
   → [COMPREHENSIVE_GUIDE.md - Feature Engineering](COMPREHENSIVE_GUIDE.md#feature-engineering)
   → [features_catalog_majnun_aqil.md](features_catalog_majnun_aqil.md)

4. **Améliorer le classifier**
   → [COMPREHENSIVE_GUIDE.md - Improvement Roadmap](COMPREHENSIVE_GUIDE.md#improvement-roadmap)
   → [ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md)

5. **Utiliser le code**
   → [COMPREHENSIVE_GUIDE.md - How to Use](COMPREHENSIVE_GUIDE.md#how-to-use)

6. **Comprendre l'historique du développement**
   → [WORKFLOW.md](WORKFLOW.md)

7. **Référence technique détaillée**
   → [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md) (go-to reference)

## 🐛 Issues connus

### ⚠️ Faux positifs élevés (82%)
- **Cause:** Classifiers apprennent dialogue = majnun (au lieu de paradoxe = majnun)
- **Solution actuelle:** Post-classification filtering réduit FP 82% → 0%
- **Améliorations futures:** Two-tier thresholding, features explicites de paradoxe
- **Référence:** [ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md)

### ⚠️ LR 71-D régression (AUC 0.804 vs 0.849)
- **Cause:** 42 features de padding (zéros) → StandardScaler NaN → coefficients parasites
- **Solution:** Remplacer padding par features réelles + réoptimiser C
- **Référence:** [COMPREHENSIVE_GUIDE.md - Known Issues](COMPREHENSIVE_GUIDE.md#known-issues--solutions)

## 📊 État du repo

- **Dernière mise à jour:** 2026-04-10
- **Version:** 0.2 (Production)
- **Structure:** Réorganisée (src/, data/, docs/, models/)
- **Environnement:** Virtual environment avec toutes dépendances
- **Git:** Tous changements poussés vers GitHub

## 🔗 GitHub

**Repository:** https://github.com/gsspt/Uqala_NLP  
**Status:** Public, production-ready

---

## 📚 Document Map (Quick Reference)

| Besoin | Document | Section |
|--------|----------|---------|
| Vue d'ensemble | README.md | — |
| Guide technique complet | COMPREHENSIVE_GUIDE.md | — |
| Performance modèles | COMPREHENSIVE_GUIDE.md | Model Performance |
| Features (71-D) | COMPREHENSIVE_GUIDE.md | Feature Engineering |
| Faux positifs | ANALYSIS_FALSE_POSITIVES.md | — |
| Amélioration roadmap | COMPREHENSIVE_GUIDE.md | Improvement Roadmap |
| Utilisation code | COMPREHENSIVE_GUIDE.md | How to Use |
| Features détail | features_catalog_majnun_aqil.md | — |
| Morphologie | guide_morphologie.md | — |
| Historique dev | WORKFLOW.md | — |

---

**Version:** Index v1.0 (2026-04-10)  
**Maintenu par:** Augustin Pot  
**Référence primaire:** [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)
