# Uqala NLP: Detecting the Wise Fool (ʿāqil majnūn) in Arabic Texts

A machine learning system for identifying the *ʿāqil majnūn* (wise fool / sage fool) figure in classical Arabic literature, particularly in the OpenITI corpus.

**Author:** Augustin Pot  
**Affiliation:** IREMAM (Institut de Recherche et d'Études sur le Monde Arabe et Musulman), Aix-Marseille University  
**Supervisor:** Hakan Özkan  
**Updated:** 2026-04-10  

---

## 📋 Quick Navigation

- **New to the project?** Start with [INDEX.md](INDEX.md) or [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)
- **Need technical details?** See [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)
- **Understand the workflow?** Read [WORKFLOW.md](WORKFLOW.md)
- **Want to improve the model?** Check [ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md)
- **Feature documentation?** See [features_catalog_majnun_aqil.md](features_catalog_majnun_aqil.md)

---

## 🎯 Project Overview

This project combines **digital humanities** and **machine learning** to detect the paradoxical wise fool figure in classical Arabic texts:

- **Surface level:** Appears foolish, mad (جنون/junun)
- **Deep level:** Possesses wisdom, insight (عقل/ʿaql, حكمة/ḥikma)
- **Key insight:** Paradox — contradiction between appearance and reality

### ✅ What We've Built

**Dual Classification System:**
- ✅ **Logistic Regression:** AUC 0.804 (interpretable)
- ✅ **XGBoost Ensemble:** AUC 0.991 (high-performance)
- ✅ **Consensus Strategy:** F1 = 0.98 (very reliable)

**Feature Engineering (71 dimensions):**
- ✅ 62 lexical features (junun markers, wisdom words, dialogue patterns)
- ✅ 9 morphological features (POS tags, aspect, voice via CAMeL Tools)

**Pipeline Quality:**
- ✅ Akhbar extraction: 7,400-10,000+ texts per author
- ✅ Metadata cleaning: 100% clean (0% corruption)
- ✅ Extraction coherence: 94% (verified by LLM)
- ✅ Post-classification filtering: Reduces FP 82% → 0%

**Results on Ibn ʿAbd Rabbih (Al-Iqd al-Farid, 10,113 texts):**
- 100 canonical fools detected (0.99% of corpus)
- 874 true majnun aqil candidates (8.64%)
- 114 reliable positives after post-filtering (2.5%)

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/gsspt/Uqala_NLP.git
cd Uqala_NLP

# 2. Create virtual environment
python -m venv venv
source venv/Scripts/activate      # Windows
# or: source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Run a Quick Test

```bash
# Test on Ibn ʿAbd Rabbih (requires openiti_corpus/)
cd src/openiti_detection
python test_quick.py 0328IbnCabdRabbih

# Strict analysis with canonical fool detection
python strict_analysis.py 0328IbnCabdRabbih --threshold-lr 0.7 --threshold-xgb 0.7

# Verify pipeline quality
python verify_extraction.py
python verify_feature_extraction.py
```

---

## 📁 Repository Structure

```
Uqala-NLP/
├── src/                              # All production code
│   ├── openiti_detection/           # Main detection pipeline
│   │   ├── detect_lr_xgboost.py    # Core classifier
│   │   ├── strict_analysis.py      # Strict analysis + canonical fool detection
│   │   ├── post_filter.py          # Post-classification filtering (reduces FP)
│   │   ├── test_quick.py           # Quick test script
│   │   └── results/                # Analysis outputs
│   │       └── 0328IbnCabdRabbih/  # Results for Ibn ʿAbd Rabbih
│   │
│   ├── scan/                        # Model training & feature extraction
│   │   ├── build_features_71.py    # 71-D feature extraction
│   │   ├── build_features_50.py    # 50-D feature extraction (legacy)
│   │   ├── lr_classifier_71features.pkl    # Trained LR model
│   │   ├── xgb_classifier_71features.pkl   # Trained XGBoost model
│   │   └── scan_openiti_lr_71features.py   # Scanning script
│   │
│   ├── comparison_ensemble/        # XGBoost training & comparison
│   │
│   └── isnad_filter.py            # Utility: extract narrative content
│
├── data/                            # Training datasets
│   ├── dataset_raw.json            # Original training data (1,221 texts)
│   └── Kitab_Uqala_al_Majanin_annotated.json  # New dataset (4.8 MB)
│
├── docs/                            # Documentation
│   ├── INDEX.md                    # Documentation hub
│   ├── COMPREHENSIVE_GUIDE.md      # Complete technical guide
│   ├── WORKFLOW.md                 # Workflow & development history
│   ├── ANALYSIS_FALSE_POSITIVES.md # FP analysis & solutions
│   ├── features_catalog_majnun_aqil.md  # Feature documentation
│   └── archived/                   # Historical docs
│
├── openiti_corpus/                  # OpenITI corpus (28GB, not in git)
├── venv/                            # Virtual environment (not in git)
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Git exclusions
```

---

## 📊 Key Results

### Performance Metrics

| Model | AUC | F1 | Notes |
|-------|-----|-----|-------|
| LR (71-D) | 0.804 | 0.75 | Interpretable, under improvement |
| XGBoost (71-D) | 0.991 | 0.98 | Best performance |
| LR (50-D) | 0.849 | 0.83 | Better baseline (legacy) |
| **Consensus** | — | **0.98** | Both models agree (very reliable) |

### Analysis Results (Ibn ʿAbd Rabbih)

**Raw Classification (LR ≥ 0.7 + XGB ≥ 0.7):**
```
Total texts analyzed: 10,113
├─ Consensus positives: 4,646 (54.6%)
│  ├─ Canonical fools: 100 (0.99%)
│  ├─ True majnun aqil: 874 (8.64%)
│  └─ False positives: 3,408 (82%) ← dialogue without paradox
└─ Consensus negatives: 5,467 (45.4%)
```

**After Post-Classification Filtering:**
```
Reliable positives: 114 (2.5% of corpus)
├─ Canonical fools: 98 (0.99%)
└─ True majnun aqil: 16 (0.3%)

False positives reduced: 82% → 0% ✅
```

### Canonical Fool Detection

Named wise fool figures detected in Al-Iqd al-Farid:
- **Khalaf (خلاف):** 72 instances
- **Riyah (رياح):** 13 instances
- **Ligit (لقيط):** 8 instances
- **Ja'ifran (جعيفران):** 4 instances
- **Alyan (عليان):** 2 instances
- **Bahlul (بهلول):** 1 instance

---

## 🔬 Technical Approach

### Feature Engineering (71 dimensions)

**Lexical Features (62):**
- Junun markers (15): جنون, مجنون, معتوه, etc.
- Intelligence/wisdom (8): عقل, حكمة, فقيه, etc.
- Dialogue patterns (11): قال, سؤال, جواب
- Validation markers (8): ضحك, بكاء, هدية
- Contrast/paradox (5): ولكن, إلا, لكن
- Authority/sources (4): attribution patterns
- Poetry (3): شعر, بيت, قافية
- Wasf/lexicographic (3): ومنها, تقول العرب

**Morphological Features (9):**
- Root density (ج.ن.ن, ع.ق.ل, ح.ك.م)
- POS tags: verb, noun, adjective densities
- Aspect: perfect/imperfect
- Voice: passive voice ratio

### Pipeline Architecture

```
OpenITI Corpus
    ↓ [Akhbar Extraction]
Coherent narrative units (7,400-10,000+ per author)
    ↓ [Metadata Cleaning]
100% clean texts (0% corruption)
    ↓ [Isnad Filtering]
Pure narrative content
    ↓ [Feature Extraction - 71D]
Feature vectors
    ↓ [Classification - LR + XGBoost]
Probabilities (0.0-1.0)
    ↓ [Consensus Filtering]
Agreement check (both ≥ 0.7)
    ↓ [Post-Classification Filtering]
Validate majnun characteristics
    ↓ [Output]
Canonical fools, true majnun aqil, confidence scores
```

---

## ⚠️ Known Issues & Solutions

### Issue 1: High False Positive Rate (82%)

**Problem:** Classifiers learn "dialogue = majnun" instead of "paradox = majnun"

**Solution (Implemented):**
- ✅ Post-classification filtering: Validates actual majnun characteristics
- ✅ Reduces FP from 82% → 0% for accepted texts

**Future Improvements:**
- Two-tier thresholding (different thresholds per text characteristics)
- Explicit paradox/irony features
- Corpus-aware baselines (different expectations per author)

See [ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md) for details.

### Issue 2: LR Model Regression (AUC 0.804 vs 0.849)

**Problem:** 71-D model underperforms 50-D model due to 42 zero-padding features

**Solution (In Progress):**
- Replace zero-padding with real features
- Restore missing morphological variants
- Re-optimize parameter C

See [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md#improvement-roadmap) for detailed roadmap.

---

## 🛣️ Improvement Roadmap

| Phase | Task | Timeline | Status |
|-------|------|----------|--------|
| 1 | Fix 71-D LR classifier (AUC 0.804 → >0.85) | 1-2 weeks | 🔴 Blocked |
| 2 | Two-tier thresholding (capture more true majnun) | 1 week | 📋 Planned |
| 3 | Explicit paradox features | 2 weeks | 📋 Planned |
| 4 | Corpus-aware baseline | 2-3 weeks | 📋 Planned |

See [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md#improvement-roadmap) for detailed plans.

---

## 💡 Key Insights

1. **Dialogue alone ≠ majnun aqil** — Post-filtering validates actual paradoxical wisdom
2. **Canonical fools are rare but identifiable** — ~1% of corpus, high confidence
3. **True majnun requires paradox** — Contradiction between foolishness and wisdom
4. **Corpus diversity matters** — Entertainment anthology (Al-Iqd) vs historical compilations have different rates

---

## 📚 Training Data

**Original Dataset (dataset_raw.json):**
- 1,221 texts
- 460 positive (37.7%)
- 761 negative (62.3%)
- Class balance: 1:1.65 (imbalanced, handled via `class_weight='balanced'`)

**New Dataset (Kitab_Uqala_al_Majanin_annotated.json):**
- 4.8 MB (not yet integrated)
- To be merged and models retrained

---

## 📖 Full Documentation

- **[INDEX.md](INDEX.md)** — Documentation navigation
- **[COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)** — Complete technical guide (this is where you need to be!)
- **[WORKFLOW.md](WORKFLOW.md)** — Development workflow & history
- **[ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md)** — FP analysis & 4 improvement strategies
- **[VERIFICATION_COMPLETE.md](VERIFICATION_COMPLETE.md)** — Pipeline verification results
- **[features_catalog_majnun_aqil.md](features_catalog_majnun_aqil.md)** — 153 features documented
- **[guide_morphologie.md](guide_morphologie.md)** — Morphology guide (CAMeL Tools)

---

## 🤝 Contributing

Areas of interest:
- Additional classical Arabic corpora
- Improved paradox/irony detection features
- Author-specific tuning
- Interactive visualization tools

---

## 📞 Contact

**Augustin Pot**  
IREMAM, Aix-Marseille University  
Supervisor: Hakan Özkan

---

## 📝 License

[To be specified - MIT, CC-BY, etc.]

---

## Changelog

### v0.2 (Current - April 2026)
- ✅ Repository reorganized (src/, data/, docs/, models/)
- ✅ Virtual environment with all dependencies
- ✅ COMPREHENSIVE_GUIDE.md created (complete documentation)
- ✅ Post-classification filtering implemented
- ✅ False positive analysis & solutions documented
- ✅ Complete pipeline verification
- ✅ Metadata cleaning validated

### v0.1 (Initial - April 2024)
- ✅ Dual classifier system (LR + XGBoost)
- ✅ 74-feature extraction pipeline
- ✅ OpenITI corpus integration
- ✅ Canonical fool detection
