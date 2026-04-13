# v80 Model Comparison: Ibn Abd Rabbih vs Ibn Mu'tazz

**Date**: 2026-04-12  
**Models**: LR v80, XGBoost v80, Ensemble (LR+XGBoost averaging)  
**Extraction**: v2_smart (semantic heuristics for akhbar boundaries)

---

## Dataset Summary

| Corpus | Author | Work | Total Akhbars | Source Path |
|--------|--------|------|---|---|
| **Ibn Abd Rabbih** | Abū 'Alī Muḥammad ibn Aḥmad ibn Isḥāq | *al-'Iqd al-Farīd* (Unique Necklace) | 10,286 | `openiti_corpus/data/0328IbnCabdRabbih` |
| **Ibn Mu'tazz** | 'Abdullāh ibn Muḥammad ibn Ya'qūb | *Ṭabaqāt al-Shuʿarāʾ* (Classes of Poets) | 36 | `openiti_corpus/data/0296IbnMuctazz` |

---

## Model Performance Comparison

### Raw Counts

| Model | Ibn Abd Rabbih | Ibn Mu'tazz |
|-------|---|---|
| **LR v80** | 606 positives | 5 positives |
| **XGBoost v80** | 66 positives | 1 positives |
| **Ensemble** | 238 positives | 2 positives |

### Positive Rates (%)

| Model | Ibn Abd Rabbih | Ibn Mu'tazz |
|-------|---|---|
| **LR v80** | 5.89% | 13.9% |
| **XGBoost v80** | 0.64% | 2.8% |
| **Ensemble** | 2.31% | 5.6% |

---

## Key Observations

### 1. **Model Disagreement Persists Across Corpora**
Both corpora show consistent pattern:
- **LR v80 is permissive** (detects ~6-14% positives)
- **XGBoost v80 is conservative** (detects ~0.6-2.8% positives)
- **Ensemble compromises** between the two

### 2. **Higher Rates in Ibn Mu'tazz**
Despite only 36 akhbars (vs 10,286 in Ibn Abd Rabbih):
- LR detects 13.9% (vs 5.89%) — **2.4× higher rate**
- XGBoost detects 2.8% (vs 0.64%) — **4.4× higher rate**
- Ensemble detects 5.6% (vs 2.31%) — **2.4× higher rate**

**Interpretation**: Ibn Mu'tazz's *Tabaqat al-Shuʿarāʾ* is a specialized collection of poet biographies and wise sayings, likely containing more paradoxical wisdom and character anecdotes than Ibn Abd Rabbih's broader anthology. This makes it naturally richer in majnun aqil-like narratives.

### 3. **Model Agreement**

| Pair | Ibn Abd Rabbih | Ibn Mu'tazz |
|-----|---|---|
| LR ∩ XGBoost | khabar_num 20 | **khabar_num 20** |
| Both models agree on the **same high-confidence positive** in Ibn Mu'tazz |

Both models independently ranked the same akhbar (khabar_num 20) as top positive:
- **LR probability**: 0.9686
- **XGBoost probability**: 0.5307
- **Content**: Majnun in ruins giving spiritual advice about fear/hope

This **single point of agreement** is a strong signal (highest confidence across both models).

---

## Ibn Mu'tazz Results Details

### XGBoost (1 positive, most conservative):
- **khabar_num 20** (prob 0.531): Majnun giving spiritual counsel in ruins ← **Most reliable positive**

### Ensemble (2 positives):
- **khabar_num 20** (prob 0.750): Majnun character (see above)
- **khabar_num 8** (prob 0.596): Foolish neighbor (معتوه) with profound poetry insights

### LR (5 positives, most permissive):
- All 5 ranked 0.726-0.969
- Include both above + 3 additional wisdom collections (khabar_num 3, 19, 28)
- These 3 extras are wisdom sayings but may lack narrative majnun aqil structure

---

## Recommendations

### For Ground Truth on Ibn Mu'tazz
Given the small corpus size (36 akhbars), **manual validation is feasible**:
1. **High confidence** (XGBoost only): khabar_num 20 ← Likely true positive
2. **Medium confidence** (Ensemble): khabar_num 8 ← Likely true positive
3. **Low confidence** (LR only): khabar_num 3, 19, 28 ← Likely false positives (wisdom without majnun aqil narrative)

### Generalization Insights
Model behavior generalizes across corpora:
- **LR v80** suffers from over-fitting on wisdom signals (paradox, dialogue, aql vocabulary)
- **XGBoost v80** over-regularizes, missing valid positives (too conservative)
- **Ensemble** balances but still requires domain validation
- **v80 v79 trend continues**: Higher rates on domain-specific collections (poets, wise sayings) than general anthologies

---

## Files Generated

```
results/
├── data/
│   ├── validation_lr_data.json           (5 positives, full text)
│   ├── validation_xgboost_data.json      (1 positive, full text)
│   ├── validation_ensemble_data.json     (2 positives, full text)
│   └── comparison_data.json              (summary statistics)
└── CORPUS_COMPARISON_v80_MODELS.md       (this file)
```

To access Ibn Mu'tazz results:
```bash
cat results/data/validation_xgboost_data.json       # Most reliable
cat results/data/validation_ensemble_data.json      # Balanced
cat results/data/validation_lr_data.json            # All detected
```

---

## Next Steps

1. **Manual annotation** of the 5 LR positives to ground truth Ibn Mu'tazz
2. **Analyze disagreement** between LR and XGBoost on khabar_num 8 (ensemble case)
3. **Compare linguistic features** of khabar_num 20 vs false positives to refine feature set
4. **Consider C1 Active Learning** pipeline for uncertain cases in larger corpora
