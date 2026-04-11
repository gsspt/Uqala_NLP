# Prochaines Étapes — v80 Validation & Enrichissement

**Date** : 2026-04-10  
**Current State** : v80 implémenté, testé, 0.869 CV AUC  
**Status** : 🟢 Ready for validation & external testing

---

## 📋 TODO LIST (Priorités)

### 1. IMMEDIATE (Before push to main)

#### 1a. CAMeL Tools Installation
**Why** : Morphological features (f65-f70) currently disabled  
**What** : `pip install camel-tools` + test  
**Expected impact** : +0.5-1% AUC (based on v79)

```bash
# Try different installation method if pip fails
pip install camel-tools --no-cache-dir
# or via conda
conda install -c conda-forge camel-tools
```

#### 1b. External Corpus Validation
**What** : Test v80 on Ibn Abd Rabbih (external test set)  
**Why** : v79 had 54.6% false positives — check if v80 improves  
**How** :
```bash
python scripts/evaluate_on_ibn_abd_rabbih.py \
  --classifier models/lr_classifier_v80.pkl \
  --output results/v80_ibn_abd_rabbih.json
```
**Success metric** : < 40% false positives (better than v79's 54.6%)

#### 1c. Top 10 Errors Review (Manual)
**What** : Sample 10 false positives + 10 false negatives from test set  
**Why** : Understand failure modes  
**How** : 
- Extract test set predictions
- Sample by confidence (closest to 0.5)
- Manual annotation: why does classifier fail?
- Document patterns in `errors_v80.md`

---

### 2. SHORT-TERM (This week)

#### 2a. Feature Enrichment (Manual)

**E3_dialogue_first_person_density** :
- Current: only قلت
- **Check** : Any other forms of "I said" distinctive? (قلت variant forms, فقلت, قلنا)
- **Action** : Scan positives for similar dialogue patterns

**E6_sacred_spaces** :
- Current: أزقة, المقابر, خرابات, قبر, سوق, مسجد
- **Check** : Any missing liminal spaces? (خرائب, قبالة, طريق, نهر?)
- **Action** : Check corpus for other space terms (ratio > 2x)

**E5_divine_personal** :
- Current: إلهي, اللهم, يا رب, يا إلهي
- **Check** : Other personal invocations? (ربي, يا ربي?)
- **Action** : Systematic scan for divine invocation patterns

#### 2b. Create v81 (Refined)
**Rationale** : Incorporate enriched lexicons from 2a  
**Target** : Same 28 features, better lexicons  
**Method** :
```python
# After manual review, update EMPIRICAL_LEXICONS in v81
# Run p1_4_logistic_regression_v80.py with updated lists
# Compare CV AUC: v80 vs v81
```

---

### 3. MEDIUM-TERM (Next 2 weeks)

#### 3a. Context-Aware Paradox Feature
**Idea** : Current paradox markers (لكن, إلا) are weak (1.1-1.5x)  
**Why** : Paradox only matters WITHIN junun context  

**Implementation** :
```python
# E8_paradox_with_junun (NEW)
def has_paradox_context(text):
    junun_positions = [match.start() for match in re.finditer(junun_pattern, text)]
    paradox_positions = [match.start() for match in re.finditer(paradox_pattern, text)]
    
    for jp in junun_positions:
        for pp in paradox_positions:
            if abs(jp - pp) < 200:  # window
                return True
    return False
```

**Expected signal** : 2-3x (tested on corpus)

#### 3b. Structural Features (Propp-like)
**Idea** : Narrative arc of khbar (introduction → action → paradox → resolution)  
**How** :
- E9_character_introduction_early (first character name in first 20%)
- E10_action_middle (high verb density in middle third)
- E11_resolution_or_ambiguity (lacks closing statement)

**Expected signal** : 1.5-2x each

#### 3c. Comparison: v80 (LR) vs v79 (XGB)
**What** : Build ensemble (LR vote + XGB vote)  
**Why** : May reduce false positives on Ibn Abd Rabbih  
**How** :
```bash
# Create A1_ensemble.py that:
# 1. Loads v80 LR classifier
# 2. Loads v79 XGB classifier
# 3. Averages probabilities (50/50 or weighted)
# 4. Tests on external corpus
```

---

### 4. LONG-TERM (v82+)

#### 4a. Active Learning Loop
**Why** : Ibn Abd Rabbih has 54.6% false positives → need targeted refinement  
**How** : Implement C1 Active Learning pipeline
- Predict on full Ibn Abd Rabbih
- Sample high-confidence errors
- Manual annotation
- Retrain v82

#### 4b. Actantial Features (Greimas)
**Idea** : Structural semantics of fool-sage relationship  
**What** :
- Protagonist role (fool, sage, observer)
- Actant relationships (fool ← → sage, fool → observer)
- Requires LLM annotation (DeepSeek)

#### 4c. Corpus Expansion
**What** : Annotate more akhbars (target: 1000+ positive examples)  
**Why** : 460 is small; 1000+ enables better model  
**How** : Systematic sampling from Nisaburi + other sources

---

## 🎯 Success Metrics

| Milestone | Metric | Target | Status |
|-----------|--------|--------|--------|
| v80 baseline | CV AUC | 0.869 | ✅ Done |
| v80 w/ CAMeL | CV AUC | > 0.870 | ⏳ Pending |
| External valid. | False Pos (Ibn Abd Rabbih) | < 40% | ⏳ Pending |
| v81 refined | CV AUC | > 0.875 | ⏳ In progress |
| v82 ensemble | False Pos | < 35% | ⏳ Future |
| Final production | False Pos | < 30% | ⏳ Future |

---

## 📝 Decision Points

**Before continuing, decide:**

**Q1** : Should we enrich features manually (2a), or accept v80 as-is?  
**Recommendation** : DO manual review (2a) — 30 min for substantial gains

**Q2** : Do we need context-aware paradox (3a)?  
**Recommendation** : Maybe later — current empirical features are strong enough

**Q3** : Ensemble (LR v80 + XGB v79)?  
**Recommendation** : YES — test on Ibn Abd Rabbih, may reduce false positives

**Q4** : Active learning loop (4a)?  
**Recommendation** : YES, but after external validation shows failures

---

## 🔄 Timeline Estimate

- **Phase 1 (now)** : CAMeL install + external valid = 2 hours
- **Phase 2 (this week)** : Manual enrichment + v81 = 3-4 hours
- **Phase 3 (next week)** : Ensemble + Ibn Abd Rabbih tuning = 4-6 hours
- **Phase 4 (ongoing)** : Active learning + corpus expansion = 10+ hours

---

## 📞 Questions for Augustin

1. **Empirical lexicons** — Are there other scene-intro or habitual verbs you see often in Nisaburi?

2. **Sacred spaces** — Besides المقابر, خرابات, أزقة, are there other liminal locations characteristic of the fool-sage?

3. **Divine invocation** — Besides إلهي, اللهم, يا رب, any other personal forms?

4. **False positives** — On Ibn Abd Rabbih, what TYPES of texts get wrongly classified as majnun aqil? (e.g., other narrative genres, specific authors?)

5. **Active learning** — Would you have time to manually annotate 50-100 high-confidence errors on Ibn Abd Rabbih?

---

## 📂 Reference Files

| File | Purpose |
|------|---------|
| `EMPIRICAL_LEXICONS.md` | All lexicons extracted + ratios |
| `FEATURE_PROPOSAL.md` | Original architecture rationale |
| `V80_RESULTS.md` | Current performance report |
| `p1_4_logistic_regression_v80.py` | Implementation code |
| `models/lr_report_v80.json` | Coefficients & metrics |

