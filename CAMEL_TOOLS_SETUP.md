# CAMeL Tools Setup & Debugging

**Status** : ✅ CAMeL Tools successfully installed and tested  
**Environment** : `C:\Users\augus\.conda\envs\uqala`  
**Python** : 3.11.15  
**Date** : 2026-04-11

---

## 📋 Summary

### Problem
The global Windows Python (3.13) couldn't find CAMeL Tools, even though it was needed for morphological feature extraction (f65-f70).

### Root Cause
- Python scripts were running with the **global Windows Python** (`C:\Users\AppData\...`)
- CAMeL Tools was installed in the **conda environment** (`~/.conda/envs/uqala`)
- These are separate Python environments with different installed packages

### Solution
Created `run_with_venv.py` — a wrapper that executes scripts with the correct Python environment.

---

## 🚀 How to Use

### Option 1: Use the wrapper (Recommended)
```bash
python3 run_with_venv.py pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 5
```

**Advantages**:
- Automatic environment management
- Works from anywhere
- Ensures CAMeL Tools is available

### Option 2: Activate venv manually
```bash
# On Windows (cmd or PowerShell)
C:\Users\augus\.conda\envs\uqala\Scripts\activate.bat

# Then run your script
python pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 5
```

### Option 3: Use full path to venv Python
```bash
C:\Users\augus\.conda\envs\uqala\python.exe pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 5
```

---

## 🔧 Installation Details

### CAMeL Tools Installation
```bash
# Installed via pip in the uqala conda environment
C:\Users\augus\.conda\envs\uqala\python.exe -m pip install camel-tools
```

### Dependencies Installed
- `camel-tools` 1.5.7 (core)
- `torch` 2.11.0 (neural network backend)
- `transformers` 4.43.4 (NLP models)
- `numpy`, `scipy`, `pandas`, `scikit-learn` (data science)
- Plus ~30 other packages

**Total size** : ~2.5 GB (torch is large)

---

## ✅ Verification

### Test 1: Direct import test
```bash
python3 run_with_venv.py test_camel_venv.py
```

**Expected output**:
```
✅ CAMeL Tools imported successfully!
✅ Morphology database loaded!
✅ Analyzer created!
  مجنون           → root: ج.ن.ن      pos: noun
  قال             → root: ق.#.ل      pos: verb
  رأيت            → root: ر.#.#      pos: verb
✅ CAMeL Tools is fully functional!
```

### Test 2: Feature extraction
```bash
python3 run_with_venv.py pipelines/level1_interpretable/p1_4_logistic_regression_v80.py --cv 1
```

**Expected**:
- First line should be: `✅ CAMeL Tools loaded`
- Features should include f65-f70 with non-zero values
- Not: `⚠️  CAMeL Tools not available`

---

## 📂 Files

| File | Purpose |
|------|---------|
| `run_with_venv.py` | Wrapper script (always use this) |
| `test_camel_venv.py` | Quick CAMeL verification |
| `pipelines/level1_interpretable/p1_4_logistic_regression_v80.py` | Main pipeline (use with wrapper) |

---

## 🔍 Troubleshooting

### Issue: "CAMeL Tools not available"
**Solution**: Always use `run_with_venv.py`:
```bash
python3 run_with_venv.py <script>
```

### Issue: "Module not found: torch"
**Solution**: Reinstall camel-tools in the venv:
```bash
C:\Users\augus\.conda\envs\uqala\python.exe -m pip install --upgrade camel-tools
```

### Issue: Database loading hangs
**Solution**: This is normal on first load (downloads models). Wait 1-2 minutes.

---

## 📊 Performance Impact

CAMeL Tools adds morphological features (f65-f70):
- `f65_root_jnn_density` : ج.ن.ن (junun root)
- `f66_root_aql_density` : ع.ق.ل (aql root)
- `f67_root_hikma_density` : ح.ك.م (hikma root)
- `f68_verb_density` : verb/total tokens
- `f69_noun_density` : noun/total tokens
- `f70_adj_density` : adj/total tokens

**Expected AUC impact** : +0.5-1% (based on v79 vs v80 without morpho)

---

## 🎯 Next Steps

1. ✅ CAMeL Tools installed
2. ✅ Wrapper created & tested
3. ⏳ Re-run v80 with morphology features
4. Compare results: v80 (without E7, with morpho)
5. Validate on external corpus

