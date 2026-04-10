# Uqala NLP - Documentation Index

Bienvenue! Ce projet détecte les figures du **fou sensé (ʿāqil majnūn)** dans les textes arabes classiques en utilisant le machine learning.

## 🚀 Démarrage rapide

**Nouveau sur le projet?** Commencez ici:

1. **[README.md](README.md)** — Vue d'ensemble du projet, résultats, structure
2. **[WORKFLOW.md](WORKFLOW.md)** — Flux complet: données → entraînement → détection → analyse

## 📚 Documentation par sujet

### Classification et Modèles
- **[ANALYSIS_FALSE_POSITIVES.md](ANALYSIS_FALSE_POSITIVES.md)** — Analyse des 82% de faux positifs et solutions (Post-filtering, Two-tier thresholding, Feature engineering)
- **[VERIFICATION_COMPLETE.md](VERIFICATION_COMPLETE.md)** — Vérification du pipeline: extraction, features, classification

### Features et Linguistique  
- **[features_catalog_majnun_aqil.md](features_catalog_majnun_aqil.md)** — Catalogue complet des 74 features (lexicales, morphologiques)
- **[guide_morphologie.md](guide_morphologie.md)** — Guide détaillé: CAMeL Tools, POS, racines, morphologie arabe

### Améliorations et Solutions
- **[EXTRACTION_IMPROVEMENTS.md](EXTRACTION_IMPROVEMENTS.md)** — Nettoyage des métadonnées OpenITI (résolu)
- **[OPTION_C_SUMMARY.md](OPTION_C_SUMMARY.md)** — Optimisation du paramètre C pour Logistic Regression (résolu)

## 📂 Structure du projet

```
src/
├── openiti_detection/    # Pipeline de détection principal
│   ├── detect_lr_xgboost.py    # Classification LR + XGBoost
│   ├── strict_analysis.py      # Analyse stricte avec détection canonical
│   ├── post_filter.py          # Post-classification filtering
│   ├── analyze_false_positives.py
│   └── results/                # Résultats d'analyse par auteur
│
├── scan/                 # Entraînement des modèles
│   ├── build_features_71.py    # Feature extraction (71-D)
│   ├── build_features_50.py    # Feature extraction (50-D, legacy)
│   ├── scan_openiti_lr_71features.py
│   ├── lr_classifier_71features.pkl
│   └── lr_report_71features.json
│
└── comparison_ensemble/  # Comparaison LR vs XGBoost
    ├── train_xgboost_71features.py
    ├── xgb_classifier_71features.pkl
    └── results/

data/
└── dataset_raw.json      # 1,221 textes annotés (460 pos, 761 neg)

models/                   # Répertoire pour modèles supplémentaires
```

## 🎯 Résultats clés

### Ibn ʿAbd Rabbih (Al-Iqd al-Farid)
- **10,113 textes** analysés
- **100 fous canoniques** (Khalaf, Bahlul, etc.) — 0.99% du corpus
- **874 candidats majnun aqil** (critères stricts)
- **3,408 faux positifs** (dialogue seul, sans paradoxe)

### Après Post-classification Filtering
- **114 positifs fiables** (2.5% du corpus)
  - 98 fous canoniques (0.99%)
  - 16 vrais majnun aqil (0.3%)
- Faux positifs réduits: **82% → 0%**

### Performance des classifiers
| Model | AUC | F1 | Notes |
|-------|-----|-----|-------|
| LR (71 features) | 0.804 | 0.75 | Interprétable |
| XGBoost (71 features) | 0.991 | 0.98 | High-performance |
| Consensus (LR + XGB) | - | 0.98 | Best reliability |

## ⚙️ Installation

```bash
# Cloner le repo
git clone https://github.com/gsspt/Uqala_NLP.git
cd Uqala_NLP

# Créer un environnement virtuel
python -m venv venv
source venv/Scripts/activate  # Windows
# ou: source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## 🔬 Utilisation rapide

### 1. Tester sur une seule œuvre

```bash
cd src/openiti_detection
python test_quick.py 0328IbnCabdRabbih
```

### 2. Analyse stricte (avec détection canonique)

```bash
python strict_analysis.py 0328IbnCabdRabbih --threshold-lr 0.7 --threshold-xgb 0.7
```

### 3. Vérifier la qualité du pipeline

```bash
python verify_extraction.py
python verify_feature_extraction.py
```

## 🐛 Problèmes connus et solutions

### Faux positifs élevés (82%)
**Cause:** Les classifiers apprennent "dialogue = majnun" au lieu de "paradoxe = majnun"

**Solutions implémentées:**
1. ✅ **Post-classification filtering** — Valide les caractéristiques réelles du majnun (paradoxe, junun markers, sagesse)
2. ✅ **Two-tier thresholding** — Seuils différents selon la présence de markers junun
3. 📋 **Feature engineering** — Ajouter des features explicites de paradoxe et ironie (en cours)

**Résultats:** Réduit les faux positifs de 82% → 0% pour les textes acceptés.

## 📊 Métriques de qualité

### Extraction d'akhbars
- ✅ **7,400-10,000+** akhbars par auteur
- ✅ **94%** cohérence narrative (vérifiée par LLM)
- ✅ **0%** corruption de métadonnées OpenITI
- ✅ **80%** unités narratives complètes

### Classification
- **Entraînement:** 1,221 textes (460 positifs, 761 négatifs)
- **Déséquilibre:** 1:1.65 (géré par class_weight='balanced')
- **Features:** 71-D (62 lexicales + 9 morphologiques)

## 🔧 Améliorations futures

### Court terme (1-2 semaines)
- [ ] Améliorer le classifier 71-features (AUC 0.804 → >0.85)
- [ ] Tester Two-tier thresholding
- [ ] Analyser d'autres auteurs (Ibn Jawzi, Al-Tabarani)

### Moyen terme (3-4 semaines)
- [ ] Features explicites de paradoxe et ironie
- [ ] Réentraîner les modèles
- [ ] Baseline par corpus (Ibn ʿAbd Rabbih ≠ Ibn Jawzi)

### Long terme
- [ ] Système de classification aware du corpus/auteur
- [ ] Interface interactive pour explorer les résultats
- [ ] Publication des résultats

## 📖 Références clés

- **Akhbar extraction:** [EXTRACTION_IMPROVEMENTS.md](EXTRACTION_IMPROVEMENTS.md)
- **Détection canonique:** [WORKFLOW.md](WORKFLOW.md) → "Canonical Fool Detection"
- **Features:** [features_catalog_majnun_aqil.md](features_catalog_majnun_aqil.md)
- **Morphologie arabe:** [guide_morphologie.md](guide_morphologie.md)

## 🤝 Contribution

Zones d'intérêt:
- Nouveaux corpus arabes classiques
- Amélioration de la détection de paradoxe/ironie
- Tuning spécifique par auteur
- Outils de visualisation interactive

## 📞 Contact

**Augustin Pot**  
IREMAM, Aix-Marseille University  
Superviseur: Hakan Özkan

---

**Version:** 0.2 (April 2024)  
**Dernière mise à jour:** 2026-04-10  
**Statut:** Production-ready (LR + XGBoost classifiers, post-filtering active)
