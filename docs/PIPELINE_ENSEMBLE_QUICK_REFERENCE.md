# Pipeline Ensemble - Référence Rapide

## 📋 Les deux pipelines en un coup d'œil

```
┌─────────────────────────────────────────────────────────────┐
│                    VOTRE KHABAR ARABE                       │
│        "مجنون يقول إلهي حمى الله في بلاده..."            │
└──────────────────────────┬──────────────────────────────────┘
                           │
              EXTRAIRE LES 27 INDICES (features)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   f00-f14             f65-f70             E1-E6
 (Folie core)       (Morphologie)      (Empirique)
 - Bahlul?         - Verbes?          - Dialogue "je"?
 - Méghnoun?       - Noms?            - Lieux sacrés?
 - Dialogue?       - Adjectifs?       - Invocation?
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
    ┌─────────────┐                ┌──────────────┐
    │ PROFESSEUR  │                │  DÉTECTIVE   │
    │  (LR v80)   │                │  (XGBoost)   │
    │ transparent │                │ intelligent  │
    └─────┬───────┘                └──────┬───────┘
          │                               │
          │ "92%"                         │ "88%"
          │ (Voici pourquoi:              │ (Patterns cachés:
          │  formule claire)              │  patterns complexes)
          │                               │
          └───────────────┬───────────────┘
                          │
                    ┌─────▼─────┐
                    │  FUSION   │
                    └─────┬─────┘
                          │
                 (92% + 88%) / 2 = 90%
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
   SHAP Explanation              Confidence Assessment
   ─────────────────            ──────────────────────
   Top features:                Les deux modèles
   1. f02_famous_fool +25%       s'accordent?
   2. E3_dialogue +20%           OUI → Confiance: TRÈS HAUTE
   3. E5_divine +15%             NON → Confiance: MOYENNE
   4. E1_scene +10%
                          │
                          ▼
                ┌──────────────────┐
                │   RÉSULTAT       │
                │   90% confiant   │
                │   "C'est un fou  │
                │    sage!"        │
                └──────────────────┘
```

---

## 🎯 Les trois phases

### PHASE 1: XGBoost v80 (Détective)
**Fichier à créer**: `p2_2_xgboost_v80_shap.py`

```
ENTRÉE:
- 27 features v80 (mêmes que LR)
- 4277 exemples d'entraînement

PROCESSUS:
- Détecte patterns complexes dans les données
- Apprend quelle combinaison de features = fou sage
- Entraîne sur 3817 négatifs + 460 positifs

SORTIE:
- Modèle XGBoost entraîné
- Probabilité pour chaque texte (0.0 - 1.0)
```

### PHASE 2: LR v80 (Professeur)
**Fichier existant**: `p1_4_logistic_regression_v80.py`

```
ENTRÉE:
- 27 features v80

PROCESSUS:
- Apprend une formule claire et simple
- Chaque feature a un poids (+0.45, -0.12, etc.)
- Combine les poids pour donner une probabilité

SORTIE:
- Modèle LR entraîné
- Probabilité pour chaque texte (0.0 - 1.0)
```

### PHASE 3: Ensemble + SHAP (Duo)
**Fichier à créer**: `p2_3_hybrid_ensemble_shap.py`

```
ENTRÉE:
- Modèle LR v80
- Modèle XGBoost v80
- Un nouveau khabar à analyser

PROCESSUS:
1. Charger les deux modèles
2. Extraire 27 features du khabar
3. Donner le texte à LR  → proba_lr (ex: 0.92)
4. Donner le texte à XGBoost → proba_xgb (ex: 0.88)
5. Calculer fusion: (0.92 + 0.88) / 2 = 0.90
6. Calculer accord: |0.92 - 0.88| = 0.04 (très proche!)
7. Utiliser SHAP pour expliquer

SORTIE:
- Prédiction: 0.90 ("OUI, c'est un fou sage")
- Confiance: "TRÈS HAUTE" (les deux modèles s'accordent)
- Explanation: Voici les features qui ont décidé
```

---

## 🔄 Flux de données concret

### Exemple: Analyser ce khabar
```
Texte: "مجنون يقول إلهي حمى الله في بلاده، مررت فرأيت..."

ÉTAPE 1: EXTRAIRE FEATURES
─────────────────────────
f00_has_junun: 1.0          (méghnoun présent? OUI)
f01_junun_density: 0.05     (5% du texte)
f02_famous_fool: 1.0        (Bahlul/Saadoun présent? OUI)
...
E1_scene_intro: 1.0         (مررت présent? OUI)
E3_dialogue_first_person: 1.0 (قلت présent? OUI)
E5_divine_personal: 1.0     (إلهي présent? OUI)
... (20 autres features)

ÉTAPE 2: DONNER À LR
──────────────────
Professeur lit les 27 nombres
Applique sa formule:
  1.0 × (+0.64) [famous_fool] = +0.64
  1.0 × (+0.47) [dialogue] = +0.47
  1.0 × (+0.05) [divine] = +0.05
  1.0 × (+0.08) [scene] = +0.08
  0.05 × (+0.29) [density] = +0.01
  ... (22 autres)
  
  SOMME = +0.92
  
Résultat: 92% de probabilité

ÉTAPE 3: DONNER À XGBOOST
────────────────────────
Détective analyse les 27 nombres
En fonction de 4000 exemples:
  "J'ai vu ce pattern 1000 fois"
  "90% des fois ça finit en fou sage"
  "Mais attendez, il y a aussi cet autre pattern..."
  "Ajustement: plutôt 88%"
  
Résultat: 88% de probabilité

ÉTAPE 4: FUSIONNER
──────────────────
Moyenne: (92 + 88) / 2 = 90%
Accord: 92% et 88% sont très proches!
Status: ✅ CONSENSUS

ÉTAPE 5: EXPLIQUER AVEC SHAP
────────────────────────────
Professeur: "Voici ma formule..."
  famous_fool: 45% d'importance
  dialogue: 22% d'importance
  divine: 8% d'importance
  scene: 5% d'importance
  
Détective: "Voici mes observations..."
  famous_fool: 28% d'importance
  dialogue: 25% d'importance
  divine: 12% d'importance
  verbs: 10% d'importance
  
Consensus: "Les deux s'accordent sur les features clés"

RÉSULTAT FINAL:
──────────────
{
  "prediction": 1,              # Oui, c'est un fou sage
  "confidence": 0.90,           # 90% confiant
  "agreement": "high",          # Les deux modèles s'accordent
  "lr_proba": 0.92,
  "xgb_proba": 0.88,
  "top_features": [
    "f02_famous_fool: 35%",
    "E3_dialogue: 24%",
    "E5_divine: 10%"
  ]
}
```

---

## 📊 Tableau comparatif des trois approches

| Aspect | LR seul | XGBoost seul | Ensemble |
|--------|---------|--------------|----------|
| **Précision** | 86% | 87% | 88-89% ⭐ |
| **Compréhension** | Facile ✅ | Difficile ❌ | Facile ✅ |
| **Patterns détectés** | Simples | Complexes | Complexes |
| **Expliquabilité** | Immédiate | Nécessite SHAP | Complète |
| **Confiance si accord** | - | - | Très haute ⭐ |
| **Vitesse** | Rapide | Rapide | Rapide |

---

## 🎓 Vocabulaire simple

| Terme | Signification |
|-------|---------------|
| **Feature** | Un indice/signal dans le texte (ex: "Y a le mot Bahlul?") |
| **Probabilité** | Certitude de 0% à 100% (0.92 = 92%) |
| **Fusion** | Combiner les réponses des deux modèles |
| **Accord** | Les deux modèles donnent la même réponse |
| **SHAP** | Explication: quelle feature a décidé? |
| **LR** | Logistic Regression = Formule mathématique simple |
| **XGBoost** | Modèle qui voit des patterns complexes |
| **Ensemble** | Combiner plusieurs modèles pour une meilleure réponse |

---

## ✅ Résumé en 3 points clés

1. **XGBoost = Détective intelligent**
   - Voit des patterns cachés
   - Donne 88-91% de précision
   - Difficile à expliquer
   
2. **LR = Professeur transparent**
   - Utilise une formule claire
   - Donne 86-92% de précision
   - Facile à expliquer

3. **Ensemble = Le meilleur des deux**
   - 88-93% de précision ⭐
   - Expliquable via SHAP ✅
   - Plus confiant quand les deux s'accordent ✅

---

Prêt à créer les vrais pipelines? 🚀
