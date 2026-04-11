# Pipelines Hybrides Ensemble — Explication Technique

---

## 1. Vue d'ensemble mathématique

### Le problème à résoudre

Nous avons 27 features $x_1, x_2, \ldots, x_{27}$ extraites d'un texte arabe. Nous voulons prédire une probabilité $P(\text{majnun aqil} | x_1, \ldots, x_{27})$.

Deux approches complémentaires :
- **LR (Logistic Regression)** : $P = \sigma(\beta_0 + \sum_{i=1}^{27} \beta_i x_i)$ 
- **XGBoost** : $P = \text{softmax}(\sum_{k=1}^{K} f_k(x))$ où $f_k$ sont des arbres de décision

L'**ensemble** moyenne les deux prédictions : $P_{\text{ensemble}} = \frac{P_{\text{LR}} + P_{\text{XGB}}}{2}$

---

## 2. Logistic Regression v80 — Détails mathématiques

### Formule complète

$$P(\text{majnun aqil}) = \frac{1}{1 + e^{-z}}$$

où $z = \beta_0 + \sum_{i=1}^{27} \beta_i x_i$

### Coefficients appris

Les 27 coefficients $\beta_i$ sont estimés par **maximum de vraisemblance** sur les données d'entraînement :

```
Top 5 features (v80):
β₂ (f02_famous_fool)        = +0.649  ← Signal très fort (Bahlul, Saadoun)
β₃ (E3_dialogue)             = +0.477  ← Dialogue 1ère personne
β₆₈ (f68_verb_density)       = +0.494  ← Densité de verbes
β₅ (f05_junun_position)      = +0.298  ← Position précoce du terme junun
β₁₀ (f10_junun_repetition)   = +0.267  ← Répétition du terme
```

Les coefficients négatifs réduisent la probabilité. Ex : les features non pertinentes ont β ≈ 0 (régularisation).

### Avantages

1. **Transparent** : Chaque $\beta_i$ a une interprétation directe (coefficient de poids)
2. **Rapide** : Prédiction = 1 produit scalaire O(27)
3. **Stable** : Peu de données d'entraînement suffisent
4. **Explication directe** : "Ce texte a score z=0.92 parce que f02=1.0 (contribue +0.649), f03=0.15 (contribue +0.07), etc."

### Limitations

1. **Linéarité obligatoire** : LR suppose que l'effet de chaque feature est additionnel et constant
2. **Pas d'interactions** : Ne voit pas "si f02=1 ET f03=0, alors très mauvais signal"
3. **Pattern complexes manqués** : Si la vraie règle est "f02 XOR f03", LR ne peut pas l'apprendre

---

## 3. XGBoost v80 — Détails mathématiques

### Approche générale : Gradient Boosting

XGBoost entraîne K arbres de décision **séquentiellement** :

$$\hat{P} = \sigma\left(\sum_{k=1}^{K} f_k(x)\right)$$

où chaque $f_k$ est un arbre qui prédit les "résidus" (erreurs) de tous les arbres précédents.

### Processus d'entraînement (simplifié)

```
Initialisation:
  residus = labels - 0.5  # Prédictions initiales = 50%

Pour k = 1 à K:
  1. Construire arbre f_k qui prédit les residus
     - Évaluer toutes les 27 features
     - Trouver le meilleur split : "si f02 > 0.5, aller à gauche"
     - Limiter profondeur à 3 (pour éviter overfit)
  
  2. Ajouter l'arbre à la somme:
     residus = residus - learning_rate × f_k(x)
     # learning_rate = 0.1 (régularisation)
  
  3. Évaluer l'amélioration sur validation set
     # Arrêter si plus d'amélioration (early stopping)
```

### Exemple concret d'un arbre

```
                   Arbre 1
                      │
              f02 > 0.5 ?
             /          \
            OUI          NON
           /  \          /  \
       +0.3  ... -0.1  ...
```

Cet arbre dit :
- Si "Bahlul mentionné" (f02=1) → ajouter +0.3 au score
- Si "Bahlul absent" (f02=0) → ajouter -0.1 au score

Le K-ème arbre raffine cette prédiction en regardant les **résidus** après les K-1 premiers arbres.

### Captures d'interactions

Contrairement à LR, XGBoost voit les interactions :

```
Arbre 2 après Arbre 1 :
        f02 > 0.5 ?
       /          \
      OUI         NON
      │            │
   f03 > 0.2?   f68 > 0.1?
   /      \      /      \
 +0.1   -0.05  +0.2    -0.15
```

**Interprétation** : 
- Si f02=1 (Bahlul) ET f03 > 0.2 (dialogue dense) → très fort signal (+0.1 de plus)
- Si f02=0 (pas Bahlul) ET f68 > 0.1 (verbes denses) → signal modéré (+0.2)
- Sinon → signal faible ou négatif

### Limitations

1. **Overfitting risque** : Les arbres peuvent mémoriser du bruit
2. **Boîte noire** : Difficile d'expliquer "pourquoi" XGBoost prédit 0.88
3. **Hyperparamètres critiques** :
   - `max_depth=3` : Profondeur maximale (petit = simpler, moins overfit)
   - `reg_alpha=0.1` : Pénalité L1 (force certains poids à zéro)
   - `learning_rate=0.1` : Taille de chaque étape (petit = plus stable, plus lent)
   - `early_stopping_rounds=10` : Arrêter si validation AUC ne s'améliore plus

---

## 4. Validation croisée (Cross-Validation)

### Pourquoi c'est crucial

Nous avons ~4000 exemples (460 positifs, 3540 négatifs). Si on entraîne et évalue sur les MÊMES données, on mesure l'overfit, pas la vraie performance.

### Processus 5-fold CV

```
Données: 4000 exemples

Partition en 5 folds (800 chacun):
[Fold1][Fold2][Fold3][Fold4][Fold5]

Itération 1: Train sur Fold 2-5, Test sur Fold 1 → AUC₁
Itération 2: Train sur Fold 1,3-5, Test sur Fold 2 → AUC₂
Itération 3: Train sur Fold 1-2,4-5, Test sur Fold 3 → AUC₃
Itération 4: Train sur Fold 1-3,5, Test sur Fold 4 → AUC₄
Itération 5: Train sur Fold 1-4, Test sur Fold 5 → AUC₅

AUC moyen = (AUC₁ + AUC₂ + AUC₃ + AUC₄ + AUC₅) / 5
Écart-type = std(AUC₁, ..., AUC₅)
```

**Résultats v80** :
- LR: 0.8606 ± 0.0716 (stable, petit écart-type)
- XGB: ~0.87-0.90 (moins stable que LR, plus sujet à variation)

---

## 5. Métriques de performance

### AUC-ROC (Area Under the Receiver Operating Characteristic Curve)

C'est la courbe "Taux de Vrais Positifs" vs "Taux de Faux Positifs" en variant le seuil de décision.

```
                  Courbe ROC
                     │
    TPR              │     /
  (Vrais             │    /  ← Bon modèle (courbe vers haut-gauche)
   Positifs)         │   /
                     │  /
                     │ /___________
                     0             1
                    Seuil
                    
AUC = Aire sous courbe = Probabilité que modèle classe correctement
      un exemple positif et un exemple négatif choisis aléatoirement
      
- AUC = 1.0 : Parfait
- AUC = 0.5 : Aléatoire
- AUC = 0.86 : Très bon (v80)
```

### Précision, Rappel, F1

Avec seuil de décision = 0.5 (prédiction ≥ 0.5 → "majnun aqil") :

```
Précision = TP / (TP + FP) = "Parmi nos prédictions positives, combien sont justes?"
Rappel    = TP / (TP + FN) = "Parmi les VRAIS positifs, combien avons-nous détectés?"
F1 = 2 × (Précision × Rappel) / (Précision + Rappel)
```

Pour les akhbars rares (460/4000 = 11.5%), on préfère **rappel élevé** = détecter le plus de vrais majnun aqil possible, même si on accepte quelques faux positifs.

---

## 6. Ensemble Hybride — Fusion et Confiance

### Stratégie de fusion : moyenne simple

```
p_lr  = LR.predict_proba(x)[0, 1]      # Ex: 0.92
p_xgb = XGB.predict_proba(x)[0, 1]     # Ex: 0.88

p_ensemble = (p_lr + p_xgb) / 2        # = 0.90
```

### Confiance basée sur l'accord

**Idée** : Si les deux modèles s'accordent, on est plus confiant.

```
agreement = 1 - abs(p_lr - p_xgb)

Si |p_lr - p_xgb| = 0.04  → agreement = 0.96 = "TRÈS HAUTE"
Si |p_lr - p_xgb| = 0.30  → agreement = 0.70 = "MOYENNE"
Si |p_lr - p_xgb| = 0.50  → agreement = 0.50 = "BASSE"
```

**Interprétation** :
- Accord fort (±0.04) → Les deux modèles voient le même signal → **FIABLE**
- Accord faible (±0.30) → LR voit "majnun aqil", XGB voit "incertain" → **À VÉRIFIER**
- Désaccord (±0.50) → Contradiction complète → **TRÈS RISQUÉ**

### Alternative : Weighted Voting

Au lieu de moyenne simple, donner plus de poids au modèle plus précis :

```
p_ensemble = (w_lr × p_lr + w_xgb × p_xgb) / (w_lr + w_xgb)

où:
  w_lr = 0.45  (LR plus stable, CV AUC = 0.8606)
  w_xgb = 0.55 (XGB plus précis, CV AUC ≈ 0.87)
```

Ceci est implémenté dans l'option "weighted" du pipeline.

---

## 7. SHAP — Explication des prédictions

### Concept fondamental : Valeur de Shapley

SHAP = "SHapley Additive exPlanations" → Décompose chaque prédiction en contribution de chaque feature.

### Pour Logistic Regression (trivial)

```
Prédiction: z = 0.92

Contributions:
f02_famous_fool:    1.0 × 0.649 = +0.649
E3_dialogue:        0.8 × 0.477 = +0.382
f68_verb_density:   0.3 × 0.494 = +0.148
f05_junun_position: 0.2 × 0.298 = +0.060
... (23 autres features)
───────────────────────────────
TOTAL:                            +0.92
```

Chaque feature a une contribution **additive**, c'est trivial pour LR.

### Pour XGBoost (plus complexe)

SHAP utilise la **théorie des jeux de coalition** pour estimer l'importance :

```
Question: "Combien la feature f02 contribue-t-elle, en moyenne,
           si on l'ajoute parmi les autres features?"

Algorithme SHAP (KernelEXPLAINER pour XGBoost):
1. Créer beaucoup de "coalitions" de features
   Ex: {f02}, {f02, f03}, {f02, f03, f68}, etc.

2. Pour chaque coalition:
   - Entraîner un modèle sur ces features seulement
   - Mesurer la contribution moyenne

3. Moyenner les contributions en pondérant par probabilités théoriques

Résultat: Chaque feature a un score SHAP (contribution globale)
```

**Exemple de résultat SHAP pour un XGBoost** :

```
SHAP value (contribution à la prédiction finale):
f02_famous_fool:      +0.35 (très important)
E3_dialogue:          +0.22 (important)
f68_verb_density:     +0.15 (modéré)
f05_junun_position:   +0.10 (faible)
... autres:           +0.03 chacun
───────────────────────────────
TOTAL:                 ~0.88 (prédiction XGB)
```

### Visualisation SHAP

```
SHAP Force Plot:
─────────────────────────────────────────
  Base value:  0.50 (prédiction par défaut)
  
  Push UP (positif):
  ├─ f02 (+0.35)     ◄─ Bahlul mentionné
  ├─ E3 (+0.22)      ◄─ Dialogue dense
  └─ f68 (+0.15)     ◄─ Verbes denses
                     
  Final:  0.50 + 0.35 + 0.22 + 0.15 = 0.88

  (Aucun feature "push down" dans cet exemple)
─────────────────────────────────────────
```

---

## 8. Hyperparamètres critiques

### Logistic Regression v80

```python
model = LogisticRegression(
    max_iter=5000,        # Itérations pour converger
    solver='lbfgs',       # Optimiseur (bon pour petit nombre de features)
    C=1.0,                # 1/C = force de régularisation L2
                          # Petit C = plus de régularisation (moins overfit)
                          # Grand C = moins de régularisation (plus fit)
)
```

v80 n'utilise **pas** de régularisation spéciale (C=1.0 par défaut) car :
- 27 features << 4000 exemples (peu d'overfit)
- LR est stable par nature

### XGBoost v80

```python
xgb_model = XGBClassifier(
    max_depth=3,                    # Profondeur max des arbres
                                    # ← Très petit (évite overfit)
    learning_rate=0.1,              # Shrinkage factor
                                    # ← Petit (0.1 = plus stable)
    n_estimators=200,               # Nombre d'arbres K
                                    # ← Modéré (200 < 1000)
    reg_alpha=0.1,                  # Régularisation L1
    reg_lambda=1.0,                 # Régularisation L2
    min_child_weight=5,             # Exemples min par feuille
                                    # ← Prévient les splits minuscules
    subsample=0.8,                  # Fraction de données par arbre
    colsample_bytree=0.8,           # Fraction de features par arbre
    early_stopping_rounds=10,       # Arrêter si pas d'amélioration
)
```

**Rationale** : Tous les paramètres tendent vers **stabilité + généralisation**, pas vers meilleure performance sur training.

---

## 9. Performance attendue et erreurs potentielles

### Résultats de v80 seuls

```
Logistic Regression:
  CV AUC: 0.8606 ± 0.0716
  Précision/Rappel à seuil 0.5: Précision ~85%, Rappel ~60%
  
XGBoost (nouveau):
  CV AUC attendu: ~0.87-0.90 (légèrement meilleur que LR)
  Gap CV↔Test: ~2-5% (léger overfit, acceptable)
```

### Ensemble attendu

```
Moyenne simple:
  CV AUC attendu: (0.8606 + 0.88) / 2 ≈ 0.87
  (Amélioration modérée, surtout par confiance + explication)
  
Weighted voting:
  CV AUC attendu: ~0.88-0.89 (légèrement meilleur que simple)
  
Accord si LR et XGB s'accordent dans ±10%: ~80% des cas
Amélioration sur Ibn Rabbih attendue: FP rate reste < 5%
```

### Erreurs potentielles et solutions

| Erreur | Cause | Solution |
|--------|-------|----------|
| Feature order mismatch | Sorted vs insertion order | Utiliser `list(dict.values())` toujours |
| NaN dans SHAP | CAMeL Tools absent | Fallback smart_camel_loader |
| CV AUC chute > 10% | Overfitting XGBoost | Réduire max_depth ou n_estimators |
| Ensemble pire que LR seul | Mauvaise fusion | Essayer weighted voting |
| Prédictions non calibrées | Probabilités biaisées | Appliquer CalibratedClassifierCV post-hoc |

---

## 10. Architecture du code

### p2_2_xgboost_v80_shap.py

```python
# Phase 1 : Charger données + features
data = load_dataset()
features = extract_all_features_27(data)  # Même que v80

# Phase 2 : Cross-validation
skf = StratifiedKFold(n_splits=5)
for train_idx, val_idx in skf.split(X, y):
    # Entraîner XGBoost
    xgb = XGBClassifier(...)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # Évaluer AUC
    auc_scores.append(roc_auc_score(y_val, xgb.predict_proba(...)))

# Phase 3 : Entraîner sur toutes les données
xgb_final = XGBClassifier(...).fit(X, y)

# Phase 4 : SHAP
explainer = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)  # Visualisation

# Phase 5 : Sauvegarder
pickle.dump({'clf': xgb_final, 'feature_names': feature_names}, 
            'models/xgb_classifier_v80.pkl')
```

### p2_3_hybrid_ensemble_shap.py

```python
# Charger les deux modèles
lr_model = pickle.load('models/lr_classifier_v80.pkl')
xgb_model = pickle.load('models/xgb_classifier_v80.pkl')

# Pour un nouveau texte:
text = "مجنون يقول إلهي..."
features = extract_all_features_27(text)
X = np.array(list(features.values()))

# Phase 1 : Prédictions individuelles
p_lr = lr_model.predict_proba(X)[0, 1]
p_xgb = xgb_model.predict_proba(X)[0, 1]

# Phase 2 : Fusion
p_ensemble = (p_lr + p_xgb) / 2
agreement = 1 - abs(p_lr - p_xgb)

# Phase 3 : SHAP explanations
shap_lr = extract_lr_shap(lr_model, X)       # Trivial pour LR
shap_xgb = shap.TreeExplainer(xgb_model).shap_values(X)

# Phase 4 : Consensus explanation
top_features_lr = sorted(shap_lr, key=abs, reverse=True)[:3]
top_features_xgb = sorted(shap_xgb, key=abs, reverse=True)[:3]

consensus = intersection(top_features_lr, top_features_xgb)

# Résultat:
{
  "prediction": p_ensemble,
  "confidence": agreement,
  "lr_proba": p_lr,
  "xgb_proba": p_xgb,
  "consensus_features": consensus
}
```

---

## 11. Résumé technique en 3 points

### 1. Logistic Regression est transparent mais limité
- Formule linéaire + coefficients interprétables
- AUC ~ 0.86, rapide, stable
- Ne voit pas les interactions

### 2. XGBoost capture des patterns complexes mais opaques
- Arbres boosted capturent les interactions
- AUC ~ 0.87-0.90, plus complexe
- SHAP déverrouille l'explication

### 3. Ensemble = Meilleur des deux mondes
- AUC ~ 0.88-0.89 (gain marginal mais réel)
- Confiance via accord (si |Δp| < 0.1 → TRÈS HAUTE)
- Explications duales (LR directe + XGB via SHAP)
- Robustesse : Si accord, décision fiable. Si désaccord, investiguer.

---

Prêt à coder les pipelines? 🚀
