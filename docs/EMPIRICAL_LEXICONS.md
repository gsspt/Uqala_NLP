# Lexiques Empiriques Extraits du Corpus

**Méthode** : Extraction directe des tokens qui distinguent positifs vs négatifs  
**Date** : 2026-04-10  
**Base** : 460 positifs vs 3817 négatifs

---

## 📋 LEXICONS PAR FEATURE

### A. SCENE INTRODUCTION VERBS
*Verbes de première personne qui introduisent une rencontre/scène*

**Signal d'ordre décroissant de pertinence** :

| Verbe | POS % | NEG % | Ratio | Notes |
|-------|-------|-------|-------|-------|
| **مررت** | 5.65 | 0.55 | **10.09x** | "Je suis passé" — très distinctif |
| **دخلت** | 12.17 | 2.78 | **4.37x** | "J'ai entré" — narratif |
| **فرأيت** | 5.87 | 0.65 | **8.83x** | "Et je vis" — observation scénique |
| **لقيت** | 2.39 | 1.13 | **2.10x** | "Je l'ai rencontré" |
| **أتيت** | 2.83 | 1.47 | **1.91x** | "Je suis venu" |
| **سقطت** | 0.87 | 0.47 | 1.81x | "Je suis tombé" |
| خرجت | 4.13 | 2.86 | 1.44x | "Je suis sorti" |
| وجدت | 2.39 | 1.78 | 1.33x | "J'ai trouvé" |

**Lexique recommandé v80** :
```
مررت, دخلت, فرأيت, لقيت, أتيت, سقطت, خرجت, وجدت, قدمت, ذهبت
```

---

### B. WITNESS OBSERVATION VERBS
*Verbes d'observation directe : "j'ai vu", "j'ai observé"*

| Verbe | POS % | NEG % | Ratio | Notes |
|-------|-------|-------|-------|-------|
| **رأيت** | 22.39 | 9.01 | **2.48x** | "J'ai vu" — core witness verb |
| **فرأيت** | 5.87 | 0.65 | **8.83x** | "Et j'ai vu" — continuation |
| **شاهدت** | 0.43 | 0.10 | 3.79x | "J'ai observé" |
| **أبصرت** | 0.43 | 0.16 | 2.60x | "J'ai discerné" |
| **عاينت** | 0.43 | 0.21 | 1.98x | "J'ai eyewitness" |
| شهدت | 0.22 | 0.34 | 0.62x | (lower) |

**Lexique recommandé v80** :
```
رأيت, فرأيت, شاهدت, أبصرت, عاينت
```

---

### C. HABITUAL ACTION VERBS
*Verbes au présent décrivant le comportement caractéristique du personnage*

| Verbe | POS % | NEG % | Ratio | Notes |
|-------|-------|-------|-------|-------|
| **يقول** | 36.30 | 20.33 | **1.78x** | "Il dit/parle" — speech as characterization |
| يركع | 0.22 | 0.00 | 21.74x | "Il s'agenouille" (very rare but distinctive) |
| يسجد | 0.22 | 0.08 | 2.45x | "Il se prosterne" |
| يحيي | 0.43 | 0.21 | 1.98x | "Il revit/vivifie" (rare) |
| يأتي | 1.74 | 1.44 | 1.20x | "Il vient" (weak) |

⚠️ **Observation** : يقول est dominant, mais les autres habitual verbs sont très rares. Le signal vient du **pattern** (présence de yafalu pour charactérisation) plutôt que des verbes spécifiques.

**Lexique recommandé v80** (enrichir manuellement) :
```
يقول — core
+ autres verbes de comportement/habitude (يفعل, يذهب, يعمل) 
```

---

### D. FORMAL ISNAD CHAINS (NEGATIVE SIGNAL)
*Chaînes de transmission formelles — TRÈS DISCRIMINANT des textes pédagogiques*

| Marqueur | POS % | NEG % | Ratio | Notes |
|----------|-------|-------|-------|-------|
| **أخبرنا** | 0.00 | 30.50 | **NEG 3050x** | "Nous a rapporté" — encyclopédie |
| **حدثنا** | 0.43 | 52.76 | **NEG 118x** | "Nous a conté" — anthologie |
| **أنبأنا** | 0.00 | 7.44 | **NEG 745x** | "Nous a informés" |
| **حدثني** | 0.65 | 13.68 | **NEG 20x** | "M'a conté" (sing) |
| **أخبرني** | 2.39 | 5.40 | NEG 2.25x | (weaker) |
| أنبأني | 0.00 | 0.08 | NEG 8.86x | |

**Signal clé** : 0% (POS) pour أخبرنا, حدثنا, أنبأنا vs 30-50% (NEG)

**Feature** : `no_formal_isnad` = BOOLEAN  
= NOT (any(marker in text) for marker in ['أخبرنا', 'حدثنا', 'أنبأنا'])

---

### E. DIALOGUE MARKERS
*Verbes/marqueurs de dialogue — rapport de parole*

| Marqueur | POS % | NEG % | Ratio | Notes |
|----------|-------|-------|-------|-------|
| **قلت** | 99.35 | 21.12 | **4.70x** | "J'ai dit" — 1st person dialogue |
| **قالت** | 13.26 | 7.78 | **1.70x** | "Elle a dit" (feminine) |
| **فقال** | 95.65 | 70.50 | 1.36x | "Et il a dit" |
| **أجاب** | 2.17 | 0.81 | 2.64x | "Il a répondu" |
| **سأل** | 16.96 | 11.37 | 1.49x | "Il a demandé" |
| وقال | 23.26 | 21.04 | 1.11x | "Et il a dit" (weaker) |
| قالوا | 11.52 | 8.65 | 1.33x | "Ils ont dit" (plural) |

⚠️ **Focus** : **قلت** est TRÈS distinctif (4.70x) — narrative à la 1ère personne

**Lexique recommandé v80** :
```
قلت, قالت, فقال, أجاب, سأل
+ قال, قالوا (pour densité générale)
```

---

### F. DIRECT ADDRESS / VOCATIVE
*Interpellation directe du personnage : يا + nom/titre*

| Adresse | POS % | NEG % | Ratio | Notes |
|---------|-------|-------|-------|-------|
| **يا بهلول** | 2.39 | 0.00 | ∞ | Direct address to famous fool |
| **يا مجنون** | 1.74 | 0.00 | ∞ | "O mad one" |
| **يا ذا** | 3.48 | 0.00 | ∞ | "O you" |
| **يا هرم** | 1.09 | 0.00 | ∞ | "O old one" |
| يا أبا | 6.74 | 3.20 | 2.10x | "O father" (also in NEG) |
| يا هذا | 1.30 | 0.50 | 2.57x | "O this one" |
| يا سعدون | 0.87 | 0.00 | ∞ | Direct to Sadun |

**Signal clé** : 0% in negatives for يا بهلول, يا مجنون, يا ذا, يا هرم

**Lexique recommandé v80** :
```
يا بهلول, يا مجنون, يا ذا, يا هرم, يا أبا, يا هذا
```

---

### G. PARADOX MARKERS
*Conjonctions et marqueurs de contraste/paradoxe*

| Marqueur | POS % | NEG % | Ratio | Notes |
|----------|-------|-------|-------|-------|
| **بل** | 51.30 | 51.27 | **1.00x** | (NO SIGNAL — omit) |
| **إلا** | 20.43 | 18.02 | 1.13x | "Sauf/mais" |
| **ولكن** | 6.74 | 4.48 | 1.50x | "Mais" (formal) |
| **لكن** | 8.91 | 6.58 | 1.35x | "Mais" |
| غير أن | 0.87 | 0.31 | 2.68x | "Otherwise that" (rare) |
| وبل | 1.09 | 1.94 | 0.56x | (weaker) |

⚠️ **Observation** : Paradox markers ne sont PAS très forts seuls. Le vrai signal c'est leur CONTEXTE (paradoxe + junun).

**Recommandation** : Feature doit être **CONTEXTUELLE**, pas juste lexicale.

---

### H. DIVINE INVOCATION (PERSONAL)
*Invocations religieuses personnelles — pas générique الله*

| Terme | POS % | NEG % | Ratio | Notes |
|-------|-------|-------|-------|-------|
| **إلهي** | 2.17 | 0.05 | **34.84x** | "My God" — very personal |
| **اللهم** | 2.61 | 2.04 | 1.27x | "O God" (invocation) |
| **يا رب** | 0.87 | 0.39 | 2.16x | "O Lord" |
| **يا إلهي** | 0.22 | 0.00 | ∞ | "O my God" |
| سبحان الله | 0.65 | 0.55 | 1.16x | (weak) |
| الحمد لله | 0.65 | 0.86 | 0.75x | (weaker) |

**Signal clé** : إلهي = 34.84x (TRÈS DISTINCTIF)

**Lexique recommandé v80** :
```
إلهي, اللهم, يا رب, يا إلهي
(drop generic الحمد لله, سبحان الله)
```

---

### I. LIMINAL & SACRED SPACES
*Lieux de rencontre avec le fou sage — espaces liminaux/sacrés*

| Lieu | POS % | NEG % | Ratio | Notes |
|------|-------|-------|-------|-------|
| **أزقة** | 1.09 | 0.00 | **108.70x** | "Alleys/streets" — very distinctive |
| **المقابر** | 3.91 | 0.08 | **44.17x** | "Cemeteries" — liminal |
| **خرابات** | 2.17 | 0.03 | **60.06x** | "Ruins" |
| **الخرابات** | 1.52 | 0.03 | **42.04x** | "The ruins" |
| **قبر** | 7.83 | 2.25 | **3.46x** | "Grave" |
| **سوق** | 4.13 | 1.31 | **3.13x** | "Market" — public space |
| **مسجد** | 4.78 | 2.28 | 2.09x | "Mosque" |
| **دار** | 21.52 | 12.10 | 1.78x | "House/home" (generic) |
| الفرات | 0.87 | 0.89 | 0.97x | "Euphrates" (no signal) |
| نهر | 0.00 | 1.68 | 0.00x | "River" (negative) |
| مقبرة | 0.65 | 0.47 | 1.35x | "Cemetery" (variant, weaker) |

**Signal clé** : أزقة, المقابر, خرابات sont TRÈS distinctifs

**Lexique recommandé v80** :
```
أزقة, المقابر, خرابات, الخرابات, قبر, سوق, مسجد, دار
(exclude: نهر, الفرات)
```

---

## 📊 SUMMARY: LEXICONS BY FEATURE

### ✅ STRONG SIGNALS (Ratio > 2)

| Feature | Top Terms | Ratio | Implementation |
|---------|-----------|-------|-----------------|
| Scene Intro | مررت, دخلت, فرأيت | 10.09x, 4.37x, 8.83x | Presence of ANY |
| Witness | رأيت, فرأيت | 2.48x, 8.83x | Density or presence |
| Direct Address | يا بهلول, يا مجنون, يا ذا | ∞, ∞, ∞ | Presence of ANY |
| Divine Personal | إلهي | 34.84x | Presence |
| Sacred Spaces | أزقة, المقابر, خرابات | 108x, 44x, 60x | Presence of ANY |
| Isnad Absence | NOT (أخبرنا, حدثنا, أنبأنا) | NEG 3050x | ZERO count |

### ⚠️ MODERATE SIGNALS (Ratio 1-2)

| Feature | Top Terms | Ratio | Notes |
|---------|-----------|-------|-------|
| Habitual Actions | يقول | 1.78x | Need contextual enrichment |
| Dialogue | قلت, قالت | 4.70x, 1.70x | Good but قلت is standout |
| Paradox | ولكن, غير أن | 1.50x, 2.68x | Context-dependent |

### ❌ WEAK/INVERSE SIGNALS

| Feature | Notes |
|---------|-------|
| بل | NO signal (1.00x) — OMIT |
| صلى (prayed) | INVERSE: 0.37x (more in NEG) |
| نهر (river) | NO signal — OMIT |

---

## 🔄 NEXT STEPS

1. **Implement** these lexicons in v80 features
2. **Validate** each feature on a hold-out sample
3. **Enrich manually** :
   - Scene intro verbs: add more variants?
   - Habitual actions: what other يفعل verbs characterize majnun aqil?
   - Divine invocation: any other personal forms missed?

4. **Context-aware features** : Some features need more than just lexicon (e.g., paradox must be paradoxe + junun, not just لكن)

