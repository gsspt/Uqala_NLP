# Proposition de Nouvelles Features — Analyse Empirique du Corpus

**Date** : 2026-04-10  
**Analystes** : Auguste + Claude  
**Méthode** : Analyse empirique sur 460 positifs vs 3817 négatifs

---

## 📊 RÉSUMÉ DES DÉCOUVERTES

### Différenciation Structurelle (Très Forte)

| Critère | Positifs | Négatifs | DIFF |
|---------|----------|----------|------|
| **Verbes d'introduction scénique** (قدمت, أتيت, وجدت...) | 24.1% | 9.3% | **2.6x** |
| **Verbes de témoin direct** (رأيت, شاهدت...) | 20.4% | 7.3% | **2.8x** |
| **Actions habituelles/caractéristiques** (يقول, يركع...) | 28.3% | 16.1% | **1.8x** |
| **Densité dialogue** (قال/قلت par texte) | 4.03 | 2.87 | **1.4x** |
| **Chaînes isnad formelles** (أخبرنا, حدثنا) | 0.00 | 0.91 | **0.0x** |

### Différenciation Lexicale (Noms Propres)

**Noms propres DISTINCTIFS des positifs** :
- بهلول (550x ratio), سعدون (292x), عليان (232x), جعيفران, سمنون, ريحانة, حيونة
- → **Signature forte du maǧnūn ʿāqil** : ensemble clos de fous célèbres

**Noms propres DISTINCTIFS des négatifs** :
- أخبرنا (129x ratio), حدثنا (114x), isnad formal chain (يحيى, أحمد, إبراهيم)
- → **Signature des anthologies** : chaînes de transmission

### Critiques des Features Actuelles

| Feature | Problème | Raison |
|---------|----------|--------|
| **hikma** (f23-f27) | Pas pertinente | Présente partout (60%+ dans négatifs aussi) |
| **validation** (f39-f46) | Bancale | Pattern instable, pas une caractéristique déterminante |
| **contraste** (f47-f51) | Pauvre lexique | Insuffisamment spécifique pour le genre |
| **autorité** (f52-f55) | Signal inversé | Faux : autorités SONT presentes face au maǧnūn |
| **ascetic actions** (f56-f58) | Wrong baseline | Poésie est distinctif mais pas assez fort |

---

## 🔬 NOUVELLE ARCHITECTURE DE FEATURES

### Piliers Conceptuels

1. **NARRATIVE STRUCTURE** (vs pédagogique/formelle)
2. **WITNESS PERSPECTIVE** (narrateur observant une scène)
3. **CHARACTER PRESENCE** (protagoniste particulier ou type)
4. **PARADOXICAL PATTERNS** (folie + sagesse, dévotion, etc)
5. **DISCOURSE MARKERS** (speech density, direct address)

---

## 📝 FEATURES PROPOSÉES (v80+)

### A. NARRATIVE STRUCTURE (5-6 features)

#### A1. Scene Introduction Signature
**Nom** : `scene_intro_verbs_presence`  
**Type** : booléen  
**Logique** : le texte COMMENCE par un verbe d'introduction scénique  
**Verbes** : قدمت, أتيت, ذهبت, سقطت, وجدت, لقيت, دخلت, مررت  
**Pourquoi** : 24.1% (POS) vs 9.3% (NEG) — signature narrative du khbar  
**Implémentation** :
```python
verbs = ['قدمت', 'أتيت', 'ذهبت', 'سقطت', 'وجدت', 'لقيت', 'دخلت', 'مررت']
# Check if any verb appears in first 10% of text
early_threshold = len(text) // 10
has_scene_intro = any(v in text[:early_threshold] for v in verbs)
```

#### A2. Formal Isnad Chain Absence
**Nom** : `no_formal_isnad`  
**Type** : booléen (NEGATION = signal positif)  
**Logique** : texte n'a PAS de chaîne d'isnad formelle  
**Marques isnad** : أخبرنا, حدثنا, أنبأنا, حدثني, أخبرني  
**Pourquoi** : 0.0% (POS) vs 0.91 (NEG) — signal TRÈS discriminant  
**Seuil** : < 1 occurrence de isnad = positif probable  
**Implémentation** :
```python
isnad_count = sum(text.count(m) for m in ['أخبرنا', 'حدثنا', 'أنبأنا', 'حدثني'])
no_formal_isnad = float(isnad_count == 0)
```

#### A3. Personal Narrative Density
**Nom** : `personal_narration_ratio`  
**Type** : densité (verbes 1ère pers. / total verbes)  
**Verbes 1ère pers.** : قدمت, أتيت, رأيت, وجدت, مررت, لقيت, دخلت, شهدت  
**Pourquoi** : narrative stance vs anthological  

#### A4. Habitual Action Pattern
**Nom** : `habitual_behavior_intensity`  
**Type** : densité ou compteur  
**Logique** : présence de verbes au présent itératif (يقول, يركع, يفعل, يسجد)  
**Pourquoi** : 28.3% (POS) vs 16.1% (NEG) — caractérisation du personnage  
**Verbes** : يقول, يفعل, يركع, يسجد, يقوم, يعمل, يذهب, يأتي, يذكر  

#### A5. Witness Verb Presence
**Nom** : `direct_witness_verbs`  
**Type** : booléen ou densité  
**Logique** : verbes d'observation directe (رأيت, شاهدت, أبصرت, عاينت, لاحظت)  
**Pourquoi** : 20.4% (POS) vs 7.3% (NEG) — authority du narrateur  

---

### B. CHARACTER IDENTIFICATION (3-4 features)

#### B1. Famous Fool Presence
**Nom** : `famous_fool_names`  
**Type** : booléen + density  
**Noms** : بهلول, سعدون, عليان, جعيفران, سمنون, ريحانة, حيونة, خلف, رياح  
**Pourquoi** : ratio 550x, 292x, 232x — TRÈS distinctif  
**Note** : keep as-is (f02), c'est pertinent  

#### B2. Named Character Protagonist Pattern
**Nom** : `named_character_focus`  
**Type** : booléen  
**Logique** : texte parle d'UN personnage nommé (pas générique)  
**Implémentation** : 
- Détecte si même nom propre/pronom app. > 3x
- Exclude generic mentions (الخليفة, الوزير, etc.)
**Pourquoi** : khbars focalisent sur une personne ; traités pédagogiques non  

#### B3. Character State Transition
**Nom** : `character_state_change`  
**Type** : booléen  
**Logique** : présence de verbes de changement d'état (أصبح, صار, ظهر, انقلب, تحول)  
**Pourquoi** : narrative arc du personnage  

---

### C. INTERACTION & DISCOURSE (4-5 features)

#### C1. Dialogue Density (Refined)
**Nom** : `dialogue_turn_density`  
**Type** : densité  
**Logique** : ratio (قال + قلت + سأل + أجاب) / total_tokens  
**Threshold** : POS 4.03 vs NEG 2.87  
**Amélioration** : count turn-taking, pas juste présence  
**Implémentation** :
```python
dialogue_markers = ['قال', 'قالت', 'قلت', 'سأل', 'أجاب', 'أجابت', 'قيل']
dialogue_density = sum(text.count(m) for m in dialogue_markers) / n_tokens
```

#### C2. Direct Address to Protagonist
**Nom** : `direct_address_intensity`  
**Type** : densité + booléen  
**Logique** : use of يا + nom (direct vocative address)  
**Termes** : يا + [name, title, descriptor]  
**Pourquoi** : intimate interaction pattern  
**Examples** : يا بهلول, يا مجنون, يا هذا  

#### C3. Reported Speech (Characteristic Saying)
**Nom** : `characteristic_utterance`  
**Type** : booléen  
**Logique** : texte rapporte UN dicton/phrase caractéristique du personnage  
**Pattern** : "[Name] يقول: [quote]" ou "قال: [distinctive phrase]"  
**Pourquoi** : khbars define character by speech  

#### C4. Question Density (Enhanced)
**Nom** : `question_intensity_refined`  
**Type** : densité  
**Logique** : ratio (ماذا + هل + كيف + ؟) / total_tokens  
**Pourquoi** : dialogue structure  
**Note** : current f34 is weak, need stronger signal  

---

### D. PARADOX & CHARACTERIZATION (3-4 features)

#### D1. Sacred Conduct Marker
**Nom** : `sacred_ascetic_conduct`  
**Type** : booléen + density  
**Logique** : actions dévotionnelles (صلى, ركع, سجد, صوم, دعاء, تسبيح)  
**MAIS** : combined with JUNUN terms (paradox signal)  
**Implémentation** :
```python
ascetic = ['صلى', 'ركع', 'سجد', 'صوم', 'تسبيح', 'دعاء', 'عبادة']
junun = ['مجنون', 'جنون']
has_ascetic = any(a in text for a in ascetic)
has_junun = any(j in text for j in junun)
sacred_conduct = float(has_ascetic and has_junun)  # BOTH = paradox
```

#### D2. Divine Reference Intensity (Refined)
**Nom** : `divine_invocation_personal`  
**Type** : density  
**Logique** : mentions of الله, الرب, في (but NOT in التعريف mode)  
**Verbes d'invocation** : يقول (يا إلهي), دعاء, استجاب, إجابة, رحمة  
**Pourquoi** : v79 f76 was too broad (الله everywhere) — make it specific  
**NEW terms** : إلهي, اللهم, يا رب, في سبيل الله, رحمة الله  

#### D3. Rejection/Negation Pattern (Refined)
**Nom** : `paradoxical_negation`  
**Type** : booléen + context  
**Logique** : NOT about absence, but about EXPECTING X but NOT happening  
**Pattern** : "[He/She] لم يفعل X" or "لا يفعل X" in context of JUNUN  
**Examples** :
- "مجنون لكنه يقول..."
- "يقول جنوناً لكن..."
- "مجنون وقد ..."
**Pourquoi** : paradox is ACTIVE contradiction  

#### D4. Moral Judgment Suspension
**Nom** : `ambiguous_moral_stance`  
**Type** : booléen  
**Logique** : text DOES NOT conclude judgment on character  
**Markers** : absence of "فهو...", "إذن...", "فهذا..."  
**Pourquoi** : khbars leave ambiguity ; didactic texts resolve it  

---

### E. LOCATION & SETTING (1-2 features)

#### E1. Outdoor/Public Setting Focus
**Nom** : `public_space_presence`  
**Type** : booléen  
**Termes** : أزقة, سوق, مجلس, بيت, دار, شارع, طريق, منبر, مسجد  
**Pourquoi** : khbars are scene-based (spatial specificity)  
**EXCLUDE** : pedagogical spaces (المكتبة, البيت المرجع)  

#### E2. Liminal/Sacred Space
**Nom** : `liminal_space_indicator`  
**Type** : booléen  
**Termes** : المقابر, الخرابات, مدفن, نهر, قبر, مقام  
**Pourquoi** : fool-sage texts often set in liminal spaces  

---

### F. TEMPORAL STRUCTURE (1-2 features)

#### F1. Immediate Present Narrative
**Nom** : `present_immediate_tense`  
**Type** : booléen + density  
**Logique** : use of present/past-immediate (كان يقول, قال, يقول) NOT future/hypothesis  
**Pourquoi** : vivid narrative vs abstract discussion  

---

## 🗑️ FEATURES TO REMOVE/DEPRECATE

### Remove Entirely
- **f23-f27 (hikma)** : not distinctive, everywhere in corpus
- **f62-f64 (wasf)** : not enough signal to be worth complexity
- **f52-f55 (authority)** : wrong signal direction (authorities ARE in khbars)

### Completely Redesign
- **f39-f46 (validation)** : too loose, redefinition needed
- **f47-f51 (contrast)** : too weak, need paradox-specific markers

### Keep & Refine
- **f00-f14 (junun)** : core concept, keep but maybe consolidate
- **f29-f38 (dialogue)** : keep but strengthen
- **f65-f70 (morphology)** : keep (good signal)
- **f74-f82 (new features)** : some good (f74 question mark), some to drop (f77-f79)

---

## 📐 QUANTITATIVE TARGETS FOR v80

| Category | # Features | Focus |
|----------|-----------|-------|
| Narrative Structure | 5-6 | Scene intro, isnad absence, witness verbs |
| Character | 3-4 | Famous fools, named protagonist |
| Discourse | 4-5 | Dialogue, direct address, speech patterns |
| Paradox | 3-4 | Sacred conduct, divine invocation, ambiguity |
| Spatial/Temporal | 2-3 | Public settings, present narrative |
| Morphology | 6 | Keep CAMeL features (f65-f70) |
| **Total** | **~25-30** | (vs 79 current) |

**Rationale** : Focus on STRUCTURAL and SEMANTIC patterns, drop redundant lexical variations.

---

## ✅ NEXT STEPS

1. **Implement 6-8 priority features** from sections A, B, C
2. **Test incrementally** against current v79 model
3. **Deep dive** on features that underperform
4. **Validation** : check that new features aren't picking up on source/author bias
5. **Corpus review** : manually verify top feature patterns match linguistic intuition

---

## 🔍 MANUAL CORPUS REVIEW NEEDED

Before finalizing, need to manually check:

1. **Do famous fool names appear in negatives at all?** (current analysis says 0%, verify)
2. **Scene intro verbs: are they TRULY scene-setting or just narrative artifact?**
3. **Isnad chains: are 0 occurrences in positives correct, or data artifact?**
4. **Habitual actions: are يقول/يركع really characteristic, or just narrative tense?**
5. **Paradox: can we formalize what makes a contradiction "paradoxical" vs just descriptive?**

**Augustin to review**: Sample 10 positives + 10 negatives with annotations for these patterns.

