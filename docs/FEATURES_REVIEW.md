# Features v79 — Revue Détaillée

**Total : 79 features** (64 lexicales + 6 morphologiques + 9 nouvelles)

---

## 📌 JUNUN — Folie (15 features: f00-f14)

### f00_has_junun
**Type** : booléen (0/1)  
**Termes activateurs** :  
```
مجنون, المجنون, مجنونا, مجانين, المجانين, مجنونة, المجنونة,
معتوه, المعتوه, معتوها, معتوهة, مدله, المدله,
هائم, الهائم, هائما, ممسوس, ممرور, مستهتر,
جنونه, جنونها, جنوني, جنونا, جنون, الجنون,
ذاهبالعقل, ذهبعقله, ذاهب, ذهب
```
**Filtres pour faux positifs** : exclut جنة (paradis), جن (djinns), سجن (prison)

### f01_junun_density
**Type** : densité (nombre tokens junun / total tokens)  
**Mesure** : saturation du texte en termes de folie

### f02_famous_fool ⭐ (Top coef: +0.638)
**Type** : booléen  
**Noms propres activateurs** :  
```
بهلول, بهلولا, سعدون, عليان, جعيفران, ريحانة,
سمنون, لقيط, حيون, حيونة, خلف, رياح
```
**Signal clé** : présence d'un sage fou célèbre

### f03_junun_count
**Type** : compteur normalisé (0-10) / 10  
**Mesure** : combien de variantes de junun apparaissent

### f04_junun_specialized
**Type** : booléen  
**Condition** : au moins un terme junun de longueur > 4 caractères  
**Signal** : utilisation de termes moins courants

### f05_junun_position
**Type** : position normalisée (0-1)  
**Mesure** : position du **premier** junun / longueur du texte  
**Signal** : apparition précoce (folie = sujet d'entrée)

### f06_junun_in_title
**Type** : booléen  
**Condition** : junun dans les 50 premiers caractères

### f07_junun_plural
**Type** : booléen  
**Condition** : contient "مجانين" ou "المجانين"

### f08_junun_in_final_third
**Type** : booléen  
**Condition** : au moins un junun dans le dernier tiers du texte

### f09_jinn_root
**Type** : booléen  
**Racine** : ج.ن.ن  
**Termes** : جنون, جنونه, جنونها, جننت, يجن, أجنّ

### f10_junun_repetition
**Type** : densité  
**Mesure** : répétitions cumulées de tous les junun / tokens  
**Signal** : insistance sur la folie

### f11_junun_morpho
**Type** : booléen  
**Variantes morphologiques** : مجنون, معتوه, هائم, ممسوس

### f12_junun_positive
**Type** : booléen inverse  
**Condition** : NOT (حتوي على : "لا مجنون", "ليس مجنون", "لم يكن مجنون")

### f13_junun_good_context
**Type** : booléen  
**Condition** : junun dans fenêtre ±20 chars d'un verbe narrative  
**Verbes contexte** : قال, رأيت, شهدت

### f14_junun_validation_prox
**Type** : booléen  
**Condition** : junun PROXIMAL (fenêtre ±100 chars) avec validation  
**Validation** : ضحك (rire), أعطى (don), بكى (pleur)

---

## 🧠 AQL / HIKMA — Sagesse (13 features: f15-f27)

### f15_has_aql
**Type** : booléen  
**Termes activateurs** :  
```
عاقل, العاقل, عقل, العقل, عقلاء, العقلاء, عقلاؤهم,
عقول, أعقل, معقول, عقلانية,
عقله, عقلها, عقلي, عقلك, عقلهم
```

### f16_aql_density
**Type** : densité (tokens aql / total)

### f17_aql_count
**Type** : compteur normalisé (0-10) / 10

### f18_paradox_junun_aql
**Type** : booléen  
**Condition** : `f00_has_junun AND f15_has_aql`  
**Signal clé** : paradoxe maǧnūn ʿāqil

### f19_junun_aql_proximity
**Type** : booléen  
**Condition** : junun et aql dans fenêtre ±80 chars

### f20_junun_aql_ratio ⭐ (Top coef: +0.353)
**Type** : ratio  
**Formule** : `density_junun / (density_aql + 0.001)`, capped at 10.0  
**Interprétation** : junun dominent sur aql

### f21_superlatives
**Type** : booléen  
**Termes** : أعقل, أحكم, أصوب, أفضل, أعلم, أصدق

### f22_aql_positive
**Type** : booléen inverse  
**Négation** : NOT (لا عاقل, ليس عاقل, بلا عقل)

### f23_has_hikma
**Type** : booléen  
**Termes** : حكمة, حكيم, حكماء, الحكمة, الحكيم, الحكماء, حكمته, حكمهم, أحكم

### f24_hikma_density
**Type** : densité (tokens hikma / total)

### f25_hikma_junun_prox
**Type** : booléen  
**Condition** : hikma proximal (±80) avec junun

### f26_hikma_qala_prox
**Type** : booléen  
**Condition** : hikma proximal (±80) avec dialogue (قال)

### f27_hikma_in_title
**Type** : booléen  
**Condition** : hikma dans les 50 premiers caractères

---

## 💬 DIALOGUE / QALA (10 features: f29-f38)
*Note : f28_has_qala supprimé v79 (signal contradictoire)*

### f29_qala_density
**Type** : densité  
**Termes dialogue** :  
```
قلت, فقلت, قلنا, قالوا, قال, قالت, قالا,
سألت, فسألت, سألني, أجبت, أجاب,
قيل له, قيل لي, فقلت له,
وقال, ويقول, يقال, أقول
```
**Signal** : texte est narratif/dialogal

### f30_has_first_person ⭐ (Important signal)
**Type** : booléen  
**Termes** :  
```
فعلت, رأيت, مررت, لقيت, وجدت, شهدت, سمعت, حدثني,
أخبرني, أنبأني, حدثنا, أخبرنا, أنبأنا,
كنت
```
**Note** : كنت = quasi-absent des négatifs (témoin oculaire)

### f31_first_person_density
**Type** : densité (tokens first person / total)

### f32_junun_near_qala
**Type** : booléen  
**Condition** : junun proximal (±80) avec dialogue

### f33_has_questions
**Type** : booléen  
**Termes** : كيف, كيفية, ماذا, ما, متى, أين, هل, من, لماذا

### f34_question_density
**Type** : densité (tokens questions / total)

### f35_question_answer
**Type** : booléen  
**Condition** : `has_questions AND has_qala`  
**Signal** : structure Q&A

### f36_dialogue_structure
**Type** : booléen  
**Condition** : count("قال") + count("قلت") >= 2  
**Signal** : vraiment dialogal

### f37_qala_position
**Type** : position normalisée  
**Mesure** : position du premier "قال" ou "قلت" / longueur texte

### f38_qala_in_final
**Type** : booléen  
**Condition** : qala dans dernier tiers du texte

---

## ✅ VALIDATION (8 features: f39-f46)
*Réactions du narrateur face au fou sage*

### Sous-catégories
```
LAUGH:   ضحك, ضحكت, ضحكوا, فضحك, فضحكوا
GIFT:    أعطى, أعطاه, وهب, جائزة, أمر, فأمر
CRY:     بكى, بكت, بكاء, دموع, دموعه
SILENCE: صمت, سكت, أطرق, طرق
```

### f39_has_validation
**Type** : booléen  
**Condition** : au moins une des 4 validations présente

### f40_validation_density
**Type** : densité

### f41_validation_laugh
**Type** : booléen

### f42_validation_gift
**Type** : booléen

### f43_validation_cry
**Type** : booléen

### f44_validation_in_final
**Type** : booléen  
**Condition** : validation dans dernier tiers

### f45_validation_multiple
**Type** : booléen  
**Condition** : au moins 2 catégories de validation présentes

### f46_validation_junun_prox
**Type** : booléen  
**Condition** : validation proximal (±100) avec junun

---

## ⚡ CONTRASTE (5 features: f47-f51)
*Retournements narratifs clés*

### Sous-catégories
```
OPPOSITION:  لكن, لكنه, لكنها, بل, وبل, ولكن
CORRECTION:  كذب, أخطأ, غلط, فبان, فتبين, أصح
REVELATION:  فإذا, وإذا, فتبين, فتبينت
```

### f47_has_contrast
**Type** : booléen

### f48_contrast_density
**Type** : densité

### f49_contrast_opposition
**Type** : booléen

### f50_contrast_correction
**Type** : booléen

### f51_contrast_revelation ⭐ (Top coef: +0.284)
**Type** : booléen  
**Termes clés** : فإذا, وإذا (dévoilement soudain du paradoxe)

---

## 👑 AUTORITÉ (4 features: f52-f55)
*Signal NÉGATIF — ce sont les anti-maǧnūn*

### Termes d'autorité
```
الخليفة, أميرالمؤمنين, أمير المؤمنين, الرشيد, المأمون,
المتوكل, المعتصم, المهدي, المنصور, الهادي, المعتضد,
الوزير, الوالي, القاضي, السلطان, الملك
```

### f52_has_authority
**Type** : booléen

### f53_authority_count
**Type** : compteur normalisé (0-5) / 5

### f54_authority_junun_prox
**Type** : booléen  
**Condition** : autorité proximal (±100) avec junun

### f55_authority_in_title
**Type** : booléen

---

## 📖 POÉSIE (3 features: f56-f58)

### Marqueurs de poésie
```
أنشد, أنشأ, أنشدني, أنشدنا, فأنشد, فأنشأ, ينشد,
الشاعر, شعره, شعرها, شعري, أبيات, قصيدة, وأنشد
```

### f56_has_shir
**Type** : booléen

### f57_shir_density
**Type** : densité

### f58_shir_alone
**Type** : booléen  
**Condition** : `has_shir AND NOT has_qala`  
**Signal** : poésie sans dialogue narrative

---

## 🗺️ SPATIAL (3 features: f59-f61)

### Prépositions spatiales
```
في, على, عند, بعد, أمام, خلف, حول, داخل, خارج
```

### f59_has_spatial
**Type** : booléen

### f60_spatial_density
**Type** : densité

### f61_spatial_variety
**Type** : ratio  
**Mesure** : nombre de prépositions différentes / 9

---

## 📚 WASF — Texte Lexicographique (3 features: f62-f64)
*Signal NÉGATIF — texte est définitionnel, pas narratif*

### Marqueurs wasf
```
ومنها, ضروب, فهو مجنون, تقول العرب, من القول,
قال الكاتب, قال المصنف, يقال, يسمى
```

### f62_has_wasf
**Type** : booléen

### f63_wasf_density
**Type** : densité

### f64_wasf_in_title
**Type** : booléen

---

## 🔤 MORPHOLOGIE ARABE — Racines & POS (6 features: f65-f70)
*Utilise CAMeL Tools (fallback à 0 si absent)*

### f65_root_jnn_density ⭐ (Top coef: +0.353)
**Type** : densité  
**Racine** : ج.ن.ن  
**Signal** : saturation morphologique en junun

### f66_root_aql_density
**Type** : densité  
**Racine** : ع.ق.ل

### f67_root_hikma_density
**Type** : densité  
**Racine** : ح.ك.م

### f68_verb_density
**Type** : densité (verbes / total tokens)

### f69_noun_density ⭐ (Top coef: +0.266)
**Type** : densité (noms / total tokens)

### f70_adj_density
**Type** : densité (adjectifs / total tokens)

**Note** : f71-f73 supprimés v79 (aspect/voix, importance XGB=0)

---

## ✨ NOUVELLES FEATURES v79 (9 features: f74-f82)

### f74_question_mark_presence ⭐⭐ (CLEF!)
**Type** : booléen  
**Condition** : présence du caractère `؟`  
**SIGNAL TRÈS FORT** : 94.7% des positifs vs 0% des négatifs  
**Interprétation** : le maǧnūn ʿāqil parle/pose des questions

### f75_question_mark_density
**Type** : densité  
**Mesure** : count("؟") / n_tokens

### f76_religious_intensity ⭐ (Top coef: +0.266)
**Type** : compteur normalisé (0-5) / 5  
**Termes religieux** :  
```
الله, ربي, ربه, رب, إلهي, اللهم,
سبحان, الرحمن, الرحيم, يا رب,
والله, بالله, لله, لوجه الله,
الجنة, القيامة, النار, العرش
```
**Signal** : 83% des positifs, beaucoup moins dans négatifs

### f77_first_person_scene
**Type** : booléen  
**Termes** :  
```
كنت, فكنت,
فرأيت, فوجدت, فمررت, فدخلت,
لقيته, صادفته, أبصرته
```
**Note** : IMPORTANCE = 0 dans LR (redondant avec f30)  
**À supprimer v80**

### f78_direct_address
**Type** : booléen  
**Termes d'interpellation** :  
```
يا بهلول, يا سعدون, يا عليان,
يا مجنون, يا ذا, يا هذا المجنون,
يا مالك, يا هرم, يا أبا
```
**Note** : IMPORTANCE = 0 (redondant avec f02)  
**À supprimer v80**

### f79_physical_reaction
**Type** : booléen  
**Termes de réaction** :  
```
شهق, شهقة, فشهق,
غشي, أغمي, أغشي عليه,
ويلي, ويلاه, يا ويلي,
تعجب, فتعجب, عجبت, فعجبت,
فزع, ارتعد, وجفت
```
**Note** : IMPORTANCE = 0 (redondant avec validation)  
**À supprimer v80**

### f80_fool_location
**Type** : booléen  
**Lieux typiques** :  
```
دار المرضى, دار المجانين,
المقابر, القبور, المقبرة,
الخرابات, الخراب,
أزقة, الأزقة,
مقيد, مقيدا,
سلسلة, القيد
```

### f81_mystical_verb
**Type** : booléen  
**Verbes de connaissance** :  
```
عرف, فعرف, عرفت, عرفه,
أدرك, فأدرك,
فهم, فهمه,
شعر, فشعر,
أبصر, فأبصر
```
**Signal** : savoir sans apprendre (mystique)

### f82_love_madness (enrichi v79)
**Type** : booléen  
**Termes** (12 termes, 1→12 depuis v76) :  
```
لليلى, ليلى,
الشوق, شوقا, شوقاً, مشتاق,
هيام, هائم,
عشق, العشق, عاشق,
المحبة, محبة,
جوى, الجوى,
متيم, فراق, صبابة
```

---

## 📊 Résumé Quantitatif

| Catégorie | Range | Total | Notes |
|-----------|-------|-------|-------|
| Junun | f00-f14 | 15 | Core concept |
| Aql/Hikma | f15-f27 | 13 | Paradoxe central |
| Dialogue | f29-f38 | 10 | (f28 supprimé) |
| Validation | f39-f46 | 8 | Témoin/réaction |
| Contraste | f47-f51 | 5 | Retournement narratif |
| Autorité | f52-f55 | 4 | Signal négatif |
| Poésie | f56-f58 | 3 | Genre |
| Spatial | f59-f61 | 3 | Contexte |
| Wasf | f62-f64 | 3 | Signal négatif |
| Morpho | f65-f70 | 6 | CAMeL Tools |
| Nouvelles | f74-f82 | 9 | v79 additions |
| **TOTAL** | | **79** | |

---

## ⚠️ Problèmes Identifiés

### À SUPPRIMER v80
- **f77_first_person_scene** : importance LR = 0 (redondant avec f30)
- **f78_direct_address** : importance LR = 0 (redondant avec f02)
- **f79_physical_reaction** : importance LR = 0 (redondant avec validation)

### À AFFINER v80
- **f76_religious_intensity** : actuellement "الله" générique, trop large
  - **Proposé** : remplacer par إلهي, اللهم, يا رب uniquement (plus spécifique)

### À INVESTIGUER
- **XGBoost overfit** : CV AUC 0.872 ± 0.045 vs Test AUC 0.948 (gap 7.7 pts)
- **Faux positifs externe** : A1_conservative = 54.6% positifs sur Ibn Abd Rabbih

