# CATALOGUE EXHAUSTIF DES FEATURES POUR LA DÉTECTION DU MAǦNŪN ʿĀQIL

**Corpus analysé** : Akhbars 161-612 du *Kitāb ʿUqalāʾ al-Maǧānīn* (452 textes positifs)

**Date d'analyse** : 2026-04-08

---

## 📊 SYNTHÈSE DES DÉCOUVERTES CLÉS

### Résultats contre-intuitifs

1. **57.1% des akhbars N'ONT PAS de terme "مجنون" explicite**
   → Le motif est souvent **implicite** (folie non nommée)

2. **Seulement 23.9% ont co-occurrence جنون + عقل**
   → Le paradoxe lexical N'EST PAS toujours explicite

3. **36.1% ont dialogue à la 1ère personne**
   → Le témoin oculaire est important mais pas systématique

4. **17.0% ont validation explicite** (rire, don, pleurs)
   → Beaucoup de fins ouvertes

5. **94.7% contiennent des interrogations**
   → La **question** est centrale au motif

### Distribution des genres

- **khabar** : 55.5% (récit narratif)
- **nadīra** : 31.4% (anecdote à chute)
- **šiʿr** : 12.6% (poésie)
- **wasf** : 0.2% (définition)

---

## 🏗️ ARCHITECTURE DES FEATURES (150+ features au total)

---

## CATÉGORIE 1 : LEXIQUE DE LA FOLIE (15 features)

### F1.1 : Présence de termes junūn génériques
**Type** : Binaire (0/1)
**Calcul** : Détecte mots : مجنون، جنون، المجنون، مجانين، جنونه
**Prévalence** : 42.9% des positifs
**Poids attendu** : +1.5 à +2.0

```python
junun_generic = ['مجنون', 'المجنون', 'مجانين', 'جنون', 'الجنون', 'جنونه']
f1_1 = int(any(term in text for term in junun_generic))
```

---

### F1.2 : Présence de termes junūn spécialisés
**Type** : Binaire
**Calcul** : Détecte : معتوه، مدله، هائم، ممسوس، ممرور، مستهتر، ذاهب العقل
**Prévalence** : ~8% des positifs
**Poids attendu** : +0.5 (plus rare mais très spécifique)

```python
junun_specialized = ['معتوه', 'المعتوه', 'مدله', 'المدله', 'هائم', 
                     'ممسوس', 'ممرور', 'مستهتر', 'ذاهب', 'ذهب']
f1_2 = int(any(term in text for term in junun_specialized))
```

---

### F1.3 : Densité de termes junūn
**Type** : Numérique (0.0 à 1.0)
**Calcul** : (Nombre de mots junūn) / (Nombre total de mots arabes)
**Distribution** : Médiane ≈ 0.0, Max ≈ 0.15
**Poids attendu** : +0.3

```python
tokens = text.split()
n_junun = sum(1 for t in tokens if t in junun_terms)
f1_3 = n_junun / max(len(tokens), 1)
```

---

### F1.4 : Noms propres de fous célèbres
**Type** : Binaire
**Calcul** : Détecte : بهلول، سعدون، عليان، جعيفران، ريحانة، سمنون، لقيط، حيون، خلف
**Prévalence** : 32.7% des positifs
**Poids attendu** : +1.8 (très discriminant)

```python
famous_fools = ['بهلول', 'بهلولا', 'سعدون', 'عليان', 'جعيفران', 
                'ريحانة', 'سمنون', 'لقيط', 'حيون', 'حيونة', 'خلف', 'رياح']
f1_4 = int(any(name in text for name in famous_fools))
```

**Note importante** : Certains noms (بهلول, سعدون) sont quasi-automatiques positifs.

---

### F1.5 : Pluriel vs singulier junūn
**Type** : Binaire
**Calcul** : Détecte pluriels : مجانين، عقلاء، حكماء
**Interprétation** : Pluriel → texte définitionnel (*wasf*) ?
**Prévalence** : ~5% des positifs
**Poids attendu** : -0.5 (signal négatif)

```python
junun_plural = ['مجانين', 'المجانين', 'عقلاء', 'حكماء']
f1_5 = int(any(pl in text for pl in junun_plural))
```

---

### F1.6 à F1.10 : Variantes morphologiques junūn
**Type** : Comptage (0 à N)
**Calcul** : Compter formes génitivées, accusatives, annexées

```python
f1_6 = text.count('مجنونا')  # Accusatif
f1_7 = text.count('مجنونة')  # Féminin
f1_8 = text.count('جنونه')   # Annexion (sa folie)
f1_9 = text.count('جنونها')  # Annexion fém.
f1_10 = text.count('بجنون')  # Prépositionnel
```

---

### F1.11 : Verbes de folie (devenir fou)
**Type** : Binaire
**Calcul** : Détecte : جُنّ، يجنّ، جُنّت، أجنّه، تجنّن
**Prévalence** : ~3% des positifs
**Poids attendu** : +0.8

```python
junun_verbs = ['جن', 'يجن', 'جنت', 'أجنه', 'تجنن', 'جنني']
f1_11 = int(any(v in text for v in junun_verbs))
```

---

### F1.12 : Adjectifs liés à junūn
**Type** : Binaire
**Calcul** : Détecte : مخبول، مهووس، مختل
**Prévalence** : <1% des positifs
**Poids attendu** : Neutre (trop rare)

---

### F1.13 : Position de junūn dans le texte
**Type** : Numérique (0.0 à 1.0)
**Calcul** : Position relative du 1er terme junūn
**Interprétation** :
- 0.0-0.3 → Début (présentation du personnage)
- 0.4-0.6 → Milieu (révélation progressive)
- 0.7-1.0 → Fin (chute paradoxale)

```python
first_junun_pos = None
for term in junun_terms:
    match = re.search(re.escape(term), text)
    if match:
        first_junun_pos = match.start() / len(text)
        break
f1_13 = first_junun_pos if first_junun_pos else 0.5  # Défaut milieu
```

---

### F1.14 : Répétition de junūn
**Type** : Numérique (0 à N)
**Calcul** : Nombre total d'occurrences de tous termes junūn
**Distribution** : Médiane = 1, Max = 8
**Poids attendu** : +0.2 (faible impact)

```python
f1_14 = sum(text.count(term) for term in junun_terms)
```

---

### F1.15 : Junūn dans titre/début d'isnad
**Type** : Binaire
**Calcul** : Détecte junūn dans 1ers 50 caractères
**Interprétation** : Texte catalogique (*wasf*) vs narratif
**Prévalence** : ~12% des positifs
**Poids attendu** : -0.8 (signal négatif si très tôt)

```python
f1_15 = int(any(term in text[:50] for term in junun_terms))
```

---

## CATÉGORIE 2 : LEXIQUE DE LA SAGESSE (10 features)

### F2.1 : Présence de termes ʿaql/ḥikma
**Type** : Binaire
**Calcul** : Détecte : عقل، العقل، حكمة، الحكمة، لب، رشد
**Prévalence** : 47.6% des positifs
**Poids attendu** : +1.2

```python
aql_terms = ['عقل', 'العقل', 'عاقل', 'العاقل', 'حكمة', 'الحكمة', 
             'حكيم', 'الحكيم', 'لب', 'اللب', 'رشد', 'رشيد', 'فطنة']
f2_1 = int(any(term in text for term in aql_terms))
```

---

### F2.2 : Densité de termes ʿaql
**Type** : Numérique (0.0 à 1.0)
**Calcul** : (Nombre de mots ʿaql) / (Nombre total de mots)
**Distribution** : Médiane ≈ 0.0, Max ≈ 0.08

```python
n_aql = sum(1 for t in tokens if t in aql_terms)
f2_2 = n_aql / max(len(tokens), 1)
```

---

### F2.3 : Co-occurrence paradoxale (junūn × ʿaql)
**Type** : Binaire
**Calcul** : Présence simultanée de junūn ET ʿaql
**Prévalence** : 23.9% des positifs
**Poids attendu** : +2.5 (très fort signal)

```python
has_junun = any(t in text for t in junun_terms)
has_aql = any(t in text for t in aql_terms)
f2_3 = int(has_junun and has_aql)
```

---

### F2.4 : Proximité junūn-ʿaql (< 80 chars)
**Type** : Binaire
**Calcul** : Co-occurrence dans fenêtre de 80 caractères
**Prévalence** : 11.5% des positifs
**Poids attendu** : +3.0 (paradoxe rapproché = noyau du motif)

```python
def count_proximity(text, terms1, terms2, window=80):
    for t1 in terms1:
        for match in re.finditer(re.escape(t1), text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            if any(t2 in text[start:end] for t2 in terms2):
                return 1
    return 0

f2_4 = count_proximity(text, junun_terms, aql_terms)
```

---

### F2.5 : Superlatifs comparatifs
**Type** : Binaire
**Calcul** : Détecte : أعقل، أحكم، أفضل، أصدق ("plus sage/raisonnable")
**Prévalence** : 9.3% des positifs
**Poids attendu** : +1.5

```python
superlatives = ['أعقل', 'أحكم', 'أفضل', 'أعلم', 'أصدق', 'أحسن', 'أشد']
f2_5 = int(any(sup in text for sup in superlatives))
```

---

### F2.6 : Négation + ʿaql ("sans raison")
**Type** : Binaire
**Calcul** : Détecte négation proche de terme ʿaql
**Prévalence** : 90.3% des positifs (!!)
**Poids attendu** : +0.5 (très fréquent donc faible poids)

```python
negations = ['لا', 'ليس', 'ما', 'لم', 'لن', 'غير']
f2_6 = 0
for neg in negations:
    for aterm in aql_terms:
        if re.search(f"{neg}.*{aterm}", text):
            f2_6 = 1
            break
```

---

### F2.7 : Termes de vertu morale
**Type** : Binaire
**Calcul** : Détecte : فضيلة، رذيلة، خير، شر، صلاح
**Interprétation** : Présence → texte moralisateur (*wasf*) ?
**Prévalence** : ~8% des positifs
**Poids attendu** : -0.7 (signal négatif)

```python
virtue_terms = ['فضيلة', 'رذيلة', 'خير', 'شر', 'صلاح', 'فساد']
f2_7 = int(any(v in text for v in virtue_terms))
```

---

### F2.8 : Termes de connaissance
**Type** : Binaire
**Calcul** : Détecte : علم، معرفة، فهم، إدراك
**Prévalence** : ~15% des positifs

```python
knowledge_terms = ['علم', 'العلم', 'معرفة', 'فهم', 'إدراك']
f2_8 = int(any(k in text for k in knowledge_terms))
```

---

### F2.9 : Position relative junūn vs ʿaql
**Type** : Catégoriel (3 valeurs)
**Calcul** : 
- 0 = junūn avant ʿaql (présentation puis révélation)
- 1 = ʿaql avant junūn (apparence sage puis folie)
- -1 = aucun des deux

```python
junun_pos = text.find('مجنون') if 'مجنون' in text else -1
aql_pos = text.find('عقل') if 'عقل' in text else -1

if junun_pos >= 0 and aql_pos >= 0:
    f2_9 = 0 if junun_pos < aql_pos else 1
else:
    f2_9 = -1
```

---

### F2.10 : Ratio junūn/ʿaql
**Type** : Numérique (0.0 à ∞)
**Calcul** : (Occurrences junūn) / (Occurrences ʿaql + 1)
**Distribution** : Médiane ≈ 0.5

```python
n_junun_total = sum(text.count(t) for t in junun_terms)
n_aql_total = sum(text.count(t) for t in aql_terms)
f2_10 = n_junun_total / (n_aql_total + 1)
```

---

## CATÉGORIE 3 : STRUCTURE DIALOGIQUE (20 features)

### F3.1 : Présence de قال
**Type** : Binaire
**Calcul** : Détecte : قال، فقال، قالت، فقالت
**Prévalence** : ~85% des positifs
**Poids attendu** : +1.0

```python
qala_variants = ['قال', 'فقال', 'قالت', 'فقالت', 'يقول']
f3_1 = int(any(q in text for q in qala_variants))
```

---

### F3.2 : Densité de قال
**Type** : Numérique (0.0 à 1.0)
**Calcul** : (Occurrences قال) / (Nombre de mots)
**Distribution** : Moyenne = 6.81%, Max = 30%
**Poids attendu** : +1.5

```python
n_qala = sum(text.count(q) for q in qala_variants)
f3_2 = n_qala / max(len(tokens), 1)
```

---

### F3.3 : Dialogue à la 1ère personne
**Type** : Binaire
**Calcul** : Détecte : قلت، فقلت، سألت، قلنا
**Prévalence** : 36.1% des positifs
**Poids attendu** : +2.0 (très discriminant)

```python
first_person = ['قلت', 'فقلت', 'سألت', 'فسألت', 'قلنا', 'سألنا']
f3_3 = int(any(fp in text for fp in first_person))
```

---

### F3.4 : Alternance énonciative (polyphonie)
**Type** : Numérique (0 à N)
**Calcul** : Nombre de changements de locuteur (approximation par comptage de قال)
**Distribution** : Médiane = 5, Max = 60

```python
f3_4 = sum(text.count(q) for q in qala_variants)
```

---

### F3.5 : Question-réponse (سأل + أجاب)
**Type** : Binaire
**Calcul** : Détecte structure سأل...قال ou سألت...فقال
**Prévalence** : ~25% des positifs
**Poids attendu** : +1.8

```python
has_sual = any(s in text for s in ['سأل', 'سألت', 'سألته', 'سئل'])
has_jawab = any(j in text for j in ['أجاب', 'فأجاب', 'فقال'])
f3_5 = int(has_sual and has_jawab)
```

---

### F3.6 : Proximité junūn-قال (< 80 chars)
**Type** : Binaire
**Calcul** : Junūn dans fenêtre de 80 chars autour de قال
**Prévalence** : 26.3% des positifs
**Poids attendu** : +2.5 (le fou EST le locuteur)

```python
f3_6 = count_proximity(text, junun_terms, qala_variants)
```

---

### F3.7 : Discours direct (guillemets, marqueurs)
**Type** : Binaire
**Calcul** : Détecte : «، »، :
**Prévalence** : ~45% des positifs

```python
f3_7 = int(any(mark in text for mark in ['«', '»', ':']))
```

---

### F3.8 : Verbes de transmission (ḥadīṯ)
**Type** : Binaire
**Calcul** : Détecte : حدثنا، أخبرنا، روى، نقل
**Interprétation** : Si présent → texte de transmission ?
**Prévalence** : ~20% des positifs
**Poids attendu** : -0.3 (léger signal négatif)

```python
transmission = ['حدثنا', 'أخبرنا', 'حدثني', 'أخبرني', 'روى', 'نقل']
f3_8 = int(any(t in text for t in transmission))
```

---

### F3.9 : Impératifs (ordres, exhortations)
**Type** : Binaire
**Calcul** : Détecte formes impératives fréquentes
**Prévalence** : ~10% des positifs

```python
imperatives = ['اتق', 'خذ', 'اعلم', 'انظر', 'اسمع']
f3_9 = int(any(imp in text for imp in imperatives))
```

---

### F3.10 : Interrogations (questions rhétoriques)
**Type** : Binaire
**Calcul** : Détecte : هل، ما، من، أين، كيف، متى
**Prévalence** : 94.7% des positifs (!)
**Poids attendu** : +0.8 (très fréquent mais discriminant)

```python
questions = ['هل', 'ما', 'من', 'أين', 'كيف', 'متى', 'لماذا', 'أي']
f3_10 = int(any(q in text for q in questions))
```

---

### F3.11 à F3.15 : Density features (densités spécifiques)

```python
# F3.11 : Densité de verbes de parole (tous types)
speech_verbs_all = qala_variants + transmission + ['ذكر', 'حكى', 'نقل']
f3_11 = sum(text.count(v) for v in speech_verbs_all) / len(tokens)

# F3.12 : Ratio 1ère pers. / 3ème pers.
first_pers_count = sum(text.count(fp) for fp in first_person)
third_pers_count = sum(text.count(q) for q in qala_variants)
f3_12 = first_pers_count / (third_pers_count + 1)

# F3.13 : Nombre de ? (points d'interrogation)
f3_13 = text.count('؟')

# F3.14 : Proximité question-ʿaql
f3_14 = count_proximity(text, questions, aql_terms, window=100)

# F3.15 : Triple proximité junūn-قال-ʿaql (fenêtre 150 chars)
# Détecte si les 3 apparaissent ensemble (noyau du motif)
def triple_proximity(text, window=150):
    for term_j in junun_terms:
        for match in re.finditer(re.escape(term_j), text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            chunk = text[start:end]
            has_qala = any(q in chunk for q in qala_variants)
            has_aql = any(a in chunk for a in aql_terms)
            if has_qala and has_aql:
                return 1
    return 0

f3_15 = triple_proximity(text)
```

---

### F3.16 : Position du premier قال
**Type** : Numérique (0.0 à 1.0)
**Calcul** : Position relative dans le texte

```python
first_qala = text.find('قال')
f3_16 = first_qala / len(text) if first_qala >= 0 else 0.5
```

---

### F3.17 : Dernier قال dans finale (dernier tiers)
**Type** : Binaire
**Calcul** : قال dans derniers 33% du texte
**Interprétation** : Chute dialogique ?

```python
last_third = text[int(len(text)*0.66):]
f3_17 = int(any(q in last_third for q in qala_variants))
```

---

### F3.18 : Verbes de demande
**Type** : Binaire
**Calcul** : سأل، طلب، استفسر

```python
request_verbs = ['سأل', 'سألت', 'طلب', 'طلبت', 'استفسر']
f3_18 = int(any(r in text for r in request_verbs))
```

---

### F3.19 : Verbes de réponse
**Type** : Binaire
**Calcul** : أجاب، رد، قال

```python
response_verbs = ['أجاب', 'فأجاب', 'رد', 'فرد']
f3_19 = int(any(r in text for r in response_verbs))
```

---

### F3.20 : Pattern question→réponse (séquence)
**Type** : Binaire
**Calcul** : Détecte séquence سأل AVANT أجاب

```python
sual_pos = text.find('سأل')
ajab_pos = text.find('أجاب')
f3_20 = int(sual_pos >= 0 and ajab_pos >= 0 and sual_pos < ajab_pos)
```

---

## CATÉGORIE 4 : POÉSIE ET FORME TEXTUELLE (12 features)

### F4.1 : Marqueurs de citation poétique
**Type** : Binaire
**Calcul** : Détecte : أنشد، فأنشد، أنشأ، ينشد
**Prévalence** : 15.3% des positifs
**Poids attendu** : +0.5 (neutre à positif)

```python
poetry_markers = ['أنشد', 'فأنشد', 'أنشأ', 'ينشد', 'أنشدني', 'وأنشد']
f4_1 = int(any(p in text for p in poetry_markers))
```

---

### F4.2 : Ratio poésie/prose (segments)
**Type** : Numérique (0.0 à 1.0)
**Calcul** : (Segments poétiques) / (Total segments)
**Distribution** : Médiane = 0.0, Moyenne = 0.744 (!)
**Poids attendu** : -1.2 (poésie pure = signal négatif)

```python
# Depuis content.segments
n_poetry = sum(1 for s in segments if s['type'] == 'poetry')
n_prose = sum(1 for s in segments if s['type'] == 'prose')
f4_2 = n_poetry / (n_poetry + n_prose + 1)
```

---

### F4.3 : Poésie SEULE (sans dialogue)
**Type** : Binaire
**Calcul** : Marqueur poétique MAIS pas de 1ère personne
**Interprétation** : Poésie amoureuse (*ġazal*) ?
**Poids attendu** : -2.0 (fort signal négatif)

```python
has_poetry = any(p in text for p in poetry_markers)
has_dialogue = any(fp in text for fp in first_person)
f4_3 = int(has_poetry and not has_dialogue)
```

---

### F4.4 : Marqueurs métriques (détection de vers)
**Type** : Numérique (0 à N)
**Calcul** : Compter lignes avec structure métrique (heuristique)
**Note** : Nécessite analyse métrique arabe (complexe)

```python
# Approximation : lignes avec rime finale (heuristique grossière)
lines = text.split('¶')  # Séparateur de vers dans ton corpus
f4_4 = len([l for l in lines if len(l.split()) >= 8])  # Vers ≈ 8+ mots
```

---

### F4.5 : Prose rimée (saǧʿ)
**Type** : Binaire
**Calcul** : Détecte patterns rythmiques (très approximatif)
**Prévalence** : ~5% des positifs

```python
# Détection simplifiée : répétition de finales similaires
# (nécessiterait analyse phonétique pour être précis)
f4_5 = 0  # Placeholder (difficile sans outils)
```

---

### F4.6 : Longueur moyenne des "lignes"
**Type** : Numérique
**Calcul** : Moyenne mots par segment prose/poetry
**Distribution** : Prose ≈ 20 mots, Poésie ≈ 10 mots

```python
prose_segs = [s['text'] for s in segments if s['type'] == 'prose']
if prose_segs:
    avg_len = sum(len(s.split()) for s in prose_segs) / len(prose_segs)
    f4_6 = avg_len
else:
    f4_6 = 0
```

---

### F4.7 : Pattern segment (prose→poetry→prose)
**Type** : Binaire
**Calcul** : Détecte alternance spécifique
**Prévalence** : ~3% des positifs

```python
seg_types = [s['type'] for s in segments if s['type'] != 'isnad']
pattern = tuple(seg_types)
f4_7 = int(pattern in [('prose', 'poetry', 'prose'), 
                       ('matn', 'poetry', 'matn')])
```

---

### F4.8 : Ratio matn/poetry
**Type** : Numérique
**Calcul** : Longueur matn prose / longueur poetry

```python
matn_len = sum(len(s['text'].split()) for s in segments if s['type'] in ['matn', 'prose'])
poetry_len = sum(len(s['text'].split()) for s in segments if s['type'] == 'poetry')
f4_8 = matn_len / (poetry_len + 1)
```

---

### F4.9 : Marqueurs de chant/récitation
**Type** : Binaire
**Calcul** : ينشد، يتلو، يرتل

```python
recitation = ['ينشد', 'يتلو', 'يرتل', 'ترتيل']
f4_9 = int(any(r in text for r in recitation))
```

---

### F4.10 : Citations poétiques attribuées
**Type** : Binaire
**Calcul** : "قال الشاعر"، "أنشد فلان"

```python
attributed_poetry = ['قال الشاعر', 'أنشد الشاعر', 'قال فلان']
f4_10 = int(any(a in text for a in attributed_poetry))
```

---

### F4.11 : Nombre de hémistiches (approximation)
**Type** : Numérique
**Calcul** : Compter séparateurs ¶

```python
f4_11 = text.count('¶')
```

---

### F4.12 : Poésie en finale (chute poétique)
**Type** : Binaire
**Calcul** : Dernier segment = poetry

```python
last_seg = segments[-1] if segments else {}
f4_12 = int(last_seg.get('type') == 'poetry')
```

---

## CATÉGORIE 5 : VALIDATION ET RÉACTION (15 features)

### F5.1 : Rire (ضحك)
**Type** : Binaire
**Calcul** : Détecte : ضحك، فضحك، يضحك
**Prévalence** : 3.1% des positifs
**Poids attendu** : +2.0 (rare mais très spécifique)

```python
laugh_markers = ['ضحك', 'فضحك', 'ضحكت', 'يضحك', 'ضاحك']
f5_1 = int(any(l in text for l in laugh_markers))
```

---

### F5.2 : Pleurs (بكى)
**Type** : Binaire
**Calcul** : Détecte : بكى، فبكى، دموع
**Prévalence** : 5.5% des positifs
**Poids attendu** : +1.5

```python
cry_markers = ['بكى', 'فبكى', 'بكت', 'يبكي', 'دموع', 'دمعة', 'بكاء']
f5_2 = int(any(c in text for c in cry_markers))
```

---

### F5.3 : Don/récompense (عطاء)
**Type** : Binaire
**Calcul** : Détecte : أعطى، وهب، جائزة
**Prévalence** : 2.9% des positifs
**Poids attendu** : +2.5 (très spécifique)

```python
gift_markers = ['أعطى', 'فأعطى', 'أعطاه', 'وهب', 'فوهب', 'جائزة', 'عطاء']
f5_3 = int(any(g in text for g in gift_markers))
```

---

### F5.4 : Admiration (إعجاب)
**Type** : Binaire
**Calcul** : Détecte : أعجب، استحسن
**Prévalence** : 1.5% des positifs
**Poids attendu** : +1.8

```python
admiration = ['أعجب', 'فأعجبه', 'أعجبني', 'استحسن', 'فاستحسن']
f5_4 = int(any(a in text for a in admiration))
```

---

### F5.5 : Silence (سكوت)
**Type** : Binaire
**Calcul** : Détecte : سكت، صمت، أطرق
**Prévalence** : 6.2% des positifs
**Poids attendu** : +1.2

```python
silence = ['سكت', 'فسكت', 'صمت', 'فصمت', 'أطرق', 'فأطرق']
f5_5 = int(any(s in text for s in silence))
```

---

### F5.6 : Surprise (تعجب)
**Type** : Binaire
**Calcul** : Détecte : تعجب، عجب
**Prévalence** : 4.2% des positifs
**Poids attendu** : +1.0

```python
surprise = ['تعجب', 'فتعجب', 'عجب', 'فعجب', 'عجبا']
f5_6 = int(any(s in text for s in surprise))
```

---

### F5.7 : Validation TOTALE (any)
**Type** : Binaire
**Calcul** : Au moins une réaction positive
**Prévalence** : 17.0% des positifs
**Poids attendu** : +1.5

```python
all_validation = (laugh_markers + cry_markers + gift_markers + 
                  admiration + silence + surprise)
f5_7 = int(any(v in text for v in all_validation))
```

---

### F5.8 : Validation en finale (dernier tiers)
**Type** : Binaire
**Calcul** : Validation dans derniers 33% du texte
**Interprétation** : Structure de chute (*nadīra*)
**Poids attendu** : +2.0

```python
last_third = text[int(len(text)*0.66):]
f5_8 = int(any(v in last_third for v in all_validation))
```

---

### F5.9 : Ordre (أمر)
**Type** : Binaire
**Calcul** : Détecte : فأمر، أمر بـ

```python
order_markers = ['فأمر', 'أمر', 'أمره', 'فأمره']
f5_9 = int(any(o in text for o in order_markers))
```

---

### F5.10 : Approbation (موافقة)
**Type** : Binaire
**Calcul** : صدق، أصاب، أحسن

```python
approval = ['صدق', 'أصاب', 'أحسن', 'فأحسن', 'وافق']
f5_10 = int(any(a in text for a in approval))
```

---

### F5.11 : Réprimande (توبيخ)
**Type** : Binaire
**Calcul** : وبّخ، لام، عاتب

```python
reprimand = ['وبخ', 'لام', 'فلامه', 'عاتب']
f5_11 = int(any(r in text for r in reprimand))
```

---

### F5.12 : Colère (غضب)
**Type** : Binaire
**Calcul** : غضب، فغضب

```python
anger = ['غضب', 'فغضب', 'غضبت', 'يغضب']
f5_12 = int(any(a in text for a in anger))
```

---

### F5.13 : Proximité validation-junūn
**Type** : Binaire
**Calcul** : Validation dans 100 chars de junūn

```python
f5_13 = count_proximity(text, all_validation, junun_terms, window=100)
```

---

### F5.14 : Type de validateur (autorité)
**Type** : Binaire
**Calcul** : Validation + figure d'autorité ensemble

```python
authority = ['الخليفة', 'الأمير', 'الوزير', 'القاضي']
has_validation = any(v in text for v in all_validation)
has_authority = any(a in text for a in authority)
f5_14 = int(has_validation and has_authority)
```

---

### F5.15 : Validation multiple (>1 type)
**Type** : Numérique
**Calcul** : Compter types de validation différents

```python
validation_types = [
    int(any(l in text for l in laugh_markers)),
    int(any(c in text for c in cry_markers)),
    int(any(g in text for g in gift_markers)),
    int(any(a in text for a in admiration)),
    int(any(s in text for s in silence)),
]
f5_15 = sum(validation_types)
```

---

## CATÉGORIE 6 : CONTRASTE ET RENVERSEMENT (18 features)

### F6.1 : Opposition (لكن)
**Type** : Binaire
**Calcul** : لكن، ولكن، لكنه، لكنها
**Prévalence** : 26.5% des positifs
**Poids attendu** : +1.5

```python
opposition = ['لكن', 'ولكن', 'لكنه', 'لكنها', 'غير', 'إلا']
f6_1 = int(any(o in text for o in opposition))
```

---

### F6.2 : Correction (بل)
**Type** : Binaire
**Calcul** : بل، وبل
**Prévalence** : 29.9% des positifs
**Poids attendu** : +1.8

```python
correction = ['بل', 'وبل', 'بل هو', 'بل كان']
f6_2 = int(any(c in text for c in correction))
```

---

### F6.3 : Révélation (فبان، فتبين)
**Type** : Binaire
**Calcul** : فبان، فتبين، فظهر، فإذا
**Prévalence** : 20.4% des positifs
**Poids attendu** : +2.0 (marqueur fort de renversement)

```python
revelation = ['فبان', 'فتبين', 'فظهر', 'فإذا', 'وإذا']
f6_3 = int(any(r in text for r in revelation))
```

---

### F6.4 : Concession (مع، رغم)
**Type** : Binaire
**Calcul** : مع أن، رغم، وإن كان
**Prévalence** : 37.2% des positifs
**Poids attendu** : +1.2

```python
concession = ['مع', 'رغم', 'وإن كان', 'مع أن']
f6_4 = int(any(c in text for c in concession))
```

---

### F6.5 : Négation (لا، ليس، ما)
**Type** : Numérique (densité)
**Calcul** : (Occurrences négation) / (Nombre mots)
**Distribution** : Médiane ≈ 0.05

```python
negations = ['لا', 'ليس', 'ما', 'لم', 'لن', 'غير']
n_neg = sum(text.count(n) for n in negations)
f6_5 = n_neg / len(tokens)
```

---

### F6.6 : Négation + junūn ("ce n'est pas de la folie")
**Type** : Binaire
**Calcul** : Négation proximale à junūn
**Prévalence** : 55.3% des positifs (!)
**Poids attendu** : +0.8

```python
f6_6 = 0
for neg in negations:
    for jterm in junun_terms:
        if re.search(f"{neg}.*{jterm}", text[:200]):  # Début de texte
            f6_6 = 1
            break
```

---

### F6.7 : Pattern paradoxal (أعقل من العقلاء)
**Type** : Binaire
**Calcul** : Superlatif + ʿaql ensemble

```python
has_super = any(s in text for s in superlatives)
has_aql = any(a in text for a in aql_terms)
f6_7 = int(has_super and has_aql)
```

---

### F6.8 : Antithèses lexicales
**Type** : Numérique (comptage)
**Calcul** : Nombre de paires antonymes présentes

```python
antonym_pairs = [
    ('جنون', 'عقل'), ('مجنون', 'عاقل'),
    ('خير', 'شر'), ('حب', 'كره'),
    ('فرح', 'حزن'), ('صواب', 'خطأ'),
    ('حق', 'باطل'), ('علم', 'جهل'),
]

f6_8 = sum(1 for w1, w2 in antonym_pairs 
           if w1 in text and w2 in text)
```

---

### F6.9 : Changement de statut (كان...ثم صار)
**Type** : Binaire
**Calcul** : كان + ثم/بعد + صار/أصبح

```python
f6_9 = int('كان' in text and any(c in text for c in ['ثم صار', 'فأصبح', 'بعد']))
```

---

### F6.10 : Formule de surprise (فإذا هو)
**Type** : Binaire
**Calcul** : فإذا هو، وإذا هو
**Prévalence** : ~8% des positifs
**Poids attendu** : +2.5 (très spécifique au renversement)

```python
surprise_formulas = ['فإذا هو', 'وإذا هو', 'فإذا به']
f6_10 = int(any(s in text for s in surprise_formulas))
```

---

### F6.11 à F6.18 : Features de contraste avancées

```python
# F6.11 : Double négation (lā...wa-lā)
f6_11 = int('لا' in text and text.count('لا') >= 2)

# F6.12 : Opposition apparence/réalité
appearance_terms = ['ظاهر', 'يظن', 'يبدو', 'يرى']
reality_terms = ['حقيقة', 'في الباطن', 'في الواقع']
f6_12 = int(any(a in text for a in appearance_terms) and 
            any(r in text for r in reality_terms))

# F6.13 : Proximité opposition-junūn
f6_13 = count_proximity(text, opposition, junun_terms)

# F6.14 : Révélation progressive (تبين، اتضح)
progressive_revelation = ['تبين', 'اتضح', 'انكشف', 'ظهر']
f6_14 = int(any(p in text for p in progressive_revelation))

# F6.15 : Contraste temporel (avant/après)
temporal_contrast = ['قبل', 'بعد', 'أولا', 'ثانيا', 'كان...فصار']
f6_15 = int(sum(text.count(t) for t in temporal_contrast) >= 2)

# F6.16 : Exception (إلا)
f6_16 = text.count('إلا')

# F6.17 : Restriction (إنما، فقط، وحده)
restriction = ['إنما', 'فقط', 'وحده', 'لا غير']
f6_17 = int(any(r in text for r in restriction))

# F6.18 : Adversatif fort (بل × opposition)
f6_18 = int('بل' in text and any(o in text for o in opposition))
```

---

## CATÉGORIE 7 : AUTORITÉ ET STATUT SOCIAL (12 features)

### F7.1 : Calife
**Type** : Binaire
**Calcul** : الخليفة، أمير المؤمنين، الرشيد، المأمون, المتوكل
**Prévalence** : 3.8% des positifs
**Poids attendu** : +2.0 (très spécifique)

```python
caliph = ['الخليفة', 'أمير المؤمنين', 'أميرالمؤمنين', 
          'الرشيد', 'المأمون', 'المتوكل', 'المعتصم', 'المهدي']
f7_1 = int(any(c in text for c in caliph))
```

---

### F7.2 : Émir/Prince
**Type** : Binaire
**Prévalence** : 5.8% des positifs

```python
emir = ['الأمير', 'أمير', 'الملك', 'السلطان']
f7_2 = int(any(e in text for e in emir))
```

---

### F7.3 : Vizir
**Type** : Binaire
**Prévalence** : 0.4% des positifs (très rare)

```python
vizier = ['الوزير', 'وزير']
f7_3 = int(any(v in text for v in vizier))
```

---

### F7.4 : Juge (qāḍī)
**Type** : Binaire
**Prévalence** : 1.5% des positifs

```python
judge = ['القاضي', 'قاض', 'قاضي']
f7_4 = int(any(j in text for j in judge))
```

---

### F7.5 : Savant/Faqīh
**Type** : Binaire
**Prévalence** : 2.2% des positifs

```python
scholar = ['العالم', 'الفقيه', 'الشيخ', 'فقيه', 'عالم']
f7_5 = int(any(s in text for s in scholar))
```

---

### F7.6 : Soufi/Ascète
**Type** : Binaire
**Prévalence** : 0.2% des positifs (très rare)

```python
sufi = ['الصوفي', 'الزاهد', 'العابد', 'صوفي', 'زاهد']
f7_6 = int(any(s in text for s in sufi))
```

---

### F7.7 : Autorité TOTALE (any)
**Type** : Binaire
**Calcul** : Au moins une figure d'autorité
**Prévalence** : 10.4% des positifs
**Poids attendu** : +1.5

```python
all_authority = caliph + emir + vizier + judge + scholar + sufi
f7_7 = int(any(a in text for a in all_authority))
```

---

### F7.8 : Proximité autorité-junūn
**Type** : Binaire
**Calcul** : Autorité dans 100 chars de junūn
**Interprétation** : Fou devant l'autorité = cœur du motif
**Poids attendu** : +3.0

```python
f7_8 = count_proximity(text, all_authority, junun_terms, window=100)
```

---

### F7.9 : Interaction autorité-fou (قال + الخليفة + مجنون)
**Type** : Binaire
**Calcul** : Triple co-occurrence

```python
has_auth = any(a in text for a in all_authority)
has_jun = any(j in text for j in junun_terms)
has_qala = any(q in text for q in qala_variants)
f7_9 = int(has_auth and has_jun and has_qala)
```

---

### F7.10 : Peuple/foule (ناس، قوم)
**Type** : Binaire
**Calcul** : الناس، القوم، الجماعة

```python
crowd = ['الناس', 'ناس', 'القوم', 'قوم', 'الجماعة', 'جماعة']
f7_10 = int(any(c in text for c in crowd))
```

---

### F7.11 : Enfants (أطفال)
**Type** : Binaire
**Calcul** : الصبيان، الأطفال، صبي

```python
children = ['الصبيان', 'صبيان', 'الأطفال', 'أطفال', 'صبي', 'طفل']
f7_11 = int(any(c in text for c in children))
```

---

### F7.12 : Hiérarchie sociale explicite
**Type** : Binaire
**Calcul** : سيد، عبد، مولى

```python
hierarchy = ['سيد', 'السيد', 'عبد', 'العبد', 'مولى', 'المولى']
f7_12 = int(any(h in text for h in hierarchy))
```

---

## CATÉGORIE 8 : ESPACE ET LIEUX (10 features)

### F8.1 : Ville nommée
**Type** : Binaire
**Calcul** : البصرة، بغداد، الكوفة، مكة، المدينة
**Prévalence** : 18.8% des positifs

```python
cities = ['البصرة', 'بغداد', 'الكوفة', 'مكة', 'المدينة', 
          'دمشق', 'مصر', 'الشام']
f8_1 = int(any(c in text for c in cities))
```

---

### F8.2 : Espace public (marché, rue)
**Type** : Binaire
**Prévalence** : 6.2% des positifs

```python
public_space = ['السوق', 'الطريق', 'الشارع', 'الباب', 'الميدان']
f8_2 = int(any(p in text for p in public_space))
```

---

### F8.3 : Espace sacré (mosquée)
**Type** : Binaire
**Prévalence** : 4.9% des positifs

```python
sacred_space = ['المسجد', 'الكعبة', 'البيت', 'الحرم']
f8_3 = int(any(s in text for s in sacred_space))
```

---

### F8.4 : Espace privé (maison)
**Type** : Binaire
**Prévalence** : 6.2% des positifs

```python
private_space = ['الدار', 'البيت', 'المنزل', 'الحجرة']
f8_4 = int(any(p in text for p in private_space))
```

---

### F8.5 : Désert/nature sauvage
**Type** : Binaire
**Prévalence** : 0.4% des positifs (très rare)

```python
wilderness = ['الصحراء', 'البادية', 'الجبل', 'البرية']
f8_5 = int(any(w in text for w in wilderness))
```

---

### F8.6 : Asile/prison (māristān)
**Type** : Binaire
**Calcul** : البيمارستان، السجن، الحبس

```python
asylum = ['البيمارستان', 'بيمارستان', 'السجن', 'سجن', 'الحبس']
f8_6 = int(any(a in text for a in asylum))
```

---

### F8.7 : Cour du calife
**Type** : Binaire
**Calcul** : المجلس، الديوان، القصر

```python
court = ['المجلس', 'مجلس', 'الديوان', 'ديوان', 'القصر', 'قصر']
f8_7 = int(any(c in text for c in court))
```

---

### F8.8 : Espace indéterminé (absent)
**Type** : Binaire
**Calcul** : Aucun lieu spécifique mentionné
**Interprétation** : Texte abstrait/définitionnel ?

```python
all_places = cities + public_space + sacred_space + private_space + wilderness + asylum + court
f8_8 = int(not any(p in text for p in all_places))
```

---

### F8.9 : Mouvement spatial (verbes)
**Type** : Binaire
**Calcul** : ذهب، جاء، مشى، دخل، خرج

```python
movement = ['ذهب', 'جاء', 'مشى', 'دخل', 'خرج', 'رجع', 'عاد']
f8_9 = int(any(m in text for m in movement))
```

---

### F8.10 : Bassora spécifiquement
**Type** : Binaire
**Calcul** : البصرة (ville la plus fréquente dans résumés)
**Prévalence** : ~8% des positifs

```python
f8_10 = int('البصرة' in text)
```

---

## CATÉGORIE 9 : CORPS ET PHYSIQUE (8 features)

### F9.1 : Nudité
**Type** : Binaire
**Prévalence** : 5.1% des positifs

```python
nudity = ['عريان', 'عريانا', 'عار', 'عاريا', 'عراة', 'تعرى']
f9_1 = int(any(n in text for n in nudity))
```

---

### F9.2 : Saleté/déchéance physique
**Type** : Binaire
**Prévalence** : 0.4% des positifs (très rare)

```python
dirtiness = ['قذر', 'وسخ', 'دنس', 'قذارة']
f9_2 = int(any(d in text for d in dirtiness))
```

---

### F9.3 : Cheveux/tête
**Type** : Binaire
**Prévalence** : 10.8% des positifs

```python
hair = ['شعر', 'شعره', 'شعرها', 'رأسه', 'رأس', 'الرأس']
f9_3 = int(any(h in text for h in hair))
```

---

### F9.4 : Vêtements
**Type** : Binaire
**Prévalence** : 4.6% des positifs

```python
clothes = ['ثوب', 'ثيابه', 'ثياب', 'خرقة', 'كساء', 'لباس']
f9_4 = int(any(c in text for c in clothes))
```

---

### F9.5 : Enchaînement/emprisonnement
**Type** : Binaire
**Calcul** : مقيد، مربوط، محبوس

```python
chains = ['مقيد', 'مربوط', 'محبوس', 'القيد', 'السلسلة', 'مكبل']
f9_5 = int(any(c in text for c in chains))
```

---

### F9.6 : Gestes transgressifs
**Type** : Binaire
**Calcul** : رمى، كسر، ضرب، هدم

```python
transgression = ['رمى', 'كسر', 'ضرب', 'هدم', 'شق']
f9_6 = int(any(t in text for t in transgression))
```

---

### F9.7 : Maladie/faiblesse physique
**Type** : Binaire
**Calcul** : مريض، سقيم، ضعيف

```python
illness = ['مريض', 'سقيم', 'ضعيف', 'مرض', 'سقم']
f9_7 = int(any(i in text for i in illness))
```

---

### F9.8 : Beauté/laideur physique
**Type** : Binaire
**Calcul** : جميل، قبيح، حسن، دميم

```python
appearance = ['جميل', 'قبيح', 'حسن', 'دميم']
f9_8 = int(any(a in text for a in appearance))
```

---

## CATÉGORIE 10 : TEMPORALITÉ ET NARRATION (10 features)

### F10.1 : Verbes de narration (كان)
**Type** : Densité
**Calcul** : (Occurrences كان) / (Nombre mots)
**Prévalence** : 34.1% ont au moins un كان

```python
kana_variants = ['كان', 'فكان', 'كانت', 'وكان']
n_kana = sum(text.count(k) for k in kana_variants)
f10_1 = n_kana / len(tokens)
```

---

### F10.2 : Succession temporelle (ثم، فـ)
**Type** : Numérique
**Calcul** : Nombre de marqueurs de succession
**Prévalence** : 43.1% des positifs

```python
succession = ['ثم', 'فـ', 'بعد', 'قبل']
f10_2 = sum(text.count(s) for s in succession)
```

---

### F10.3 : Condition (إن، إذا، لو)
**Type** : Binaire
**Prévalence** : 75.7% des positifs (!)
**Poids attendu** : +0.5

```python
condition = ['إن', 'إذا', 'لو', 'لما']
f10_3 = int(any(c in text for c in condition))
```

---

### F10.4 : Surprise temporelle (فإذا)
**Type** : Binaire
**Prévalence** : 20.6% des positifs
**Poids attendu** : +1.5

```python
temporal_surprise = ['فإذا', 'وإذا', 'إذا هو']
f10_4 = int(any(t in text for t in temporal_surprise))
```

---

### F10.5 : Passé narratif vs présent gnomique
**Type** : Ratio
**Calcul** : (Verbes passé) / (Verbes présent)
**Note** : Nécessite analyse morphologique

```python
# Approximation : كان (passé) vs يكون (présent)
past_count = text.count('كان') + text.count('كانت')
present_count = text.count('يكون') + text.count('هو')
f10_5 = past_count / (present_count + 1)
```

---

### F10.6 : Durée (زمن طويل، سنين)
**Type** : Binaire
**Calcul** : زمن، سنة، سنين، أيام

```python
duration = ['زمن', 'زمنا', 'سنة', 'سنين', 'أيام', 'شهر', 'عام']
f10_6 = int(any(d in text for d in duration))
```

---

### F10.7 : Instantanéité (فجأة، حالا)
**Type** : Binaire
**Calcul** : فجأة، في الحال، لحظة

```python
instant = ['فجأة', 'في الحال', 'حالا', 'لحظة', 'بغتة']
f10_7 = int(any(i in text for i in instant))
```

---

### F10.8 : Répétition (كلما، مرة)
**Type** : Binaire
**Calcul** : كلما، في كل مرة

```python
repetition = ['كلما', 'كل مرة', 'مرارا', 'تكرار']
f10_8 = int(any(r in text for r in repetition))
```

---

### F10.9 : Début de récit (كان في...)
**Type** : Binaire
**Calcul** : كان في الـ (1ers 50 chars)

```python
f10_9 = int('كان' in text[:50])
```

---

### F10.10 : Clôture narrative (final)
**Type** : Binaire
**Calcul** : Marqueurs de fin dans dernier tiers

```python
closure = ['توفي', 'مات', 'انتهى', 'تم', 'آخر']
last_third = text[int(len(text)*0.66):]
f10_10 = int(any(c in last_third for c in closure))
```

---

## CATÉGORIE 11 : RELIGION ET SPIRITUALITÉ (8 features)

### F11.1 : Citations coraniques
**Type** : Binaire
**Prévalence** : 1.5% des positifs (rare)

```python
quran = ['قال الله', 'قوله تعالى', 'في القرآن', 'الآية']
f11_1 = int(any(q in text for q in quran))
```

---

### F11.2 : Citations de ḥadīṯ
**Type** : Binaire
**Prévalence** : 0.4% des positifs (très rare)

```python
hadith = ['قال النبي', 'قال رسول الله', 'في الحديث', 'روي عن النبي']
f11_2 = int(any(h in text for h in hadith))
```

---

### F11.3 : Invocations (duʿāʾ)
**Type** : Binaire
**Calcul** : اللهم، رحمه الله، جزاك الله

```python
invocation = ['اللهم', 'رحمه الله', 'جزاك الله', 'بارك الله', 'حفظه الله']
f11_3 = int(any(i in text for i in invocation))
```

---

### F11.4 : Prière rituelle (صلاة)
**Type** : Binaire
**Calcul** : صلى، يصلي، صلاة

```python
prayer = ['صلى', 'يصلي', 'صلاة', 'الصلاة']
f11_4 = int(any(p in text for p in prayer))
```

---

### F11.5 : Monde/dunya (الدنيا)
**Type** : Binaire
**Calcul** : الدنيا، الدنيا والآخرة

```python
dunya = ['الدنيا', 'دنيا', 'الآخرة', 'آخرة']
f11_5 = int(any(d in text for d in dunya))
```

---

### F11.6 : Amour divin (حب الله)
**Type** : Binaire
**Calcul** : حب الله، عشق الله، محبة الله

```python
divine_love = ['حب الله', 'عشق الله', 'محبة الله', 'حبيب الله']
f11_6 = int(any(d in text for d in divine_love))
```

---

### F11.7 : Ascèse (زهد)
**Type** : Binaire
**Calcul** : زهد، زاهد، ترك الدنيا

```python
asceticism = ['زهد', 'الزهد', 'زاهد', 'الزاهد', 'ترك الدنيا']
f11_7 = int(any(a in text for a in asceticism))
```

---

### F11.8 : Extase/ivresse spirituelle
**Type** : Binaire
**Calcul** : سكر، مخمور، وجد، سكران (contexte mystique)

```python
ecstasy = ['سكر', 'سكران', 'وجد', 'الوجد', 'غيبة']
f11_8 = int(any(e in text for e in ecstasy))
```

---

## CATÉGORIE 12 : STATISTIQUES GÉNÉRALES (15 features)

### F12.1 : Longueur totale (mots)
**Type** : Numérique (log-transformé)
**Distribution** : Médiane = 47 mots, Moyenne = 73 mots

```python
f12_1 = math.log(len(tokens) + 1)
```

---

### F12.2 : Longueur totale (caractères)
**Type** : Numérique (log-transformé)

```python
f12_2 = math.log(len(text) + 1)
```

---

### F12.3 : Ratio caractères/mots (longueur moyenne mot)
**Type** : Numérique
**Distribution** : Moyenne ≈ 4.5 lettres/mot

```python
f12_3 = len(text) / (len(tokens) + 1)
```

---

### F12.4 : Densité de mots arabes (vs ponctuation)
**Type** : Numérique

```python
arabic_chars = len(re.findall(r'[\u0621-\u064A\u0671-\u06D3]', text))
f12_4 = arabic_chars / (len(text) + 1)
```

---

### F12.5 : Type-token ratio (diversité lexicale)
**Type** : Numérique (0.0 à 1.0)
**Calcul** : (Mots uniques) / (Total mots)

```python
unique_tokens = len(set(tokens))
f12_5 = unique_tokens / (len(tokens) + 1)
```

---

### F12.6 : Hapax ratio (mots apparaissant 1 seule fois)
**Type** : Numérique

```python
from collections import Counter
token_counts = Counter(tokens)
hapax = sum(1 for c in token_counts.values() if c == 1)
f12_6 = hapax / (len(tokens) + 1)
```

---

### F12.7 : Répétitions lexicales
**Type** : Numérique
**Calcul** : Proportion de mots répétés

```python
repeated = sum(1 for c in token_counts.values() if c > 1)
f12_7 = repeated / len(token_counts)
```

---

### F12.8 : Densité de noms propres (approximation)
**Type** : Numérique
**Calcul** : Mots commençant par majuscule / Total
**Note** : Peu fiable en arabe sans NER

```python
# Approximation via liste de noms connus
names = famous_fools + ['محمد', 'علي', 'عمر', 'حسن']
n_names = sum(1 for t in tokens if t in names)
f12_8 = n_names / (len(tokens) + 1)
```

---

### F12.9 : Densité de chiffres
**Type** : Numérique
**Calcul** : Proportion de tokens numériques

```python
numbers = ['واحد', 'اثنان', 'ثلاثة', 'عشرة', 'مائة', 'ألف']
n_numbers = sum(1 for t in tokens if t in numbers or t.isdigit())
f12_9 = n_numbers / (len(tokens) + 1)
```

---

### F12.10 : Nombre de phrases (approximation)
**Type** : Numérique
**Calcul** : Nombre de . و ؛

```python
f12_10 = text.count('.') + text.count('؛') + text.count('،')
```

---

### F12.11 : Longueur moyenne phrase
**Type** : Numérique

```python
n_sentences = f12_10 if f12_10 > 0 else 1
f12_11 = len(tokens) / n_sentences
```

---

### F12.12 : Densité de conjonctions
**Type** : Numérique

```python
conjunctions = ['و', 'أو', 'ثم', 'لكن', 'بل', 'أم']
n_conj = sum(text.count(c) for c in conjunctions)
f12_12 = n_conj / (len(tokens) + 1)
```

---

### F12.13 : Densité de prépositions
**Type** : Numérique

```python
prepositions = ['في', 'من', 'إلى', 'على', 'عن', 'ب', 'ل']
n_prep = sum(text.count(p) for p in prepositions)
f12_13 = n_prep / (len(tokens) + 1)
```

---

### F12.14 : Ratio verbes/noms (approximation)
**Type** : Numérique
**Note** : Nécessite morphologie (CAMeL Tools)

```python
# Placeholder (nécessite vraie analyse POS)
f12_14 = 0.5  # Défaut neutre
```

---

### F12.15 : Densité lexicale (mots pleins vs mots outils)
**Type** : Numérique
**Calcul** : 1 - (mots outils / total)

```python
function_words = ['في', 'من', 'إلى', 'على', 'عن', 'الذي', 'التي', 
                  'هذا', 'ذلك', 'أن', 'إن', 'قد', 'لم']
n_function = sum(text.count(fw) for fw in function_words)
f12_15 = 1 - (n_function / (len(tokens) + 1))
```

---

## CATÉGORIE 13 : FEATURES SPÉCIFIQUES AU GENRE (10 features)

### F13.1 : Marqueurs wasf (définitionnel)
**Type** : Binaire
**Calcul** : ومنها، ضروب، تقول العرب، يقال
**Interprétation** : SIGNAL NÉGATIF fort
**Poids attendu** : -3.0

```python
wasf_markers = ['ومنها', 'والفعل منه', 'والاسم', 'ضروب', 
                'تقول العرب', 'ومن أمثالهم', 'يقال له ذلك']
f13_1 = int(any(w in text for w in wasf_markers))
```

---

### F13.2 : Témoignage oculaire (mubāshara)
**Type** : Binaire
**Prévalence** : 25.7% des positifs
**Poids attendu** : +1.5

```python
witness = ['رأيت', 'فرأيت', 'مررت', 'فمررت', 'لقيت', 'فلقيت']
f13_2 = int(any(w in text for w in witness))
```

---

### F13.3 : Chute narrative (nadīra finale)
**Type** : Binaire
**Calcul** : Validation OU pointe finale dans dernier tiers

```python
last_third = text[int(len(text)*0.66):]
has_validation_end = any(v in last_third for v in all_validation)
has_qala_end = any(q in last_third for q in qala_variants)
f13_3 = int(has_validation_end or has_qala_end)
```

---

### F13.4 : Structure khabar classique (isnad + matn)
**Type** : Binaire
**Calcul** : Présence de transmetteurs dans 1ers 20%

```python
first_fifth = text[:int(len(text)*0.2)]
f13_4 = int(any(t in first_fifth for t in transmission))
```

---

### F13.5 : Densité de noms de transmetteurs
**Type** : Numérique
**Calcul** : حدثنا + أخبرنا par 100 mots

```python
n_trans = sum(text.count(t) for t in transmission)
f13_5 = n_trans / (len(tokens) / 100)
```

---

### F13.6 : Genre déclaré (depuis métadonnées)
**Type** : Catégoriel (one-hot encoding)
**Calcul** : khabar=1, nadira=2, shir=3, wasf=4, hikma=5

```python
# Depuis khabar.get('genre')
genre_map = {'khabar': 1, 'nadira': 2, 'shir': 3, 'wasf': 4, 'hikma': 5}
f13_6 = genre_map.get(khabar.get('genre', ''), 0)
```

---

### F13.7 : Résumé contient "amour"
**Type** : Binaire
**Calcul** : Mot "amour" dans résumé français
**Interprétation** : Folie d'amour (ġazal) ?
**Poids attendu** : -0.8

```python
# Depuis khabar.get('resume')
resume = khabar.get('resume', '').lower()
f13_7 = int('amour' in resume)
```

---

### F13.8 : Résumé contient "narrateur"
**Type** : Binaire
**Calcul** : Mot "narrateur" dans résumé
**Poids attendu** : +1.0

```python
f13_8 = int('narrateur' in resume)
```

---

### F13.9 : Résumé contient "récite"/"déclame"
**Type** : Binaire
**Calcul** : Mots de récitation poétique
**Poids attendu** : +0.5 (contexte poétique mais valide)

```python
f13_9 = int(any(w in resume for w in ['récite', 'déclame', 'poème']))
```

---

### F13.10 : Longueur du résumé
**Type** : Numérique (log)
**Calcul** : log(nombre de mots du résumé)
**Interprétation** : Résumés longs = textes complexes ?

```python
resume_words = len(resume.split())
f13_10 = math.log(resume_words + 1)
```

---

## 🎯 RÉSUMÉ FINAL : 150 FEATURES AU TOTAL

| Catégorie | Nombre | Exemples clés |
|-----------|--------|--------------|
| 1. Lexique folie | 15 | has_junun, junun_density, famous_fools |
| 2. Lexique sagesse | 10 | has_aql, paradox_cooccurrence, proximity_junun_aql |
| 3. Structure dialogique | 20 | qala_density, first_person, junun_near_qala |
| 4. Poésie | 12 | poetry_markers, poetry_ratio, poetry_alone |
| 5. Validation | 15 | laugh, gift, validation_in_final |
| 6. Contraste | 18 | opposition, correction, revelation, surprise |
| 7. Autorité | 12 | caliph, authority_junun_proximity |
| 8. Espaces | 10 | cities, public_space, asylum |
| 9. Corps | 8 | nudity, chains, transgression |
| 10. Temporalité | 10 | kana_density, succession, temporal_surprise |
| 11. Religion | 8 | quran, divine_love, asceticism |
| 12. Stats générales | 15 | log_length, type_token_ratio, lexical_density |
| 13. Genre | 10 | wasf_markers, witness, genre_declared |

**TOTAL** : **153 features**

---

## 📌 RECOMMANDATIONS POUR L'IMPLÉMENTATION

### Priorités de développement

**PHASE 1 (Essentiel - 40 features)** :
- Toute la catégorie 1 (lexique folie)
- F2.1 à F2.5 (lexique sagesse)
- F3.1 à F3.6 (dialogue)
- F5.1 à F5.8 (validation)
- F6.1 à F6.4 (contraste)
- F12.1, F12.5, F12.15 (stats générales)

**PHASE 2 (Important - 50 features)** :
- Catégories 7, 8, 9, 10 (autorité, lieux, corps, temps)
- Features de proximité contextuelle (fenêtres)
- Features de position (début/milieu/fin)

**PHASE 3 (Avancé - 63 features)** :
- Features morphologiques (nécessite CAMeL Tools)
- Features syntaxiques
- Features de patterns séquentiels

---

## 🔬 MÉTHODE DE VALIDATION

Pour chaque feature :

1. **Calcul de prévalence** sur positifs vs négatifs
2. **Test de corrélation** avec label (chi2, mutual information)
3. **Inspection qualitative** : afficher 10 exemples positifs avec valeur maximale
4. **Validation philologique** : cette feature fait-elle sens ?
5. **Test d'importance** dans modèle ML (permutation importance)

---

## 💡 INSIGHTS CLÉS POUR TON PIPELINE

### Features à fort pouvoir discriminant attendu

1. **F3.6** : junūn_near_qala (+3.0) — Le fou EST le locuteur
2. **F7.8** : authority_junun_proximity (+3.0) — Fou devant autorité
3. **F2.4** : junun_aql_proximity (+3.0) — Paradoxe rapproché
4. **F6.10** : surprise_formula "فإذا هو" (+2.5)
5. **F13.1** : wasf_markers (-3.0) — SIGNAL NÉGATIF

### Features à valider empiriquement

- F3.10 : questions (94.7% prévalence = peut-être trop fréquent ?)
- F6.6 : negation_junun (55.3% = surprenant)
- F10.3 : condition (75.7% = très fréquent)

### Combinaisons à tester

- F1.4 × F3.3 : Fou nommé + dialogue 1ère personne
- F2.3 × F7.7 : Paradoxe + autorité
- F5.8 × F6.3 : Validation finale + révélation

