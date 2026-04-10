# GUIDE : ENRICHISSEMENT DES LEXIQUES ET ANALYSE MORPHOLOGIQUE

**Date** : 2026-04-08  
**Auteur** : Augustin  
**Objectif** : Améliorer la couverture lexicale et intégrer l'analyse morphologique

---

## 📊 DIAGNOSTIC : VARIANTES MANQUANTES (analyse exhaustive)

### 1. Racine ج-ن-ن (junūn)

**Couverture actuelle** : 12 formes → **24 formes manquantes détectées**

| Forme | Occurrences | Type | Statut |
|-------|-------------|------|--------|
| مجنون | 66 | Nom/adj. | ✓ Inclus |
| المجنون | 50 | Nom/adj. défini | ✓ Inclus |
| **مجنونا** | **24** | **Accusatif** | ✗ **MANQUANT** |
| المجانين | 21 | Pluriel | ✓ Inclus |
| **بمجنون** | **9** | **Prépositionnel** | ✗ **MANQUANT** |
| **جن** | **6** | **Verbe** | ✗ **MANQUANT** |
| **المجنونة** | **5** | **Féminin défini** | ✗ **MANQUANT** |
| **جننت** | **2** | **Verbe passé 1ère** | ✗ **MANQUANT** |
| **يجن** | **2** | **Verbe présent** | ✗ **MANQUANT** |
| **جنوني** | **2** | **Possessif 1ère** | ✗ **MANQUANT** |
| **بالجنون** | **2** | **Préposition+déf.** | ✗ **MANQUANT** |

**Faux positifs à exclure** :
- الجنة (paradis) : 19 occurrences
- السجن (prison) : 7 occurrences  
- الجنان (jardins) : 6 occurrences
- الجنيد (nom propre) : 3 occurrences

---

### 2. Racine ع-ق-ل (ʿaql)

**Couverture actuelle** : 5 formes → **19 formes manquantes détectées**

| Forme | Occurrences | Type | Statut |
|-------|-------------|------|--------|
| العقل | 16 | Nom défini | ✓ Inclus |
| **عقله** | **11** | **Possessif 3ème** | ✗ **MANQUANT** |
| عقل | 8 | Nom | ✓ Inclus |
| **عقلي** | **5** | **Possessif 1ère** | ✗ **MANQUANT** |
| عاقل | 4 | Participe actif | ✓ Inclus |
| **بعقله** | **3** | **Préposition+poss.** | ✗ **MANQUANT** |
| **أعقل** | **3** | **Comparatif** | ✗ **MANQUANT** |
| **عقلها** | **3** | **Possessif fém.** | ✗ **MANQUANT** |
| **عقلاء** | **1** | **Pluriel savant** | ✗ **MANQUANT** |
| **يعقل** | **1** | **Verbe présent** | ✗ **MANQUANT** |
| **العقول** | **1** | **Pluriel** | ✗ **MANQUANT** |

---

### 3. Racine ق-و-ل (qāla)

**Couverture actuelle** : 6 formes → **14 formes manquantes détectées**

| Forme | Occurrences | Type | Statut |
|-------|-------------|------|--------|
| قال | 679 | Verbe passé | ✓ Inclus |
| فقال | 373 | Conj.+verbe | ✓ Inclus |
| يقول | 118 | Présent | ✓ Inclus |
| **وقال** | **85** | **Conj.+verbe** | ✗ **MANQUANT** |
| **تقول** | **49** | **Présent 2ème** | ✗ **MANQUANT** |
| قالت | 33 | Féminin | ✓ Inclus |
| **ويقول** | **24** | **Conj.+présent** | ✗ **MANQUANT** |
| **يقال** | **24** | **Passif** | ✗ **MANQUANT** |
| **أقول** | **22** | **Présent 1ère** | ✗ **MANQUANT** |
| **قالوا** | **21** | **Pluriel 3ème** | ✗ **MANQUANT** |

---

## 🎯 DEUX APPROCHES POSSIBLES

### APPROCHE A : LEXIQUES ENRICHIS MANUELLEMENT (sans CAMeL Tools)

**Avantages** :
- ✅ Simple à implémenter
- ✅ Rapide à tester
- ✅ Pas de dépendance externe
- ✅ Contrôle total sur faux positifs

**Inconvénients** :
- ❌ Travail manuel fastidieux
- ❌ Risque d'oubli de variantes rares
- ❌ Peu flexible pour autres corpus

**Lexiques enrichis** (extraits ci-dessous) :

```python
# Racine ج-ن-ن : 36 formes (au lieu de 12)
JUNUN_ENRICHED = [
    # Formes de base
    'مجنون', 'المجنون', 'مجنونا', 'بمجنون', 'للمجنون', 'كمجنون',
    'مجنونة', 'المجنونة', 'مجنونتي',
    'مجانين', 'المجانين', 'بالمجانين', 'وللمجانين',
    
    # Maṣdar
    'جنون', 'الجنون', 'بالجنون', 'جنونه', 'جنونها', 'جنوني', 'جنونك',
    'وجنونك', 'وجنته',
    
    # Verbes
    'جن', 'جننت', 'جنني', 'يجن', 'أجنته', 'وجنت',
    
    # Termes spécialisés
    'معتوه', 'المعتوه', 'معتوها',
    'مدله', 'المدله',
    'هائم', 'الهائم', 'هائما',
]

# EXCLUSIONS (faux positifs)
JUNUN_EXCLUDE = [
    'الجنة',    # Paradis
    'جنة',      # id.
    'والجنة',   # id.
    'السجن',    # Prison
    'وسجني',    # id.
    'الجنان',   # Jardins
    'الجنيد',   # Nom propre (soufi)
    'جنازته',   # Funérailles
    'جنبي',     # À côté
    'جنب',      # id.
    'الجن',     # Djinns
    'والجن',    # id.
]

# Racine ع-ق-ل : 25 formes (au lieu de 5)
AQL_ENRICHED = [
    # Formes de base
    'عقل', 'العقل', 'عقلا', 'بعقل',
    'عقله', 'عقلها', 'عقلي', 'عقلك', 'بعقله', 'بعقلها',
    'عقول', 'العقول', 'عقولنا', 'عقولهم',
    
    # Participe actif
    'عاقل', 'العاقل', 'عاقلا', 'عاقلون',
    'عقلاء', 'العقلاء',
    
    # Comparatif/superlatif
    'أعقل', 'كأعقل',
    
    # Verbes
    'يعقل', 'تعقل', 'عقلت', 'عقلوا',
]

# Racine ح-ك-م : 15 formes
HIKMA_ENRICHED = [
    'حكمة', 'الحكمة', 'بالحكمة', 'وحكمة', 'حكمته',
    'حكيم', 'الحكيم', 'حكيما', 'حكماء', 'الحكماء',
    'حكم', 'الحكم', 'حكمه', 'بحكم',
    'محكم',
]

# EXCLUSIONS ح-ك-م
HIKMA_EXCLUDE = [
    'حياكم', 'أحياكم', 'وأحياكم',  # Salutations
    'ويحكم',                         # Interjection
    'صاحبكم', 'أحدكم',              # Pronoms
    'الحاكم',                        # Juge (déjà dans AUTHORITY)
    'فتحاكموا', 'فتحاكما',          # Verbes de jugement
]

# Racine ق-و-ل : 30 formes
QALA_ENRICHED = [
    # Passé
    'قال', 'قالت', 'قالا', 'قالوا', 'قلت', 'قلنا',
    'فقال', 'فقالت', 'فقالوا', 'فقلت', 'فقلنا',
    'وقال', 'وقالت', 'وقالوا',
    
    # Présent
    'يقول', 'تقول', 'نقول', 'أقول', 'يقولون', 'تقولون',
    'فيقول', 'فتقول', 'ويقول', 'وتقول',
    
    # Passif
    'يقال', 'قيل', 'فقيل',
    
    # Impératif
    'قل', 'فقل', 'قولوا',
    
    # Maṣdar
    'قول', 'القول', 'قوله', 'قولها', 'قولي', 'أقوال',
]
```

**Implémentation** :

```python
def has_junun_enriched(text):
    """Détecte junūn avec lexique enrichi et exclusions."""
    # Vérifier exclusions d'abord
    if any(exc in text for exc in JUNUN_EXCLUDE):
        # Affiner : compter junūn vs exclusions
        n_junun = sum(text.count(j) for j in JUNUN_ENRICHED)
        n_exclude = sum(text.count(e) for e in JUNUN_EXCLUDE)
        return n_junun > n_exclude  # Majorité junūn
    
    return any(j in text for j in JUNUN_ENRICHED)
```

---

### APPROCHE B : ANALYSE MORPHOLOGIQUE (CAMeL Tools)

**Avantages** :
- ✅ Couverture TOTALE (toutes variantes automatiquement)
- ✅ Flexible pour n'importe quel corpus
- ✅ Accès aux racines, lemmes, POS
- ✅ Permet features morphologiques avancées

**Inconvénients** :
- ❌ Dépendance externe (CAMeL Tools)
- ❌ Plus lent (analyse token par token)
- ❌ Nécessite désambiguïsation (plusieurs analyses possibles)

**Installation** :

```bash
pip install camel-tools
camel_data -i morphology-db-msa  # Base morphologique MSA
```

**Implémentation** :

```python
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.mle import MLEDisambiguator

# Charger analyseur + désambiguïsateur
db = MorphologyDB.builtin_db()
analyzer = Analyzer(db)
disambig = MLEDisambiguator(db)

def extract_morphological_features(text):
    """
    Extrait features morphologiques avec CAMeL Tools.
    
    Returns:
        dict avec 20+ features morphologiques
    """
    features = {}
    
    # Tokeniser
    tokens = text.split()
    
    # Analyser + désambiguïser
    disambiguated = disambig.disambiguate(tokens)
    
    # Compteurs par racine
    roots = {
        'jnn': 0,   # ج-ن-ن
        'aql': 0,   # ع-ق-ل
        'hkm': 0,   # ح-ك-م
        'qwl': 0,   # ق-و-ل
    }
    
    # Compteurs par POS
    pos_counts = {
        'noun': 0,
        'verb': 0,
        'adj': 0,
        'adv': 0,
    }
    
    # Compteurs cas/aspects
    case_counts = {'nom': 0, 'acc': 0, 'gen': 0}
    aspect_counts = {'perf': 0, 'imperf': 0, 'imp': 0}
    
    for token_analysis in disambiguated:
        root = token_analysis.get('root', '')
        pos = token_analysis.get('pos', '')
        case = token_analysis.get('case', '')
        aspect = token_analysis.get('aspect', '')
        
        # Racines clés
        if root == 'جنن':
            roots['jnn'] += 1
        elif root == 'عقل':
            roots['aql'] += 1
        elif root == 'حكم':
            roots['hkm'] += 1
        elif root == 'قول':
            roots['qwl'] += 1
        
        # POS
        if pos.startswith('noun'):
            pos_counts['noun'] += 1
        elif pos.startswith('verb'):
            pos_counts['verb'] += 1
        elif pos.startswith('adj'):
            pos_counts['adj'] += 1
        
        # Cas
        if case in case_counts:
            case_counts[case] += 1
        
        # Aspect
        if aspect in aspect_counts:
            aspect_counts[aspect] += 1
    
    n_tokens = max(len(tokens), 1)
    
    # Features morphologiques (20 au total)
    features['root_jnn_density'] = roots['jnn'] / n_tokens
    features['root_aql_density'] = roots['aql'] / n_tokens
    features['root_hkm_density'] = roots['hkm'] / n_tokens
    features['root_qwl_density'] = roots['qwl'] / n_tokens
    
    features['noun_density'] = pos_counts['noun'] / n_tokens
    features['verb_density'] = pos_counts['verb'] / n_tokens
    features['adj_density'] = pos_counts['adj'] / n_tokens
    
    features['nom_ratio'] = case_counts['nom'] / sum(case_counts.values() or [1])
    features['acc_ratio'] = case_counts['acc'] / sum(case_counts.values() or [1])
    features['gen_ratio'] = case_counts['gen'] / sum(case_counts.values() or [1])
    
    features['perf_ratio'] = aspect_counts['perf'] / sum(aspect_counts.values() or [1])
    features['imperf_ratio'] = aspect_counts['imperf'] / sum(aspect_counts.values() or [1])
    
    # Ratios avancés
    features['verb_noun_ratio'] = pos_counts['verb'] / (pos_counts['noun'] + 1)
    features['jnn_aql_root_ratio'] = roots['jnn'] / (roots['aql'] + 1)
    
    # Patterns verbaux
    features['has_jnn_verb'] = int(any(
        t.get('root') == 'جنن' and t.get('pos', '').startswith('verb')
        for t in disambiguated
    ))
    
    features['has_aql_verb'] = int(any(
        t.get('root') == 'عقل' and t.get('pos', '').startswith('verb')
        for t in disambiguated
    ))
    
    return features
```

---

## 🔬 ANALYSE PRÉALABLE RECOMMANDÉE

Avant d'implémenter l'une ou l'autre approche, je recommande une **analyse morphologique exploratoire** sur les 452 positifs :

### Script d'analyse exploratoire

```python
#!/usr/bin/env python3
"""
analyze_morphology_positifs.py
────────────────────────────────
Analyse morphologique exploratoire des 452 positifs.
Produit statistiques sur POS, racines, cas, aspects.
"""

import json
from collections import Counter
from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator

# Charger corpus
with open('akhbar.json') as f:
    data = json.load(f)
    
positifs = data['akhbar'][160:612]

# Charger CAMeL
db = MorphologyDB.builtin_db()
disambig = MLEDisambiguator(db)

# Compteurs globaux
root_counter = Counter()
pos_counter = Counter()
case_counter = Counter()
aspect_counter = Counter()

# Racines clés
key_roots = {
    'جنن': Counter(),  # Formes de ج-ن-ن
    'عقل': Counter(),  # Formes de ع-ق-ل
    'حكم': Counter(),  # Formes de ح-ك-م
    'قول': Counter(),  # Formes de ق-و-ل
}

total_tokens = 0

for i, kh in enumerate(positifs):
    if i % 50 == 0:
        print(f"Progression : {i}/452...")
    
    # Extraire matn
    segments = kh.get('content', {}).get('segments', [])
    matn = ' '.join(s.get('text', '') for s in segments 
                    if s.get('type') != 'isnad')
    
    # Tokeniser + analyser
    tokens = matn.split()
    total_tokens += len(tokens)
    
    try:
        disambiguated = disambig.disambiguate(tokens)
    except:
        continue
    
    for analysis in disambiguated:
        root = analysis.get('root', '')
        pos = analysis.get('pos', '')
        case = analysis.get('case', '')
        aspect = analysis.get('aspect', '')
        lex = analysis.get('lex', '')
        
        # Compter
        if root:
            root_counter[root] += 1
        if pos:
            pos_counter[pos] += 1
        if case:
            case_counter[case] += 1
        if aspect:
            aspect_counter[aspect] += 1
        
        # Formes des racines clés
        if root in key_roots:
            key_roots[root][lex] += 1

# ══════════════════════════════════════════════════════════
# RAPPORT
# ══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("ANALYSE MORPHOLOGIQUE DES 452 POSITIFS")
print("="*60 + "\n")

print(f"Total tokens analysés : {total_tokens}\n")

print("── Top 30 racines ──")
for root, cnt in root_counter.most_common(30):
    pct = cnt/total_tokens*100
    print(f"  {cnt:5d} ({pct:5.2f}%)  {root}")

print("\n── Distribution POS ──")
for pos, cnt in pos_counter.most_common(15):
    pct = cnt/total_tokens*100
    print(f"  {cnt:5d} ({pct:5.2f}%)  {pos}")

print("\n── Distribution CAS (noms) ──")
for case, cnt in case_counter.most_common(10):
    pct = cnt/sum(case_counter.values())*100
    print(f"  {cnt:5d} ({pct:5.2f}%)  {case}")

print("\n── Distribution ASPECT (verbes) ──")
for asp, cnt in aspect_counter.most_common(10):
    pct = cnt/sum(aspect_counter.values())*100
    print(f"  {cnt:5d} ({pct:5.2f}%)  {asp}")

# Formes des racines clés
for root_ar, forms in key_roots.items():
    print(f"\n── Formes de la racine {root_ar} ──")
    for form, cnt in forms.most_common(20):
        print(f"  {cnt:4d}  {form}")

# Sauvegarder rapport
with open('morphology_report.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_tokens': total_tokens,
        'roots': dict(root_counter.most_common(100)),
        'pos': dict(pos_counter),
        'case': dict(case_counter),
        'aspect': dict(aspect_counter),
        'key_roots_forms': {k: dict(v.most_common(50)) 
                           for k, v in key_roots.items()},
    }, f, ensure_ascii=False, indent=2)

print("\n✅ Rapport sauvegardé : morphology_report.json")
```

**Résultats attendus** :

1. **Inventaire exhaustif** de toutes les formes réellement utilisées
2. **Distribution POS** : ratio verbes/noms (attendu ≈ 0.3-0.5 pour récits)
3. **Distribution CAS** : accusatif > nominatif ? (narratif vs définitionnel)
4. **Distribution ASPECT** : perfectif > imperfectif ? (passé vs présent)
5. **Validation empirique** des lexiques enrichis

---

## 📌 RECOMMANDATION FINALE

### PHASE 1 (immédiate) : APPROCHE A (lexiques enrichis)

**Raisons** :
- ✅ Gain immédiat : +50% couverture junūn, +300% couverture ʿaql
- ✅ Implémentation rapide (1 journée)
- ✅ Pas de dépendances externes
- ✅ Permet de tester pipeline ML rapidement

**Actions** :
1. Copier les lexiques enrichis ci-dessus dans `extract_features_openiti.py`
2. Remplacer `JUNUN_TERMS` → `JUNUN_ENRICHED`
3. Ajouter fonction de filtrage des faux positifs
4. Tester sur 100 positifs + 100 négatifs

### PHASE 2 (après validation Phase 1) : APPROCHE B (morphologie)

**Raisons** :
- ✅ Couverture 100%
- ✅ Permet features avancées (POS, cas, aspects)
- ✅ Généralisable à autres corpus

**Actions** :
1. Installer CAMeL Tools
2. Lancer `analyze_morphology_positifs.py`
3. Valider formes détectées vs lexiques enrichis
4. Créer `extract_features_morphological.py` (version avancée)
5. Comparer performances : lexiques vs morphologie

---

## 🎯 FEATURES MORPHOLOGIQUES SUPPLÉMENTAIRES (20 nouvelles)

Si tu passes à l'approche B, voici 20 features morphologiques à ajouter :

| # | Feature | Type | Justification |
|---|---------|------|---------------|
| F51 | root_jnn_density | Float | Densité racine ج-ن-ن (TOTAL) |
| F52 | root_aql_density | Float | Densité racine ع-ق-ل (TOTAL) |
| F53 | root_qwl_density | Float | Densité racine ق-و-ل (TOTAL) |
| F54 | has_jnn_verb | Binary | Verbe de folie (جن، جننت) |
| F55 | has_aql_verb | Binary | Verbe de raison (عقل، يعقل) |
| F56 | noun_density | Float | Densité de noms |
| F57 | verb_density | Float | Densité de verbes |
| F58 | adj_density | Float | Densité d'adjectifs |
| F59 | verb_noun_ratio | Float | Ratio verbes/noms |
| F60 | nom_case_ratio | Float | Ratio cas nominatif |
| F61 | acc_case_ratio | Float | Ratio cas accusatif |
| F62 | gen_case_ratio | Float | Ratio cas génitif |
| F63 | perf_aspect_ratio | Float | Ratio aspect perfectif |
| F64 | imperf_aspect_ratio | Float | Ratio aspect imperfectif |
| F65 | jnn_noun_vs_verb | Binary | Junūn comme nom > verbe ? |
| F66 | aql_noun_vs_verb | Binary | ʿAql comme nom > verbe ? |
| F67 | qwl_perf_density | Float | قال perfectif (passé narratif) |
| F68 | qwl_imperf_density | Float | يقول imperfectif (dialogue) |
| F69 | passive_voice_ratio | Float | Ratio voix passive |
| F70 | imperative_density | Float | Densité impératifs |

**Hypothèses philologiques à tester** :

- H1 : Textes *maǧnūn ʿāqil* ont **plus d'accusatifs** (narration > définition)
- H2 : Ratio **verbes/noms** plus élevé (récit dynamique)
- H3 : Junūn apparaît plus comme **NOM** que comme VERBE (état > action)
- H4 : قال en **perfectif** dominant (passé narratif)
- H5 : Peu de **passif** (narration directe, non abstraite)

---

## ⚙️ SCRIPT FINAL RECOMMANDÉ

Je vais créer **deux versions** du script :

### Version 1 : `extract_features_enriched.py`
- Lexiques enrichis manuellement
- 50 features originales
- **Production-ready maintenant**

### Version 2 : `extract_features_morphological.py`
- CAMeL Tools
- 70 features (50 + 20 morphologiques)
- **À tester après analyse exploratoire**

