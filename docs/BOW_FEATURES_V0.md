# Features BoW v0 — Documentation philologique et statistique

**Fichier d'implémentation** : `src/uqala_nlp/features/bow_features_v0.py`  
**Date de création** : 2026-04-16  
**Méthode de découverte** : analyse statistique directe du corpus de positifs via BoW + CAMeL Tools (lemmatisation MLE) + exploration multi-axe (6 analyses)

---

## Résumé

Ces 10 features (B01–B10) ont été découvertes par trois analyses successives :

| Script | Méthode | Découvertes clés |
|--------|---------|-----------------|
| `bow_feature_discovery.py` | Fréquences brutes | أنشأ, ذات يوم, الصبيان |
| `bow_camel_feature_discovery.py` | Lemmatisation MLE | أنشأ (confirmé), حب, ج.ن.ن root |
| `deep_feature_exploration.py` | Co-occurrence, position, PMI | B01 (positionnel), عض, موسوس, مقيد, négatifs |

Toutes les statistiques citées ont été calculées sur :
- **460 positifs** (Nisaburi, corpus de référence)
- **3 817 négatifs** (sources multiples)
- **Validation externe** : 66 positifs XGB d'Ibn Abd Rabbih (corpus hors entraînement)

---

## Features positives (B01–B10)

---

### B01 — `f_junun_first_third` ⭐⭐⭐ (priorité maximale)

**Implémentation** : présence d'un terme de folie dans le premier tiers du texte

```python
first_third = set(tokens[:n//3])
B01 = float(bool(JUNUN_ALL & first_third))
```

**Statistiques** :
- Précision : **0.867** — 87% des textes avec مجنون tôt sont des vrais positifs
- Rappel : **39.6%** — couvre presque 40% du corpus positif
- LR : 53.9x
- XGB% : 3% (cross-corpus faible — les positifs XGB d'Ibn Abd Rabbih identifient le fou plus tard)

**Justification philologique** :  
Dans le genre du *majnun aqil*, le personnage du fou est présenté dès l'ouverture du récit — c'est lui le sujet de l'anecdote. Dans les textes négatifs (hadith, histoire, adab général), مجنون peut apparaître de façon incidente dans n'importe quelle position.

L'information positionnelle (**où** مجنون apparaît, pas seulement **si**) ajoute +13 points de précision par rapport à la feature `f00_has_junun` de v80.

**Exemples** :
- ✅ "ذهبنا إلى دير هزقل ننظر إلى **المجانين**..." (position 1/3, positif confirmé)
- ❌ "...قاتل من يجنّ عليه ثم ذكر **مجنونا** في السوق..." (position 3/3, adab)

---

### B02 — `f_poetry_intro` ⭐⭐⭐

**Implémentation** : présence d'une formule de transition vers la poésie

```python
POETRY_INTRO_BIGRAMS = [
    ('أنشأ', 'يقول'), ('ثم', 'أنشأ'), ('وهو', 'يقول'), ...
]
```

**Statistiques** :
- Précision : **0.819–0.957** (selon la variante)
- Rappel : 12.8% (أنشأ seul) / ~18% (toutes variantes)
- LR : 37.7x
- XGB% : 4%

**Justification philologique** :  
La structure narrative canonique du *khabar* de fou sage est :
1. Mise en scène (témoin rencontre le fou)
2. Dialogue (question → réponse paradoxale en prose)
3. **Transition poétique** : "ثم أنشأ يقول..." suivi d'un poème

Cette formule **أنشأ يقول** / **ثم أنشأ** marque le changement de registre prose→vers qui clôt la plupart des anecdotes. Elle est quasi absente des textes négatifs car le hadith et l'histoire n'ont pas cette structure.

---

### B03 — `f_date_scene` ⭐⭐

**Implémentation** : détection des marqueurs temporels de mise en scène

```python
DATE_SCENE = {'ذات يوم', 'ذات ليلة', 'في يوم من الأيام', ...}
```

**Statistiques** :
- Précision : **0.706**
- Rappel : 7.8%
- LR : 19.9x

**Justification** :  
"ذات يوم" (un jour) est une formule de *khabar* qui introduit une scène sans verbe de déplacement. Elle complète les `SCENE_INTRO_VERBS` de v80 (مررت, دخلت...) pour les cas où le témoin est déjà en situation, pas en mouvement.

---

### B04 — `f_love_field` ⭐⭐⭐

**Implémentation** : champ sémantique de l'amour et du désir

```python
LOVE_FIELD = {'حب','عشق','هوى','وجد','غرام','ليلى','هيام','محبة','ولع',...}
```

**Statistiques** :
- Précision : 0.238 (modeste)
- Rappel : **22.2%** — couvre plus d'un cinquième du corpus positif
- LR : 2.6x
- **XGB% : 23%** — validation externe forte

**Justification philologique** :  
Le *junūn al-ʿishq* (folie d'amour) est le **sous-genre le plus fréquent** du majnun aqil. Le fou amoureux (Majnun Layla, Ibn Abi ʿAmmar...) est un archétype distinct du fou mystique. Le champ حب/عشق/هوى/وجد est peu discriminant seul (prec=0.24) mais sa forte co-présence avec les autres features B permet de distinguer le sous-genre.

**Attention** : ne pas utiliser seul comme feature — faible précision. Utile en combinaison.

---

### B05 — `f_crowd_presence` ⭐⭐

**Implémentation** : présence d'une foule / assemblée de témoins

```python
CROWD_FIELD = {'ناس','الناس','قوم','القوم','جمع','جماعة','خلق','عامة',...}
```

**Statistiques** :
- Précision : 0.114 (très faible seule)
- Rappel : 24.1%
- **XGB% : 33%** — la plus haute validation externe de toutes les features

**Justification philologique** :  
Dans le récit de majnun aqil, le fou sage est un personnage public — il performe sa sagesse devant une foule qui valide (ou refuse de valider) sa parole. La foule sert de caisse de résonance. 33% des positifs XGB contiennent ناس/قوم, ce qui confirme la généralisation cross-corpus.

**Attention** : ناس/قوم apparaissent aussi dans tous les genres. Feature utile uniquement en combinaison.

---

### B06 — `f_children_scene` ⭐⭐

**Implémentation** : enfants comme témoins / poursuivants du fou

```python
CHILDREN_FIELD = {'صبيان','الصبيان','غلمان','الغلمان','صبي','أطفال',...}
```

**Statistiques** :
- Précision : **0.786**
- Rappel : 4.8%
- LR : 30.4x
- XGB% : 2%

**Justification philologique** :  
Topos narratif classique : le fou est suivi dans les rues par des enfants qui se moquent de lui (يلعبون به, يضحكون عليه). Ce contraste enfants-qui-moquent / sage-qui-parle est une mise en scène typique du genre. Très peu de textes négatifs mentionnent الصبيان dans ce contexte.

---

### B07 — `f_moussous` ⭐

**Implémentation** : détection du type "fou obsessionnel"

```python
{'موسوس','الموسوس','مموسوس','المموسوس','موسوسا'}
```

**Statistiques** :
- PMI : 2.976
- Précision : **0.846**
- Rappel : 2.4%

**Justification philologique** :  
*Al-muwaswis* (الموسوس) désigne celui qui est atteint de *waswās* — les "voix intérieures" ou pensées obsessionnelles. C'est un type de folie distinct de *junūn* (possession/dérèglement), issu de la terminologie médicale médiévale (Ibn Sīnā, Rāzī). Non couvert par les formes JUNUN_TERMS de v80.

---

### B08 — `f_chained_fool` ⭐

**Implémentation** : fou physiquement contraint / enchaîné

```python
{'مقيد','المقيد','مقيدا','مقيده','مقيدون','مقيدين'}
```

**Statistiques** :
- PMI : 2.917
- Précision : **0.812**
- Rappel : 2.8%

**Justification philologique** :  
Le fou *enchaîné* (maqīd) est une figure récurrente : son corps est contraint mais son esprit libre. Ce paradoxe physique/mental est au cœur du genre. Les textes qui mentionnent مقيد dans un contexte de folie sont presque toujours des récits de majnun aqil.

---

### B09 — `f_bite_behavior` ⭐⭐

**Implémentation** : co-occurrence de عض dans une fenêtre ±7 autour de مجنون

```python
_window_cooc(text, JUNUN_ALL, {'عض','يعض','فعض','عضه',...}, window=7)
```

**Statistiques** :
- Précision : **0.893** (en co-occurrence)
- Rappel : 5.4%
- LR : **69.2x**

**Justification philologique** :  
"عضّ على يديه" (mordre ses mains) est un geste de douleur/désespoir dans la littérature arabe classique. "عضّ القوم" peut signifier agresser. Dans le voisinage direct de مجنون, ce comportement physique est hautement discriminant — il ne s'explique que dans le contexte du fou en crise. LR=69x en fait l'un des signaux les plus forts.

---

### B10 — `f_dhahib_aql` ⭐

**Implémentation** : syntagme "ذاهب العقل" / "ذهب عقله" (fenêtre ±4)

```python
_window_cooc(text, {'ذاهب','ذاهبا','ذهب'}, {'عقله','عقلها','عقل',...}, window=4)
```

**Statistiques** :
- PMI : 2.480
- Précision : **0.600**
- Rappel : 2.6%

**Justification** :  
Paraphrase non-lexicale de la folie : "dont la raison est partie" sans utiliser مجنون. Couvre les cas où l'auteur décrit la folie narrativement plutôt que nominalement.

---

## Intégration avec v80

Les features BoW v0 sont **complémentaires** aux features v80 :

| Dimension | v80 | BoW v0 |
|-----------|-----|--------|
| Junun terms | liste fixe (20 formes) | position dans texte (B01) |
| Scène | verbes de mouvement (E1-E2) | marqueur temporel (B03) |
| Dialogue | densité قال, 1ère personne | co-occurrence عض+مجنون (B09) |
| Poésie | absent | formule transition (B02) |
| Sous-genres | absent | amour (B04), enchaîné (B08), موسوس (B07) |
| Faux positifs | f12_junun_positive | — |

**Fusion recommandée** :
```python
from src.uqala_nlp.features.bow_features_v0 import extract_bow_features_v0
from pipelines.level1_interpretable.p1_4_logistic_regression_v80 import extract_all_features_27

v80_feats = list(extract_all_features_27(text).values())     # 27 features
bow_feats  = list(extract_bow_features_v0(text).values())    # 10 features
combined   = v80_feats + bow_feats                           # 37 features total
```

---

## Roadmap

- **v0** (actuel) : 10 features, string matching, pas de dépendance CAMeL
- **v1** : intégration des racines CAMeL (ج.ن.ن density via MLE) pour B01 amélioré
- **v2** : features positionnelles fines (poésie en dernier tiers, scène en premier tiers)
- **v3** : features de co-occurrence CAMeL (lemmes dans fenêtre) pour réduire faux positifs morphologiques

---

## Utilisation rapide

```bash
# Test de validation
python src/uqala_nlp/features/bow_features_v0.py

# Dans un script d'analyse
python -c "
from src.uqala_nlp.features.bow_features_v0 import extract_bow_features_v0
text = 'ذهبنا ذات يوم إلى المجانين فرأينا فتى منهم يعقل ثم أنشأ يقول...'
feats = extract_bow_features_v0(text)
for k, v in feats.items():
    print(f'{k}: {v}')
"
```
