"""
bow_features_v0.py
==================
Features de la série B (BoW) — version 0.

Découvertes par analyse statistique directe du corpus de positifs
via BoW + CAMeL Tools (lemmatisation MLE) + exploration multi-axe.
Ces features sont INDÉPENDANTES du pipeline v80 et peuvent être
combinées avec lui en FeatureUnion.

Chaque feature est documentée avec :
  - prec  = P(positif | terme présent)
  - rec   = P(terme | positif)
  - LR    = ratio de vraisemblance (rec_pos / rec_neg)
  - XGB%  = % de positifs XGB d'Ibn Abd Rabbih qui en ont (validation externe)

Usage :
    from src.uqala_nlp.features.bow_features_v0 import extract_bow_features_v0
    features = extract_bow_features_v0(text)
    # → dict ordonné de 10 features (B01–B10)
"""

import re

# ── Normalisation légère (sans altérer alef/hamza pour matching) ──────────────
def _norm(text: str) -> str:
    """Supprime diacritiques et marqueurs OpenITI. Garde l'arabe intact."""
    text = re.sub(r'PageV\d+P\d+', ' ', text)
    text = re.sub(r'ms\d+', ' ', text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)   # tashkeel
    return re.sub(r'\s+', ' ', text).strip()

def _tokens(text: str) -> list:
    return text.split()

def _window_cooc(text: str, anchor_set: set, target_set: set, window: int = 7) -> bool:
    """Vrai si un terme de target_set apparaît dans une fenêtre ±window autour d'un terme anchor."""
    toks = _tokens(text)
    for i, t in enumerate(toks):
        if t in anchor_set:
            start = max(0, i - window)
            end   = min(len(toks), i + window + 1)
            if target_set & set(toks[start:end]):
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# LEXIQUES
# ══════════════════════════════════════════════════════════════════════════════

# Toutes les formes du radical ج.ن.ن
JUNUN_ALL = {
    'مجنون','المجنون','مجنونا','مجنونه','مجنونة','المجنونة',
    'مجانين','المجانين','جنون','الجنون','جنونه','جنونها','جنوني','جنونا',
    'معتوه','المعتوه','معتوها','معتوهة',
    'هائم','الهائم','هائما',
    'ممسوس','ممرور','مستهتر','مدله','المدله',
    'موسوس','المموسوس','الموسوس',    # B07
    'مقيد','المقيد','مقيدا',         # B08
}

# Formules de transition poétique
POETRY_INTRO = {
    'أنشأ','وأنشأ','فأنشأ','ثم',
    'أنشد','وأنشد','فأنشد',
    'ينشد','يقول','يتمثل',
}
POETRY_INTRO_BIGRAMS = [
    ('أنشأ', 'يقول'), ('ثم', 'أنشأ'), ('وهو', 'يقول'),
    ('فأنشأ', 'يقول'), ('وأنشأ', 'يقول'), ('ثم', 'أنشد'),
]

# Champ amour / folie amoureuse
LOVE_FIELD = {
    'حب','الحب','حبه','حبها','حبي','أحبه','أحبها',
    'عشق','العشق','عشقه','عشقها','عاشق','العاشق',
    'هوى','الهوى','هواه','هواها',
    'وجد','الوجد','وجده',
    'غرام','الغرام','غرامه',
    'صبابة','الصبابة',
    'ليلى',                    # Majnun Layla
    'هيام','الهيام',
    'محبة','المحبة','محبته',
    'ولع','الولع','ولهان',
}

# Foule / assemblée de témoins
CROWD_FIELD = {
    'ناس','الناس','للناس',
    'قوم','القوم','قومه',
    'جمع','الجمع','مجمع','جماعة','الجماعة',
    'خلق','الخلق',
    'عامة','العامة',
}

# Enfants comme témoins
CHILDREN_FIELD = {
    'صبيان','الصبيان','للصبيان',
    'غلمان','الغلمان',
    'صبي','الصبي','صبيا',
    'ولدان','الولدان',
    'أطفال','الأطفال',
}

# Comportement physique du fou — mordre (عض) près de مجنون
BITE_TERMS = {
    'عض','يعض','فعض','عضه','عضها','يعضون',
}

# ذاهب العقل — forme explicite de la perte de raison
DHAHIB_AQL = {
    'ذاهب','ذاهبا','ذهب',
}
AQL_TERMS = {
    'عقله','عقلها','عقلي','عقل','العقل',
}

# Scène temporelle
DATE_SCENE = {
    'ذات يوم', 'ذات ليلة',
    'في يوم من الأيام', 'في بعض الأيام', 'في ليلة من الليالي',
    'يوما من الأيام',
}


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION DES FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_bow_features_v0(text: str) -> dict:
    """
    Extrait les 10 features BoW v0 d'un akhbar arabe.

    Paramètre :
        text : str — texte brut (avec ou sans isnad, OpenITI ou propre)

    Retourne :
        dict ordonné — clés B01..B10
        Valeurs : float [0.0, 1.0] ou int {0, 1}
    """
    t = _norm(text)
    toks = _tokens(t)
    n = len(toks)
    if n == 0:
        return {k: 0.0 for k in _FEATURE_KEYS}

    features = {}

    # ── B01 : مجنون dans le premier tiers ─────────────────────────────────────
    # prec=0.867  rec=39.6%  LR=53.9x
    # Quand l'identification du fou intervient tôt, le texte est presque
    # certainement un récit de majnun aqil (personnage nommé dès l'intro).
    first_third = set(toks[:n // 3 + 1])
    features['B01_junun_first_third'] = float(bool(JUNUN_ALL & first_third))

    # ── B02 : Formule de transition poétique (أنشأ يقول) ──────────────────────
    # prec=0.819-0.957  rec=12.8%  LR=37.7x
    # Le fou RÉPOND en prose puis SE MET à réciter de la poésie.
    # Ce changement de registre (prose → vers) est un marqueur structurel du genre.
    has_poetry_word = bool(POETRY_INTRO & set(toks))
    has_poetry_bigram = any(
        a in t and b in t for a, b in POETRY_INTRO_BIGRAMS
    )
    features['B02_poetry_intro'] = float(has_poetry_word or has_poetry_bigram)

    # ── B03 : Marqueur temporel de scène (ذات يوم) ────────────────────────────
    # prec=0.706  rec=7.8%  LR=19.9x
    # Formule d'entrée en scène temporelle sans verbe de mouvement.
    # Complète les SCENE_INTRO_VERBS de v80 qui nécessitent un déplacement.
    features['B03_date_scene'] = float(any(s in t for s in DATE_SCENE))

    # ── B04 : Champ sémantique amour / folie amoureuse ────────────────────────
    # prec=0.238  rec=22.2%  XGB=23%
    # Le sous-genre junūn al-ʿishq est la forme la plus fréquente du majnun aqil.
    # حب/عشق/هوى/وجد/ليلى couvrent 22% des positifs et 23% des XGB externes.
    features['B04_love_field'] = float(bool(LOVE_FIELD & set(toks)))

    # ── B05 : Présence d'une foule / assemblée ────────────────────────────────
    # prec=0.114  rec=24.1%  XGB=33%
    # Le plus haut XGB% de toutes les analyses. La foule valide la performance
    # du fou sage (témoins = public du paradoxe).
    features['B05_crowd_presence'] = float(bool(CROWD_FIELD & set(toks)))

    # ── B06 : Enfants comme témoins ───────────────────────────────────────────
    # prec=0.786  rec=4.8%  LR=30.4x
    # Topos classique : le fou est suivi et moqué par des enfants dans les rues,
    # ce qui contraste avec la sagesse de ses propos.
    features['B06_children_scene'] = float(bool(CHILDREN_FIELD & set(toks)))

    # ── B07 : موسوس — type de folie obsessionnelle ────────────────────────────
    # PMI=2.976  prec=0.846  rec=2.4%
    # Type distinct de مجنون : obsédé, qui entend des voix (waswās).
    # Non couvert par v80 — nomenclature médicale médiévale.
    features['B07_moussous'] = float(any(
        tok in {'موسوس','الموسوس','مموسوس','المموسوس','موسوسا'} for tok in toks
    ))

    # ── B08 : مقيد — fou enchaîné / état physique de réclusion ───────────────
    # PMI=2.917  prec=0.812  rec=2.8%
    # Le fou interné, physiquement contraint. Marque l'aspect institutionnel
    # de la folie (asile, chaînes) qui contraste avec la liberté de sa parole.
    features['B08_chained_fool'] = float(any(
        tok in {'مقيد','المقيد','مقيدا','مقيده','مقيدون','مقيدين'} for tok in toks
    ))

    # ── B09 : عض — mordre dans le voisinage de مجنون ─────────────────────────
    # prec=0.893  rec=5.4%  LR=69.2x (en fenêtre ±7)
    # Comportement physique typique : le fou mord sa main de douleur
    # (عضّ على يديه) ou agresse. Signal très fort quand co-présent avec مجنون.
    features['B09_bite_behavior'] = float(
        _window_cooc(t, JUNUN_ALL, BITE_TERMS, window=7)
    )

    # ── B10 : ذاهب + عقل — expression explicite de la perte de raison ─────────
    # PMI=2.480  prec=0.600  rec=2.6%
    # Syntagme "ذاهب العقل" ou "ذهب عقله" = paraphrase de la folie
    # sans utiliser مجنون. Couvre des cas non capturés par f11_junun_morpho.
    features['B10_dhahib_aql'] = float(
        _window_cooc(t, DHAHIB_AQL, AQL_TERMS, window=4)
    )

    return features


# ── Clés dans l'ordre canonique ───────────────────────────────────────────────
_FEATURE_KEYS = [
    'B01_junun_first_third',
    'B02_poetry_intro',
    'B03_date_scene',
    'B04_love_field',
    'B05_crowd_presence',
    'B06_children_scene',
    'B07_moussous',
    'B08_chained_fool',
    'B09_bite_behavior',
    'B10_dhahib_aql',
]

def feature_names() -> list:
    """Retourne les noms des features dans l'ordre canonique."""
    return list(_FEATURE_KEYS)


# ── Validation rapide ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    SAMPLES = [
        # Vrai positif — majnun aqil (khabar 3710, Ibn Abd Rabbih)
        ("VRAI",
         "ذهبنا إلى دير هزقل ننظر إلى المجانين فإذا المجانين كلهم قد رأونا "
         "ونظرنا إلى فتى منهم قد غسل ثوبه ونظفه وجلس ناحية عنهم "
         "فقلنا إن كان ففهذا فوقفنا به فسلمنا عليه فلم يرد السلام "
         "فقلنا له ما تجد فقال الله يعلم أنني كمد لا أستطيع أبث ما أجد"),
        # Vrai positif — folie amoureuse
        ("VRAI_AMOUR",
         "ذات يوم مررت برجل في السوق وكان يعشق فتاة ويهيم بها حبا "
         "فسألته عن حاله فأنشأ يقول شعرا في وصف عشقه وحبه وهواه"),
    ]

    print(f"\n{'─'*60}")
    print(f"  {'Feature':<30} {'VRAI':>8} {'VRAI_AMOUR':>12}")
    print(f"{'─'*60}")

    results = {label: extract_bow_features_v0(text) for label, text in SAMPLES}
    for key in _FEATURE_KEYS:
        vals = [f"{results[label][key]:.2f}" for label, _ in SAMPLES]
        print(f"  {key:<30} {vals[0]:>8} {vals[1]:>12}")
    print(f"{'─'*60}")
