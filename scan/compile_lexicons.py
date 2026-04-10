"""
compile_lexicons.py
────────────────────
Construit les lexiques actantiels depuis actantial_model.json
et actantial_annotations.json.

Méthode : graines linguistiquement cadrées (ne pas enrichir avec
les matns complets — trop de bruit). LLR appliqué pour pondérer
et valider chaque token.

LLR(t) = log( P(t | canonique) / P(t | corpus_complet) )
  > 0  → surreprésenté dans le motif canonique
  < 0  → sous-représenté (peu discriminant)

Sortie : scan/actantial_lexicons.json

Usage :
  python scan/compile_lexicons.py
"""

import json, re, math, pathlib, sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE   = pathlib.Path(__file__).parent.parent
ANN    = BASE / "approche_actantielle" / "actantial_annotations.json"
MODEL  = BASE / "approche_actantielle" / "actantial_model.json"
AKHBAR = BASE / "corpus" / "akhbar.json"
OUT    = BASE / "scan" / "actantial_lexicons.json"

# ── Utilitaires ───────────────────────────────────────────────────────────────
def tok_ar(text):
    if not text or str(text).lower() in ('null','none',''):
        return []
    return [t for t in re.findall(r'[\u0621-\u064A\u0671-\u06D3]+', text)
            if len(t) >= 2]

def _val(obj, *keys):
    for k in keys:
        if not isinstance(obj, dict): return None
        obj = obj.get(k)
    if obj is None: return None
    v = str(obj).strip().lower()
    return None if v in {'null','none','','absent','absente','n/a'} else v

def llr(c_in, n_in, c_bg, n_bg, alpha=0.5):
    p_in = (c_in + alpha) / (n_in + alpha * 2)
    p_bg = (c_bg + alpha) / (n_bg + alpha * 2)
    return round(math.log(p_in / p_bg), 4)

# ── Chargement ────────────────────────────────────────────────────────────────
print("Chargement des données…")
anns  = [r for r in json.load(open(ANN,   encoding='utf-8')) if '_error' not in r]
model = json.load(open(MODEL, encoding='utf-8'))
akh   = json.load(open(AKHBAR, encoding='utf-8'))['akhbar']

scores_par_num = {int(k): v
    for k, v in model['scores_canonicite']['scores_par_num'].items()}

def get_matn(a):
    segs = a.get('content', {}).get('segments', [])
    return ' '.join(s['text'] for s in segs
                    if s.get('type') != 'isnad' and s.get('text','').strip())

matn_par_num = {a['num']: get_matn(a) for a in akh}
annotated = [{'r': r, 'num': r['_num'],
              'score': scores_par_num.get(r['_num'], 0),
              'matn': matn_par_num.get(r['_num'], '')}
             for r in anns]

canoniques     = [a for a in annotated if a['score'] >= 6]
non_canoniques = [a for a in annotated if a['score'] <= 1]
print(f"  Canoniques (≥6) : {len(canoniques)}  |  Périphériques (≤1) : {len(non_canoniques)}")

# Fréquences sur corpus complet et corpus canonique
all_tok   = Counter(t for a in annotated for t in tok_ar(a['matn']))
canon_tok = Counter(t for a in canoniques for t in tok_ar(a['matn']))
n_all, n_can = sum(all_tok.values()), sum(canon_tok.values())
print(f"  Tokens total : {n_all:,}  |  Tokens canoniques : {n_can:,}\n")

def score_seeds(seeds, min_canon=2, min_llr=-99):
    """
    Calcule LLR pour une liste de tokens-graines.
    Garde ceux avec fréquence canon ≥ min_canon et LLR ≥ min_llr.
    Retourne dict {token: llr} trié par LLR décroissant.
    """
    out = {}
    for tok in set(seeds):
        c_can = canon_tok.get(tok, 0)
        c_all = all_tok.get(tok, 0)
        if c_can < min_canon:
            continue
        s = llr(c_can, n_can, c_all, n_all)
        if s >= min_llr:
            out[tok] = s
    return dict(sorted(out.items(), key=lambda x: -x[1]))

def show(name, lex, top=12):
    print(f"  {name} : {len(lex)} tokens retenus")
    for t, s in list(lex.items())[:top]:
        print(f"    {s:+.3f}  {t}")
    print()

# ── Lexique 1 : L_junun ───────────────────────────────────────────────────────
# Termes désignant explicitement le maǧnūn ou ses synonymes
# Source : termes_arabes_par_fonction du modèle + noms propres de fous connus

# Termes génériques de la folie
L_junun_seeds = {
    # Adjectifs / noms génériques
    'مجنون','المجنون','مجنونا','مجانين','المجانين',
    'معتوه','المعتوه','معتوها','معتوهين',
    'مدله','المدله','مدلها',
    'هائم','الهائم','هائما',
    'ممسوس','الممسوس',
    'ممرور','الممرور',
    'مستهتر','المستهتر',
    # Racine ج-ن-ن
    'جنون','الجنون','جنونه','جنونها','جنوني','جنونا','جنونك',
    'جن','أجن','مجنونة','جانا','جانوا',
    # Racine ع-ق-ل (folie = perte de raison)
    'ذاهب','ذهب','العقل','عقله','عقلها','ذهبعقله',
    # Noms propres canoniques (du modèle)
    'بهلول','بهلولا','سعدون','عليان','جعيفران','ريحانة','سمنون','لقيط',
    'حيون','حيونة','يوسف','مالك','نور',
}

# Enrichir depuis termes_arabes_par_fonction du modèle actantiel
for fn, terms in model['termes_arabes_par_fonction'].items():
    for t_raw, _ in terms:
        for t in tok_ar(t_raw):
            if len(t) >= 3:
                L_junun_seeds.add(t)

L_junun = score_seeds(L_junun_seeds, min_canon=2, min_llr=0.05)
show("L_junun", L_junun)

# ── Lexique 2 : L_dialogue ────────────────────────────────────────────────────
# Marqueurs de structure de dialogue (سؤال وجواب)
# Verbes de parole qui signalent un échange entre deux interlocuteurs
L_dialogue_seeds = {
    # Formules d'échange direct
    'قلت','فقلت','قلنا','قلتم',           # je/nous dis(ons)
    'سألت','فسألت','سألني','سألوه',        # j'ai demandé
    'سئل','يسأل','تسألني',                 # il fut interrogé
    'فقال','وقال','فقالت','وقالت',         # il/elle dit
    'فأجاب','أجاب','فأجابه','فأجابني',     # il répondit
    'فقلت له','قيل له','قيل لي',           # on lui dit
    'فقال له','قال له','قالت له',          # il dit à
}
L_dialogue = score_seeds(L_dialogue_seeds, min_canon=2, min_llr=0.0)
show("L_dialogue", L_dialogue)

# ── Lexique 3 : L_shir ────────────────────────────────────────────────────────
# Marqueurs de citation poétique
L_shir_seeds = {
    'أنشد','أنشأ','أنشدني','أنشدنا','فأنشد','فأنشأ','ينشد','تنشد',
    'الشاعر','شاعر','شعر','شعره','شعرها','شعري',
    'أبيات','بيت','قصيدة','قصيد','أبياتا',
    'وزن','قافية','وافر','طويل','بسيط',   # noms de mètres
}
L_shir = score_seeds(L_shir_seeds, min_canon=2, min_llr=0.0)
show("L_shir", L_shir)

# ── Lexique 4 : L_autorite ────────────────────────────────────────────────────
# Figures d'autorité politique, religieuse ou savante
L_autorite_seeds = {
    # Titres
    'الخليفة','خليفة','أمير','الأمير','أميرالمؤمنين',
    'الوزير','وزير','الوالي','والي','القاضي','قاض',
    'الملك','ملك','السلطان','سلطان','الحاكم',
    # Califes abbasides fréquents dans le corpus
    'الرشيد','هارون','المأمون','المتوكل','المعتصم',
    'المهدي','الهادي','المنصور','المهتدي','المعتضد',
    # Termes de généralisation
    'الإمام','إمام','الأمراء','الخلفاء','الوزراء',
}
# Enrichir depuis termes_arabes des actants 'autorite'
for a in annotated:
    for actant in a['r'].get('actants') or []:
        if _val(actant, 'position_relative') == 'autorite':
            ta = actant.get('terme_arabe') or ''
            for t in tok_ar(ta):
                if len(t) >= 3:
                    L_autorite_seeds.add(t)

L_autorite = score_seeds(L_autorite_seeds, min_canon=2, min_llr=0.0)
show("L_autorite", L_autorite)

# ── Lexique 5 : L_validation_forte ───────────────────────────────────────────
# Réception explicite de la parole : don, larmes, silence, rire
L_val_seeds = {
    # Don / ʿaṭāʾ
    'فأمر','أمرله','أعطاه','فأعطاه','فأعطى','وهب','فوهب',
    'جائزة','الجائزة','عطاء','فأمربه','خلعة',
    # Pleurs
    'فبكى','فبكت','بكى','بكاء','فأبكى','عبرة','دموع','دمعه',
    # Rire / admiration
    'فضحك','ضحك','فأعجبه','أعجبه','فاستحسنه','استحسنه',
    'فاستطرفه','فأطربه','فطرب','سر','يسر','ففرح',
    # Silence / stupeur
    'فسكت','فصمت','فأطرق','تعجب','فتعجب','فاندهش',
    'فلم يرد','فأمسك','فخرس',
}
L_validation_forte = score_seeds(L_val_seeds, min_canon=2, min_llr=0.0)
show("L_validation_forte", L_validation_forte)

# ── Lexique 6 : L_narrateur ───────────────────────────────────────────────────
# Marqueurs de commentaire éditorial du narrateur (al-Nīsābūrī)
L_narr_seeds = {
    # Insertions narratives
    'وكان','فكان','وكانت','وكنت',
    'وهو','وإذا','فإذا','وإذاهو',
    'وقيل','فقيل','يقال','يقول','قيل',
    # Formules d'explication
    'يعني','أي','يريد','والمعنى','ذلك','وذلك','وهذا',
    'لأن','لأنه','لأنها','بمعنى',
    # Renvoi intertextuel
    'وقدذكرنا','ذكرناه','وقدمضى','كماقلنا',
}
L_narrateur = score_seeds(L_narr_seeds, min_canon=2, min_llr=0.0)
show("L_narrateur", L_narrateur)

# ── Lexique 7 : L_renversement ────────────────────────────────────────────────
# Marqueurs lexicaux de renversement ou paradoxe
L_revers_seeds = {
    # Connecteurs adversatifs
    'لكن','لكنه','لكنها','بل','لا بل','وإنما',
    'ولكن','غير','إلا','إلاأن','إلاأنه',
    # Marqueurs de surprise / découverte
    'فإذاهو','فإذابه','وإذاهو','وإذابه',
    'فبان','فتبين','فظهر','فتكشف',
    # Supériorité / retournement
    'أعقل','أحكم','أصوب','أصح','أفضل','أعلم',
    'أكثر','خير','صادق','صدق','حق','أصدق',
    # Erreur révélée
    'أخطأ','كذب','غلط','جهل','لم يعلم',
}
L_renversement = score_seeds(L_revers_seeds, min_canon=2, min_llr=0.0)
show("L_renversement", L_renversement)

# ── Lexique 8 : L_folie_amour ────────────────────────────────────────────────
# Marqueurs de la folie d'amour (signal différenciateur — pas alibi_de_parole)
L_amour_seeds = {
    'الهوى','هواه','هواها','هواي',
    'العشق','عشق','عشقه','عشقها',
    'الحب','حب','حبه','حبها','حبيب','حبيبه','حبيبها',
    'الغرام','غرام','الصبابة','صبابة','الشوق','شوق',
    # Noms féminins canoniques de la lyrique d'amour
    'ليلى','لبنى','سلمى','مي','هند','عزة','بثينة',
    # Maǧnūn Laylā
    'قيس','العامري',
}
L_folie_amour = score_seeds(L_amour_seeds, min_canon=2, min_llr=0.0)
show("L_folie_amour", L_folie_amour)

# ── Lexique 9 : L_mubashara ──────────────────────────────────────────────────
# Formule du témoin oculaire (رأيت مجنوناً)
L_mubashara_seeds = {
    'رأيت','رأى','فرأيت','فرأى',
    'مررت','مررنا','فمررت','مررتبه',
    'لقيت','لقي','فلقيت','لقيني',
    'وجدت','وجد','فوجدت','فوجد',
    'أبصرت','أبصر','رأيته','رأيتها',
    'شهدت','شهد','حضرت','حضر',
}
L_mubashara = score_seeds(L_mubashara_seeds, min_canon=2, min_llr=0.0)
show("L_mubashara", L_mubashara)

# ── Lexique 10 : L_wasf ───────────────────────────────────────────────────────
# Signal NÉGATIF — définition taxonomique (non narratif, wasf)
# Surreprésenté dans les textes périphériques (non-motif)
# On calcule LLR vs fond INVERSÉ : canon_tok → non_can_tok
non_can_tok = Counter(t for a in non_canoniques for t in tok_ar(a['matn']))
n_noncan = sum(non_can_tok.values())

L_wasf_seeds = {
    'ومنها','منها','والفعل','منه','والاسم',
    'يقال','يقال له','يقالله','يسمى','فهو',
    'تقول العرب','ومن أمثالهم','أمثالهم','وقالوا','العرب',
    'ضرب من','ضروب','وهو الذي','وهم الذين',
    'وصفه','تعريفه','معناه','يعني به',
}

def score_seeds_neg(seeds, min_noncan=2):
    """LLR négatif : sur-représentation dans le corpus non canonique."""
    out = {}
    for tok in set(seeds):
        c_nc = non_can_tok.get(tok, 0)
        c_all = all_tok.get(tok, 0)
        if c_nc < min_noncan:
            continue
        s = llr(c_nc, n_noncan, c_all, n_all)
        if s > 0:
            out[tok] = round(s, 4)
    return dict(sorted(out.items(), key=lambda x: -x[1]))

L_wasf = score_seeds_neg(L_wasf_seeds, min_noncan=2)
show("L_wasf (signal négatif = texte définitionnel)", L_wasf)

# ── Sauvegarde ────────────────────────────────────────────────────────────────
lexicons = {
    'meta': {
        'n_canoniques':     len(canoniques),
        'n_non_canoniques': len(non_canoniques),
        'n_zone_grise':     len(annotated) - len(canoniques) - len(non_canoniques),
        'total_tokens_corpus':   n_all,
        'total_tokens_canon':    n_can,
        'total_tokens_noncanon': n_noncan,
        'methode': 'LLR (log-rapport de vraisemblance) sur graines linguistiques cadrées',
    },
    'lexiques': {
        'L_junun':            L_junun,
        'L_dialogue':         L_dialogue,
        'L_shir':             L_shir,
        'L_autorite':         L_autorite,
        'L_validation_forte': L_validation_forte,
        'L_narrateur':        L_narrateur,
        'L_renversement':     L_renversement,
        'L_folie_amour':      L_folie_amour,
        'L_mubashara':        L_mubashara,
        'L_wasf':             L_wasf,
    },
    'scores_par_num': scores_par_num,
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(lexicons, f, ensure_ascii=False, indent=2)

print(f"Lexiques sauvegardés → {OUT}")
total_tok = sum(len(v) for v in lexicons['lexiques'].values())
print(f"  10 lexiques  |  {total_tok} tokens retenus au total")
