"""
deep_feature_exploration.py
============================
Exploration approfondie de features discriminantes pour le genre majnun aqil.

Axes d'analyse :
  1. Champs sémantiques  — groupes de lemmes par thème narratif
  2. Co-occurrence       — voisinage (fenêtre ±5) des termes junun
  3. Position dans texte — début / milieu / fin (features narratives)
  4. PMI                 — Pointwise Mutual Information (plus robuste que LR)
  5. Trigrammes          — séquences de 3 lemmes
  6. Contrastes          — termes fortement négatifs (signal inverse)

Correction bug CAMeL : رأى → راوَنْد est un artefact; corrigé par table.
"""

import json, re, math, sys
from collections import defaultdict, Counter
import pathlib

sys.stdout.reconfigure(encoding='utf-8')

BASE     = pathlib.Path(__file__).resolve().parent.parent
CACHE    = BASE / "results" / "camel_lemma_cache.json"
OUT_JSON = BASE / "results" / "deep_feature_candidates.json"

# ── Correction des lemmes erronés (bug CAMeL sur verbes faibles) ───────────────
LEMMA_FIXES = {
    'راوَنْد': 'رَأَى',   # رأى, رأيت, فرأيت → راوند (bug)
    'وَرَى':   'رَأَى',   # يرى → ورى (bug)
    'NTWS':    None,       # token non analysé → ignorer
}

def fix_lemma(lem: str) -> str | None:
    fixed = LEMMA_FIXES.get(lem, lem)
    if fixed is None:
        return None
    # Supprimer diacritiques pour comparaison uniforme
    return re.sub(r'[\u064B-\u065F\u0670]', '', fixed).strip() or None

# ── Champs sémantiques à tester ────────────────────────────────────────────────
# Chaque champ = liste de lemmes (sans diacritiques)
SEMANTIC_FIELDS = {
    # — Genre majnun aqil —
    'F_LOVE':        ['حب','عشق','هوى','وجد','غرام','محبه','صباب','ليلى','هيام','عاشق'],
    'F_WANDER':      ['هام','طاف','جال','هرب','ضل','تاه','سار','مشى','تجول'],
    'F_APPEAR_FOOL': ['اشعث','رث','عار','نحيف','هزيل','شعث','خلق','بال'],
    'F_POETRY':      ['انشا','انشد','شعر','بيت','قصيده','ابيات','قافيه','ينشد'],
    'F_CHILDREN':    ['صبي','ولد','غلام','اطفال','صبيان','غلمان'],
    'F_CROWD':       ['ناس','قوم','جمع','جماعه','خلق'],
    'F_EMOTION_POS': ['بكى','تعجب','عجب','دهش','تفكر','تامل'],
    'F_WISDOM':      ['حكمه','عقل','فهم','ذكاء','لطيف','فطن','كلام','حكيم'],
    'F_ASCETIC':     ['زهد','توكل','دنيا','اخره','فناء','بلاء','تقوى','خوف','رجاء'],
    'F_QUESTION':    ['ما','كيف','لم','من','هل'],   # interrogatifs
    'F_GRIEF':       ['حزن','بكاء','دمع','كمد','نحيب','حسره','الم'],
    'F_PARADOX':     ['لكن','مع','بينما','رغم','حين','عند','اذ'],
    'F_RUIN_SPACE':  ['خراب','قبر','طريق','باب','زاويه','ناحيه','منعزل'],
    'F_GIFT':        ['اعطى','اهدى','بذل','كرم','عطا','هديه'],
    # — Signal négatif —
    'F_AUTHORITY':   ['خليفه','ملك','وزير','سلطان','امير','حاكم','قاضي'],
    'F_FORMAL_REL':  ['حدثنا','اخبرنا','روى','عن','اسناد'],
}

# ── Chargement cache ───────────────────────────────────────────────────────────
print("Chargement du cache CAMeL...")
with open(CACHE, encoding='utf-8') as f:
    cache = json.load(f)

train = cache['train']
xgb   = cache['xgb']

# Appliquer correction des lemmes
def get_fixed_lemmas(record):
    return [fl for l in record.get('lemmas_content', [])
            if (fl := fix_lemma(l)) is not None]

def get_fixed_roots(record):
    return [r for r in record.get('roots_content', [])
            if r and '#' not in r and len(r) >= 3 and r != 'NTWS']

pos_records = [r for r in train if r['label'] == 1]
neg_records = [r for r in train if r['label'] == 0]
N_POS, N_NEG = len(pos_records), len(neg_records)
N_XGB = len(xgb)
print(f"  {N_POS} pos | {N_NEG} neg | {N_XGB} XGB")

# ── ANALYSE 1 : Champs sémantiques ────────────────────────────────────────────
print("\n══ ANALYSE 1 : Champs sémantiques ══")

field_results = []
for field_name, field_terms in SEMANTIC_FIELDS.items():
    field_set = set(field_terms)
    n_pos = sum(1 for r in pos_records if field_set & set(get_fixed_lemmas(r)))
    n_neg = sum(1 for r in neg_records if field_set & set(get_fixed_lemmas(r)))
    n_xgb = sum(1 for r in xgb       if field_set & set(get_fixed_lemmas(r)))

    prec = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
    rec  = n_pos / N_POS
    lr   = rec / (n_neg / N_NEG) if n_neg > 0 else rec * N_NEG
    xgb_pct = n_xgb / N_XGB * 100

    field_results.append({
        'field': field_name, 'precision': round(prec,3),
        'recall': round(rec,3), 'lr_ratio': round(lr,2),
        'n_pos': n_pos, 'xgb_pct': round(xgb_pct,1),
        'terms': field_terms[:5]
    })

field_results.sort(key=lambda x: x['precision']*x['recall'], reverse=True)
print(f"\n  {'Champ':<20} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'n_pos':>6} {'XGB%':>6}  Exemple de termes")
print(f"  {'─'*85}")
for f in field_results:
    xgb_s = f"{f['xgb_pct']:.0f}%" if f['xgb_pct'] > 0 else "—"
    sign = "↑" if f['field'] not in ('F_AUTHORITY','F_FORMAL_REL') else "↓"
    print(f"  {sign} {f['field']:<18} {f['precision']:>6.3f} {f['recall']:>6.3f} "
          f"{f['lr_ratio']:>7.1f} {f['n_pos']:>6} {xgb_s:>6}  {', '.join(f['terms'][:4])}")

# ── ANALYSE 2 : Co-occurrence (fenêtre ±5 autour des termes junun) ─────────────
print("\n══ ANALYSE 2 : Co-occurrence ±5 tokens autour des termes junun ══")

JUNUN_LEMMAS = {'مجنون','بهلول','سعدون','معتوه','هائم','ممسوس','مدله'}

def get_junun_neighbors(lemmas: list, window=5) -> set:
    """Retourne les lemmes dans une fenêtre ±window autour des termes junun."""
    neighbors = set()
    for i, l in enumerate(lemmas):
        if l in JUNUN_LEMMAS:
            start = max(0, i - window)
            end   = min(len(lemmas), i + window + 1)
            neighbors.update(lemmas[start:end])
    neighbors -= JUNUN_LEMMAS
    return neighbors

pos_cooc = defaultdict(set)
neg_cooc = defaultdict(set)
xgb_cooc = defaultdict(set)

for i, r in enumerate(pos_records):
    lemmas = get_fixed_lemmas(r)
    for nb in get_junun_neighbors(lemmas):
        pos_cooc[nb].add(i)

for i, r in enumerate(neg_records):
    lemmas = get_fixed_lemmas(r)
    for nb in get_junun_neighbors(lemmas):
        neg_cooc[nb].add(i)

for i, r in enumerate(xgb):
    lemmas = get_fixed_lemmas(r)
    for nb in get_junun_neighbors(lemmas):
        xgb_cooc[nb].add(i)

# Stats sur les co-occurrences
cooc_stats = []
for term, pos_set in pos_cooc.items():
    n_pos = len(pos_set)
    n_neg = len(neg_cooc.get(term, set()))
    n_xgb = len(xgb_cooc.get(term, set()))
    if n_pos < 4:
        continue
    prec = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
    rec  = n_pos / N_POS
    lr   = rec / (n_neg / N_NEG) if n_neg > 0 else rec * N_NEG
    if prec < 0.40 or rec < 0.04:
        continue
    cooc_stats.append({
        'term': term, 'precision': round(prec,3), 'recall': round(rec,3),
        'lr_ratio': round(lr,2), 'n_pos': n_pos, 'n_neg': n_neg,
        'xgb_pct': round(n_xgb/N_XGB*100,1),
        'certainty': round(prec*rec,5)
    })

cooc_stats.sort(key=lambda x: x['certainty'], reverse=True)
print(f"\n  Termes dans le voisinage de مجنون/بهلول... (prec≥0.40, rec≥0.04)")
print(f"\n  {'Terme':<22} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'n_pos':>6} {'XGB%':>6}")
print(f"  {'─'*62}")
for c in cooc_stats[:30]:
    xgb_s = f"{c['xgb_pct']:.0f}%" if c['xgb_pct'] > 0 else "—"
    print(f"  ★ {c['term']:<20} {c['precision']:>6.3f} {c['recall']:>6.3f} "
          f"{c['lr_ratio']:>7.1f} {c['n_pos']:>6} {xgb_s:>6}")

# ── ANALYSE 3 : Position dans le texte ───────────────────────────────────────
print("\n══ ANALYSE 3 : Features positionnelles ══")

KEY_TERMS_POS = {
    'scene_intro':  {'مر','دخل','رأى','خرج','جاء','أتى'},
    'junun':        JUNUN_LEMMAS,
    'poetry_intro': {'أنشأ','أنشد'},
    'question_ya':  {'يا'},
    'wisdom':       {'حكمه','عقل','كلام'},
    'grief':        {'بكى','حزن','دمع'},
}

def position_feature(lemmas, term_set, zone='first'):
    """zone: 'first' = 1er tiers, 'last' = dernier tiers."""
    n = len(lemmas)
    if n == 0:
        return False
    if zone == 'first':
        chunk = set(lemmas[:n//3])
    else:
        chunk = set(lemmas[2*n//3:])
    return bool(chunk & term_set)

pos_features = {}
for fname, fterms in KEY_TERMS_POS.items():
    for zone in ('first', 'last'):
        key = f"{fname}_{zone}"
        n_pos = sum(1 for r in pos_records if position_feature(get_fixed_lemmas(r), fterms, zone))
        n_neg = sum(1 for r in neg_records if position_feature(get_fixed_lemmas(r), fterms, zone))
        n_xgb = sum(1 for r in xgb       if position_feature(get_fixed_lemmas(r), fterms, zone))
        prec = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
        rec  = n_pos / N_POS
        lr   = rec / (n_neg / N_NEG) if n_neg > 0 else rec * N_NEG
        pos_features[key] = {
            'precision': round(prec,3), 'recall': round(rec,3),
            'lr_ratio': round(lr,2), 'n_pos': n_pos,
            'xgb_pct': round(n_xgb/N_XGB*100,1)
        }

print(f"\n  {'Feature positionnelle':<35} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'XGB%':>6}")
print(f"  {'─'*62}")
for k, v in sorted(pos_features.items(), key=lambda x: x[1]['precision']*x[1]['recall'], reverse=True):
    xgb_s = f"{v['xgb_pct']:.0f}%" if v['xgb_pct'] > 0 else "—"
    print(f"  {k:<35} {v['precision']:>6.3f} {v['recall']:>6.3f} "
          f"{v['lr_ratio']:>7.1f} {xgb_s:>6}")

# ── ANALYSE 4 : PMI (Pointwise Mutual Information) ───────────────────────────
print("\n══ ANALYSE 4 : PMI — termes les plus discriminants ══")

# PMI = log2[ P(term ∩ pos) / (P(term) * P(pos)) ]
N_TOTAL = N_POS + N_NEG
P_POS   = N_POS / N_TOTAL

# Construire index global (pour P(term))
all_term_docs = defaultdict(set)
for i, r in enumerate(train):
    lemmas = set(get_fixed_lemmas(r))
    for l in lemmas:
        all_term_docs[l].add(i)

pmi_stats = []
pos_term_docs = defaultdict(set)
for i, r in enumerate(pos_records):
    for l in set(get_fixed_lemmas(r)):
        pos_term_docs[l].add(i)

for term, pos_set in pos_term_docs.items():
    n_pos_t  = len(pos_set)
    n_total_t = len(all_term_docs[term])
    if n_pos_t < 5 or n_total_t < 8:
        continue

    p_term     = n_total_t / N_TOTAL
    p_term_pos = n_pos_t / N_TOTAL
    p_joint    = p_term_pos  # P(term AND pos) ≈ n_pos_term / N_total

    if p_term > 0 and P_POS > 0:
        pmi = math.log2(p_joint / (p_term * P_POS) + 1e-10)
    else:
        continue

    prec = n_pos_t / n_total_t
    rec  = n_pos_t / N_POS
    n_xgb = len(xgb_cooc.get(term, set()) | {i for i,r in enumerate(xgb) if term in set(get_fixed_lemmas(r))})

    pmi_stats.append({
        'term': term, 'pmi': round(pmi,3),
        'precision': round(prec,3), 'recall': round(rec,3),
        'n_pos': n_pos_t, 'n_total': n_total_t,
        'xgb_pct': round(n_xgb/N_XGB*100,1)
    })

pmi_stats.sort(key=lambda x: x['pmi'], reverse=True)

# Filtrer hors v80
V80 = {'مجنون','بهلول','سعدون','معتوه','هائم','ممسوس','مدله','جنون',
        'مررت','دخل','رأى','قال','يا','إلهي','رب','مقابر','خراب','مسجد','قبر','سوق'}

print(f"\n  Top PMI — nouvelles features (hors v80)")
print(f"\n  {'Terme':<22} {'PMI':>6} {'Prec':>6} {'Rapp':>6} {'n_pos':>6} {'XGB%':>6}")
print(f"  {'─'*62}")
count = 0
pmi_new = []
for c in pmi_stats:
    if c['term'] in V80 or len(c['term']) < 2:
        continue
    if c['precision'] < 0.30:
        continue
    xgb_s = f"{c['xgb_pct']:.0f}%" if c['xgb_pct'] > 0 else "—"
    print(f"  ★ {c['term']:<20} {c['pmi']:>6.3f} {c['precision']:>6.3f} "
          f"{c['recall']:>6.3f} {c['n_pos']:>6} {xgb_s:>6}")
    pmi_new.append(c)
    count += 1
    if count >= 35:
        break

# ── ANALYSE 5 : Trigrammes de lemmes ─────────────────────────────────────────
print("\n══ ANALYSE 5 : Trigrammes de lemmes ══")

def trigrams(lst):
    return [f"{lst[i]} {lst[i+1]} {lst[i+2]}" for i in range(len(lst)-2)]

pos_tri = defaultdict(set)
neg_tri = defaultdict(set)
xgb_tri = defaultdict(set)

for i, r in enumerate(pos_records):
    toks = get_fixed_lemmas(r)
    for t in set(trigrams(toks)):
        pos_tri[t].add(i)
for i, r in enumerate(neg_records):
    toks = get_fixed_lemmas(r)
    for t in set(trigrams(toks)):
        neg_tri[t].add(i)
for i, r in enumerate(xgb):
    toks = get_fixed_lemmas(r)
    for t in set(trigrams(toks)):
        xgb_tri[t].add(i)

tri_stats = []
for term, pos_set in pos_tri.items():
    n_pos = len(pos_set)
    n_neg = len(neg_tri.get(term, set()))
    if n_pos < 4:
        continue
    prec = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
    rec  = n_pos / N_POS
    if prec < 0.50 or rec < 0.03:
        continue
    n_xgb = len(xgb_tri.get(term, set()))
    tri_stats.append({
        'term': term, 'precision': round(prec,3), 'recall': round(rec,3),
        'lr_ratio': round(rec/(n_neg/N_NEG) if n_neg>0 else rec*N_NEG, 2),
        'n_pos': n_pos, 'xgb_pct': round(n_xgb/N_XGB*100,1)
    })

tri_stats.sort(key=lambda x: x['precision']*x['recall'], reverse=True)
print(f"\n  {'Trigramme':<40} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'XGB%':>6}")
print(f"  {'─'*70}")
for c in tri_stats[:30]:
    xgb_s = f"{c['xgb_pct']:.0f}%" if c['xgb_pct'] > 0 else "—"
    print(f"  ★ {c['term']:<38} {c['precision']:>6.3f} {c['recall']:>6.3f} "
          f"{c['lr_ratio']:>7.1f} {xgb_s:>6}")

# ── ANALYSE 6 : Signaux négatifs forts ───────────────────────────────────────
print("\n══ ANALYSE 6 : Signaux NÉGATIFS (termes qui excluent le genre) ══")

neg_signals = []
neg_term_docs = defaultdict(set)
for i, r in enumerate(neg_records):
    for l in set(get_fixed_lemmas(r)):
        neg_term_docs[l].add(i)

for term, neg_set in neg_term_docs.items():
    n_neg_t = len(neg_set)
    n_pos_t = len(pos_term_docs.get(term, set()))
    if n_neg_t < 20 or n_pos_t > 10:
        continue
    neg_prec = n_neg_t / (n_neg_t + n_pos_t)
    neg_rec  = n_neg_t / N_NEG
    if neg_prec < 0.85 and neg_rec < 0.05:
        continue
    if neg_prec < 0.80:
        continue
    neg_signals.append({
        'term': term, 'neg_precision': round(neg_prec,3),
        'neg_recall': round(neg_rec,3), 'n_neg': n_neg_t, 'n_pos': n_pos_t
    })

neg_signals.sort(key=lambda x: x['neg_precision']*x['neg_recall'], reverse=True)
print(f"\n  {'Terme':<25} {'Prec(neg)':>10} {'Rapp(neg)':>10} {'n_neg':>7} {'n_pos':>7}")
print(f"  {'─'*62}")
for c in neg_signals[:25]:
    print(f"  ↓ {c['term']:<23} {c['neg_precision']:>10.3f} {c['neg_recall']:>10.3f} "
          f"{c['n_neg']:>7} {c['n_pos']:>7}")

# ── Synthèse finale ───────────────────────────────────────────────────────────
print(f"\n{'═'*75}")
print("  SYNTHÈSE — Features recommandées pour v81")
print(f"{'═'*75}")

best = []

# Meilleurs champs sémantiques
top_fields = [f for f in field_results
              if f['precision'] >= 0.40 and f['recall'] >= 0.06
              and f['field'] not in ('F_AUTHORITY','F_FORMAL_REL')]
for f in top_fields[:5]:
    best.append({'source':'field', 'name': f['field'],
                 'prec': f['precision'], 'rec': f['recall'],
                 'lr': f['lr_ratio'], 'xgb': f['xgb_pct'],
                 'desc': ', '.join(f['terms'][:4])})

# Meilleures co-occurrences
for c in cooc_stats[:5]:
    if c['precision'] >= 0.50:
        best.append({'source':'cooc', 'name': c['term'],
                     'prec': c['precision'], 'rec': c['recall'],
                     'lr': c['lr_ratio'], 'xgb': c['xgb_pct'], 'desc': 'voisinage junun'})

# Meilleurs trigrammes
for c in tri_stats[:5]:
    if c['precision'] >= 0.70:
        best.append({'source':'trigram', 'name': c['term'],
                     'prec': c['precision'], 'rec': c['recall'],
                     'lr': c['lr_ratio'], 'xgb': c['xgb_pct'], 'desc': 'séquence narrative'})

best.sort(key=lambda x: x['prec']*x['rec'], reverse=True)
print(f"\n  {'Source':<10} {'Feature':<35} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'XGB%':>6}")
print(f"  {'─'*72}")
for b in best:
    xgb_s = f"{b['xgb']:.0f}%" if b['xgb'] > 0 else "—"
    print(f"  {b['source']:<10} {b['name']:<35} {b['prec']:>6.3f} {b['rec']:>6.3f} "
          f"{b['lr']:>7.1f} {xgb_s:>6}")

# Sauvegarde
output = {
    'semantic_fields':    field_results,
    'cooccurrence':       cooc_stats[:50],
    'positional':         pos_features,
    'pmi_new':            pmi_new[:50],
    'trigrams':           tri_stats[:40],
    'negative_signals':   neg_signals[:30],
    'recommended':        best,
}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅  {OUT_JSON}")
