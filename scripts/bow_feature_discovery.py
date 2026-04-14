"""
bow_feature_discovery.py
========================
Découverte de nouvelles features à partir du vocabulaire des POSITIFS.

Stratégie :
  1. Extraire tous les n-grammes (mots + bigrammes) présents dans les positifs
  2. Pour chaque terme, calculer directement sur les corpus bruts :
       - précision   = docs_pos / (docs_pos + docs_neg)
       - rappel      = docs_pos / total_positifs
       - ratio LR    = P(terme|pos) / P(terme|neg)
       - score cert. = précision × rappel
  3. Validation cross-corpus : présence dans les positifs XGB d'Ibn Abd Rabbih
  4. Filtres de certitude :
       - précision ≥ 0.40
       - rappel ≥ 0.05
       - n_pos ≥ 5 documents
       - non redondant avec les features v80 existantes

Pas de classifieur — analyse directe du corpus des positifs.
"""

import json
import re
import sys
import pathlib
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

BASE = pathlib.Path(__file__).resolve().parent.parent

DATASET  = BASE / "data" / "raw" / "dataset_raw.json"
XGB_POS  = BASE / "results" / "0328IbnCabdRabbih" / "xgboost_v80_validation_proper_akhbars.json"
OUT_JSON = BASE / "results" / "bow_feature_candidates.json"

# ── Features v80 déjà existantes ──────────────────────────────────────────────
V80_EXISTING_TERMS = {
    'مجنون','المجنون','مجانين','المجانين','معتوه','المعتوه','هائم','ممسوس',
    'ممرور','مستهتر','جنون','الجنون','مدله','جنونه','جنونها',
    'بهلول','سعدون','عليان','جعيفران','ريحانة','سمنون','لقيط','حيون',
    'مررت','دخلت','فرأيت','لقيت','أتيت','خرجت','وجدت','رأيت',
    'شاهدت','أبصرت',
    'قلت','فقلت','قلنا',
    'يا هذا','يا مجنون','يا بهلول','يا سعدون','يا ذا','يا هرم',
    'إلهي','اللهم','يا رب','يا إلهي',
    'المقابر','خرابات','الخرابات','أزقة','قبر','سوق','مسجد',
}

# ── Normalisation arabe ────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = re.sub(r'PageV\d+P\d+', ' ', text)
    text = re.sub(r'ms\d+', ' ', text)
    text = re.sub(r'#[^\n]*', ' ', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)   # diacritiques
    text = re.sub(r'[أإآ]', 'ا', text)   # alef
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    text = re.sub(r'[^\u0600-\u06FF\s؟]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text: str) -> list:
    return [t for t in text.split() if len(t) >= 2]

def get_ngrams(tokens: list, n: int) -> list:
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ── Chargement ─────────────────────────────────────────────────────────────────
print("Chargement dataset_raw.json...")
with open(DATASET, encoding='utf-8') as f:
    data = json.load(f)

pos_texts_raw = [d['text_ar'] for d in data if d['label'] == 1]
neg_texts_raw = [d['text_ar'] for d in data if d['label'] == 0]
print(f"  Positifs : {len(pos_texts_raw)} | Négatifs : {len(neg_texts_raw)}")

pos_texts = [normalize(t) for t in pos_texts_raw]
neg_texts = [normalize(t) for t in neg_texts_raw]

N_POS = len(pos_texts)
N_NEG = len(neg_texts)

print("Chargement positifs XGB (Ibn Abd Rabbih)...")
with open(XGB_POS, encoding='utf-8') as f:
    xgb_data = json.load(f)
xgb_texts = [normalize(p['text']) for p in xgb_data['positives_detailed']]
N_XGB = len(xgb_texts)
print(f"  {N_XGB} positifs XGB chargés")

# ── Construire l'index des termes (depuis les positifs uniquement) ─────────────
print("\nExtraction du vocabulaire des positifs (uni + bigrammes)...")

# Pour chaque terme : ensemble des doc_ids où il apparaît
pos_term_docs  = defaultdict(set)   # term → {doc_id dans positifs}
neg_term_docs  = defaultdict(set)   # term → {doc_id dans négatifs}
xgb_term_docs  = defaultdict(set)   # term → {doc_id dans XGB positifs}

for i, text in enumerate(pos_texts):
    toks = tokenize(text)
    terms = set(toks) | set(get_ngrams(toks, 2))
    for t in terms:
        pos_term_docs[t].add(i)

for i, text in enumerate(neg_texts):
    toks = tokenize(text)
    terms = set(toks) | set(get_ngrams(toks, 2))
    for t in terms:
        neg_term_docs[t].add(i)

for i, text in enumerate(xgb_texts):
    toks = tokenize(text)
    terms = set(toks) | set(get_ngrams(toks, 2))
    for t in terms:
        xgb_term_docs[t].add(i)

print(f"  {len(pos_term_docs)} termes distincts dans les positifs")

# ── Calcul des statistiques pour chaque terme des positifs ───────────────────
print("Calcul précision / rappel / ratio LR...")

candidates = []

for term, pos_docs in pos_term_docs.items():
    n_pos_docs = len(pos_docs)
    n_neg_docs = len(neg_term_docs.get(term, set()))
    n_xgb_docs = len(xgb_term_docs.get(term, set()))

    if n_pos_docs < 5:
        continue

    prec     = n_pos_docs / (n_pos_docs + n_neg_docs)
    rec      = n_pos_docs / N_POS
    neg_rate = n_neg_docs / N_NEG if n_neg_docs > 0 else 1 / (N_NEG * 2)
    lr_ratio = rec / neg_rate
    certainty = prec * rec
    xgb_pct  = n_xgb_docs / N_XGB * 100

    # Filtre de base
    if prec < 0.30 or rec < 0.04:
        continue

    # Redondance v80
    already_v80 = term in V80_EXISTING_TERMS or any(
        tok in V80_EXISTING_TERMS for tok in term.split()
    )

    candidates.append({
        'term': term,
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'lr_ratio': round(lr_ratio, 2),
        'certainty_score': round(certainty, 5),
        'n_pos_docs': n_pos_docs,
        'n_neg_docs': n_neg_docs,
        'xgb_hits': n_xgb_docs,
        'xgb_pct': round(xgb_pct, 1),
        'ngram_len': len(term.split()),
        'already_in_v80': already_v80,
    })

candidates.sort(key=lambda x: x['certainty_score'], reverse=True)
print(f"  {len(candidates)} termes passent les filtres de base")

# ── Affichage : top 50 tous termes ─────────────────────────────────────────────
print(f"\n{'Terme':<35} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'n_pos':>6} {'XGB%':>6} {'v80?':>5}")
print("─" * 75)
for c in candidates[:50]:
    mark = "✓" if c['already_in_v80'] else "★"
    xgb_s = f"{c['xgb_pct']:.0f}%" if c['xgb_pct'] > 0 else "  —"
    print(f"{mark} {c['term']:<33} {c['precision']:>6.3f} {c['recall']:>6.3f} "
          f"{c['lr_ratio']:>7.1f} {c['n_pos_docs']:>6} {xgb_s:>6}")

# ── Nouvelles features (hors v80) ─────────────────────────────────────────────
new_cands = [c for c in candidates if not c['already_in_v80']]
print(f"\n\n══ NOUVELLES FEATURES (hors v80) — {len(new_cands)} candidats ══\n")
print(f"{'Terme':<35} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'n_pos':>6} {'XGB%':>6}")
print("─" * 70)
for c in new_cands[:50]:
    xgb_s = f"{c['xgb_pct']:.0f}%" if c['xgb_pct'] > 0 else "  —"
    print(f"★ {c['term']:<33} {c['precision']:>6.3f} {c['recall']:>6.3f} "
          f"{c['lr_ratio']:>7.1f} {c['n_pos_docs']:>6} {xgb_s:>6}")

# ── Recommandations strictes (certitude maximale) ────────────────────────────
print("\n\n══ RECOMMANDATIONS FINALES (prec≥0.50, rec≥0.05) ══\n")

strict = [c for c in new_cands if c['precision'] >= 0.50 and c['recall'] >= 0.05]
strict_xgb = [c for c in strict if c['xgb_pct'] > 0]
strict_noxgb = [c for c in strict if c['xgb_pct'] == 0]

print(f"Validés cross-corpus (présents dans XGB Ibn Abd Rabbih) : {len(strict_xgb)}")
for c in strict_xgb:
    print(f"  {c['term']:<33}  prec={c['precision']:.3f}  rec={c['recall']:.3f}"
          f"  LR={c['lr_ratio']:.1f}x  XGB={c['xgb_pct']:.0f}%  n_pos={c['n_pos_docs']}")

print(f"\nSolides (pas dans XGB) : {len(strict_noxgb)}")
for c in strict_noxgb[:15]:
    print(f"  {c['term']:<33}  prec={c['precision']:.3f}  rec={c['recall']:.3f}"
          f"  LR={c['lr_ratio']:.1f}x  n_pos={c['n_pos_docs']}")

# ── Sauvegarde ─────────────────────────────────────────────────────────────────
output = {
    'method': 'direct corpus frequency (no classifier)',
    'n_pos': N_POS,
    'n_neg': N_NEG,
    'n_xgb_positives': N_XGB,
    'total_candidates_filtered': len(candidates),
    'new_candidates': len(new_cands),
    'recommended_cross_corpus': len(strict_xgb),
    'all_new_candidates': new_cands[:80],
    'recommended': strict_xgb,
    'solid_no_xgb': strict_noxgb[:20],
}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Résultats → {OUT_JSON}")
