"""
bow_camel_feature_discovery.py
================================
Découverte de features BoW avec lemmatisation CAMeL Tools.

Workflow :
  1. Lemmatisation MLE (contexte) → lemme, racine, POS par token
  2. Filtrage POS : garder NOUN, VERB, ADJ, PROPN seulement
  3. Trois représentations BoW :
       A. Lemmes  : unigrammes + bigrammes (principale)
       B. Racines : unigrammes (généralisation morphologique)
       C. Lemme+POS bigrammes : capture les séquences syntaxiques
  4. Stats de certitude : précision × rappel, ratio LR
  5. Validation cross-corpus : présence dans les positifs XGB Ibn Abd Rabbih
  6. Cache JSON pour relance rapide sans re-lemmatiser

Usage :
  python bow_camel_feature_discovery.py           # traitement complet
  python bow_camel_feature_discovery.py --cached  # utiliser le cache existant
"""

import json
import re
import sys
import time
import pathlib
import argparse
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

BASE      = pathlib.Path(__file__).resolve().parent.parent
DATASET   = BASE / "data" / "raw" / "dataset_raw.json"
XGB_POS   = BASE / "results" / "0328IbnCabdRabbih" / "xgboost_v80_validation_proper_akhbars.json"
CACHE     = BASE / "results" / "camel_lemma_cache.json"
OUT_JSON  = BASE / "results" / "bow_camel_feature_candidates.json"

# POS à garder (contenu lexical uniquement)
CONTENT_POS = {'noun', 'noun_prop', 'verb', 'adj', 'noun_quant', 'noun_num'}
CONTENT_UD  = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'}

# Features v80 déjà existantes (termes bruts + lemmes attendus)
V80_EXISTING = {
    # Junun
    'مجنون', 'مَجْنُون', 'مجانين', 'معتوه', 'مَعْتُوه', 'هائم', 'ممسوس',
    'ممرور', 'مستهتر', 'جنون', 'جُنُون', 'مدله', 'مُدَلَّه',
    # Famous fools
    'بهلول', 'بَهْلُول', 'سعدون', 'سَعْدُون', 'عليان', 'جعيفران', 'ريحانة', 'سمنون',
    # Scene verbs (lemmes)
    'مَرّ', 'دَخَل', 'رَأَى', 'لَقِي', 'أَتَى', 'خَرَج', 'وَجَد',
    # Dialogue
    'قال', 'قُلْت',
    # Sacred spaces
    'مَقْبَرَة', 'خَرابَة', 'مَسْجِد', 'قَبْر', 'سُوق',
    # Divine
    'إِلٰه', 'رَبّ',
}

# ── Nettoyage OpenITI (sans normalisation alef — CAMeL en a besoin) ────────────
def clean_openiti(text: str) -> str:
    text = re.sub(r'PageV\d+P\d+', ' ', text)
    text = re.sub(r'ms\d+', ' ', text)
    text = re.sub(r'#[^\n]*', ' ', text)
    text = re.sub(r'\|', ' ', text)
    # Diacritiques seulement
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text: str) -> list:
    """Tokens arabes ≥ 2 caractères."""
    return [t for t in text.split() if len(t) >= 2]

# ── Lemmatisation d'un texte complet ──────────────────────────────────────────
def lemmatize_text(text: str, mle) -> dict:
    """
    Retourne un dict avec :
      lemmas_content   : liste des lemmes (tokens de contenu seulement)
      roots_content    : liste des racines (tokens de contenu seulement)
      lempos_content   : liste de "lemme_POS" (tokens de contenu)
      lemmas_all       : tous les lemmes (y compris mots grammaticaux)
    """
    tokens = tokenize(text)
    if not tokens:
        return {'lemmas_content': [], 'roots_content': [],
                'lempos_content': [], 'lemmas_all': []}

    try:
        analyses = mle.disambiguate(tokens)
    except Exception:
        return {'lemmas_content': [], 'roots_content': [],
                'lempos_content': [], 'lemmas_all': []}

    lemmas_content = []
    roots_content  = []
    lempos_content = []
    lemmas_all     = []

    for ana in analyses:
        if not ana.analyses:
            continue
        top  = ana.analyses[0].analysis
        lex  = top.get('lex')   or ''
        pos  = top.get('pos')   or ''
        ud   = top.get('ud')    or ''
        root = top.get('root')  or ''

        # Lemme : supprimer diacritiques pour comparaison
        lex_bare = re.sub(r'[\u064B-\u065F\u0670]', '', lex).strip()

        if lex_bare:
            lemmas_all.append(lex_bare)

        # Filtrer sur POS contenu
        # ud peut être composé : "ADP+NOUN" → extraire la partie principale
        ud_parts = set(ud.split('+'))
        pos_ok = (pos in CONTENT_POS) or bool(ud_parts & CONTENT_UD)

        if pos_ok and lex_bare and len(lex_bare) >= 2:
            lemmas_content.append(lex_bare)
            if root and '#' not in root and len(root) >= 3:
                roots_content.append(root)
            lempos_content.append(f"{lex_bare}_{pos[:3]}")  # ex: مجنون_nou

    return {
        'lemmas_content': lemmas_content,
        'roots_content':  roots_content,
        'lempos_content': lempos_content,
        'lemmas_all':     lemmas_all,
    }

# ── Bigrammes ─────────────────────────────────────────────────────────────────
def bigrams(lst: list) -> list:
    return [f"{lst[i]} {lst[i+1]}" for i in range(len(lst)-1)]

# ── Analyse BoW : précision / rappel / LR ────────────────────────────────────
def compute_stats(pos_term_docs, neg_term_docs, xgb_term_docs,
                  N_POS, N_NEG, N_XGB, min_prec=0.30, min_rec=0.04, min_n=5):
    results = []
    for term, pos_set in pos_term_docs.items():
        n_pos = len(pos_set)
        n_neg = len(neg_term_docs.get(term, set()))
        n_xgb = len(xgb_term_docs.get(term, set()))

        if n_pos < min_n:
            continue

        prec     = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
        rec      = n_pos / N_POS
        neg_rate = n_neg / N_NEG if n_neg > 0 else 1 / (N_NEG * 2)
        lr_ratio = rec / neg_rate
        certainty = prec * rec
        xgb_pct   = n_xgb / N_XGB * 100

        if prec < min_prec or rec < min_rec:
            continue

        already = term in V80_EXISTING or any(t in V80_EXISTING for t in term.split())

        results.append({
            'term':          term,
            'precision':     round(prec, 4),
            'recall':        round(rec, 4),
            'lr_ratio':      round(lr_ratio, 2),
            'certainty':     round(certainty, 5),
            'n_pos':         n_pos,
            'n_neg':         n_neg,
            'xgb_hits':      n_xgb,
            'xgb_pct':       round(xgb_pct, 1),
            'already_v80':   already,
        })

    return sorted(results, key=lambda x: x['certainty'], reverse=True)

# ── Affichage d'une table ─────────────────────────────────────────────────────
def print_table(rows, title, n=40):
    print(f"\n{'═'*75}")
    print(f"  {title}")
    print(f"{'═'*75}")
    print(f"  {'Terme':<30} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'n_pos':>6} {'XGB%':>6} {'v80':>4}")
    print(f"  {'─'*68}")
    for c in rows[:n]:
        mark  = "✓" if c['already_v80'] else "★"
        xgbs  = f"{c['xgb_pct']:.0f}%" if c['xgb_pct'] > 0 else "—"
        print(f"  {mark} {c['term']:<28} {c['precision']:>6.3f} {c['recall']:>6.3f} "
              f"{c['lr_ratio']:>7.1f} {c['n_pos']:>6} {xgbs:>6}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument('--cached', action='store_true', help='Utiliser le cache existant')
parser.add_argument('--min-prec', type=float, default=0.35)
parser.add_argument('--min-rec',  type=float, default=0.04)
args = parser.parse_args()

# ── Chargement dataset ────────────────────────────────────────────────────────
print("Chargement dataset_raw.json...")
with open(DATASET, encoding='utf-8') as f:
    data = json.load(f)

records = [(d['text_ar'], d['label']) for d in data]
N_POS = sum(1 for _, l in records if l == 1)
N_NEG = sum(1 for _, l in records if l == 0)
print(f"  Positifs : {N_POS} | Négatifs : {N_NEG}")

print("Chargement positifs XGB (Ibn Abd Rabbih)...")
with open(XGB_POS, encoding='utf-8') as f:
    xgb_data = json.load(f)
xgb_raw = [p['text'] for p in xgb_data['positives_detailed']]
N_XGB = len(xgb_raw)
print(f"  {N_XGB} positifs XGB")

# ── Phase 1 : Lemmatisation (ou cache) ───────────────────────────────────────
if args.cached and CACHE.exists():
    print(f"\nChargement du cache : {CACHE}")
    with open(CACHE, encoding='utf-8') as f:
        cache = json.load(f)
    lemmatized      = cache['train']
    lemmatized_xgb  = cache['xgb']
    print(f"  {len(lemmatized)} textes + {len(lemmatized_xgb)} XGB chargés depuis cache")
else:
    print("\nPhase 1 — Lemmatisation MLE (CAMeL Tools)...")
    from camel_tools.disambig.mle import MLEDisambiguator
    mle = MLEDisambiguator.pretrained('calima-msa-r13')
    print("  MLE chargé")

    lemmatized = []
    t0 = time.time()
    for i, (text, label) in enumerate(records):
        cleaned = clean_openiti(text)
        result  = lemmatize_text(cleaned, mle)
        result['label'] = label
        lemmatized.append(result)
        if (i+1) % 500 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i+1) * (len(records) - i - 1)
            print(f"  {i+1}/{len(records)} ({elapsed:.0f}s écoulées, ~{remaining:.0f}s restantes)")

    print(f"  Corpus principal terminé en {time.time()-t0:.0f}s")

    print("  Lemmatisation des positifs XGB...")
    lemmatized_xgb = []
    for text in xgb_raw:
        cleaned = clean_openiti(text)
        lemmatized_xgb.append(lemmatize_text(cleaned, mle))

    # Sauvegarde cache
    print(f"  Sauvegarde cache → {CACHE}")
    with open(CACHE, 'w', encoding='utf-8') as f:
        json.dump({'train': lemmatized, 'xgb': lemmatized_xgb},
                  f, ensure_ascii=False)

# ── Phase 2 : Construction des index de termes ───────────────────────────────
print("\nPhase 2 — Construction des index BoW...")

# Pour chaque représentation : term → {doc_ids dans positifs/négatifs/xgb}
def build_index(lemmatized_list, lemmatized_xgb_list, key, with_bigrams=True):
    pos_docs = defaultdict(set)
    neg_docs = defaultdict(set)
    xgb_docs = defaultdict(set)

    for i, rec in enumerate(lemmatized_list):
        tokens = rec.get(key, [])
        terms  = set(tokens)
        if with_bigrams:
            terms |= set(bigrams(tokens))
        label = rec['label']
        for t in terms:
            if label == 1:
                pos_docs[t].add(i)
            else:
                neg_docs[t].add(i)

    for i, rec in enumerate(lemmatized_xgb_list):
        tokens = rec.get(key, [])
        terms  = set(tokens)
        if with_bigrams:
            terms |= set(bigrams(tokens))
        for t in terms:
            xgb_docs[t].add(i)

    return pos_docs, neg_docs, xgb_docs

# A. Lemmes (unigrammes + bigrammes)
print("  A. Lemmes content (uni + bi)...")
pos_lem, neg_lem, xgb_lem = build_index(
    lemmatized, lemmatized_xgb, 'lemmas_content', with_bigrams=True)
print(f"     {len(pos_lem)} termes distincts dans les positifs")

# B. Racines (unigrammes seulement)
print("  B. Racines (uni)...")
pos_root, neg_root, xgb_root = build_index(
    lemmatized, lemmatized_xgb, 'roots_content', with_bigrams=False)

# C. Lemme+POS bigrammes
print("  C. Lemme+POS (bi)...")
pos_lp, neg_lp, xgb_lp = build_index(
    lemmatized, lemmatized_xgb, 'lempos_content', with_bigrams=True)

# ── Phase 3 : Calcul des statistiques ────────────────────────────────────────
print("\nPhase 3 — Calcul précision/rappel/LR...")

stats_lem  = compute_stats(pos_lem,  neg_lem,  xgb_lem,  N_POS, N_NEG, N_XGB,
                           args.min_prec, args.min_rec)
stats_root = compute_stats(pos_root, neg_root, xgb_root, N_POS, N_NEG, N_XGB,
                           args.min_prec, args.min_rec)
stats_lp   = compute_stats(pos_lp,   neg_lp,   xgb_lp,   N_POS, N_NEG, N_XGB,
                           args.min_prec, args.min_rec)

print(f"  Lemmes    : {len(stats_lem)} candidats")
print(f"  Racines   : {len(stats_root)} candidats")
print(f"  Lemme+POS : {len(stats_lp)} candidats")

# ── Affichage ─────────────────────────────────────────────────────────────────

# Tous les candidats lemmes (top 50)
print_table(stats_lem,  "A. LEMMES — tous candidats (uni+bi)", n=50)

# Nouveaux seulement (hors v80)
new_lem  = [c for c in stats_lem  if not c['already_v80']]
new_root = [c for c in stats_root if not c['already_v80']]
new_lp   = [c for c in stats_lp   if not c['already_v80']]

print_table(new_lem,  f"A. LEMMES — nouvelles features ({len(new_lem)} candidats)", n=40)
print_table(new_root, f"B. RACINES — nouvelles features ({len(new_root)} candidats)", n=20)
print_table(new_lp,   f"C. LEMME+POS bigrammes — nouvelles features ({len(new_lp)} candidats)", n=25)

# Recommandations finales (certitude maximale + validation cross-corpus)
print(f"\n{'═'*75}")
print("  RECOMMANDATIONS FINALES (prec≥0.50, rec≥0.05, présents en XGB)")
print(f"{'═'*75}")

best = [c for c in new_lem
        if c['precision'] >= 0.50 and c['recall'] >= 0.05 and c['xgb_pct'] > 0]
best += [c for c in new_root
         if c['precision'] >= 0.50 and c['recall'] >= 0.05
         and c['xgb_pct'] > 0 and not any(b['term']==c['term'] for b in best)]

best.sort(key=lambda x: x['certainty'], reverse=True)

if best:
    print(f"\n  {'Terme':<30} {'Repr':<8} {'Prec':>6} {'Rapp':>6} {'LR':>7} {'XGB%':>6} {'n_pos':>6}")
    print(f"  {'─'*65}")
    for c in best:
        repr_type = 'root' if c in new_root else 'lemme'
        print(f"  ★ {c['term']:<28} {repr_type:<8} {c['precision']:>6.3f} {c['recall']:>6.3f} "
              f"{c['lr_ratio']:>7.1f} {c['xgb_pct']:>5.0f}% {c['n_pos']:>6}")
else:
    print("\n  Aucun candidat ne passe tous les filtres stricts.")
    print("  Meilleurs candidats (prec≥0.45, rec≥0.04) :")
    relaxed = sorted(
        [c for c in new_lem if c['precision'] >= 0.45 and c['recall'] >= 0.04],
        key=lambda x: x['certainty'], reverse=True)
    for c in relaxed[:10]:
        xgbs = f"{c['xgb_pct']:.0f}%" if c['xgb_pct'] > 0 else "—"
        print(f"  ★ {c['term']:<30} prec={c['precision']:.3f} rec={c['recall']:.3f} "
              f"LR={c['lr_ratio']:.1f}x XGB={xgbs}")

# ── Sauvegarde résultats ──────────────────────────────────────────────────────
output = {
    'method': 'CAMeL Tools MLE lemmatization + BoW (content POS only)',
    'n_pos': N_POS, 'n_neg': N_NEG, 'n_xgb': N_XGB,
    'filters': {'min_prec': args.min_prec, 'min_rec': args.min_rec},
    'counts': {
        'lemma_candidates': len(stats_lem),
        'root_candidates': len(stats_root),
        'lempos_candidates': len(stats_lp),
        'new_lemma': len(new_lem),
        'new_root': len(new_root),
    },
    'lemma_all':     stats_lem[:80],
    'lemma_new':     new_lem[:50],
    'root_new':      new_root[:30],
    'lempos_new':    new_lp[:30],
    'recommended':   best,
}
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Résultats → {OUT_JSON}")
print(f"   Cache     → {CACHE}")
print(f"   Relancer avec --cached pour ré-analyser sans re-lemmatiser.")
