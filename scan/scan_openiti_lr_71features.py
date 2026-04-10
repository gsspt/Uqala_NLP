#!/usr/bin/env python3
"""
scan_openiti_lr_71features.py
─────────────────────────────
Scanne openiti_corpus avec LR classifier 71 real features.

Utilise 62 features lexicales réelles + 9 morphologiques réelles (camel-tools).

Pipeline :
  1. Charge le classifier entraîné (71 features)
  2. Extrait tous les akhbars du corpus
  3. Pour chaque: extrait 71 features réelles (62 lexical + 9 morpho)
  4. Applique scaling et LR prediction
  5. Sauvegarde les candidats ≥ seuil

Sortie :
  scan/openiti_lr_71features_candidates.json

Usage :
  python scan/scan_openiti_lr_71features.py --threshold 0.5
"""

import json
import re
import pathlib
import sys
import argparse
import pickle
import unicodedata
import time
from collections import Counter
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS CAMEL TOOLS
# ══════════════════════════════════════════════════════════════════════════════

try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    print("✅ CAMeL Tools loaded")
except ImportError as e:
    print(f"❌ CAMeL Tools import failed: {e}")
    sys.exit(1)

morpho_db = MorphologyDB.builtin_db()
analyzer = Analyzer(morpho_db)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE      = pathlib.Path(__file__).parent.parent
CLF_PATH  = BASE / "scan" / "lr_classifier_71features.pkl"

# Corpus path configurable
DEFAULT_CORPUS = BASE / "openiti_corpus" / "data"
DEFAULT_OUT    = BASE / "scan" / "openiti_lr_71features_candidates.json"
DEFAULT_PROGRESS = BASE / "scan" / "openiti_lr_71features_progress.json"

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (71 REAL FEATURES)
# ══════════════════════════════════════════════════════════════════════════════
# Import from build_features_71.py or define here

JUNUN_TERMS = [
    'مجنون','المجنون','مجنونا','مجانين','المجانين','مجنونة','المجنونة',
    'معتوه','المعتوه','معتوها','معتوهة','مدله','المدله',
    'هائم','الهائم','هائما','ممسوس','ممرور','مستهتر',
    'جنونه','جنونها','جنوني','جنونا','جنون','الجنون',
    'ذاهبالعقل','ذهبعقله','ذاهب','ذهب',
]

FAMOUS_FOOLS = [
    'بهلول','بهلولا','سعدون','عليان','جعيفران','ريحانة',
    'سمنون','لقيط','حيون','حيونة','خلف','رياح',
]

AQL_TERMS = [
    'عاقل','العاقل','عقل','العقل','عقلاء','العقلاء','عقلاؤهم',
    'عقول','أعقل','معقول','عقلانية','حكمة','حكيم','عقلائي',
    'عقله','عقلها','عقلي','عقلك','عقلهم',
]

HIKMA_TERMS = [
    'حكمة','حكيم','حكماء','الحكمة','الحكيم','الحكماء','حكمته','حكمهم','أحكم',
]

QALA_VARIANTS = [
    'قلت','فقلت','قلنا','قالوا','قال','قالت','قالا',
    'سألت','فسألت','سألني','أجبت','أجاب',
    'قيل له','قيل لي','فقلت له',
    'وقال','ويقول','يقال','أقول',
]

FIRST_PERSON = [
    'فعلت','رأيت','مررت','لقيت','وجدت','شهدت','سمعت','حدثني',
    'أخبرني','أنبأني','حدثنا','أخبرنا','أنبأنا',
]

QUESTIONS = ['كيف','كيفية','ماذا','ما','متى','أين','هل','من','لماذا']

VALIDATION = {
    'laugh': ['ضحك','ضحكت','ضحكوا','فضحك','فضحكوا'],
    'gift': ['أعطى','أعطاه','وهب','جائزة','أمر','فأمر'],
    'cry': ['بكى','بكت','بكاء','دموع','دموعه'],
    'silence': ['صمت','سكت','أطرق','طرق'],
}

CONTRAST = {
    'opposition': ['لكن','لكنه','لكنها','بل','وبل','ولكن'],
    'correction': ['كذب','أخطأ','غلط','فبان','فتبين','أصح'],
    'revelation': ['فإذا','وإذا','فتبين','فتبينت'],
}

AUTHORITY = [
    'الخليفة','أميرالمؤمنين','أمير المؤمنين','الرشيد','المأمون',
    'المتوكل','المعتصم','المهدي','المنصور','الهادي','المعتضد',
    'الوزير','الوالي','القاضي','السلطان','الملك',
]

SHIR_MARKERS = [
    'أنشد','أنشأ','أنشدني','أنشدنا','فأنشد','فأنشأ','ينشد',
    'الشاعر','شعره','شعرها','شعري','أبيات','قصيدة','وأنشد',
]

SPATIAL = ['في','على','عند','بعد','أمام','خلف','حول','داخل','خارج']

WASF_MARKERS = [
    'ومنها','ضروب','فهو مجنون','تقول العرب','من القول',
    'قال الكاتب','قال المصنف','يقال','يسمى',
]

RE_TOK = re.compile(r'[\u0621-\u064A\u0671-\u06D3]+')

def has_junun_filtered(text):
    """Détecte junūn, filtre les faux positifs"""
    if not any(t in text for t in JUNUN_TERMS):
        return False
    if any(fp in text for fp in ['الجنة ', ' الجنة', 'الجن ', 'السجن ']):
        return False
    return True

def has_aql_filtered(text):
    """Détecte aql"""
    return any(t in text for t in AQL_TERMS)

def count_proximity(text, list1, list2, window=80):
    """Compte co-occurrences dans une fenêtre"""
    count = 0
    for term1 in list1:
        idx = text.find(term1)
        if idx < 0:
            continue
        for term2 in list2:
            start = max(0, idx - window)
            end = min(len(text), idx + len(term1) + window)
            if term2 in text[start:end]:
                count += 1
                break
    return count

def extract_features_71(text):
    """
    Extrait 71 features RÉELLES (62 lexical + 9 morphological)
    Same as build_features_71.py, optimized for speed
    """
    features = {}
    tokens = RE_TOK.findall(text)
    n_tokens = len(tokens) if tokens else 1

    # ─── JUNUN (15 features: f00-f14) ─────────────────────────────
    features['f00_has_junun'] = float(has_junun_filtered(text))
    features['f01_junun_density'] = sum(1 for t in tokens if t in JUNUN_TERMS) / n_tokens
    features['f02_famous_fool'] = float(any(name in text for name in FAMOUS_FOOLS))

    junun_count = sum(1 for t in JUNUN_TERMS if t in text)
    features['f03_junun_count'] = min(float(junun_count), 10.0) / 10

    specialized = [t for t in JUNUN_TERMS if len(t) > 4]
    features['f04_junun_specialized'] = float(any(s in text for s in specialized))

    first_junun = None
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0 and (first_junun is None or idx < first_junun):
            first_junun = idx
    features['f05_junun_position'] = first_junun / max(len(text), 1) if first_junun else 0.5

    features['f06_junun_in_title'] = float(any(term in text[:50] for term in JUNUN_TERMS))
    features['f07_junun_plural'] = float('مجانين' in text or 'المجانين' in text)

    third = len(text) // 3
    features['f08_junun_in_final_third'] = float(any(t in text[2*third:] for t in JUNUN_TERMS))

    jinn_root = ['جنون','جنونه','جنونها','جننت','يجن','أجنّ']
    features['f09_jinn_root'] = float(any(j in text for j in jinn_root))

    junun_rep = sum(text.count(t) for t in JUNUN_TERMS)
    features['f10_junun_repetition'] = min(float(junun_rep) / n_tokens, 1.0)

    features['f11_junun_morpho'] = float(any(m in text for m in ['مجنون','معتوه','هائم','ممسوس']))

    neg_junun = any(ng in text for ng in ['لا مجنون','ليس مجنون','لم يكن مجنون'])
    features['f12_junun_positive'] = float(not neg_junun)

    junun_context = 0
    for t in JUNUN_TERMS:
        idx = text.find(t)
        if idx >= 0:
            context = text[max(0, idx-20):idx+len(t)+20]
            if any(c in context for c in ['قال','رأيت','شهدت']):
                junun_context += 1
    features['f13_junun_good_context'] = float(junun_context > 0)

    val_all = VALIDATION['laugh'] + VALIDATION['gift'] + VALIDATION['cry']
    features['f14_junun_validation_prox'] = float(count_proximity(text, JUNUN_TERMS, val_all, 100) > 0)

    # ─── AQL (8 features: f15-f22) ──────────────────────────────
    features['f15_has_aql'] = float(has_aql_filtered(text))
    features['f16_aql_density'] = sum(1 for t in tokens if t in AQL_TERMS) / n_tokens

    aql_count = sum(1 for t in AQL_TERMS if t in text)
    features['f17_aql_count'] = min(float(aql_count), 10.0) / 10

    features['f18_paradox_junun_aql'] = float(features['f00_has_junun'] and features['f15_has_aql'])
    features['f19_junun_aql_proximity'] = float(count_proximity(text, JUNUN_TERMS, AQL_TERMS, 80) > 0)

    features['f20_junun_aql_ratio'] = features['f01_junun_density'] / (features['f16_aql_density'] + 0.001)
    features['f20_junun_aql_ratio'] = min(features['f20_junun_aql_ratio'], 10.0)

    superlatives = ['أعقل','أحكم','أصوب','أفضل','أعلم','أصدق']
    features['f21_superlatives'] = float(any(s in text for s in superlatives))

    neg_aql = any(ng in text for ng in ['لا عاقل','ليس عاقل','بلا عقل'])
    features['f22_aql_positive'] = float(not neg_aql)

    # ─── HIKMA (5 features: f23-f27) ────────────────────────────
    features['f23_has_hikma'] = float(any(t in text for t in HIKMA_TERMS))
    hikma_count = sum(1 for t in HIKMA_TERMS if t in text)
    features['f24_hikma_density'] = hikma_count / n_tokens

    features['f25_hikma_junun_prox'] = float(count_proximity(text, HIKMA_TERMS, JUNUN_TERMS, 80) > 0)
    features['f26_hikma_qala_prox'] = float(count_proximity(text, HIKMA_TERMS, QALA_VARIANTS, 80) > 0)
    features['f27_hikma_in_title'] = float(any(term in text[:50] for term in HIKMA_TERMS))

    # ─── DIALOGUE/QALA (11 features: f28-f38) ───────────────────
    features['f28_has_qala'] = float(any(q in text for q in QALA_VARIANTS))
    qala_count = sum(1 for q in QALA_VARIANTS if q in text)
    features['f29_qala_density'] = qala_count / n_tokens

    features['f30_has_first_person'] = float(any(fp in text for fp in FIRST_PERSON))
    fp_count = sum(1 for fp in FIRST_PERSON if fp in text)
    features['f31_first_person_density'] = fp_count / n_tokens

    features['f32_junun_near_qala'] = float(count_proximity(text, JUNUN_TERMS, QALA_VARIANTS, 80) > 0)

    features['f33_has_questions'] = float(any(q in text for q in QUESTIONS))
    q_count = sum(1 for q in QUESTIONS if q in text)
    features['f34_question_density'] = q_count / n_tokens

    has_sual = any(s in text for s in QUESTIONS)
    has_jawab = any(j in text for j in QALA_VARIANTS)
    features['f35_question_answer'] = float(has_sual and has_jawab)

    qala_count_multiple = text.count('قال') + text.count('قلت') >= 2
    features['f36_dialogue_structure'] = float(qala_count_multiple)

    first_qala = text.find('قال')
    if first_qala < 0:
        first_qala = text.find('قلت')
    features['f37_qala_position'] = first_qala / max(len(text), 1) if first_qala >= 0 else 0.5

    features['f38_qala_in_final'] = float(any(q in text[2*third:] for q in QALA_VARIANTS))

    # ─── VALIDATION (8 features: f39-f46) ───────────────────────
    all_validation = VALIDATION['laugh'] + VALIDATION['gift'] + VALIDATION['cry'] + VALIDATION['silence']
    features['f39_has_validation'] = float(any(v in text for v in all_validation))

    val_count = sum(1 for v in all_validation if v in text)
    features['f40_validation_density'] = val_count / n_tokens

    has_laugh = any(l in text for l in VALIDATION['laugh'])
    has_gift = any(g in text for g in VALIDATION['gift'])
    has_cry = any(c in text for c in VALIDATION['cry'])
    features['f41_validation_laugh'] = float(has_laugh)
    features['f42_validation_gift'] = float(has_gift)
    features['f43_validation_cry'] = float(has_cry)

    features['f44_validation_in_final'] = float(any(v in text[2*third:] for v in all_validation))

    val_types = [has_laugh, has_gift, has_cry]
    features['f45_validation_multiple'] = float(sum(val_types) > 1)

    features['f46_validation_junun_prox'] = float(count_proximity(text, all_validation, JUNUN_TERMS, 100) > 0)

    # ─── CONTRASTE (5 features: f47-f51) ────────────────────────
    all_contrast = CONTRAST['opposition'] + CONTRAST['correction'] + CONTRAST['revelation']
    features['f47_has_contrast'] = float(any(c in text for c in all_contrast))

    c_count = sum(1 for c in all_contrast if c in text)
    features['f48_contrast_density'] = c_count / n_tokens

    has_opp = any(o in text for o in CONTRAST['opposition'])
    has_corr = any(c in text for c in CONTRAST['correction'])
    has_rev = any(r in text for r in CONTRAST['revelation'])
    features['f49_contrast_opposition'] = float(has_opp)
    features['f50_contrast_correction'] = float(has_corr)
    features['f51_contrast_revelation'] = float(has_rev)

    # ─── AUTORITÉ (4 features: f52-f55) ─────────────────────────
    features['f52_has_authority'] = float(any(a in text for a in AUTHORITY))
    auth_count = sum(1 for a in AUTHORITY if a in text)
    features['f53_authority_count'] = min(float(auth_count), 5.0) / 5

    features['f54_authority_junun_prox'] = float(count_proximity(text, AUTHORITY, JUNUN_TERMS, 100) > 0)
    features['f55_authority_in_title'] = float(any(a in text[:50] for a in AUTHORITY))

    # ─── POÉSIE (3 features: f56-f58) ───────────────────────────
    features['f56_has_shir'] = float(any(s in text for s in SHIR_MARKERS))
    shir_count = sum(1 for s in SHIR_MARKERS if s in text)
    features['f57_shir_density'] = shir_count / n_tokens

    features['f58_shir_alone'] = float(features['f56_has_shir'] and not features['f28_has_qala'])

    # ─── SPATIAL (3 features: f59-f61) ───────────────────────────
    features['f59_has_spatial'] = float(any(s in text for s in SPATIAL))
    spatial_count = sum(1 for s in SPATIAL if s in text)
    features['f60_spatial_density'] = spatial_count / n_tokens

    unique_spatial = sum(1 for s in SPATIAL if s in text)
    features['f61_spatial_variety'] = unique_spatial / len(SPATIAL)

    # ─── WASF (3 features: f62-f64) ────────────────────────────
    features['f62_has_wasf'] = float(any(w in text for w in WASF_MARKERS))
    wasf_count = sum(1 for w in WASF_MARKERS if w in text)
    features['f63_wasf_density'] = wasf_count / n_tokens

    features['f64_wasf_in_title'] = float(any(w in text[:80] for w in WASF_MARKERS))

    # ─── MORPHOLOGICAL (9 features: f65-f73) ────────────────────
    tokens_split = text.split()
    n_tokens_split = len(tokens_split) if tokens_split else 1

    jnn_count = aql_count = hikma_count = 0
    verb_count = noun_count = adj_count = 0
    perf_count = imperf_count = passive_count = 0

    for token in tokens_split:
        try:
            analyses = analyzer.analyze(token)
            if analyses:
                a = analyses[0]
                root = a.get('root', '')
                pos = a.get('pos', '')
                asp = a.get('asp', '')
                vox = a.get('vox', '')

                if root == 'ج.ن.ن':
                    jnn_count += 1
                if root == 'ع.ق.ل':
                    aql_count += 1
                if root == 'ح.ك.م':
                    hikma_count += 1

                if pos == 'verb':
                    verb_count += 1
                if pos == 'noun':
                    noun_count += 1
                if pos == 'adj':
                    adj_count += 1

                if asp == 'perf':
                    perf_count += 1
                if asp == 'imperf':
                    imperf_count += 1
                if vox == 'pass':
                    passive_count += 1
        except:
            pass

    features['f65_root_jnn_density'] = jnn_count / n_tokens_split
    features['f66_root_aql_density'] = aql_count / n_tokens_split
    features['f67_root_hikma_density'] = hikma_count / n_tokens_split
    features['f68_verb_density'] = verb_count / n_tokens_split
    features['f69_noun_density'] = noun_count / n_tokens_split
    features['f70_adj_density'] = adj_count / n_tokens_split
    features['f71_perf_density'] = perf_count / n_tokens_split
    features['f72_imperf_density'] = imperf_count / n_tokens_split
    features['f73_passive_voice_ratio'] = passive_count / n_tokens_split

    return features

# ══════════════════════════════════════════════════════════════════════════════
# CORPUS SCANNING
# ══════════════════════════════════════════════════════════════════════════════

def extract_akhbars(filepath):
    """Extrait akhbars depuis fichier OpenITI"""
    def count_ar(s):
        return sum(1 for c in s if unicodedata.category(c) == 'Lo' and '\u0600' <= c <= '\u06FF')

    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except:
        return []

    content_started = False
    units, current = [], []

    for line in lines:
        line = line.rstrip('\n')
        if '#META#Header#End#' in line:
            content_started = True
            continue
        if not content_started:
            continue
        if line.startswith('~~'):
            current.append(line[2:].strip())
        elif line.startswith('# '):
            if current:
                text = ' '.join(current)
                if 80 <= count_ar(text) <= 3000:
                    units.append(text)
            current = [line[2:].strip()] if not line.startswith('# |') else []
        else:
            if current:
                text = ' '.join(current)
                if 80 <= count_ar(text) <= 3000:
                    units.append(text)
            current = []

    if current:
        text = ' '.join(current)
        if 80 <= count_ar(text) <= 3000:
            units.append(text)

    return units

def scan(corpus_dir, threshold=0.5, resume=False, out_path=None, progress_path=None):
    """Scanne corpus avec LR 71-features"""

    if out_path is None:
        out_path = DEFAULT_OUT
    if progress_path is None:
        progress_path = DEFAULT_PROGRESS

    if not CLF_PATH.exists():
        print(f"ERROR: Classifier not found → {CLF_PATH}")
        return

    print(f"  Corpus: {corpus_dir}")
    print(f"  Output: {out_path}\n")

    print(f"  Loading classifier (71 features)…")
    with open(CLF_PATH, 'rb') as f:
        clf_data = pickle.load(f)
    clf = clf_data['clf']
    scaler = clf_data['scaler']

    all_files = sorted(corpus_dir.rglob('*-ara1'))
    n_files = len(all_files)
    print(f"  OpenITI files: {n_files}\n")

    all_candidates = []
    total_akhbars = 0
    done_files = set()

    if resume and pathlib.Path(progress_path).exists():
        state = json.load(open(progress_path))
        all_candidates = state['candidates']
        total_akhbars = state['total_akhbars']
        done_files = set(state['done_files'])
        print(f"  Resuming: {len(done_files)}/{n_files} files processed")

    remaining = [fp for fp in all_files if fp.name not in done_files]
    print(f"  Remaining: {len(remaining)}")
    print(f"  Threshold: {threshold:.0%}\n")

    t_start = time.time()

    for fi, fp in enumerate(remaining):
        akhbars = extract_akhbars(fp)
        if not akhbars:
            done_files.add(fp.name)
            continue

        # Extract features & score
        features = []
        for a in akhbars:
            feat = extract_features_71(a)
            features.append(list(feat.values()))

        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = scaler.transform(features)
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        probs = clf.predict_proba(features_scaled)[:, 1]
        total_akhbars += len(akhbars)
        done_files.add(fp.name)

        # Save candidates
        n_before = len(all_candidates)
        for j, (raw, prob) in enumerate(zip(akhbars, probs)):
            if prob >= threshold:
                all_candidates.append({
                    'file': fp.name,
                    'author': fp.parent.parent.name,
                    'work': fp.parent.name,
                    'idx': j,
                    'lr_score': float(prob),
                    'text': raw[:400],
                })

        n_found = len(all_candidates) - n_before

        # Save progress
        if (fi + 1) % 20 == 0:
            json.dump({
                'threshold': threshold,
                'total_akhbars': total_akhbars,
                'done_files': list(done_files),
                'candidates': all_candidates,
            }, open(progress_path, 'w'))

        # Progress
        if (fi + 1) % 50 == 0 or n_found > 0:
            elapsed = time.time() - t_start
            pct = len(done_files) / n_files * 100
            rate = (fi + 1) / elapsed if elapsed > 0 else 0
            eta_s = (len(remaining) - fi - 1) / rate if rate > 0 else 0
            eta_min = int(eta_s // 60)

            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}%  {len(done_files)}/{n_files}  "
                  f"|  {total_akhbars:,} akhbars  |  {len(all_candidates)} candidates")

    # Finalize
    json.dump({
        'threshold': threshold,
        'total_akhbars': total_akhbars,
        'done_files': list(done_files),
        'candidates': all_candidates,
    }, open(progress_path, 'w'))

    all_candidates.sort(key=lambda x: -x['lr_score'])

    print(f"\n  ── LR Scan Results (71 features) ──")
    print(f"  Akhbars scanned: {total_akhbars:,}")
    print(f"  Candidates ≥ {threshold:.0%}: {len(all_candidates)}")
    print(f"  Detection rate: {len(all_candidates)/total_akhbars*100:.2f}%")

    json.dump({
        'threshold': threshold,
        'total_akhbars': total_akhbars,
        'n_candidates': len(all_candidates),
        'candidates': all_candidates,
    }, open(out_path, 'w'), ensure_ascii=False, indent=2)

    print(f"\n  Candidates saved: {out_path}")
    if pathlib.Path(progress_path).exists():
        pathlib.Path(progress_path).unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default=str(DEFAULT_CORPUS),
                        help=f'Corpus directory (default: {DEFAULT_CORPUS})')
    parser.add_argument('--threshold', type=float, default=0.5, help='LR score threshold')
    parser.add_argument('--resume', action='store_true', help='Resume from progress file')
    args = parser.parse_args()

    corpus_path = pathlib.Path(args.corpus)
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found → {corpus_path}")
        exit(1)

    # Déterminer les paths de sortie basés sur le corpus
    if 'targeted' in args.corpus.lower():
        out_path = BASE / "scan" / "openiti_lr_71features_targeted_candidates.json"
        progress_path = BASE / "scan" / "openiti_lr_71features_targeted_progress.json"
    else:
        out_path = DEFAULT_OUT
        progress_path = DEFAULT_PROGRESS

    print(f"{'='*70}")
    print(f"LR Scan with 71 Features (Lexical + Morphological)")
    print(f"{'='*70}\n")

    scan(corpus_path, threshold=args.threshold, resume=args.resume,
         out_path=out_path, progress_path=progress_path)
