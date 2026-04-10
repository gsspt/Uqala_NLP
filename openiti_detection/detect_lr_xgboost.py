#!/usr/bin/env python3
"""
detect_lr_xgboost.py
────────────────────
Détecte les figures du fou sensé (ʿāqil majnūn) dans le corpus openITI
en appliquant DEUX classifiers: LR (interprétable) et XGBoost (performant).

Utilise les modèles entraînés sur dataset_raw.json avec 74 features réelles
(62 lexicales + 9 morphologiques via CAMeL Tools).

Usage:
  python openiti_detection/detect_lr_xgboost.py --threshold-lr 0.5 --threshold-xgb 0.5

Sorties:
  openiti_detection/results/lr_predictions.json
  openiti_detection/results/xgb_predictions.json
  openiti_detection/results/comparison.json (consensus entre LR et XGBoost)
"""

import json
import sys
import pathlib
import pickle
import argparse
import unicodedata
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
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

# Importer isnad_filter pour séparer isnad du matn
try:
    _base_path = pathlib.Path(__file__).parent.parent
    sys.path.insert(0, str(_base_path))
    from isnad_filter import get_matn
    print("✅ isnad_filter loaded")
except ImportError as e:
    print(f"⚠️  isnad_filter not available, analyzing full text (including isnads)")
    def get_matn(text):
        return text

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE = pathlib.Path(__file__).parent.parent
LR_MODEL_PATH = BASE / "scan" / "lr_classifier_71features.pkl"
XGB_MODEL_PATH = BASE / "comparison_ensemble" / "results" / "xgb_classifier_71features.pkl"
CORPUS_PATH = BASE / "openiti_targeted"  # Corpus ciblé pour détection
OUT_DIR = pathlib.Path(__file__).parent / "results"

OUT_DIR.mkdir(exist_ok=True)

LR_RESULTS_PATH = OUT_DIR / "lr_predictions.json"
XGB_RESULTS_PATH = OUT_DIR / "xgb_predictions.json"
COMPARISON_PATH = OUT_DIR / "comparison.json"
PROGRESS_PATH = OUT_DIR / "progress.json"

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS (même que build_features_71.py)
# ══════════════════════════════════════════════════════════════════════════════

JUNUN_TERMS = [
    'مجنون','المجنون','مجنونا','مجانين','المجانين','مجنونة','المجنونة',
    'معتوه','المعتوه','معتوها','معتوهة','مدله','المدله',
    'هائم','الهائم','هائما','ممسوس','ممرور','مستهتر',
]

FAMOUS_FOOLS = [
    'بهلول','بهلولا','سعدون','عليان','جعيفران','ريحانة',
    'سمنون','لقيط','حيون','حيونة','خلف','رياح',
]

AQL_TERMS = [
    'عقل','العقل','عقلا','عاقل','العاقل','عقلي','عقلك','عقله','عقلها',
    'عقلاء','معقول','المعقول','يعقل','تعقل',
]

HIKMA_TERMS = ['حكم','حكمة','الحكمة','حكيم','الحكيم','حكما','حكيما']

QALA_TERMS = ['قال','وقال','فقال','ويقول','يقول','قلت','أقول','قالوا','يقال']

VALIDATION_TERMS = [
    'ضحك','يضحك','فضحك','ضاحك','الضحك',
    'أهدى','أهداه','هدية','هدايا',
    'بكى','يبكي','بكاء','الدموع',
]

CONTRAST_TERMS = ['لكن','لكنه','غير','لولا','بينما','والعكس','بل','إذا']

AUTHORITY_TERMS = [
    'الخليفة','الخليفتان','الخليفة الراشد','خليفة','الأمير','الملك','السلطان',
    'والي','الوالي','الوزير','القاضي','القضاة',
]

SHIR_TERMS = ['شعر','الشعر','شاعر','شاعران','شاعري','بيت شعر','بيتان','أبيات']

WASF_TERMS = [
    'ومنها','ضروب','تقول العرب','قالوا','الأول','الثاني',
    'من ضروب','من قول','من مقولة',
]

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_arabic(text):
    """Normalise l'arabe (diacritiques, variantes)."""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

def clean_text(text):
    """Nettoie le texte."""
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_term_occurrence(text, terms):
    """Compte les occurrences d'une liste de termes dans le texte."""
    text_normalized = normalize_arabic(text.lower())
    count = 0
    for term in terms:
        term_normalized = normalize_arabic(term.lower())
        count += len([m for m in __import__('re').finditer(
            r'\b' + __import__('re').escape(term_normalized) + r'\b',
            text_normalized
        )])
    return count

def extract_features_74(text):
    """
    Extrait les 74 features réelles (62 lexicales + 9 morphologiques + 3 wasf)
    Retourne un vecteur numpy de dimension 74.
    """
    text_normalized = normalize_arabic(text.lower())
    text_length = len(text_normalized)

    # Initialiser le vecteur de features
    features = {}

    # LEXICAL FEATURES (62)
    # ──────────────────────

    # JUNUN block (15)
    features['f00_has_junun'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in JUNUN_TERMS]) else 0.0
    features['f01_junun_density'] = count_term_occurrence(text, JUNUN_TERMS) / max(text_length / 100, 1.0)
    features['f02_famous_fool'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in FAMOUS_FOOLS]) else 0.0
    features['f03_junun_count'] = count_term_occurrence(text, JUNUN_TERMS) / max(text_length / 200, 1.0)
    features['f04_junun_specialized'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['مجنون', 'معتوه', 'هائم']]) else 0.0
    features['f05_junun_position'] = 0.5 if count_term_occurrence(text, JUNUN_TERMS) > 0 else 0.0
    features['f06_junun_in_title'] = 1.0 if any(t in text_normalized[:200] for t in
        [normalize_arabic(x.lower()) for x in JUNUN_TERMS]) else 0.0
    features['f07_junun_plural'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['مجانين', 'المجانين']]) else 0.0
    features['f08_jinn_root'] = 1.0 if any(x in text_normalized for x in ['جنن', 'جنون']) else 0.0
    features['f09_junun_morpho'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['معتوه', 'هائم', 'ممسوس']]) else 0.0
    features['f10_junun_positive'] = 1.0 if features['f00_has_junun'] else 0.0
    features['f11_junun_good_context'] = 1.0 if features['f00_has_junun'] and any(
        t in text_normalized for t in [normalize_arabic(x.lower()) for x in QALA_TERMS]) else 0.0
    features['f12_junun_validation_prox'] = 1.0 if features['f00_has_junun'] and any(
        t in text_normalized for t in [normalize_arabic(x.lower()) for x in VALIDATION_TERMS]) else 0.0
    features['f13_junun_near_qala'] = 1.0 if features['f00_has_junun'] and any(
        t in text_normalized for t in [normalize_arabic(x.lower()) for x in QALA_TERMS]) else 0.0
    features['f14_junun_variant_forms'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['مجنونا', 'بمجنون', 'المجنونة']]) else 0.0

    # AQL block (8)
    features['f15_has_aql'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in AQL_TERMS]) else 0.0
    features['f16_aql_density'] = count_term_occurrence(text, AQL_TERMS) / max(text_length / 100, 1.0)
    features['f17_aql_count'] = count_term_occurrence(text, AQL_TERMS) / max(text_length / 200, 1.0)
    features['f18_paradox_junun_aql'] = 1.0 if (features['f00_has_junun'] and features['f15_has_aql']) else 0.0
    features['f19_junun_aql_proximity'] = 1.0 if features['f18_paradox_junun_aql'] else 0.0
    features['f20_junun_aql_ratio'] = (count_term_occurrence(text, JUNUN_TERMS) /
        max(count_term_occurrence(text, AQL_TERMS), 1.0))
    features['f21_superlatives'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['أعقل', 'أحكم', 'أصوب']]) else 0.0
    features['f22_aql_variant_forms'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['عقله', 'عقلي', 'عقلك']]) else 0.0

    # HIKMA block (5)
    features['f23_has_hikma'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in HIKMA_TERMS]) else 0.0
    features['f24_hikma_density'] = count_term_occurrence(text, HIKMA_TERMS) / max(text_length / 100, 1.0)
    features['f25_hikma_junun_prox'] = 1.0 if (features['f00_has_junun'] and features['f23_has_hikma']) else 0.0
    features['f26_hikma_qala_prox'] = 1.0 if (features['f23_has_hikma'] and any(
        t in text_normalized for t in [normalize_arabic(x.lower()) for x in QALA_TERMS])) else 0.0
    features['f27_hikma_in_title'] = 1.0 if any(t in text_normalized[:200] for t in
        [normalize_arabic(x.lower()) for x in HIKMA_TERMS]) else 0.0

    # DIALOGUE block (11)
    features['f28_has_qala'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in QALA_TERMS]) else 0.0
    features['f29_qala_density'] = count_term_occurrence(text, QALA_TERMS) / max(text_length / 100, 1.0)
    features['f30_has_first_person'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['قلت', 'أقول', 'قالوا']]) else 0.0
    features['f31_first_person_density'] = count_term_occurrence(text, ['قلت', 'أقول']) / max(text_length / 200, 1.0)
    features['f32_junun_near_qala'] = 1.0 if (features['f00_has_junun'] and features['f28_has_qala']) else 0.0
    features['f33_has_questions'] = 1.0 if '؟' in text else 0.0
    features['f34_question_density'] = text.count('؟') / max(text_length / 200, 1.0)
    features['f35_question_answer'] = 1.0 if ('؟' in text and '،' in text) else 0.0
    features['f36_dialogue_structure'] = 1.0 if count_term_occurrence(text, QALA_TERMS) >= 2 else 0.0
    features['f37_qala_position'] = 1.0 if features['f28_has_qala'] else 0.0
    features['f38_qala_in_final'] = 1.0 if any(t in text_normalized[-200:] for t in
        [normalize_arabic(x.lower()) for x in QALA_TERMS]) else 0.0

    # VALIDATION block (8)
    features['f39_has_validation'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in VALIDATION_TERMS]) else 0.0
    features['f40_validation_density'] = count_term_occurrence(text, VALIDATION_TERMS) / max(text_length / 100, 1.0)
    features['f41_validation_laugh'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['ضحك', 'يضحك', 'ضاحك']]) else 0.0
    features['f42_validation_gift'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['أهدى', 'هدية', 'هدايا']]) else 0.0
    features['f43_validation_cry'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['بكى', 'يبكي', 'بكاء']]) else 0.0
    features['f44_validation_in_final'] = 1.0 if any(t in text_normalized[-200:] for t in
        [normalize_arabic(x.lower()) for x in VALIDATION_TERMS]) else 0.0
    features['f45_validation_multiple'] = 1.0 if count_term_occurrence(text, VALIDATION_TERMS) >= 2 else 0.0
    features['f46_validation_junun_prox'] = 1.0 if (features['f00_has_junun'] and
        features['f39_has_validation']) else 0.0

    # CONTRAST block (5)
    features['f47_has_contrast'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in CONTRAST_TERMS]) else 0.0
    features['f48_contrast_density'] = count_term_occurrence(text, CONTRAST_TERMS) / max(text_length / 100, 1.0)
    features['f49_contrast_opposition'] = 1.0 if 'لكن' in text_normalized else 0.0
    features['f50_contrast_correction'] = 1.0 if 'بل' in text_normalized else 0.0
    features['f51_contrast_revelation'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in ['فإذا', 'وإذا']]) else 0.0

    # AUTHORITY block (4)
    features['f52_has_authority'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in AUTHORITY_TERMS]) else 0.0
    features['f53_authority_count'] = count_term_occurrence(text, AUTHORITY_TERMS) / max(text_length / 200, 1.0)
    features['f54_authority_junun_prox'] = 1.0 if (features['f00_has_junun'] and
        features['f52_has_authority']) else 0.0
    features['f55_authority_in_title'] = 1.0 if any(t in text_normalized[:200] for t in
        [normalize_arabic(x.lower()) for x in AUTHORITY_TERMS]) else 0.0

    # POETRY block (3)
    features['f56_has_shir'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in SHIR_TERMS]) else 0.0
    features['f57_shir_density'] = count_term_occurrence(text, SHIR_TERMS) / max(text_length / 100, 1.0)
    features['f58_shir_alone'] = 1.0 if features['f56_has_shir'] and not features['f00_has_junun'] else 0.0

    # WASF/NEGATION block (3) — restored from model_14
    features['f59_has_wasf'] = 1.0 if any(t in text_normalized for t in
        [normalize_arabic(x.lower()) for x in WASF_TERMS]) else 0.0
    features['f60_wasf_density'] = count_term_occurrence(text, WASF_TERMS) / max(text_length / 100, 1.0)
    features['f61_wasf_in_title'] = 1.0 if any(t in text_normalized[:200] for t in
        [normalize_arabic(x.lower()) for x in WASF_TERMS]) else 0.0

    # Additional features (4)
    features['f62_text_length_norm'] = min(text_length / 500, 1.0)
    features['f63_positive_indicator'] = (features['f00_has_junun'] + features['f18_paradox_junun_aql']) / 2
    features['f64_narrative_structure'] = 1.0 if features['f28_has_qala'] and features['f39_has_validation'] else 0.0

    # MORPHOLOGICAL FEATURES via CAMeL Tools (9)
    # ────────────────────────────────────────────

    try:
        analyses = analyzer.analyze(text)

        # Root densities
        jnn_count = sum(1 for a in analyses if 'j.n.n' in str(a.get('root', '')).lower())
        aql_count = sum(1 for a in analyses if 'ʿ.q.l' in str(a.get('root', '')).lower())
        hikma_count = sum(1 for a in analyses if 'ḥ.k.m' in str(a.get('root', '')).lower())

        features['f65_root_jnn_density'] = jnn_count / max(len(analyses), 1.0)
        features['f66_root_aql_density'] = aql_count / max(len(analyses), 1.0)
        features['f67_root_hikma_density'] = hikma_count / max(len(analyses), 1.0)

        # POS densities
        verb_count = sum(1 for a in analyses if 'verb' in str(a.get('pos', '')).lower())
        noun_count = sum(1 for a in analyses if 'noun' in str(a.get('pos', '')).lower())
        adj_count = sum(1 for a in analyses if 'adj' in str(a.get('pos', '')).lower())

        features['f68_verb_density'] = verb_count / max(len(analyses), 1.0)
        features['f69_noun_density'] = noun_count / max(len(analyses), 1.0)
        features['f70_adj_density'] = adj_count / max(len(analyses), 1.0)

        # Aspect & voice
        perf_count = sum(1 for a in analyses if 'perf' in str(a.get('aspect', '')).lower())
        pass_count = sum(1 for a in analyses if 'pass' in str(a.get('voice', '')).lower())

        features['f71_perf_density'] = perf_count / max(len(analyses), 1.0)
        features['f72_passive_voice_ratio'] = pass_count / max(verb_count, 1.0)

    except Exception as e:
        # Si CAMeL Tools échoue, mettre des 0
        for i in range(65, 73):
            features[f'f{i:02d}'] = 0.0

    # Dernière feature
    features['f73_consensus_validation'] = (features['f18_paradox_junun_aql'] +
        features['f39_has_validation']) / 2

    # Retourner un vecteur ordonné (récupérer toutes les features créées)
    feature_vector = []
    for key in sorted(features.keys(), key=lambda x: (int(x[1:3]), x)):
        feature_vector.append(features[key])

    # Assurer que nous avons exactement 74 features
    while len(feature_vector) < 74:
        feature_vector.append(0.0)

    return np.array(feature_vector[:74], dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
# CORPUS LOADING
# ══════════════════════════════════════════════════════════════════════════════

def count_arabic_chars(s):
    """Compte les caractères arabes dans une chaîne."""
    return sum(1 for c in s if '\u0600' <= c <= '\u06FF')

def clean_openiti_metadata(text):
    """
    Nettoie les marqueurs de métadonnées openITI du texte.

    Supprime:
    - Marqueurs de manuscrit: ms#### (ex: ms0001, ms0118)
    - Marqueurs de pagination: PageV##P### (ex: PageV01P023)
    - Citations coraniques: ^ ... ^ (ex: ^ (texte) ^)
    - Références de versets: [ ... ] (ex: [ Surah : ], [ الدين و ])
    - Sections de poésie: % (marqueurs isolés et paires)
    - Séparateurs: | en fin de ligne ou isolés
    """
    import re

    # Supprimer ms#### (marqueurs de manuscrit)
    text = re.sub(r'\bms\d{4,5}\b', '', text)

    # Supprimer PageV##P### (marqueurs de pagination)
    text = re.sub(r'PageV\d{2}P\d{3,4}', '', text)

    # Supprimer ^ ... ^ (citations coraniques avec parenthèses)
    text = re.sub(r'\^\s*\([^)]*\)\s*\^', '', text)

    # Supprimer ^ isolés (restes de citations)
    text = re.sub(r'\s\^\s', ' ', text)
    text = re.sub(r'^\^', '', text)
    text = re.sub(r'\^$', '', text)

    # Supprimer [...] (références de versets et autres)
    text = re.sub(r'\[\s*[^\]]*\]', '', text)

    # Supprimer % ... % (paires de poésie)
    text = re.sub(r'%[^%]*%', '', text, flags=re.DOTALL)

    # Supprimer % isolés (restes de marqueurs de poésie)
    text = re.sub(r'\s%\s', ' ', text)
    text = re.sub(r'^\s*%\s*', '', text)
    text = re.sub(r'\s*%\s*$', '', text)
    text = re.sub(r'%', '', text)

    # Supprimer | isolés ou en fin de ligne
    text = re.sub(r'\s*\|\s*', ' ', text)

    # Nettoyer les espaces multiples
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def extract_akhbars_from_file(filepath):
    """
    Extrait les akhbars (unités narratives) d'un fichier openITI.
    Applique isnad_filter pour extraire uniquement le matn (contenu narratif).

    Format openITI:
    - Lignes commençant par ~~ : contenu
    - Lignes commençant par # : en-têtes de section
    - #META#Header#End# : fin de l'en-tête

    Nettoyage OpenITI:
    - Supprime ms####, PageV##P###, ^...^, [...], %...%, | marqueurs

    Filtre: 80-3000 caractères arabes par akhbar (après suppression isnad + métadonnées)
    """
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return []

    content_started = False
    akhbars = []
    current = []

    for line in lines:
        line = line.rstrip('\n')

        # Passer jusqu'à la fin du header
        if '#META#Header#End#' in line:
            content_started = True
            continue
        if not content_started:
            continue

        # Accumulate content lines (starting with ~~)
        if line.startswith('~~'):
            current.append(line[2:].strip())

        # New section (# ...) or end of file
        elif line.startswith('# '):
            # Save current akhbar if not empty
            if current:
                full_text = ' '.join(current)
                # Nettoyer les métadonnées openITI
                full_text = clean_openiti_metadata(full_text)
                # Appliquer isnad_filter pour extraire le matn
                matn = get_matn(full_text)
                ar_count = count_arabic_chars(matn)
                if 80 <= ar_count <= 3000:
                    akhbars.append(matn)
            # Start new akhbar
            current = [] if line.startswith('# |') else [line[2:].strip()]
        else:
            # Empty line or other content
            if current and len(line.strip()) > 0:
                current.append(line.strip())

    # Don't forget the last akhbar
    if current:
        full_text = ' '.join(current)
        # Nettoyer les métadonnées openITI
        full_text = clean_openiti_metadata(full_text)
        # Appliquer isnad_filter pour extraire le matn
        matn = get_matn(full_text)
        ar_count = count_arabic_chars(matn)
        if 80 <= ar_count <= 3000:
            akhbars.append(matn)

    return akhbars

def stream_openiti_corpus():
    """
    Streaming generator — charge les akhbars à la volée (pas de chargement d'ensemble).
    Yield chaque akhbar un par un.
    """
    if not CORPUS_PATH.exists():
        print(f"⚠️  Corpus path not found: {CORPUS_PATH}")
        return

    # Scanner tous les fichiers du corpus
    for filepath in CORPUS_PATH.rglob("*"):
        if filepath.is_file() and not filepath.name.startswith('.'):
            try:
                file_akhbars = extract_akhbars_from_file(filepath)

                for text in file_akhbars:
                    if len(text) > 50:
                        yield {
                            'path': str(filepath.relative_to(CORPUS_PATH)),
                            'text': text,
                            'length': len(text)
                        }
            except Exception as e:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Detect wise fool figures with LR and XGBoost')
    parser.add_argument('--threshold-lr', type=float, default=0.5, help='LR threshold')
    parser.add_argument('--threshold-xgb', type=float, default=0.5, help='XGBoost threshold')
    args = parser.parse_args()

    # Vérifier les modèles
    if not LR_MODEL_PATH.exists():
        print(f"❌ LR model not found: {LR_MODEL_PATH}")
        sys.exit(1)

    if not XGB_MODEL_PATH.exists():
        print(f"❌ XGBoost model not found: {XGB_MODEL_PATH}")
        sys.exit(1)

    # Charger les modèles
    print("Loading models…")
    with open(LR_MODEL_PATH, 'rb') as f:
        lr_data = pickle.load(f)

    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_data = pickle.load(f)

    lr_model = lr_data['clf']
    lr_scaler = lr_data['scaler']
    xgb_model = xgb_data['clf']
    xgb_scaler = xgb_data['scaler']

    # Streaming: traiter au fur et à mesure (pas de chargement d'ensemble)
    print("Processing openiti_targeted corpus (streaming)…\n")

    lr_results = []
    xgb_results = []
    consensus_hits = []

    start_time = time.time()
    idx = 0
    last_report_time = start_time

    for akhbar in stream_openiti_corpus():
        idx += 1

        # Afficher la progression toutes les 5 secondes (simple)
        current_time = time.time()
        if current_time - last_report_time >= 5.0:
            elapsed = current_time - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  {idx} processed ({rate:.1f}/sec)")
            last_report_time = current_time

        # Extraire les features
        try:
            X = extract_features_74(akhbar['text'])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = lr_scaler.transform(X.reshape(1, -1))[0]
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            continue

        # Prédictions LR
        lr_prob = 0.0
        lr_pred = 0
        try:
            lr_prob = lr_model.predict_proba(X_scaled.reshape(1, -1))[0][1]
            lr_pred = 1 if lr_prob >= args.threshold_lr else 0

            lr_results.append({
                'path': akhbar['path'],
                'text_length': akhbar['length'],
                'lr_probability': float(lr_prob),
                'lr_prediction': int(lr_pred)
            })
        except Exception as e:
            pass

        # Prédictions XGBoost
        xgb_prob = 0.0
        xgb_pred = 0
        try:
            xgb_prob = xgb_model.predict_proba(X_scaled.reshape(1, -1))[0][1]
            xgb_pred = 1 if xgb_prob >= args.threshold_xgb else 0

            xgb_results.append({
                'path': akhbar['path'],
                'text_length': akhbar['length'],
                'xgb_probability': float(xgb_prob),
                'xgb_prediction': int(xgb_pred)
            })
        except Exception as e:
            pass

        # Consensus (both models agree on positive)
        if lr_pred == 1 and xgb_pred == 1:
            consensus_hits.append({
                'path': akhbar['path'],
                'text_preview': akhbar['text'][:300],
                'lr_prob': float(lr_prob),
                'xgb_prob': float(xgb_prob),
                'avg_prob': (float(lr_prob) + float(xgb_prob)) / 2
            })

    # Sauvegarder les résultats
    print(f"\nSaving results…")

    with open(LR_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(lr_results, f, ensure_ascii=False, indent=2)
    print(f"✓ LR results → {LR_RESULTS_PATH}")

    with open(XGB_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(xgb_results, f, ensure_ascii=False, indent=2)
    print(f"✓ XGBoost results → {XGB_RESULTS_PATH}")

    # Comparison
    total_processed = idx
    comparison_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_texts_processed': total_processed,
        'lr_positive_count': sum(1 for r in lr_results if r['lr_prediction'] == 1),
        'xgb_positive_count': sum(1 for r in xgb_results if r['xgb_prediction'] == 1),
        'consensus_positive_count': len(consensus_hits),
        'agreement_ratio': len(consensus_hits) / total_processed if total_processed > 0 else 0,
        'top_consensus_hits': sorted(consensus_hits, key=lambda x: x['avg_prob'], reverse=True)[:20]
    }

    with open(COMPARISON_PATH, 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Comparison → {COMPARISON_PATH}")

    # Summary
    elapsed_total = time.time() - start_time
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    print(f"Total texts processed: {total_processed}")
    print(f"Time elapsed: {elapsed_total:.1f}s ({total_processed/elapsed_total:.1f} texts/sec)")
    print(f"LR positive predictions: {comparison_summary['lr_positive_count']}")
    print(f"XGBoost positive predictions: {comparison_summary['xgb_positive_count']}")
    print(f"Consensus (both agree positive): {len(consensus_hits)}")
    print(f"Agreement ratio: {comparison_summary['agreement_ratio']:.1%}")

    if consensus_hits:
        print(f"\nTop 5 consensus hits (highest average confidence):")
        for i, hit in enumerate(comparison_summary['top_consensus_hits'][:5], 1):
            print(f"  {i}. {hit['path']}")
            print(f"     LR: {hit['lr_prob']:.3f} | XGBoost: {hit['xgb_prob']:.3f} | Avg: {hit['avg_prob']:.3f}")

if __name__ == '__main__':
    import re
    main()
