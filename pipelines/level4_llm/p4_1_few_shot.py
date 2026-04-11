#!/usr/bin/env python3
"""
p4_1_few_shot.py - Few-shot classification using DeepSeek with REAL examples from Nisaburi
"""

import json
import sys
import pathlib
import os
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

if not DEEPSEEK_API_KEY:
    print("ERROR: DEEPSEEK_API_KEY not found in .env")
    sys.exit(1)

BASE = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE / "results" / "0328IbnCabdRabbih"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "src"))
from uqala_nlp.preprocessing.akhbar_extraction import extract_akhbars_from_file
from uqala_nlp.preprocessing.isnad_filter import split_isnad

# REAL EXAMPLES FROM NISABURI CORPUS
FEW_SHOT_EXAMPLES = [
    {
        "text": "أخبرنا محمد قال: أخبرنا الحسن قال: أنشدنا أبو محمد أحمد بن محمد بن إسحاق الجيرنجي بمرو قال: أنشدنا عبدالله بن بهلول: وما عاقل في الناس يحمد أمره ويذكر إلا وهو في الحب أحمق",
        "label": True,
        "explanation": "TRUE: Bahlul (famous fool) speaks paradoxical wisdom - no wise person praised except when in love foolishly"
    },
    {
        "text": "قال عطاء : رأيت سعدون يتفلّى ذات يوم في الشمس فانكشفت عورته فقلتُ له : استر يا أبا عثمان",
        "label": True,
        "explanation": "TRUE: Saadun (famous fool) exhibits apparent madness (exposing himself) - dialogue implies hidden wisdom"
    },
    {
        "text": "رأيت سعدون المجنون وبيده فحمة وهو يكتب بها على الجدار",
        "label": True,
        "explanation": "TRUE: Saadun explicitly called 'al-majnun' - mysterious foolish action implies wisdom"
    },
    {
        "text": "جاء رجل إلى ابن عقيل فقال إني كلما انغمس في النهر غمستين وثلاثا لا أتيقن أنه قد غمسني الماء - ومن المنقول عن أبي الوفاء بن عقيل رضي الله عنه",
        "label": False,
        "explanation": "FALSE: Religious/legal commentary - mentions madness concept but no wise fool narrative"
    },
    {
        "text": "قال أبي يوسف القاضي: الناس ثلاثة: مجنون، ونصف مجنون، وعاقل، فأما المجنون فأنت معه في راحة",
        "label": False,
        "explanation": "FALSE: Philosophical classification - mentions 'majnun' but not narrative about wise fool character"
    },
]

SYSTEM_PROMPT = """Vous êtes expert en littérature arabe classique. Identifiez le "majnun aqil" (fou sage).

Critères pour VRAI majnun aqil:
1. Identification comme fou/insensé (explicite)
2. Comportement excentrique ou folie apparente
3. Sagesse cachée ou vérités paradoxales
4. Interaction narrative montrant le paradoxe
5. Souvent personnage historique (Bahlul, Saadun, etc.)

Répondez UNIQUEMENT: TRUE/FALSE | raison (une phrase)"""

def query_deepseek(text):
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        return None

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    examples_text = "EXEMPLES (corpus Nisaburi réel):
"
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"
{i}. [{ex['label']}] {ex['text'][:70]}...
"

    user_message = f"""{examples_text}
CLASSIFIEZ:
{text}

TRUE/FALSE | raison"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def extract_text_for_prediction(filename, khabar_num):
    corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"
    for filepath in corpus_root.rglob("*-ara1"):
        if filename in filepath.name:
            akhbars = extract_akhbars_from_file(str(filepath))
            if 0 <= khabar_num < len(akhbars):
                akhbar, _ = split_isnad(akhbars[khabar_num])
                return akhbar
    return None


def classify_predictions():
    print("
" + "="*80)
    print("DEEPSEEK FEW-SHOT (exemples réels Nisaburi)")
    print("="*80)

    results_file = RESULTS_DIR / "ensemble_v80_validation_results.json"
    if not results_file.exists():
        print(f"ERROR: {results_file}")
        return None

    with open(results_file, encoding='utf-8') as f:
        results = json.load(f)

    predictions = results['predictions_sample']
    print(f"
[1] Chargement: {len(predictions)} prédictions")

    print(f"[2] Classification avec DeepSeek...")

    classifications = []
    true_count = 0
    false_count = 0

    with tqdm(total=len(predictions)) as pbar:
        for pred in predictions:
            text = extract_text_for_prediction(pred['filename'], pred['khabar_num'])
            if not text:
                pbar.update(1)
                continue

            response = query_deepseek(text)
            if not response:
                pbar.update(1)
                continue

            is_true = response.startswith("TRUE")
            if is_true:
                true_count += 1
            else:
                false_count += 1

            reason = response.split("|", 1)[1].strip() if "|" in response else "N/A"
            classifications.append({
                'ensemble_score': pred['pred_ensemble'],
                'agreement': pred['agreement'],
                'classification': is_true,
                'reason': reason,
            })

            pbar.update(1)

    print(f"
" + "="*80)
    print("RÉSULTATS")
    print("="*80)

    print(f"
Classification DeepSeek:")
    print(f"  VRAI majnun aqil:  {true_count}")
    print(f"  FAUX (générique):  {false_count}")

    if true_count + false_count > 0:
        precision = 100 * true_count / (true_count + false_count)
        print(f"
Précision: {precision:.1f}%")
        estimated = int(10286 * precision / 100)
        print(f"Estimé dans Ibn Abd Rabbih: ~{estimated} majnun aqil vrais")

    output_file = RESULTS_DIR / "deepseek_few_shot_nisaburi.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'true_positives': true_count,
            'false_positives': false_count,
            'precision': true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0,
            'few_shot_source': 'Nisaburi real examples',
            'classifications': classifications,
        }, f, ensure_ascii=False, indent=2)

    print(f"
Résultats: {output_file.name}")

if __name__ == '__main__':
    classify_predictions()
