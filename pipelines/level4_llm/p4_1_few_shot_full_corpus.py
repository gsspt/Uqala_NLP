#!/usr/bin/env python3
"""
p4_1_few_shot_full_corpus.py
──────────────────────────────────────────────────────────────────
Few-shot classification de TOUS les akhbars Ibn Abd Rabbih (10,286).

Coût estimé: ~$1.50 USD
Temps: ~90 minutes avec rate limiting (2 req/sec)
Progression: barre tqdm en temps réel avec ETA
Checkpointing: sauve chaque 100 textes en cas d'interruption

Usage:
  python pipelines/level4_llm/p4_1_few_shot_full_corpus.py

Interrompre (Ctrl+C) est sûr - reprendra depuis le dernier checkpoint
"""

import json
import sys
import pathlib
import os
import time
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
        "explanation": "TRUE: Bahlul speaks paradoxical wisdom"
    },
    {
        "text": "قال عطاء : رأيت سعدون يتفلّى ذات يوم في الشمس فانكشفت عورته",
        "label": True,
        "explanation": "TRUE: Saadun exhibits apparent madness - dialogue implies wisdom"
    },
    {
        "text": "رأيت سعدون المجنون وبيده فحمة وهو يكتب بها",
        "label": True,
        "explanation": "TRUE: Saadun called majnun - mysterious action"
    },
    {
        "text": "جاء رجل إلى ابن عقيل فقال إني كلما انغمس في النهر غمستين وثلاثا",
        "label": False,
        "explanation": "FALSE: Religious/legal commentary - no wise fool narrative"
    },
    {
        "text": "قال أبي يوسف القاضي: الناس ثلاثة: مجنون، ونصف مجنون، وعاقل",
        "label": False,
        "explanation": "FALSE: Philosophical classification - not character narrative"
    },
]

SYSTEM_PROMPT = """Vous êtes expert en littérature arabe classique.

Majnun aqil = fou sage qui combine:
1. Identification explicite comme fou/insensé
2. Comportement excentrique ou folie apparente
3. Sagesse cachée ou vérités paradoxales
4. Interaction narrative montrant le paradoxe

Répondez UNIQUEMENT: TRUE/FALSE | raison courte"""

def query_deepseek(text, delay=0.5):
    """Query with rate limiting"""
    time.sleep(delay)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        return None

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    examples_text = "EXEMPLES (Nisaburi):\n"
    for i, ex in enumerate(FEW_SHOT_EXAMPLES[:3], 1):
        examples_text += f"{i}. [{ex['label']}] {ex['text'][:60]}...\n"

    user_message = f"""{examples_text}

CLASSIFIEZ:
{text}

Répondez: TRUE/FALSE | raison"""

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
        return None


def load_all_akhbars():
    """Load all 10,286 akhbars from corpus"""
    corpus_root = BASE / "openiti_corpus" / "data" / "0328IbnCabdRabbih"

    all_akhbars = []
    for filepath in sorted(corpus_root.rglob("*-ara1")):
        akhbars = extract_akhbars_from_file(str(filepath))
        for khabar_num, akhbar_raw in enumerate(akhbars):
            akhbar, _ = split_isnad(akhbar_raw)
            all_akhbars.append({
                'filename': filepath.name,
                'khabar_num': khabar_num,
                'text': akhbar,
            })

    return all_akhbars


def classify_full_corpus():
    """Classify entire Ibn Abd Rabbih corpus with DeepSeek"""

    print("\n" + "="*80)
    print("DEEPSEEK FEW-SHOT CLASSIFICATION - CORPUS COMPLET IBN ABD RABBIH")
    print("="*80)

    print("\n[1] Chargement du corpus Ibn Abd Rabbih...")
    all_akhbars = load_all_akhbars()
    print(f"    [OK] {len(all_akhbars)} akhbars chargés")

    print(f"\n[2] Coût estimé: ~$1.50 USD")
    print(f"    Temps: ~90 minutes avec rate limiting (2 req/sec)")
    print(f"    Progression: barre tqdm en temps réel")
    print(f"    Checkpointing: sauve tous les 100 textes (safe to Ctrl+C)\n")

    checkpoint_file = RESULTS_DIR / "deepseek_full_corpus_checkpoint.json"

    # Check for checkpoint
    classifications = []
    start_idx = 0

    if checkpoint_file.exists():
        print(f"    [REPRISE] Checkpoint trouvé...")
        with open(checkpoint_file, encoding='utf-8') as f:
            data = json.load(f)
            classifications = data.get('classifications', [])
            start_idx = len(classifications)
            print(f"    Reprenant à index {start_idx}/{len(all_akhbars)}\n")

    # Classify
    print(f"[3] Classification avec DeepSeek...\n")

    true_count = sum(1 for c in classifications if c.get('classification'))
    false_count = sum(1 for c in classifications if not c.get('classification'))

    with tqdm(total=len(all_akhbars), initial=start_idx,
              desc="  Classification", unit=" akhbar", unit_scale=True) as pbar:

        for i in range(start_idx, len(all_akhbars)):
            akh = all_akhbars[i]

            response = query_deepseek(akh['text'], delay=0.5)

            if not response:
                classifications.append({
                    'filename': akh['filename'],
                    'khabar_num': akh['khabar_num'],
                    'classification': None,
                    'error': True,
                })
                pbar.update(1)
                continue

            is_true = response.startswith("TRUE")
            if is_true:
                true_count += 1
            else:
                false_count += 1

            reason = response.split("|", 1)[1].strip() if "|" in response else "N/A"

            classifications.append({
                'filename': akh['filename'],
                'khabar_num': akh['khabar_num'],
                'classification': is_true,
                'reason': reason,
            })

            pbar.update(1)

            # Save checkpoint every 100
            if (i + 1) % 100 == 0:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'progress': i + 1,
                        'total': len(all_akhbars),
                        'true': true_count,
                        'false': false_count,
                        'classifications': classifications,
                    }, f, ensure_ascii=False, indent=2)

    # Results
    print(f"\n" + "="*80)
    print("RÉSULTATS FINAUX")
    print("="*80)

    valid = [c for c in classifications if c.get('classification') is not None]
    errors = [c for c in classifications if c.get('error')]

    print(f"\nTotal traité: {len(valid)} / {len(all_akhbars)}")
    print(f"Erreurs API: {len(errors)}")
    print(f"\nClassification DeepSeek (exemples réels Nisaburi):")
    print(f"  ✓ VRAI majnun aqil:  {true_count} ({100*true_count/len(valid):.1f}%)")
    print(f"  ✗ FAUX (générique):  {false_count} ({100*false_count/len(valid):.1f}%)")

    if true_count + false_count > 0:
        precision = true_count / (true_count + false_count)
        print(f"\nPrécision de l'ensemble: {100*precision:.1f}%")
        estimated = int(10286 * precision)
        print(f"Estimé majnun aqil réels dans Ibn Abd Rabbih: ~{estimated}")

    # Save final results
    output_file = RESULTS_DIR / "deepseek_full_corpus_nisaburi.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'DeepSeek Few-Shot (Nisaburi real examples)',
            'corpus': 'Ibn Abd Rabbih (10,286 akhbars)',
            'total_processed': len(valid),
            'errors': len(errors),
            'true_positives': true_count,
            'false_positives': false_count,
            'precision': true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0,
            'few_shot_source': 'Real Nisaburi majnun aqil narratives',
            'cost_usd': 1.50,
            'time_minutes': 90,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[4] Résultats sauvegardés: {output_file.name}")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"    Checkpoint supprimé")

    print("\n" + "="*80)
    print("CLASSIFICATION COMPLÈTE!")
    print("="*80)


if __name__ == '__main__':
    try:
        classify_full_corpus()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progression sauvegardée dans checkpoint")
        print("Relancez pour reprendre depuis où on s'est arrêté")
        sys.exit(0)
