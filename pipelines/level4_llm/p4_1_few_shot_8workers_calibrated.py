#!/usr/bin/env python3
"""
p4_1_few_shot_8workers_calibrated.py
──────────────────────────────────────────────────────────────────
Few-shot classification WITH CALIBRATED EXAMPLES from dataset_raw.json

SELECTED EXAMPLES (best from Nisaburi corpus):
- TRUE: Oyais (fool paradox), Qais & Layla (love paradox)
- FALSE: Philosophical definitions, no narrative

8 workers parallèles, rate limiting 0.1sec (10 req/sec max)

Usage:
  python pipelines/level4_llm/p4_1_few_shot_8workers_calibrated.py
"""

import json
import sys
import pathlib
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
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

# CALIBRATED EXAMPLES FROM DATASET_RAW.JSON (Nisaburi corpus)
FEW_SHOT_EXAMPLES = [
    # TRUE: Majnun Aqil - clear narratives with paradox
    {
        "text": "قدمت الكوفة ولم يكن لي هم إلا أويس القرني أطلبه وأسأل عنه حتى سقطت عليه جالساً وحده على شاطئ الفرات نصف النهار يتوضأ ويغسل ثوبه، فعرفته بالنعت الذي نعت لي، فإذا رجل لحيم آدم شديد الأدمة محلوق الرأس كث اللحية عليه إزار من صوف ورداء من صوف كريه الوجه مهيب المنظر جداً. فقلت: ومن أين عرفتني وعرفت اسمي واسم أبي ووالله ما رأيتك قط قبل اليوم؟ فقال: نبأني العليم الخبير، عرفت روحي روحك حين كلمت نفسي نفسك، إن الأرواح لها أنفس كأنفس الأحياء.",
        "label": True,
        "explanation": "TRUE majnun aqil: Oyais - appears crude/foolish in aspect (dark, unkempt) but demonstrates paradoxical spiritual knowledge (recognizes stranger through spiritual insight, not intellect)."
    },
    {
        "text": "وما عاقل في الناس يحمد أمره ويذكر إلا وهو في الحب أحمق وما من فتى ما ذاق بؤس معيشة من الناس إلا ذاقها حين يعشق",
        "label": True,
        "explanation": "TRUE majnun aqil: Bahlul - paradoxical wisdom about love: no wise person escapes becoming fool when in love. Wisdom expressed through recognition of this paradox."
    },
    {
        "text": "احتبس علينا القطر بالبصرة فخرجنا نستسقي فإذا بسعدون المجنون، فلما أبصرني قال : يا عطاء إلى أين؟ قلت : خرجنا نستسقي . قال: بقلوب سماوية أم بقلوب خاوية؟ قلت: بقلوب سماوية قال : لا تُبهرج فإن الناقد بصير! . قلت : ما هو إلا ما حكيت لك، فاستسق لنا. فرفع رأسه إلى السماء وقال: أقسمت عليك إلا سقيتنا الغيث",
        "label": True,
        "explanation": "TRUE majnun aqil: Saadoun (identified as majnun) - profound dialogue about prayer: criticizes false spirituality, demonstrates true wisdom through paradoxical questions (heavenly vs empty hearts)."
    },
    {
        "text": "قيل لليلى : حبك للمجنون أكثر من حبه لك؟. فقالت: بل حبي له . قيل : وكيف؟ قالت : لأن حبه لي كان مشهوراً وحبي له كان مستوراً.",
        "label": True,
        "explanation": "TRUE majnun aqil: Qais (called majnun) - paradox of love: his madness public/visible, hers hidden/veiled. Wisdom in understanding the distinction between apparent and hidden states."
    },

    # FALSE: Philosophical/definitional texts - NO narrative, NO character
    {
        "text": "والمجنون عند الناس من يُسمع ويسبّ ويرمي ويخرق الثوب، أو من يخالفهم في عاداتهم فيجيء بما يُنكرون ؛ ولذلك دعت الأمم الرسل مجانين لأنهم شقوا عَصاهم فنابذوهم وأتوا بخلاف ما هو فيه. قال الله تعالى : كذبت قبلهم قوم نوح فكذبوا عبدنا وقالوا مجنون وازدجر",
        "label": False,
        "explanation": "FALSE: Exegetical definition - defines WHAT majnun means socially (someone who breaks norms) but no NARRATIVE, no CHARACTER with paradoxical wisdom. Pure theological commentary."
    },
    {
        "text": "وكما شاب صفات أهل الدنيا بأضدادها، كذلك شاب عقولهم بالجنون، فلا يخلو العاقل فيها من ضَرْب من الجنون. ولذلك أشار النبي صلى الله عليه وسلم إلى من أبلى شبابه في المعصية فسماه مجنوناً.",
        "label": False,
        "explanation": "FALSE: Philosophical commentary - discusses junun as metaphorical concept (everyone has some madness) but NO NARRATIVE about a specific fool character, NO paradoxical wisdom demonstration."
    },
]

SYSTEM_PROMPT = """Vous êtes expert en littérature arabe classiques.

Majnun aqil (fou sage) = un personnage qui:
1. Est explicitement identifié/reconnu comme fou, majnun, ou excentrique
2. Mais démontre sagesse, perspicacité, ou vérités profondes EN PARADOXE avec cette folie
3. À travers UN RÉCIT NARRATIF avec dialogue, action, ou poésie
4. Où le paradoxe fou-sage est le point clé du passage

Répondez UNIQUEMENT: TRUE/FALSE | raison en 1-2 phrases"""

# GLOBAL RATE LIMITING (thread-safe)
class RateLimiter:
    def __init__(self, min_interval=0.1):
        self.min_interval = min_interval
        self.last_request = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request = time.time()

rate_limiter = RateLimiter(min_interval=0.1)  # 10 req/sec max


def query_deepseek(text):
    """Query with thread-safe rate limiting"""
    rate_limiter.wait()

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    examples_text = "EXEMPLES CALIBRÉS (Nisaburi corpus):\n\n"
    examples_text += "VRAI majnun aqil:\n"
    for i, ex in enumerate(FEW_SHOT_EXAMPLES[:3], 1):
        examples_text += f"{i}. {ex['text'][:80]}...\n"
    examples_text += "\nFAUX (définitions philosophiques, pas de récit):\n"
    for i, ex in enumerate(FEW_SHOT_EXAMPLES[3:], 1):
        examples_text += f"{i}. {ex['text'][:80]}...\n"

    user_message = f"""{examples_text}

CLASSIFIEZ CE TEXTE:
{text}

Répondez: TRUE/FALSE | raison courte"""

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
            _, akhbar = split_isnad(akhbar_raw)
            all_akhbars.append({
                'filename': filepath.name,
                'khabar_num': khabar_num,
                'text': akhbar,
                'idx': len(all_akhbars),
            })

    return all_akhbars


def process_akhbar(akh):
    """Process single akhbar - returns result dict"""
    response = query_deepseek(akh['text'])

    if not response:
        return {
            'idx': akh['idx'],
            'filename': akh['filename'],
            'khabar_num': akh['khabar_num'],
            'text': akh['text'],
            'classification': None,
            'error': True,
        }

    is_true = response.startswith("TRUE")
    reason = response.split("|", 1)[1].strip() if "|" in response else "N/A"

    return {
        'idx': akh['idx'],
        'filename': akh['filename'],
        'khabar_num': akh['khabar_num'],
        'text': akh['text'],
        'classification': is_true,
        'reason': reason,
        'error': False,
    }


def classify_full_corpus():
    """Classify entire corpus with 8 workers"""

    print("\n" + "="*80)
    print("DEEPSEEK FEW-SHOT CLASSIFICATION (CALIBRATED EXAMPLES)")
    print("="*80)

    print("\n[1] Chargement du corpus Ibn Abd Rabbih...")
    all_akhbars = load_all_akhbars()
    print(f"    [OK] {len(all_akhbars)} akhbars chargés")

    print(f"\n[2] Configuration optimisée:")
    print(f"    Workers: 8")
    print(f"    Rate limit: 0.1 sec entre requêtes (10 req/sec max)")
    print(f"    Temps estimé: ~20 minutes")
    print(f"    Coût: ~$1.50 USD\n")

    checkpoint_file = RESULTS_DIR / "deepseek_8workers_calibrated_checkpoint.json"

    # Check for checkpoint
    classifications = {}
    start_idx = 0

    if checkpoint_file.exists():
        print(f"    [REPRISE] Checkpoint trouvé...")
        with open(checkpoint_file, encoding='utf-8') as f:
            data = json.load(f)
            classifications = {int(k): v for k, v in data.get('classifications', {}).items()}
            start_idx = max(classifications.keys()) + 1 if classifications else 0
            print(f"    Reprenant à index {start_idx}/{len(all_akhbars)}\n")

    # Classify with 8 workers
    print(f"[3] Classification avec 8 workers...\n")

    remaining_akhbars = all_akhbars[start_idx:]
    true_count = sum(1 for c in classifications.values() if c.get('classification'))
    false_count = sum(1 for c in classifications.values() if not c.get('classification'))

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(process_akhbar, akh): akh['idx']
            for akh in remaining_akhbars
        }

        with tqdm(total=len(remaining_akhbars), desc="  Classification",
                  unit=" akhbar", unit_scale=True) as pbar:

            for future in as_completed(future_to_idx):
                result = future.result()
                idx = result['idx']

                classifications[idx] = result

                if not result.get('error'):
                    if result['classification']:
                        true_count += 1
                    else:
                        false_count += 1

                pbar.update(1)

                # Save checkpoint every 100
                if len(classifications) % 100 == 0:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'progress': len(classifications),
                            'total': len(all_akhbars),
                            'true': true_count,
                            'false': false_count,
                            'classifications': classifications,
                        }, f, ensure_ascii=False, indent=2)

    # Results
    print(f"\n" + "="*80)
    print("RÉSULTATS FINAUX")
    print("="*80)

    valid = [c for c in classifications.values() if not c.get('error')]
    errors = [c for c in classifications.values() if c.get('error')]

    print(f"\nTotal traité: {len(valid)} / {len(all_akhbars)}")
    print(f"Erreurs API: {len(errors)}")
    print(f"\nClassification DeepSeek (8 workers - CALIBRATED):")
    print(f"  ✓ VRAI majnun aqil:  {true_count} ({100*true_count/len(valid):.1f}%)")
    print(f"  ✗ FAUX (générique):  {false_count} ({100*false_count/len(valid):.1f}%)")

    if true_count + false_count > 0:
        precision = true_count / (true_count + false_count)
        print(f"\nPrécision de l'ensemble: {100*precision:.1f}%")
        estimated = int(10286 * precision)
        print(f"Estimé majnun aqil réels dans Ibn Abd Rabbih: ~{estimated}")

    # Collect positives with full text + reason for analysis
    positives_detail = sorted(
        [c for c in classifications.values() if c.get('classification') and not c.get('error')],
        key=lambda x: x['idx']
    )

    # Save final results
    output_file = RESULTS_DIR / "deepseek_full_corpus_8workers_calibrated.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'DeepSeek Few-Shot (8 workers - CALIBRATED)',
            'corpus': 'Ibn Abd Rabbih (10,286 akhbars)',
            'extraction_method': 'akhbar_extraction_v2_smart',
            'calibration': 'Nisaburi corpus examples (Oyais, Qais & Layla)',
            'total_processed': len(valid),
            'errors': len(errors),
            'true_positives': true_count,
            'false_positives': false_count,
            'precision': true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0,
            'few_shot_source': 'Dataset_raw.json (Nisaburi corpus) - Calibrated selection',
            'workers': 8,
            'rate_limit_sec': 0.1,
            'cost_usd': 1.50,
            'time_minutes_estimated': 20,
            'positives': [
                {
                    'idx': c['idx'],
                    'filename': c['filename'],
                    'khabar_num': c['khabar_num'],
                    'text': c['text'],
                    'reason': c.get('reason', 'N/A'),
                }
                for c in positives_detail
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[4] Résultats sauvegardés: {output_file.name}")
    print(f"    {true_count} positifs avec texte intégral + raison DeepSeek")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"    Checkpoint supprimé")

    print("\n" + "="*80)
    print("CLASSIFICATION COMPLÈTE (8 WORKERS - CALIBRATED)!")
    print("="*80)


if __name__ == '__main__':
    try:
        classify_full_corpus()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progression sauvegardée dans checkpoint")
        print("Relancez pour reprendre depuis où on s'est arrêté")
        sys.exit(0)
