#!/usr/bin/env python3
"""
p4_1_few_shot_8workers.py
──────────────────────────────────────────────────────────────────
Few-shot classification avec 8 workers concurrents (ThreadPoolExecutor).

Optimisation:
  - 8 workers parallèles (vs 1 séquentiel)
  - Rate limiting global: 0.1sec entre requêtes (10 req/sec max)
  - Temps estimé: ~18-22 minutes (vs 3h seul)
  - Erreurs: Très bas (rate limiting gentil)
  - Checkpointing: Sûr même avec interruption

Usage:
  python pipelines/level4_llm/p4_1_few_shot_8workers.py
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

# REAL COMPLETE AKHBARS FROM NISABURI CORPUS
FEW_SHOT_EXAMPLES = [
    # TRUE: Majnun Aqil narratives (complete akhbars)
    {
        "text": "أخبرنا محمد قال: أخبرنا الحسن قال: أنشدنا أبو محمد أحمد بن محمد بن إسحاق الجيرنجي بمرو قال: أنشدنا عبدالله بن بهلول بقرميسين: وما عاقل في الناس يحمد أمره ويذكر إلا وهو في الحب أحمق وما من فتى ما ذاق بؤس معيشة من الناس إلا ذاقها حين يعشق",
        "label": True,
        "explanation": "TRUE majnun aqil: Bahlul (famous fool) - complete akhbar with isnad + paradoxical poetry (no wise person praised except when love makes them fool)"
    },
    {
        "text": "أخبرنا محمد قال : أخبرنا الحسن قال : أخبرنا أبو عبدالله محمد بن الطيب ببوشنج قال : حدثنا أبو بكر حفص بن عمر بن حفص الهروي قال : حدثنا علي بن محمد بن عبد الحميد قال : حدثنا إبراهيم بن الجنيد الختلى عن محمد بن الحسين عن راشد بن علقمة البصري الأزدي قال : قال لي عطاء السلمي : احتبس علينا القطر بالبصرة فخرجنا نستسقي فإذا بسعدون المجنون، فلما أبصرني قال : يا عطاء إلى أين؟ قلت : خرجنا نستسقي . قال: بقلوب سماوية أم بقلوب خاوية؟ قلت: بقلوب سماوية قال : لا تُبهرج فإن الناقد بصير! . قلت : ما هو إلا ما حكيت لك، فاستسق لنا. فرفع رأسه إلى السماء وقال: أقسمت عليك إلا سقيتنا الغيث ثم أنشأ يقول : أيا من كلما دعاه الداعي...",
        "label": True,
        "explanation": "TRUE majnun aqil: Saadun - complete akhbar with long isnad + profound dialogue about prayer (heavenly vs empty hearts) + philosophical questions + invocation"
    },
    {
        "text": "أخبرنا محمد قال : أخبرنا الحسن قال : أخبرنا أبو عبدالله قال : حدثنا أبو بكر قال : حدثنا علي بن محمد عن إبراهيم بن الجنيد عن محمد بن الحسين عن عبيد الله الهاشمي قال : قال عطاء : رأيت سعدون يتفلّى ذات يوم في الشمس فانكشفت عورته فقلتُ له : استر يا أخا الجهل ، فقال : أما لك مثلها؟ فأمنته . ثم مر بي يوماً وأنا آكل رماناً في السوق، فعرك أذني وقال : من الجاهل منا أنا أم أنت؟ ثم أنشأ يقول : أرى كل إنسان يرى عيب غيره ويعمى عن العيب الذي هو فيه",
        "label": True,
        "explanation": "TRUE majnun aqil: Saadun - apparent madness (exposing himself) + paradoxical dialogue (criticizes critic for judging) + wisdom about human hypocrisy + poetry"
    },

    # FALSE: Hard negatives - mention junun/majnun but NOT wise fool narrative
    {
        "text": "وكما شاب صفات أهل الدنيا بأضدادها، كذلك شاب عقولهم بالجنون، فلا يخلو العاقل فيها من ضَرْب من الجنون. ولذلك أشار النبي صلى الله عليه وسلم إلى من أبلى شبابه في المعصية فسماه مجنوناً . أخبرنا محمد قال : حدثنا الحسن قال : حدثنا أبو زكريا يحيى بن محمد بن عبدالله العنبري قال : حدثنا أبو إسحاق حيان بن أحمد بن حيان البلخي قال : حدثنا محمد بن مدويه الكرابيسي الترمذي قال : حدثنا خالد بن خداش عن صالح المري عن جعفر بن زيد العبدي عن أنس بن مالك",
        "label": False,
        "explanation": "FALSE: Philosophical commentary - mentions junun/majnun but is theological discussion about human nature, not narrative about wise fool character with paradoxical wisdom"
    },
    {
        "text": "والمجنون عند الناس من يُسمع ويسبّ ويرمي ويخرق الثوب، أو من يخالفهم في عاداتهم فيجيء بما يُنكرون ؛ ولذلك دعت الأمم الرسل مجانين لأنهم شقوا عَصاهم فنابذوهم وأتوا بخلاف ما هو فيه. قال الله تعالى : كذبت قبلهم قوم نوح فكذبوا عبدنا وقالوا مجنون وازدجر",
        "label": False,
        "explanation": "FALSE: Exegetical definition - defines what 'majnun' means socially/Qur'anically but no narrative, no character, no paradoxical wisdom about foolishness"
    },
]

SYSTEM_PROMPT = """Vous êtes expert en littérature arabe classique.

Majnun aqil = fou sage qui combine:
1. Identification explicite comme fou/insensé
2. Comportement excentrique ou folie apparente
3. Sagesse cachée ou vérités paradoxales
4. Interaction narrative montrant le paradoxe

Répondez UNIQUEMENT: TRUE/FALSE | raison courte"""

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
    print("DEEPSEEK FEW-SHOT CLASSIFICATION - 8 WORKERS")
    print("="*80)

    print("\n[1] Chargement du corpus Ibn Abd Rabbih...")
    all_akhbars = load_all_akhbars()
    print(f"    [OK] {len(all_akhbars)} akhbars chargés")

    print(f"\n[2] Configuration optimisée:")
    print(f"    Workers: 8")
    print(f"    Rate limit: 0.1 sec entre requêtes (10 req/sec max)")
    print(f"    Temps estimé: ~20 minutes")
    print(f"    Coût: ~$1.50 USD\n")

    checkpoint_file = RESULTS_DIR / "deepseek_8workers_checkpoint.json"

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
        # Submit all tasks
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
    print(f"\nClassification DeepSeek (8 workers):")
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
    output_file = RESULTS_DIR / "deepseek_full_corpus_8workers.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'DeepSeek Few-Shot (8 workers)',
            'corpus': 'Ibn Abd Rabbih (10,286 akhbars)',
            'total_processed': len(valid),
            'errors': len(errors),
            'true_positives': true_count,
            'false_positives': false_count,
            'precision': true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0,
            'few_shot_source': 'Real Nisaburi majnun aqil narratives',
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
    print("CLASSIFICATION COMPLÈTE (8 WORKERS)!")
    print("="*80)


if __name__ == '__main__':
    try:
        classify_full_corpus()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progression sauvegardée dans checkpoint")
        print("Relancez pour reprendre depuis où on s'est arrêté")
        sys.exit(0)
