"""
scan_openiti.py
───────────────
Détection du maǧnūn ʿāqil dans le corpus OpenITI.
Pipeline en 3 phases indépendantes (chacune reprend là où elle s'est arrêtée) :

  Phase 1 — SEGMENT & FILTER  (local, rapide ~10 min)
    Segmente tous les fichiers OpenITI en unités narratives.
    Filtre par lexique de folie + contexte de parole.
    Sortie : scan_candidates.json  (~43k unités)

  Phase 2 — EXTRACT FEATURES  (API DeepSeek, ~2h, ~5$)
    Extrait le schéma actantiel pour chaque candidat.
    Sortie : scan_features.json

  Phase 3 — CLASSIFY & RANK   (local, <1 min)
    Applique le modèle XGBoost entraîné.
    Sortie : scan_results.json  (résultats classés par score)

Usage :
  python scan_openiti.py --phase 1
  python scan_openiti.py --phase 2          # reprend si interrompu
  python scan_openiti.py --phase 3
  python scan_openiti.py --phase all        # tout enchaîner

Options :
  --openiti DIR     Répertoire OpenITI/data  (défaut : openiti_corpus/data)
  --min-score FLOAT Seuil score P(positif) pour inclure dans résultats  (défaut : 0.45)
  --max-results INT Nombre max de résultats dans scan_results.json       (défaut : 500)
  --exclude-train   Exclure les sources utilisées en entraînement
                    (adhkiya, hamqa, cuyun, nisaburi)
"""

import json, os, re, time, pathlib, argparse, threading, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE        = pathlib.Path(__file__).parent
ENV         = BASE / ".env"
CANDIDATES  = BASE / "scan_candidates.json"
FEATURES    = BASE / "scan_features.json"
RESULTS     = BASE / "scan_results.json"
MODEL_PATH  = BASE / "model_features.joblib"

# Sources utilisées en entraînement — à exclure du scan par défaut
TRAIN_SOURCES = {
    "0597IbnJawzi",          # Adhkiyāʾ + Ḥamqā
    "0276IbnQutaybaDinawari", # ʿUyūn al-Aḫbār
    # al-Nīsābūrī n'est pas dans OpenITI sous ce nom mais par sécurité :
    "0406IbnHayyanQurtubi",
}

# ── Lexique de la folie (filtre pré-DeepSeek) ─────────────────────────────────
# Termes désignant explicitement un fou dans un contexte narratif arabe classique
LEXIQUE_FOU = re.compile(
    # Désignations directes
    r'مجنون|مجانين|المجانين'
    r'|معتوه|معاتيه'
    r'|موله|مُولَّه|مولّه'           # muwallah — égaré (mysticisme/amour)
    r'|أحمق|حمقى|الحمق'
    r'|أبله|البله'
    r'|مخبول|مخابيل'
    r'|مغفّل|مغفل|مغافيل'            # mugaffal — simplet hébété
    r'|مصروع'                        # épileptique/possédé
    r'|هائم|هيمان'                   # errant, fou d'amour/mystique
    r'|واله|الواله'                  # égaré d'amour ou de Dieu
    # Syntagmes figés
    r'|عقلاء المجانين|عقلاء الجنون'
    r'|مدعي الجنون|يدّعي الجنون|يدعي الجنون'  # feindre la folie
    r'|يتجانن|تجانن|تجاهل'
    # Expressions périphrastiques
    r'|مصاب في عقله|مصاب بعقله|أصيب بعقله'
    r'|به مسّ|به مس|أصابه مس|مسّه جنون|مسه جنون'
    r'|ذهب عقله|زال عقله|اختل عقله|فقد عقله|فسد عقله'
    r'|لا يعقل|ما يعقل|لا عقل له'
)
VERBE_NARRATION = re.compile(r'قال|فقال|حكى|روى|أخبر|حدثن|أنبأن')

# ── Constantes de parsing ─────────────────────────────────────────────────────
MIN_AR = 80
MAX_AR = 2500
RE_PAGE   = re.compile(r'PageV\d+P\d+')
RE_MS     = re.compile(r'\bms\d+\b')
RE_SPACES = re.compile(r'[ \t]+')

def count_arabic(text):
    return sum(1 for c in text if '\u0600' <= c <= '\u06FF')

def clean_text(raw):
    t = RE_PAGE.sub('', raw)
    t = RE_MS.sub('', t)
    return RE_SPACES.sub(' ', t).strip()

def is_chapter_header(line):
    s = line.lstrip('# ').lstrip()
    return s.startswith('|') or s.startswith('باب') or s.startswith('فصل') or s.startswith('كتاب')

def parse_openiti_file(filepath):
    """Retourne liste de paragraphes nettoyés."""
    try:
        text = filepath.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return []

    lines = text.splitlines()
    content_started = False
    paragraphs, current = [], []

    for line in lines:
        if '#META#Header#End#' in line:
            content_started = True
            continue
        if not content_started:
            continue
        if line.startswith('~~'):
            current.append(line[2:].strip())
        elif line.startswith('# '):
            if current:
                paragraphs.append(clean_text(' '.join(current)))
            current = [] if is_chapter_header(line) else [line[2:].strip()]
        else:
            if current:
                paragraphs.append(clean_text(' '.join(current)))
                current = []

    if current:
        paragraphs.append(clean_text(' '.join(current)))
    return paragraphs

def extract_metadata(filepath):
    """Extrait auteur/titre/date depuis le nom de fichier et le chemin OpenITI."""
    # Structure : data/AuthorDir/AuthorDir.TitleDir/AuthorDir.TitleDir.SourceID-ara1
    parts = filepath.stem.split('.')  # ['0597IbnJawzi', 'Adhkiya', 'JK000910-ara1']
    author_id = parts[0] if parts else '?'
    title_id  = parts[1] if len(parts) > 1 else '?'
    # Extraire date depuis l'ID auteur (4 premiers chiffres)
    date_match = re.match(r'^(\d{4})', author_id)
    date = int(date_match.group(1)) if date_match else 0
    return {
        'author_id': author_id,
        'title_id':  title_id,
        'date_hijri': date,
        'file': filepath.name,
    }

# ── PHASE 1 : Segmentation + filtrage lexical ─────────────────────────────────
def phase1_segment(openiti_dir, exclude_train):
    print("\n═══ PHASE 1 : Segmentation et filtrage lexical ═══")
    data_dir = pathlib.Path(openiti_dir)
    files    = [f for f in data_dir.rglob('*ara1')
                if f.is_file() and not f.suffix == '.yml'
                and not f.name.endswith('.yml')]

    if exclude_train:
        files = [f for f in files
                 if not any(src in str(f) for src in TRAIN_SOURCES)]
        print(f"  Sources d'entraînement exclues : {', '.join(TRAIN_SOURCES)}")

    print(f"  Fichiers à scanner : {len(files):,}")

    candidates = []
    n_files_done = 0
    n_paras_total = 0
    n_cands = 0

    for fp in files:
        paras = parse_openiti_file(fp)
        meta  = extract_metadata(fp)
        n_files_done += 1

        for i, p in enumerate(paras):
            n_ar = count_arabic(p)
            if n_ar < MIN_AR or n_ar > MAX_AR:
                continue
            n_paras_total += 1
            # Filtre lexique folie + contexte narratif
            if LEXIQUE_FOU.search(p) and VERBE_NARRATION.search(p):
                n_cands += 1
                candidates.append({
                    'id':         f"{meta['author_id']}_{meta['title_id']}_{i:05d}",
                    'text_ar':    p,
                    'author_id':  meta['author_id'],
                    'title_id':   meta['title_id'],
                    'date_hijri': meta['date_hijri'],
                    'file':       meta['file'],
                    'para_idx':   i,
                })

        if n_files_done % 500 == 0:
            print(f"  {n_files_done:,}/{len(files):,} fichiers | {n_paras_total:,} narratifs | {n_cands:,} candidats")

    print(f"\n  Terminé : {n_paras_total:,} paragraphes narratifs → {n_cands:,} candidats ({n_cands/max(n_paras_total,1):.1%})")
    with open(CANDIDATES, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print(f"  Sortie : {CANDIDATES}")
    return candidates

# ── PHASE 2 : Extraction features DeepSeek ────────────────────────────────────
def _load_env():
    if not ENV.exists():
        return
    for line in ENV.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

MODEL_API = "deepseek-chat"
API_URL   = "https://api.deepseek.com/chat/completions"
WORKERS   = 15
SAVE_EVERY = 50

SYSTEM_PROMPT = """Tu es un expert en littérature arabe classique.
On t'envoie un extrait de texte arabe (un akhbar ou anecdote courte).
Réponds UNIQUEMENT avec un objet JSON valide, sans markdown.

{
  "locuteur_statut": "fou" | "marginal" | "sage" | "autorite" | "neutre" | "?",
  "interlocuteur_autorite": true | false,
  "type_autorite": "politique" | "religieux" | "savant" | "riche" | "foule" | "aucun" | "?",
  "prise_de_parole": "directe" | "detournee" | "silence_expressif" | "acte_symbolique" | "absente",
  "registre_verite": "coranique" | "proverbial" | "absurde" | "litteral" | "ironique" | "narratif_neutre" | "?",
  "structure_chute": "retournement" | "confirmation_autorite" | "ambigue" | "absence" | "?",
  "reaction_autorite": "desequilibree" | "renforcee" | "absente" | "indecidable" | "?",
  "presence_folie_explicite": true | false,
  "presuppose_sagesse_cachee": true | false,
  "note": "observation libre max 1 phrase"
}"""

def call_api(text_ar, entry_id, api_key):
    payload = {
        "model": MODEL_API,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text_ar[:3000]},
        ],
        "temperature": 0.0,
        "max_tokens": 350,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(5):
        try:
            r = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt < 4:
                time.sleep(8 * (attempt + 1))
            else:
                return None
    return None

def phase2_extract():
    _load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERREUR : DEEPSEEK_API_KEY manquante dans .env")
        return

    if not CANDIDATES.exists():
        print("ERREUR : scan_candidates.json manquant — lancer phase 1 d'abord")
        return

    print("\n═══ PHASE 2 : Extraction features (DeepSeek) ═══")
    with open(CANDIDATES, encoding='utf-8') as f:
        candidates = json.load(f)

    # Resume
    done = {}
    if FEATURES.exists():
        with open(FEATURES, encoding='utf-8') as f:
            for e in json.load(f):
                done[e['id']] = e

    todo = [e for e in candidates if e['id'] not in done]
    print(f"  Candidats total   : {len(candidates):,}")
    print(f"  Déjà traités      : {len(done):,}")
    print(f"  À traiter         : {len(todo):,}")
    print(f"  Workers           : {WORKERS}")

    _lock   = threading.Lock()
    _counter = [0]

    def process(entry):
        features = call_api(entry['text_ar'], entry['id'], api_key)
        result   = {**entry, 'features': features}
        with _lock:
            done[entry['id']] = result
            _counter[0] += 1
            n = len(done)
            if _counter[0] % SAVE_EVERY == 0:
                _save_features(done)
                print(f"  [{n:,}/{len(candidates):,}] sauvegarde…")
            elif _counter[0] % 200 == 0:
                print(f"  [{n:,}/{len(candidates):,}]")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process, e): e for e in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  [ERR] {e}")

    _save_features(done)
    n_ok  = sum(1 for e in done.values() if e.get('features'))
    n_err = sum(1 for e in done.values() if not e.get('features'))
    print(f"\n  Terminé : {n_ok:,} succès, {n_err:,} erreurs")
    print(f"  Sortie  : {FEATURES}")

def _save_features(done_dict):
    with open(FEATURES, 'w', encoding='utf-8') as f:
        json.dump(list(done_dict.values()), f, ensure_ascii=False, indent=2)

# ── PHASE 3 : Classification et ranking ───────────────────────────────────────
FEATURE_ENCODINGS = {
    "locuteur_statut":           {"fou":0,"marginal":1,"sage":2,"autorite":3,"neutre":4,"?":2},
    "interlocuteur_autorite":    {True:1,False:0,"true":1,"false":0},
    "type_autorite":             {"politique":0,"religieux":1,"savant":2,"riche":3,"foule":4,"aucun":5,"?":5},
    "prise_de_parole":           {"directe":0,"detournee":1,"silence_expressif":2,"acte_symbolique":3,"absente":4},
    "registre_verite":           {"coranique":0,"proverbial":1,"absurde":2,"litteral":3,"ironique":4,"narratif_neutre":5,"?":5},
    "structure_chute":           {"retournement":0,"confirmation_autorite":1,"ambigue":2,"absence":3,"?":3},
    "reaction_autorite":         {"desequilibree":0,"renforcee":1,"absente":2,"indecidable":3,"?":2},
    "presence_folie_explicite":  {True:1,False:0,"true":1,"false":0},
    "presuppose_sagesse_cachee": {True:1,False:0,"true":1,"false":0},
}
FEATURE_KEYS = list(FEATURE_ENCODINGS.keys())

def encode(entry):
    f = entry.get('features')
    if not f:
        return None
    row = []
    for key in FEATURE_KEYS:
        val = f.get(key)
        m   = FEATURE_ENCODINGS[key]
        row.append(float(m.get(val, m.get(str(val).lower() if val is not None else '?',
                                          list(m.values())[-1]))))
    return row

def phase3_classify(min_score, max_results):
    import numpy as np
    try:
        import joblib
    except ImportError:
        print("ERREUR : pip install joblib")
        return

    if not FEATURES.exists():
        print("ERREUR : scan_features.json manquant — lancer phase 2 d'abord")
        return
    if not MODEL_PATH.exists():
        print("ERREUR : model_features.joblib manquant — lancer train_classifier.py d'abord")
        return

    print("\n═══ PHASE 3 : Classification et ranking ═══")
    model = joblib.load(MODEL_PATH)

    with open(FEATURES, encoding='utf-8') as f:
        data = json.load(f)

    valid = [(e, encode(e)) for e in data if e.get('features')]
    valid = [(e, r) for e, r in valid if r is not None]
    print(f"  Entrées avec features : {len(valid):,}")

    X = np.array([r for _, r in valid])
    entries = [e for e, _ in valid]

    proba = model.predict_proba(X)[:, 1]  # P(maǧnūn ʿāqil)

    # Assembler résultats
    results = []
    for entry, score in zip(entries, proba):
        if score >= min_score:
            f = entry.get('features', {})
            results.append({
                'score':      round(float(score), 4),
                'id':         entry['id'],
                'author_id':  entry['author_id'],
                'title_id':   entry['title_id'],
                'date_hijri': entry['date_hijri'],
                'file':       entry['file'],
                'text_ar':    entry['text_ar'],
                'features': {
                    'locuteur_statut':          f.get('locuteur_statut'),
                    'presence_folie_explicite': f.get('presence_folie_explicite'),
                    'structure_chute':          f.get('structure_chute'),
                    'presuppose_sagesse_cachee':f.get('presuppose_sagesse_cachee'),
                    'registre_verite':          f.get('registre_verite'),
                    'prise_de_parole':          f.get('prise_de_parole'),
                    'note':                     f.get('note', ''),
                },
            })

    results.sort(key=lambda x: -x['score'])
    results = results[:max_results]

    # Stats par auteur/siècle
    from collections import Counter
    by_author  = Counter(r['author_id'] for r in results)
    by_century = Counter(r['date_hijri'] // 100 for r in results)

    print(f"\n  Résultats (score ≥ {min_score}) : {len(results):,}")
    print(f"\n  Top 15 auteurs :")
    for author, n in by_author.most_common(15):
        print(f"    {author:45s} {n:4d}")
    print(f"\n  Distribution par siècle (hijri) :")
    for century in sorted(by_century):
        bar = "█" * by_century[century]
        print(f"    {century}e s.  {bar}  ({by_century[century]})")

    print(f"\n  Top 20 résultats :")
    for r in results[:20]:
        print(f"    [{r['score']:.3f}] {r['author_id']:30s} | {r['title_id']:25s} | "
              f"statut={r['features']['locuteur_statut']:8s} | "
              f"chute={r['features']['structure_chute']}")

    output = {
        'params': {'min_score': min_score, 'max_results': max_results},
        'stats': {
            'n_candidates':  len(data),
            'n_with_features': len(valid),
            'n_results':     len(results),
            'by_author':     dict(by_author.most_common(30)),
            'by_century':    {str(k*100): v for k, v in sorted(by_century.items())},
        },
        'results': results,
    }
    with open(RESULTS, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Sortie : {RESULTS}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all',
                        choices=['1','2','3','all'],
                        help='Phase à exécuter (défaut : all)')
    parser.add_argument('--openiti', default='openiti_corpus/data',
                        help='Répertoire OpenITI/data')
    parser.add_argument('--min-score', type=float, default=0.45)
    parser.add_argument('--max-results', type=int, default=500)
    parser.add_argument('--exclude-train', action='store_true', default=True,
                        help='Exclure sources utilisées en entraînement (défaut : oui)')
    parser.add_argument('--include-train', dest='exclude_train', action='store_false')
    args = parser.parse_args()

    if args.phase in ('1', 'all'):
        phase1_segment(args.openiti, args.exclude_train)
    if args.phase in ('2', 'all'):
        phase2_extract()
    if args.phase in ('3', 'all'):
        phase3_classify(args.min_score, args.max_results)

if __name__ == '__main__':
    main()
