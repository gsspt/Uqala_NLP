"""
scan_targeted.py
────────────────
Scan ciblé du corpus OpenITI : auteurs adab/histoire/soufi pertinents
pour la détection du motif maǧnūn ʿāqil.

Inclut une validation automatique du modèle au démarrage :
  → teste sur des positifs/négatifs connus
  → s'arrête si le modèle n'est pas calibré

Sortie : scan_targeted_results.json

Usage :
  python scan_targeted.py
  python scan_targeted.py --threshold 0.7 --resume
"""

import json, pathlib, re, time, sys
import numpy as np
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).parent.parent))
from isnad_filter import get_matn, split_isnad

BASE      = pathlib.Path(__file__).parent.parent
MODEL_DIR = BASE / "camelbert_majnun"

# ── Auteurs ciblés ────────────────────────────────────────────────────────────
# Sélection : grands recueils d'adab, compilations narratives, sources soufies,
# histoires et biographies — période 0200-0900 H.
TARGET_AUTHORS = {
    # Adab narratif
    "0255Jahiz":              "al-Jāḥiẓ (adab encyclopédique)",
    "0276IbnQutaybaDinawari": "Ibn Qutayba (ʿUyūn al-Akhbār, adab)",
    "0328IbnCabdRabbih":      "Ibn ʿAbd Rabbih (ʿIqd al-Farīd)",
    "0414AbuHayyanTawhidi":   "Abū Ḥayyān al-Tawḥīdī (Imtāʿ, Basāʾir)",
    "0850ShihabDinIbshihi":   "al-Ibshīhī (Mustaṭraf — contient ch. ʿuqalāʾ)",

    # Sources soufies (parole marginale / folie sainte)
    "0412Sulami":             "al-Sulamī (Ṭabaqāt al-Ṣūfiyya)",
    "0465IbnHawazinQushayri": "al-Qushayrī (Risāla)",

    # Compilations biographiques et historiques
    "0310Tabari":             "al-Ṭabarī (Tārīkh)",
    "0463KhatibBaghdadi":     "al-Khaṭīb al-Baghdādī (Tārīkh Baghdād)",
    "0571IbnCasakir":         "Ibn ʿAsākir (Tārīkh Dimashq)",
    "0597IbnJawzi":           "Ibn al-Jawzī (Adhkiyāʾ, Ḥamqā, etc.)",
    "0626YaqutHamawi":        "Yāqūt al-Ḥamawī (Muʿjam al-Udabāʾ)",
    "0681IbnKhallikan":       "Ibn Khallikān (Wafayāt al-Aʿyān)",
    "0733Nuwayri":            "al-Nuwayrī (Nihāyat al-Arab)",
    "0748Dhahabi":            "al-Dhahabī (Tārīkh al-Islām, Siyar)",

    # Poésie avec cadre narratif (akhbars de poètes)
    "0360Tabarani":           "al-Ṭabarānī (recueils hadith/biographie)",
    "0542IbnBassamShantarini":"Ibn Bassām (Dhakhīra — adab andalou)",
    "0279Baladhuri":          "al-Balādhurī (Ansāb al-Ashrāf)",
}

# ── Utilitaires texte ─────────────────────────────────────────────────────────
RE_PAGE   = re.compile(r'PageV\d+P\d+')
RE_MS     = re.compile(r'\bms\d+\b')
RE_SPACES = re.compile(r'[ \t]+')

def clean_text(t):
    t = RE_PAGE.sub('', t)
    t = RE_MS.sub('', t)
    return RE_SPACES.sub(' ', t).strip()

def extract_akhbars(filepath):
    import unicodedata
    def count_ar(s):
        return sum(1 for c in s if unicodedata.category(c) == 'Lo' and '\u0600' <= c <= '\u06FF')
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
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
                text = clean_text(' '.join(current))
                if 80 <= count_ar(text) <= 3000:
                    units.append(text)
            rest = line[2:].strip()
            if line.startswith('# |') or rest.startswith('باب') or rest.startswith('فصل'):
                current = []
            else:
                current = [rest]
        else:
            if current:
                text = clean_text(' '.join(current))
                if 80 <= count_ar(text) <= 3000:
                    units.append(text)
            current = []
    if current:
        text = clean_text(' '.join(current))
        if 80 <= count_ar(text) <= 3000:
            units.append(text)
    return units

def predict_batch(texts, model, tokenizer, device, batch_size=64):
    import torch
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, truncation=True, padding='max_length',
                        max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        all_probs.append((e / e.sum(axis=1, keepdims=True))[:, 1])
    return np.concatenate(all_probs)

# ── Validation du modèle ──────────────────────────────────────────────────────
def validate_model(model, tokenizer, device):
    """
    Teste le modèle sur des exemples connus depuis dataset_raw.json.
    Retourne True si le modèle est calibré, False sinon.
    """
    dataset_path = BASE / "dataset_raw.json"
    if not dataset_path.exists():
        print("  [!] dataset_raw.json introuvable — validation ignorée")
        return True

    data = json.load(open(dataset_path, encoding='utf-8'))
    positives = [e for e in data if e['label'] == 1 and e['source'] == 'nisaburi'][:15]
    negatives = [e for e in data if e['label'] == 0][:15]

    texts = [get_matn(e['text_ar']) for e in positives + negatives]
    labels = [1]*len(positives) + [0]*len(negatives)

    probs = predict_batch(texts, model, tokenizer, device, batch_size=32)

    pos_scores = probs[:len(positives)]
    neg_scores = probs[len(positives):]

    mean_pos = float(pos_scores.mean())
    mean_neg = float(neg_scores.mean())

    print(f"\n  ── Validation du modèle ──")
    print(f"  Score moyen positifs (Nīsābūrī) : {mean_pos:.3f}")
    print(f"  Score moyen négatifs            : {mean_neg:.3f}")
    print(f"  Séparation                      : {mean_pos - mean_neg:.3f}")

    # Détail
    for e, s in zip(positives[:3], pos_scores[:3]):
        print(f"  POS [{s:.2f}] {get_matn(e['text_ar'])[:80]}…")
    for e, s in zip(negatives[:3], neg_scores[:3]):
        print(f"  NEG [{s:.2f}] {get_matn(e['text_ar'])[:80]}…")

    if mean_pos < 0.55:
        print(f"\n  ✗ MODÈLE NON CALIBRÉ : score positifs trop bas ({mean_pos:.2f})")
        print(f"    → Relancer finetune_camelbert.py sur Colab avant de scanner")
        return False
    if mean_pos - mean_neg < 0.2:
        print(f"\n  ✗ MODÈLE NON DISCRIMINANT : séparation insuffisante ({mean_pos-mean_neg:.2f})")
        print(f"    → Le modèle ne distingue pas bien positifs/négatifs")
        return False

    print(f"\n  ✓ Modèle validé — lancement du scan")
    return True

# ── Sauvegarde progression ────────────────────────────────────────────────────
def _save(candidates, done_files, total_akhbars, threshold, progress_path):
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump({'threshold': threshold, 'total_akhbars': total_akhbars,
                   'done_files': done_files, 'candidates': candidates}, f, ensure_ascii=False)

# ── Scan ciblé ────────────────────────────────────────────────────────────────
def scan(corpus_dir, threshold=0.7, batch_size=64, resume=False,
         output_path=None, progress_path=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Trouver le modèle
    def find_model():
        for candidate in [MODEL_DIR, MODEL_DIR / "camelbert_majnun"]:
            if (candidate / "config.json").exists():
                return candidate
        for cfg in sorted(MODEL_DIR.rglob("config.json")):
            return cfg.parent
        return MODEL_DIR

    model_path = find_model()
    if not (model_path / "config.json").exists():
        print(f"ERREUR : modèle introuvable dans {model_path}")
        print("         Vérifier le dossier camelbert_majnun/")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device    : {device}")
    print(f"  Modèle    : {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    # ── Validation avant scan ──────────────────────────────────────────────
    if not validate_model(model, tokenizer, device):
        sys.exit(1)

    # ── Collecter les fichiers ciblés ──────────────────────────────────────
    all_files = []
    for author_dir, desc in TARGET_AUTHORS.items():
        dp = corpus_dir / author_dir
        if not dp.exists():
            print(f"  [MANQUANT] {author_dir}")
            continue
        files = sorted(dp.rglob('*-ara1'))
        all_files.extend(files)
        print(f"  {len(files):3d} fichiers  {author_dir:35s} {desc}")

    n_files = len(all_files)
    print(f"\n  Total : {n_files} fichiers à scanner")
    print(f"  Seuil : {threshold:.0%}")

    # ── Reprise ────────────────────────────────────────────────────────────
    all_candidates, total_akhbars, done_files = [], 0, set()
    PROGRESS = progress_path or BASE / "scan_targeted_progress.json"
    OUTPUT   = output_path   or BASE / "scan_targeted_results.json"

    if resume and PROGRESS.exists():
        state = json.load(open(PROGRESS, encoding='utf-8'))
        if state['threshold'] != threshold:
            print(f"  [!] Seuil différent ({state['threshold']:.0%}) — relancer sans --resume")
            return
        all_candidates = state['candidates']
        total_akhbars  = state['total_akhbars']
        done_files     = set(state['done_files'])
        print(f"\n  Reprise : {len(done_files)}/{n_files} déjà traités"
              f"  |  {len(all_candidates)} candidats")
    elif resume:
        print("  [!] Pas de sauvegarde — démarrage normal")

    remaining = [fp for fp in all_files if fp.name not in done_files]
    print(f"\n  Fichiers restants : {len(remaining)}")
    print()

    OUTPUT   = output_path   or BASE / "scan_targeted_results.json"
    PROGRESS = progress_path or BASE / "scan_targeted_progress.json"

    t_start = time.time()
    SAVE_EVERY = 20  # + sauvegarde immédiate si candidat trouvé

    for fi, fp in enumerate(remaining):
        akhbars = extract_akhbars(fp)
        if not akhbars:
            done_files.add(fp.name)
            continue

        matns = [get_matn(a) for a in akhbars]
        probs = predict_batch(matns, model, tokenizer, device, batch_size)
        total_akhbars += len(akhbars)
        done_files.add(fp.name)

        n_before = len(all_candidates)
        for j, (raw, matn, prob) in enumerate(zip(akhbars, matns, probs)):
            if prob >= threshold:
                isnad, _ = split_isnad(raw)
                all_candidates.append({
                    'file':        fp.name,
                    'author':      fp.parent.parent.name,
                    'work':        fp.parent.name,
                    'idx_in_file': j,
                    'score':       float(prob),
                    'isnad':       isnad[:200] if isnad else '',
                    'matn':        matn[:500],
                    'text_raw':    raw[:500],
                })
        n_found = len(all_candidates) - n_before

        if (fi + 1) % SAVE_EVERY == 0:
            _save(all_candidates, list(done_files), total_akhbars, threshold, PROGRESS)

        elapsed = time.time() - t_start
        if (fi + 1) % 10 == 0 or n_found > 0:
            fi_global = len(done_files)
            pct     = fi_global / n_files * 100
            rate    = (fi + 1) / elapsed if elapsed > 0 else 0
            eta_s   = (len(remaining) - fi - 1) / rate if rate > 0 else 0
            eta_min = int(eta_s // 60)
            eta_sec = int(eta_s % 60)
            bar     = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}%  {fi_global}/{n_files}"
                  f"  |  {total_akhbars:,} akhbars"
                  f"  |  {len(all_candidates)} candidats"
                  f"  |  ETA {eta_min}m{eta_sec:02d}s")
            if n_found > 0:
                author = fp.parent.parent.name
                work   = fp.parent.name
                desc   = TARGET_AUTHORS.get(author, author)
                print(f"    ↳ {desc}")
                print(f"       {work}")
                top = sorted(all_candidates[-n_found:], key=lambda x: -x['score'])
                for c in top[:3]:
                    preview = c['matn'].strip()[20:180].replace('\n', ' ')
                    print(f"       [{c['score']:.2f}] …{preview}…")

    # Fin
    _save(all_candidates, list(done_files), total_akhbars, threshold, PROGRESS)
    all_candidates.sort(key=lambda x: -x['score'])

    print(f"\n  ── Résultats finaux ──")
    print(f"  Akhbars scannés : {total_akhbars:,}")
    print(f"  Candidats ≥{threshold:.0%} : {len(all_candidates)}")
    print(f"  Taux            : {len(all_candidates)/max(total_akhbars,1)*100:.2f}%")
    print()
    for t in [0.95, 0.90, 0.80, 0.70]:
        n = sum(1 for c in all_candidates if c['score'] >= t)
        print(f"    ≥ {t:.0%} : {n:4d}")

    from collections import Counter
    print(f"\n  Par auteur (score ≥ 0.80) :")
    authors = Counter(c['author'] for c in all_candidates if c['score'] >= 0.80)
    for author, n in authors.most_common():
        desc = TARGET_AUTHORS.get(author, author)
        print(f"    {n:4d}  {desc}")

    print(f"\n  Top 15 candidats :")
    for c in all_candidates[:15]:
        desc = TARGET_AUTHORS.get(c['author'], c['author'])
        print(f"  [{c['score']:.2f}] {desc}")
        print(f"         {c['matn'].strip()[:150]}")
        print()

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump({'threshold': threshold, 'total_scanned': total_akhbars,
                   'n_candidates': len(all_candidates), 'candidates': all_candidates},
                  f, ensure_ascii=False, indent=2)
    print(f"  Résultats : {OUTPUT}")
    if PROGRESS.exists():
        PROGRESS.unlink()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',    default='openiti_corpus/data/')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--batch',     type=int,   default=64)
    parser.add_argument('--resume',    action='store_true')
    parser.add_argument('--output',    default=None, help='Chemin du fichier de résultats (défaut: dossier du script)')
    parser.add_argument('--progress',  default=None, help='Chemin du fichier de progression (défaut: dossier du script)')
    args = parser.parse_args()

    corpus_path = pathlib.Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = BASE / args.corpus
    if not corpus_path.exists():
        print(f"ERREUR : {corpus_path}")
        sys.exit(1)

    output_path   = pathlib.Path(args.output)   if args.output   else BASE / "scan_targeted_results.json"
    progress_path = pathlib.Path(args.progress) if args.progress else BASE / "scan_targeted_progress.json"

    scan(corpus_path, threshold=args.threshold,
         batch_size=args.batch, resume=args.resume,
         output_path=output_path, progress_path=progress_path)
