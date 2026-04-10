"""
scan_camelbert.py
─────────────────
Scanne le corpus OpenITI avec le modèle CAMeLBERT fine-tuné.
Détecte les akhbars porteurs du motif maǧnūn ʿāqil, y compris
les textes où la folie n'est pas désignée explicitement.

Pipeline :
  1. Extraire tous les akhbars du corpus (par structure isnad)
  2. Stripping isnad → matn
  3. CAMeLBERT → score de probabilité
  4. Sauvegarder les candidats au-dessus du seuil

Sortie : scan_results.json — triés par score décroissant

Usage :
  python scan_camelbert.py
  python scan_camelbert.py --threshold 0.6 --corpus openiti_corpus/data/
  python scan_camelbert.py --resume          ← reprendre après interruption
"""

import json, pathlib, argparse, re, time
import numpy as np
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).parent.parent))
from isnad_filter import get_matn, split_isnad

BASE      = pathlib.Path(__file__).parent.parent
OUTPUT      = BASE / "scan_results.json"
PROGRESS    = BASE / "scan_progress.json"   # fichier de reprise

def _find_model_dir() -> pathlib.Path:
    """Trouve le dossier modèle même si l'arborescence est imbriquée."""
    candidates = [
        BASE / "camelbert_majnun",
        BASE / "camelbert_majnun" / "camelbert_majnun",
    ]
    # Chercher aussi les checkpoints
    for base in candidates:
        if base.exists():
            # Vérifier si c'est un modèle valide (contient config.json)
            if (base / "config.json").exists():
                return base
            # Chercher dans les sous-dossiers (checkpoints)
            for sub in sorted(base.rglob("config.json")):
                return sub.parent
    return BASE / "camelbert_majnun"

MODEL_DIR = _find_model_dir()

# ── Extraction des akhbars depuis un fichier OpenITI ─────────────────────────
RE_PAGE    = re.compile(r'PageV\d+P\d+')
RE_MS      = re.compile(r'\bms\d+\b')
RE_SPACES  = re.compile(r'[ \t]+')

def clean_text(t):
    t = RE_PAGE.sub('', t)
    t = RE_MS.sub('', t)
    return RE_SPACES.sub(' ', t).strip()

def extract_akhbars(filepath: pathlib.Path) -> list[dict]:
    """
    Extrait les unités narratives d'un fichier OpenITI mARkdown.
    Stratégie : paragraphes # entre 80 et 3000 caractères arabes.
    """
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


# ── Inférence par batch ───────────────────────────────────────────────────────
def predict_batch(texts: list[str], model, tokenizer, device, batch_size=64) -> np.ndarray:
    """Retourne array de probabilités P(positif) pour chaque texte."""
    import torch
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        all_probs.append(probs[:, 1])

    return np.concatenate(all_probs)


# ── Sauvegarde de progression ─────────────────────────────────────────────────
def _save_progress(candidates, done_files, total_akhbars, threshold):
    with open(PROGRESS, 'w', encoding='utf-8') as f:
        json.dump({
            'threshold':     threshold,
            'total_akhbars': total_akhbars,
            'done_files':    done_files,
            'candidates':    candidates,
        }, f, ensure_ascii=False)

def _load_progress():
    if not PROGRESS.exists():
        return None
    with open(PROGRESS, encoding='utf-8') as f:
        return json.load(f)

# ── Scan d'un répertoire OpenITI ─────────────────────────────────────────────
def scan(corpus_dir: pathlib.Path, threshold: float = 0.5, batch_size: int = 64,
         resume: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not MODEL_DIR.exists():
        print(f"ERREUR : modèle introuvable dans {MODEL_DIR}")
        print("         Lancer d'abord : python finetune_camelbert.py")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device : {device}")
    print(f"  Chargement du modèle {MODEL_DIR.name}…")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # Collecter tous les fichiers OpenITI
    all_files = sorted(corpus_dir.rglob('*-ara1'))
    n_files = len(all_files)

    # ── Reprise ───────────────────────────────────────────────────────────────
    all_candidates = []
    total_akhbars  = 0
    done_files     = set()

    if resume and PROGRESS.exists():
        state = _load_progress()
        if state['threshold'] != threshold:
            print(f"  [!] Seuil différent dans la sauvegarde ({state['threshold']:.0%} vs {threshold:.0%})")
            print(f"      Lancer sans --resume pour repartir de zéro")
            return
        all_candidates = state['candidates']
        total_akhbars  = state['total_akhbars']
        done_files     = set(state['done_files'])
        print(f"  Reprise : {len(done_files)}/{n_files} fichiers déjà traités")
        print(f"            {total_akhbars:,} akhbars  |  {len(all_candidates)} candidats")
    elif resume:
        print(f"  [!] Aucune sauvegarde trouvée — démarrage normal")

    # Filtrer les fichiers déjà traités
    remaining = [fp for fp in all_files if fp.name not in done_files]

    print(f"  Fichiers à scanner : {len(remaining)} (/{n_files} total)")
    print(f"  Seuil de détection : {threshold:.0%}")
    print()

    t_start = time.time()
    SAVE_EVERY = 20    # sauvegarde intermédiaire tous les N fichiers

    for fi, fp in enumerate(remaining):
        akhbars = extract_akhbars(fp)
        if not akhbars:
            done_files.add(fp.name)
            continue

        # Stripping isnad → matns
        matns = [get_matn(a) for a in akhbars]

        # Prédiction
        probs = predict_batch(matns, model, tokenizer, device, batch_size)
        total_akhbars += len(akhbars)
        done_files.add(fp.name)

        # Garder les candidats > threshold
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

        # Sauvegarde intermédiaire
        if (fi + 1) % SAVE_EVERY == 0:
            _save_progress(all_candidates, list(done_files), total_akhbars, threshold)

        # Progression tous les 10 fichiers ou à chaque nouveau candidat
        elapsed = time.time() - t_start
        if (fi + 1) % 10 == 0 or n_found > 0:
            fi_global = len(done_files)
            pct     = fi_global / n_files * 100
            rate    = (fi + 1) / elapsed if elapsed > 0 else 0
            eta_s   = (len(remaining) - fi - 1) / rate if rate > 0 else 0
            eta_min = int(eta_s // 60)
            eta_sec = int(eta_s % 60)
            bar     = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}%  {fi_global}/{n_files} fichiers"
                  f"  |  {total_akhbars:,} akhbars"
                  f"  |  {len(all_candidates)} candidats"
                  f"  |  ETA {eta_min}m{eta_sec:02d}s")
            # Détail des candidats trouvés dans ce fichier
            if n_found > 0:
                top = sorted(all_candidates[-n_found:], key=lambda x: -x['score'])
                author = fp.parent.parent.name
                work   = fp.parent.name
                print(f"    ↳ {author} / {work}  ({n_found} candidat{'s' if n_found>1 else ''})")
                for c in top[:3]:
                    # Afficher le matn (après stripping isnad) plutôt que le début du texte brut
                    matn = c['matn'].strip()
                    # Prendre 150 caractères au milieu du texte pour éviter les débuts d'isnad résiduels
                    if len(matn) > 60:
                        preview = matn[20:170].replace('\n', ' ')
                    else:
                        preview = matn.replace('\n', ' ')
                    print(f"      [{c['score']:.2f}] …{preview}…")

    # Sauvegarde finale de progression + nettoyage
    _save_progress(all_candidates, list(done_files), total_akhbars, threshold)

    # Trier par score décroissant
    all_candidates.sort(key=lambda x: -x['score'])

    print(f"\n  ── Résultats ──")
    print(f"  Total akhbars scannés : {total_akhbars}")
    print(f"  Candidats ≥ {threshold:.0%} : {len(all_candidates)}")

    # Distribution par seuil
    for t in [0.9, 0.8, 0.7, 0.6, 0.5]:
        n = sum(1 for c in all_candidates if c['score'] >= t)
        print(f"    ≥ {t:.0%} : {n}")

    # Top 10 auteurs
    from collections import Counter
    authors = Counter(c['author'] for c in all_candidates if c['score'] >= 0.7)
    print(f"\n  Top auteurs (score ≥ 70%) :")
    for author, count in authors.most_common(10):
        print(f"    {count:4d}  {author}")

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump({
            'threshold':      threshold,
            'total_scanned':  total_akhbars,
            'n_candidates':   len(all_candidates),
            'candidates':     all_candidates,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Résultats : {OUTPUT}")
    # Supprimer le fichier de progression une fois terminé
    if PROGRESS.exists():
        PROGRESS.unlink()
        print(f"  (scan_progress.json supprimé)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',    default='openiti_corpus/data/', help='Répertoire OpenITI')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch',     type=int,   default=64)
    parser.add_argument('--resume',    action='store_true', help='Reprendre après interruption')
    args = parser.parse_args()

    corpus_path = BASE / args.corpus
    if not corpus_path.exists():
        print(f"ERREUR : répertoire corpus introuvable : {corpus_path}")
        exit(1)

    scan(corpus_path, threshold=args.threshold, batch_size=args.batch, resume=args.resume)
