"""
score_openiti_lr_bert.py
────────────────────────
Stage 2 du pipeline LR + CAMeLBERT : Applique BERT fine-tuné aux candidats LR.

Prend les candidats détectés par scan_openiti_lr.py et les score avec le modèle
CAMeLBERT fine-tuné (trained_majnun).

Pipeline :
  1. Charge les candidats LR depuis openiti_lr_candidates.json
  2. Applique isnad filtering (get_matn) à chaque candidat
  3. Batch-score avec CAMeLBERT
  4. Combine scores LR et BERT
  5. Trie et sauvegarde les résultats finaux

Sortie :
  scan/openiti_lr_bert_results.json — résultats finaux (LR + BERT)

Usage :
  python scan/score_openiti_lr_bert.py
  python scan/score_openiti_lr_bert.py --threshold_bert 0.5
"""

import json, pathlib, sys, argparse, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE        = pathlib.Path(__file__).parent.parent
LR_CANDS    = BASE / "scan" / "openiti_lr_candidates.json"
BERT_DIR    = BASE / "camelbert_majnun"
OUT         = BASE / "scan" / "openiti_lr_bert_results.json"

# Charger isnad_filter
sys.path.insert(0, str(BASE))
from isnad_filter import get_matn

# ── Chargement BERT ────────────────────────────────────────────────────────────
def load_bert():
    """Charge CAMeLBERT fine-tuné"""
    if not BERT_DIR.exists():
        print(f"ERREUR : modèle BERT introuvable → {BERT_DIR}")
        return None, None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device : {device}")
        print(f"  Chargement CAMeLBERT fine-tuné…")

        tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
        model.to(device)
        model.eval()

        return tokenizer, model, device
    except ImportError:
        print("ERREUR : pip install torch transformers")
        return None, None, None

def bert_score_batch(texts, tokenizer, model, device, batch_size=32, max_length=256):
    """Retourne array de P(motif) pour chaque texte"""
    import torch

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True,
            max_length=max_length, return_tensors='pt'
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        all_probs.append(probs[:, 1])

    return np.concatenate(all_probs)

# ── Main ───────────────────────────────────────────────────────────────────────
def score_lr_candidates(threshold_bert=0.5):
    """Applique BERT aux candidats LR"""

    if not LR_CANDS.exists():
        print(f"ERREUR : candidats LR introuvables → {LR_CANDS}")
        print(f"  Lancer d'abord : python scan/scan_openiti_lr.py")
        return

    print(f"  Chargement candidats LR…")
    lr_data = json.load(open(LR_CANDS, encoding='utf-8'))
    candidates = lr_data['candidates']
    print(f"  {len(candidates)} candidats LR")

    # Charger BERT
    tokenizer, model, device = load_bert()
    if tokenizer is None:
        return

    # Appliquer isnad filtering et BERT scoring
    print(f"\nFiltrage isnad et inférence BERT…")
    matns = []
    for c in candidates:
        matn = get_matn(c['text'])
        if len(matn) < 30:
            matn = c['text']
        matns.append(matn)

    print(f"  BERT batch scoring ({len(matns)} textes)…")
    t0 = time.time()
    bert_scores = bert_score_batch(matns, tokenizer, model, device, batch_size=32)
    elapsed = time.time() - t0
    print(f"  {elapsed:.1f}s ({len(matns)/elapsed:.1f} tex/s)")

    # Combiner LR + BERT
    print(f"\nCombinaisonnage des scores…")
    for c, bert_prob in zip(candidates, bert_scores):
        c['bert_score'] = round(float(bert_prob), 4)
        # Score combiné : 40% LR + 60% BERT
        c['combined_score'] = round(
            c['lr_score'] * 0.4 + c['bert_score'] * 0.6, 4
        )

    # Filtrer et trier
    filtered = [c for c in candidates if c['bert_score'] >= threshold_bert]
    filtered.sort(key=lambda x: -x['combined_score'])

    print(f"\n  ── Résultats ──")
    print(f"  Candidats LR : {len(candidates)}")
    print(f"  Après BERT (≥{threshold_bert:.0%}) : {len(filtered)}")

    # Distribution BERT
    for t in [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
        n = sum(1 for c in filtered if c['bert_score'] >= t)
        print(f"    ≥ {t:.0%} : {n}")

    # Top auteurs
    from collections import Counter
    authors = Counter(c['author'] for c in filtered)
    print(f"\n  Top auteurs (BERT ≥ 70%):")
    for auth, cnt in authors.most_common(10):
        scores = [c['bert_score'] for c in filtered if c['author'] == auth]
        mean_bert = sum(scores) / len(scores)
        mean_lr = np.mean([c['lr_score'] for c in filtered if c['author'] == auth])
        print(f"    {cnt:4d}  bert={mean_bert:.2f}  lr={mean_lr:.2f}  {auth}")

    # Top 20 candidats
    print(f"\n  ── Top 20 Candidats ──")
    for i, c in enumerate(filtered[:20], 1):
        text = c['text'].replace('\n', ' ')[:150]
        print(f"\n  {i:2d}. [lr={c['lr_score']:.2f}  bert={c['bert_score']:.2f}  "
              f"comb={c['combined_score']:.2f}]")
        print(f"      {c['author']} / {c['work']}")
        print(f"      {text}…")

    # Sauvegarder
    output = {
        'threshold_bert': threshold_bert,
        'n_lr_candidates': len(candidates),
        'n_bert_filtered': len(filtered),
        'candidates': filtered,
    }
    json.dump(output, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n  Résultats : {OUT}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold_bert', type=float, default=0.5)
    args = parser.parse_args()

    score_lr_candidates(threshold_bert=args.threshold_bert)
