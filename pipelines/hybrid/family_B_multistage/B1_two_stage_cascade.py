"""
Pipeline Hybride B1 — Cascade deux étages (rapide → précis).

Principe : Un premier modèle léger filtre 90% des textes rapidement.
Un second modèle lourd affine les 10% restants.

    Étage 1 : LR (74 features) — Seuil = 0.30 (recall élevé)
              → Filtre 90% du corpus en quelques secondes
    Étage 2 : CAMeLBERT embeddings + SVM — Seuil = 0.75 (précision élevée)
              → Affine sur les 10% qui passent l'étage 1

Avantage économique :
    CAMeLBERT sur 200,000 textes = ~20h GPU
    CAMeLBERT sur 20,000 textes (après étage 1) = ~2h GPU

Performance attendue :
    Precision ~82%, Recall ~65%, F1 ~0.73
    Temps ~3x plus rapide que CAMeLBERT seul

STATUT : À IMPLÉMENTER
    Prérequis : Pipeline 3.3 (CAMeLBERT) implémenté

Référence : Discussion Claude 8 avr. — Pipeline B1 (Two-Stage Cascade)
"""

from __future__ import annotations


def run_stage1(corpus: list[dict], lr_model, threshold: float = 0.30) -> list[dict]:
    """
    Étage 1 : Filtrage rapide par LR.
    Garde seulement les textes avec prob_lr >= threshold.

    TODO: Réutiliser A1_conservative.py avec seuil abaissé.
    """
    raise NotImplementedError()


def run_stage2(candidates: list[dict], camelbert_model, tokenizer,
               threshold: float = 0.75) -> list[dict]:
    """
    Étage 2 : Classification précise par CAMeLBERT.
    Appliqué uniquement sur les candidats de l'étage 1.

    TODO: Utiliser p3_3_camelbert.predict().
    """
    raise NotImplementedError()


def run(corpus: list[dict], lr_model, camelbert_model, tokenizer) -> list[dict]:
    """
    Exécute la cascade complète en deux étapes.

    TODO: Enchaîner run_stage1() et run_stage2().
    """
    raise NotImplementedError()
