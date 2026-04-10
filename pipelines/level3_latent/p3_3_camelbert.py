"""
Pipeline 3.3 — CAMeLBERT fine-tuning (interprétabilité nulle).

Principe : Fine-tuner le modèle CAMeLBERT (BERT pré-entraîné sur arabe)
sur notre corpus annoté (460 positifs + négatifs).

Modèle de base : CAMeL-Lab/bert-base-arabic-camelbert-mix
    Entraîné sur corpus arabe mixte (classique + moderne)
    128M paramètres

Avantages :
    - Précision théorique ~75-85% (si corpus équilibré)
    - Capture contexte long (512 tokens)
    - État de l'art pour classification de textes arabes

Limites :
    - Boîte noire totale (12 couches × 768 dimensions = 96M poids)
    - Impossible de justifier une décision individuelle
    - Biais sur lexique observé : غزل → faux positifs fréquents
    - GPU requis pour fine-tuning (~2-4h sur GPU moderne)

Performance attendue : Precision ~78%, Recall ~72%, F1 ~0.75
    (attention : biais possible sur certains registres)

ÉVALUATION ÉPISTÉMOLOGIQUE :
    Ce pipeline est le MOINS adapté pour une thèse de philologie.
    Il maximise la performance mais sacrifie toute justifiabilité.
    Utiliser UNIQUEMENT comme terme de comparaison supérieur.

STATUT : À IMPLÉMENTER (OPTIONNEL — dernière priorité)
    Prérequis : GPU disponible, corpus négatif constitué

Référence : Discussion Claude 8 avr. — Pipeline 3.3
"""

from __future__ import annotations


CAMELBERT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def fine_tune(train_dataset, eval_dataset, output_dir: str = "models/camelbert_finetuned",
              epochs: int = 3, batch_size: int = 16):
    """
    Fine-tune CAMeLBERT pour la classification binaire maǧnūn ʿāqil.

    TODO: Utiliser HuggingFace Trainer.
          Nécessite GPU (CUDA ou MPS).
    """
    raise NotImplementedError(
        "Fine-tuning CAMeLBERT nécessite GPU. "
        "Voir docs/workflow.md §Phase 4 pour les prérequis."
    )


def predict(text: str, model, tokenizer) -> dict:
    """
    Prédit si un texte est maǧnūn ʿāqil.

    Returns:
        {'label': int, 'prob_positive': float, 'prob_negative': float}

    TODO: Implémenter.
    """
    raise NotImplementedError()


def extract_cls_embedding(text: str, model, tokenizer) -> 'np.ndarray':
    """
    Extrait l'embedding [CLS] (représentation du texte sans fine-tuning).
    Utile pour le Pipeline F1 (distillation).

    TODO: Implémenter.
    """
    raise NotImplementedError()
