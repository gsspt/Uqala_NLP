"""
Interface d'annotation pour l'active learning (Pipeline C1).

Usage :
    python scripts/annotate.py --cycle 1 --model models/lr_classifier_71features.pkl
    python scripts/annotate.py --resume --cycle 3

Ce script interactif présente les khabars incertains un par un,
avec une explication SHAP des features influentes, et enregistre
les annotations avec justifications.

STATUT : À IMPLÉMENTER
    Dépend de : pipelines/hybrid/family_C_iterative/C1_active_learning.py
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from uqala_nlp.config import MODELS_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Interface d'annotation pour l'active learning"
    )
    parser.add_argument("--cycle", type=int, required=True,
                        help="Numéro du cycle d'active learning (1-10)")
    parser.add_argument("--model", type=str,
                        default=str(MODELS_DIR / "lr_classifier_71features.pkl"),
                        help="Chemin vers le modèle entraîné")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Nombre de khabars à annoter par cycle")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre depuis le dernier checkpoint")
    parser.add_argument("--output-dir", type=str,
                        default=str(RESULTS_DIR / "active_learning"),
                        help="Répertoire de sortie")
    args = parser.parse_args()

    raise NotImplementedError(
        "annotate.py n'est pas encore implémenté. "
        "Voir pipelines/hybrid/family_C_iterative/C1_active_learning.py "
        "pour l'implémentation de l'interface d'annotation."
    )


if __name__ == "__main__":
    main()
