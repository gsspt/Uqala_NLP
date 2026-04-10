"""
Pipeline Hybride C1 — Active Learning itératif.

Principe fondamental :
    Au lieu d'annoter 10,000 khabars aléatoirement,
    l'algorithme sélectionne les 50 exemples sur lesquels
    il est le MOINS certain → chaque annotation est maximalement
    informative pour améliorer le modèle.

Boucle en 10 cycles (2h/cycle) :
    1. Entraîner modèle sur corpus annoté courant
    2. Prédire sur corpus non-annoté
    3. Sélectionner 50 cas les plus incertains (prob ≈ 0.5)
    4. Annoter manuellement ces 50 cas (avec explication SHAP)
    5. Ajouter au corpus annoté
    6. Ré-entraîner → retour à 1

Résultat attendu après 10 cycles (500 annotations ciblées) :
    F1 ≈ 0.85 (vs F1 ≈ 0.70 avec 500 annotations aléatoires)
    Gain : +15% F1 pour même effort

Contribution méthodologique pour la thèse :
    Les cas incertains révèlent les frontières conceptuelles du motif.
    → "Après 10 cycles, 3 clusters de cas ambigus émergent :
       1. Anecdotes ascétiques (zuhhād)
       2. Ḥikam gnomiques paradoxales
       3. Nawādir humoristiques"

STATUT : À IMPLÉMENTER — PRIORITÉ HAUTE
    C'est le pipeline principal recommandé pour la thèse.

Référence : Discussion Claude 8 avr. — Pipeline H2 (Active Learning Itératif)
           + explication détaillée en 4 étapes
"""

from __future__ import annotations
from pathlib import Path
import json


def uncertainty_score(prob: float) -> float:
    """Mesure d'incertitude : distance à 0.5 (0 = très incertain, 0.5 = très certain)."""
    return abs(prob - 0.5)


def select_uncertain_batch(probs: list[float], indices: list[int],
                            batch_size: int = 50) -> list[int]:
    """
    Sélectionne les `batch_size` exemples les plus incertains.

    Args:
        probs: Probabilités prédites pour la classe positive
        indices: Indices dans le corpus non-annoté
        batch_size: Nombre d'exemples à sélectionner

    Returns:
        Liste d'indices des exemples sélectionnés

    TODO: Implémenter. Trier par abs(prob - 0.5) croissant.
    """
    raise NotImplementedError()


def annotation_interface(khabars: list[dict], clf, explainer,
                          feature_names: list[str],
                          output_file: str) -> list[dict]:
    """
    Interface CLI d'annotation manuelle avec explication SHAP.

    Affiche pour chaque khabar :
        - Texte complet (matn)
        - Prédiction + confiance
        - Top-5 features influentes (SHAP)
        - Demande label [0/1] + justification optionnelle

    Sauvegarde progressivement dans output_file (JSONL).

    TODO: Implémenter l'interface interactive.
    """
    raise NotImplementedError()


def run_cycle(cycle_num: int, X_annotated, y_annotated,
              X_unannotated, corpus_unannotated: list[dict],
              output_dir: str = "results/active_learning") -> dict:
    """
    Exécute un cycle complet d'active learning.

    Returns:
        Dict avec métriques du cycle (precision, recall, F1, n_new_annotations)

    TODO: Implémenter les étapes 1-5 décrites dans le module docstring.
    """
    raise NotImplementedError()


def run_full_pipeline(corpus_annotated: list[dict], corpus_unannotated: list[dict],
                      n_cycles: int = 10, batch_size: int = 50,
                      output_dir: str = "results/active_learning") -> dict:
    """
    Exécute les n_cycles cycles d'active learning complets.

    Sauvegarde après chaque cycle pour reprendre si interruption.
    Génère un rapport final avec courbe d'apprentissage.

    TODO: Implémenter en enchaînant run_cycle().
    """
    raise NotImplementedError()
