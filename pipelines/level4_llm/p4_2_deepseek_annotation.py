"""
Pipeline 4.2 — Annotation automatique par DeepSeek-R1 + seuillage.

Principe : Déléguer l'annotation actantielle au LLM DeepSeek-R1
(modèle de raisonnement), puis filtrer par seuil de confiance.

Ce pipeline a deux usages distincts :
    A) Annotation du corpus pour créer les features actantielles
       → Alimente le Pipeline 1.3 et les hybrides (family_A, family_C)
    B) Classification directe (pipeline autonome)
       → Baselin de comparaison uniquement

Avantages :
    - Scalable (API parallèle)
    - Exploite les capacités de "raisonnement" de DeepSeek-R1
    - Produit une annotation structurée (vs. classification brute)

Limites :
    - Hallucinations possibles sur termes rares
    - Justifications non-vérifiables
    - Coût API variable

DOUBLE USAGE :
    1. Annotation actantielle (alimente src/uqala_nlp/features/actantial.py)
    2. Classification directe (pipeline autonome de comparaison)

STATUT : À IMPLÉMENTER
    Priorité : HAUTE pour l'annotation actantielle
    Prérequis : Clé API DeepSeek (DEEPSEEK_API_KEY dans .env)

Référence : Discussion Claude 8 avr. — Pipeline 4.2
"""

from __future__ import annotations
import json
from pathlib import Path


ANNOTATION_PROMPT = """Annote ce khabar arabe selon le schéma JSON suivant.
Réponds UNIQUEMENT avec le JSON, sans autre texte.

SCHÉMA :
{
  "forme_textuelle": "khabar|nadira|shicr|hikma|wasf|autre",
  "junun_operateur": {
    "sous_type": "majnun|mucattah|mudallah|absent",
    "nomination": "nomme|anonyme|absent",
    "fonction": "alibi_de_parole|perception_spirituelle|transgression_sociale|feinte|folie_amour|absent"
  },
  "parole_centrale": {
    "type_acte": "sual_jawab|qawl_hikma|insulte_paradoxale|silence_signifiant|recitation_shicr|absent",
    "contenu_paradoxal": true/false
  },
  "mecanisme_narratif": "renversement|paradoxe_lexical|revelation|ironie|absent",
  "validation": {
    "validateur": "calife|savant|foule|narrateur|absent",
    "mode": "rire|don|admiration|assentiment|pleurs|absent"
  },
  "est_majnun_aqil": true/false,
  "confiance": 0.0-1.0
}

KHABAR :
{text}"""


def annotate_khabar(text: str, client) -> dict:
    """
    Annote un khabar avec le schéma actantiel complet.

    TODO: Implémenter avec l'API DeepSeek.
    """
    raise NotImplementedError()


def annotate_corpus(corpus: list[dict], client, output_file: str,
                    resume: bool = True) -> list[dict]:
    """
    Annote un corpus complet avec checkpoint (reprend si interruption).

    Args:
        corpus: Liste de khabars {'id', 'text_ar', ...}
        client: Client API DeepSeek
        output_file: Fichier JSONL de sortie (un enregistrement par ligne)
        resume: Reprendre depuis le dernier checkpoint

    TODO: Implémenter avec sauvegarde progressive + gestion erreurs.
    """
    raise NotImplementedError()


def validate_annotations(annotated_corpus: list[dict],
                          gold_corpus: list[dict]) -> dict:
    """
    Évalue la qualité des annotations LLM vs. annotations gold.

    Returns:
        {'agreement': float, 'kappa': float, 'confusion_matrix': ...}

    TODO: Implémenter pour valider les annotations DeepSeek.
    """
    raise NotImplementedError()
