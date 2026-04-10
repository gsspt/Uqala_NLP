"""
Features actantielles — Modèle de Greimas appliqué au maǧnūn ʿāqil.

Ce module encode la structure narrative profonde de chaque khabar :
- Rôle fonctionnel des actants (opérateur de folie, interlocuteur, validateur)
- Type d'acte de parole central (sual_jawab, qawl_hikma, etc.)
- Mécanisme de validation narrative
- Forme textuelle (khabar, nadira, šiʿr)

Les annotations actantielles sont produites par LLM (Claude/DeepSeek)
via le pipeline D2 (family_D_human_loop/D2_llm_assisted.py),
puis converties en vecteurs numériques ici.

Référence méthodologique :
    Greimas, A.J. (1966). Sémantique structurale.
    Adaptation au corpus d'adab : voir docs/features_catalog.md §7

STATUT : EN COURS D'IMPLÉMENTATION
    - Schéma JSON d'annotation défini ✓
    - Annotation de 50 khabars test via DeepSeek : À FAIRE
    - Conversion JSON → features numériques : À FAIRE
    - Validation philologique : À FAIRE
"""

from __future__ import annotations
from typing import Any


# ──────────────────────────────────────────────────────────────
# Schéma d'annotation actantielle (produit par LLM)
# ──────────────────────────────────────────────────────────────
ANNOTATION_SCHEMA = {
    "forme_textuelle": ["khabar", "nadira", "shicr", "hikma", "wasf", "autre"],
    "junun_operateur": {
        "sous_type": ["majnun", "mucattah", "mudallah", "macruh", "mustas_tuh", "absent"],
        "nomination": ["nomme", "anonyme", "isnad_seulement"],
        "fonction": [
            "alibi_de_parole",      # La folie libère la parole
            "perception_spirituelle",  # Accès au sacré/caché
            "transgression_sociale",   # Violation des normes
            "feinte",                  # Folie simulée
            "folie_amour",             # Junun al-hubb
            "absent",
        ],
    },
    "actants": [
        {
            "role_fonctionnel": "str",  # description libre
            "position_relative": ["autorite", "pair", "tiers", "absent"],
            "terme_arabe": "str",
        }
    ],
    "parole_centrale": {
        "type_acte": [
            "sual_jawab",      # Question / réponse
            "qawl_hikma",      # Maxime de sagesse
            "insulte_paradoxale",
            "silence_signifiant",
            "recitation_shicr",
            "absent",
        ],
        "contenu_paradoxal": "bool",
    },
    "mecanisme_narratif": [
        "renversement",        # Retournement de situation
        "paradoxe_lexical",    # Coprésence junūn + ʿaql
        "revelation",          # Dévoilement d'une vérité cachée
        "ironie",              # Second degré
        "absent",
    ],
    "validation": {
        "validateur": ["calife", "savant", "foule", "narrateur", "absent"],
        "mode": ["rire", "don", "admiration", "assentiment", "pleurs", "absent"],
    },
    "scene_enonciative": {
        "lieu": ["cour", "rue", "mosquee", "desert", "marche", "prison", "autre", "absent"],
        "temoignage_direct": "bool",  # رأيت / سمعت
    },
}


def annotation_to_features(annotation: dict[str, Any]) -> dict[str, float]:
    """
    Convertit une annotation actantielle JSON en vecteur de features numériques.

    Args:
        annotation: Dict produit par le pipeline LLM (schéma ANNOTATION_SCHEMA)

    Returns:
        Dict de features numériques (float 0.0 / 1.0)

    TODO: Implémenter la conversion pour chaque champ du schéma.
    """
    raise NotImplementedError(
        "annotation_to_features() n'est pas encore implémentée. "
        "Voir pipeline D2 pour l'annotation et ce module pour la conversion."
    )


def extract_actantial_features(text: str, use_llm: bool = False) -> dict[str, float]:
    """
    Extrait les features actantielles d'un texte arabe.

    En mode use_llm=True, appelle l'API LLM pour annoter le texte.
    En mode use_llm=False, utilise les patterns regex comme approximation.

    Args:
        text: Texte arabe (matn uniquement, isnād filtré)
        use_llm: Utiliser le LLM pour l'annotation (lent, précis)
                 ou les regex (rapide, approximatif)

    Returns:
        Dict de 14 features actantielles (float)

    TODO: Implémenter les deux modes.
    """
    raise NotImplementedError(
        "extract_actantial_features() n'est pas encore implémentée. "
        "Voir docs/features_catalog.md pour la liste complète des features."
    )
