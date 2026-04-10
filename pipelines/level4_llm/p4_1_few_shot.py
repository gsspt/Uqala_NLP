"""
Pipeline 4.1 — Détection par few-shot prompting (GPT-4 / Claude).

Principe : Fournir au LLM une définition du motif + exemples annotés,
et lui demander de classifier chaque khabar avec justification.

Avantage unique : Le LLM produit une justification en langue naturelle,
ce qui peut guider la lecture philologique — mais cette justification
n'est pas vérifiable.

Avantages :
    - Précision potentielle ~80-90% (si prompt bien calibré)
    - Capture nuances narratives complexes (ironie, suspension morale)
    - Justification en langue naturelle (arabe ou français)

Limites :
    - Totalement opaque (poids du modèle inconnus)
    - Non-déterministe (résultats varient selon temperature)
    - Coût API prohibitif pour corpus 10,000+ textes
    - Hallucinations possibles

ÉVALUATION ÉPISTÉMOLOGIQUE :
    Justifications = non-vérifiables, non-reproductibles.
    Utile pour validation qualitative sur échantillon (50-100 textes).
    Ne pas utiliser pour le corpus final de thèse.

STATUT : PARTIELLEMENT IMPLÉMENTÉ
    - Script LLM de base : voir D2_llm_assisted.py (Family D)
    - Prompt few-shot structuré : À FAIRE
    - Évaluation systématique sur 100 textes : À FAIRE

Référence : Discussion Claude 8 avr. — Pipeline 4.1
"""

from __future__ import annotations
import json
import os


SYSTEM_PROMPT = """Tu es un expert en littérature d'adab arabe médiévale.

Tu dois déterminer si un khabar relève du motif « maǧnūn ʿāqil » (sage-fou).

Critères définitoires :
1. PARADOXE : Tension entre folie apparente et sagesse sous-jacente
2. TRANSGRESSION : Comportement déviant qui révèle une vérité
3. SUSPENSION MORALE : Le narrateur ne juge pas explicitement le fou
4. STRUCTURE NARRATIVE : Récit avec actants (pas simple maxime)

Réponds UNIQUEMENT en JSON valide :
{
  "label": 1 ou 0,
  "confiance": 0.0 à 1.0,
  "paradoxe": true/false,
  "transgression": true/false,
  "suspension_morale": true/false,
  "justification": "explication en 1-2 phrases"
}"""


def build_few_shot_prompt(khabar_text: str, positive_examples: list[dict],
                           negative_examples: list[dict]) -> str:
    """
    Construit un prompt few-shot avec exemples positifs et négatifs.

    Args:
        khabar_text: Texte à classifier
        positive_examples: 3-5 exemples positifs annotés
        negative_examples: 3-5 exemples négatifs annotés

    TODO: Implémenter la construction du prompt.
    """
    raise NotImplementedError()


def classify_with_llm(text: str, client, model: str = "gpt-4") -> dict:
    """
    Classifie un khabar avec un LLM en few-shot.

    TODO: Implémenter avec openai.ChatCompletion.create ou claude API.
    """
    raise NotImplementedError()


def batch_classify(texts: list[str], client, batch_size: int = 10,
                   output_file: str = "results/llm_predictions.jsonl") -> list[dict]:
    """
    Classifie un lot de textes (avec checkpoint pour reprendre si interruption).

    TODO: Implémenter avec gestion des erreurs API et sauvegarde progressive.
    """
    raise NotImplementedError()
