# Niveau 4 — LLM (interprétabilité nulle)

Pipelines basés sur des modèles de langage génératifs.
Les décisions sont produites par des modèles dont les poids
sont inconnus ou non-interprétables.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `p4_1_few_shot.py` | Few-shot prompting (GPT-4 / Claude) | Partiel |
| `p4_2_deepseek_annotation.py` | Annotation actantielle par DeepSeek-R1 | À implémenter |

## Usage recommandé

Ces pipelines ont **deux usages distincts** :

### Usage A : Annotation du corpus (haute priorité)
`p4_2_deepseek_annotation.py` est utilisé pour produire les **annotations
actantielles** qui alimentent `src/uqala_nlp/features/actantial.py`.
Ce n'est pas une classification autonome : c'est une aide à la structuration
des données.

### Usage B : Baseline de comparaison (basse priorité)
`p4_1_few_shot.py` permet de mesurer les performances théoriques maximales
atteignables (avec LLM + prompt optimal).
À utiliser sur un échantillon de 100 textes, pas sur le corpus complet.

## Note épistémologique

> « Un algorithme totalement opaque produisant une justification en langue
> naturelle ne constitue pas une preuve philologique. La justification du LLM
> est non-vérifiable, non-reproductible, et potentiellement hallucinée. »

L'usage de ces pipelines doit être **explicitement justifié** dans le chapitre
méthodologique de la thèse, avec reconnaissance de leurs limites.

## Prérequis
- Clé API `OPENAI_API_KEY` ou `DEEPSEEK_API_KEY` dans `.env`
- `pip install openai`
