# Famille D — Humain dans la boucle

Pipelines où la décision humaine est intégrée dans le workflow,
avec des outils d'explication pour guider le philologue.

| Fichier | Description | Statut |
|---------|-------------|--------|
| `D1_explainable_loop.py` | Annotation avec explication SHAP en temps réel | Partiel |
| `D1_explainable_loop_simple.py` | Version simplifiée sans SHAP | Partiel |
| `D1_visualize_importance.py` | Visualisation de l'importance des features | Partiel |
| `D2_llm_assisted.py` | LLM pré-annote, expert valide/corrige | Partiel |

## D1 — Human-in-the-Loop avec SHAP

Pour chaque khabar présenté, l'interface affiche :
```
════════════════════════════════════════
Khabar #12043 — Prob = 0.51 (INCERTAIN)
Auteur : 0255Jahiz / Hayawan

Matn : وحدثني شيخ من أهل الحجاز أن رجلاً...

Top features influentes :
  paradox_pairs          : +0.18
  has_isnad              : +0.12
  poetry_ratio           : -0.15  ← trop de poésie
  qala_density           : +0.08

Label [0/1] : _
Justification (optionnel) : _
════════════════════════════════════════
```

**Avantage pour la thèse** : Chaque annotation est accompagnée d'une
justification SHAP traçable → documentation méthodologique rigoureuse.

## D2 — LLM-Assisted Annotation

Le LLM propose une annotation actantielle + classification,
le philologue valide ou corrige.

**Gain de temps estimé** : 3-5x plus rapide que l'annotation pure.
**Risque** : Biais du LLM sur certains registres → valider systématiquement.
