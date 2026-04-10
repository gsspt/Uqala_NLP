"""
Configuration centralisée du projet.

Toutes les constantes, chemins et seuils sont définis ici
pour éviter les valeurs hardcodées dans les scripts.
"""

import os
import pathlib

# ──────────────────────────────────────────────────────────────
# Chemins racine
# ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent.parent  # racine du repo
SRC = ROOT / "src"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ──────────────────────────────────────────────────────────────
# Données
# ──────────────────────────────────────────────────────────────
DATASET_RAW = DATA_DIR / "raw" / "dataset_raw.json"
DATASET_ANNOTATED = DATA_DIR / "annotated" / "Kitab_Uqala_al_Majanin_annotated.json"
NEGATIVES_DIR = DATA_DIR / "negatives"
OPENITI_CORPUS = ROOT / "openiti_corpus"  # corpus OpenITI (non versionné)

# ──────────────────────────────────────────────────────────────
# Modèles entraînés
# ──────────────────────────────────────────────────────────────
LR_MODEL = MODELS_DIR / "lr_classifier_71features.pkl"
XGB_MODEL = MODELS_DIR / "xgb_classifier_71features.pkl"
LR_REPORT = MODELS_DIR / "lr_report_71features.json"
XGB_REPORT = MODELS_DIR / "xgb_report_71features.json"

# ──────────────────────────────────────────────────────────────
# Seuils de classification
# ──────────────────────────────────────────────────────────────
THRESHOLDS = {
    "conservative": {"lr": 0.85, "xgb": 0.85},  # haute précision
    "balanced": {"lr": 0.70, "xgb": 0.70},       # équilibré
    "max_recall": {"lr": 0.40, "xgb": 0.40},     # rappel maximal
}
DEFAULT_THRESHOLD = "balanced"

# ──────────────────────────────────────────────────────────────
# API Keys (chargées depuis l'environnement)
# ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ──────────────────────────────────────────────────────────────
# Active Learning
# ──────────────────────────────────────────────────────────────
AL_CYCLES = 10          # nombre de cycles d'active learning
AL_BATCH_SIZE = 50      # textes à annoter par cycle
AL_UNCERTAINTY_RANGE = (0.35, 0.65)  # plage d'incertitude (prob)

# ──────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────
FEATURE_WINDOW = 80     # fenêtre de co-occurrence (caractères)
MIN_TEXT_LENGTH = 50    # longueur minimale du matn (mots)
MAX_TEXT_LENGTH = 600   # longueur maximale du matn (mots)
