"""
Utilitaires pour le traitement du texte arabe classique.

Fonctions partagées entre tous les modules du projet :
normalisation, tokenisation, détection poésie, comptage de mots.
"""

import re
import unicodedata


# Diacritiques arabes à supprimer pour la normalisation
_DIACRITICS = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]'
)

# Marqueurs de vers/poésie
_VERSE_MARKERS = re.compile(r'%~%|~~|#\$#|[∥‖]')


def normalize_arabic(text: str) -> str:
    """Normalise le texte arabe : NFD + suppression diacritiques + espaces."""
    text = unicodedata.normalize("NFD", text)
    text = _DIACRITICS.sub("", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenise un texte arabe en liste de mots (whitespace-based)."""
    return normalize_arabic(text).split()


def word_count(text: str) -> int:
    """Retourne le nombre de mots dans le texte."""
    return len(tokenize(text))


def poetry_ratio(text: str) -> float:
    """
    Estime la proportion de vers dans le texte.
    Détecte les marqueurs OpenITI (%~%) et les hémistiches.
    """
    lines = text.split('\n')
    if not lines:
        return 0.0
    verse_lines = sum(1 for line in lines if _VERSE_MARKERS.search(line))
    return verse_lines / max(len(lines), 1)


def count_occurrences(text: str, terms: list[str]) -> int:
    """Compte les occurrences (totales) d'une liste de termes dans le texte."""
    normalized = normalize_arabic(text)
    return sum(normalized.count(term) for term in terms)


def has_any(text: str, terms: list[str]) -> bool:
    """Retourne True si au moins un terme est présent dans le texte."""
    normalized = normalize_arabic(text)
    return any(term in normalized for term in terms)


def window_cooccurrence(text: str, terms_a: list[str], terms_b: list[str],
                        window: int = 80) -> int:
    """
    Compte les co-occurrences de (terms_a, terms_b) dans une fenêtre de `window` caractères.
    """
    normalized = normalize_arabic(text)
    count = 0
    for term_a in terms_a:
        for m in re.finditer(re.escape(term_a), normalized):
            start = max(0, m.start() - window)
            end = min(len(normalized), m.end() + window)
            context = normalized[start:end]
            if any(tb in context for tb in terms_b):
                count += 1
    return count
