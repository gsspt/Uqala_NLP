"""Tests pour le module features."""

import pytest


def test_arabic_utils_normalize():
    from uqala_nlp.utils.arabic import normalize_arabic
    text = "مُجَنُّونٌ"  # Avec diacritiques
    normalized = normalize_arabic(text)
    assert "ُ" not in normalized
    assert "مجنون" in normalized


def test_arabic_utils_word_count():
    from uqala_nlp.utils.arabic import word_count
    text = "رأيت مجنوناً بالبصرة"
    assert word_count(text) == 3


def test_arabic_utils_has_any():
    from uqala_nlp.utils.arabic import has_any
    text = "رأيت مجنوناً بالبصرة"
    assert has_any(text, ["مجنون", "معتوه"]) is True
    assert has_any(text, ["بهلول", "سعدون"]) is False


def test_arabic_utils_window_cooccurrence():
    from uqala_nlp.utils.arabic import window_cooccurrence
    text = "كان مجنوناً ولكنه أظهر حكمة بالغة"
    count = window_cooccurrence(text, ["مجنون"], ["حكمة"], window=100)
    assert count >= 1


def test_actantial_features_stub():
    """Vérifie que les stubs lèvent NotImplementedError (pas d'import silencieux)."""
    from uqala_nlp.features.actantial import extract_actantial_features
    with pytest.raises(NotImplementedError):
        extract_actantial_features("test", use_llm=False)
