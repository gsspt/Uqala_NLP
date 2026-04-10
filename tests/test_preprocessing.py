"""Tests pour le module preprocessing."""

import pytest
from uqala_nlp.preprocessing.isnad_filter import get_matn, split_isnad


def test_get_matn_removes_isnad():
    text = "حدثنا محمد بن علي عن أبي بكر قال رأيت مجنوناً بالبصرة"
    matn = get_matn(text)
    assert "حدثنا" not in matn
    assert "مجنوناً" in matn


def test_get_matn_pure_narrative():
    """Un texte sans isnād doit être retourné tel quel."""
    text = "رأيت مجنوناً بالبصرة فسألته عن الحكمة"
    matn = get_matn(text)
    assert len(matn) > 0


def test_split_isnad_returns_tuple():
    text = "حدثنا محمد عن علي قال رأيت رجلاً"
    result = split_isnad(text)
    assert isinstance(result, tuple)
    assert len(result) == 2
