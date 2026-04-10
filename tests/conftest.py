"""Configuration partagée pour les tests pytest."""

import json
import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture
def sample_positive_khabar():
    """Khabar positif minimal pour les tests unitaires."""
    return {
        "id": "test_pos_001",
        "text_ar": "رأيت مجنوناً بالبصرة فسألته عن الحكمة فقال العقل في ترك العقل فضحك الخليفة وأعطاه ألف درهم",
        "label": 1,
    }


@pytest.fixture
def sample_negative_khabar():
    """Khabar négatif minimal pour les tests unitaires."""
    return {
        "id": "test_neg_001",
        "text_ar": "قال الشاعر في وصف الربيع والخضرة والأزهار الجميلة والطير المغرد",
        "label": 0,
    }


@pytest.fixture
def sample_wasf_khabar():
    """Khabar de type wasf (définitionnel) — doit être négatif."""
    return {
        "id": "test_wasf_001",
        "text_ar": "المجنون هو من ذهب عقله ومن أنواعه المعتوه والمدله والأحمق",
        "label": 0,
    }
