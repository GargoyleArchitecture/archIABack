"""Tests F5-T4: función pura `estimate_difficulty`."""
from src.graph.services.difficulty import estimate_difficulty


def test_baseline_no_bonuses():
    """LLM=3, conceptos OK, snippet corto → la base se respeta."""
    result = estimate_difficulty(
        num_concepts=2,
        current_mastery=0.5,
        snippet_length=200,
        llm_difficulty=3,
    )
    assert result == 3


def test_many_concepts_adds_one():
    """num_concepts > 5 → +1 sobre la base del LLM."""
    result = estimate_difficulty(
        num_concepts=8,
        current_mastery=0.5,
        snippet_length=200,
        llm_difficulty=2,
    )
    assert result == 3


def test_long_snippet_adds_one():
    """snippet_length > 800 → +1 sobre la base."""
    result = estimate_difficulty(
        num_concepts=2,
        current_mastery=0.5,
        snippet_length=1500,
        llm_difficulty=4,
    )
    assert result == 5


def test_clamp_upper_bound():
    """LLM=5 + ambos bonuses no debe pasar de 5 (clamp)."""
    result = estimate_difficulty(
        num_concepts=10,
        current_mastery=0.5,
        snippet_length=2000,
        llm_difficulty=5,
    )
    assert result == 5


def test_clamp_lower_bound():
    """LLM=0 (fuera de rango) se debe clamp-ear a 1."""
    result = estimate_difficulty(
        num_concepts=1,
        current_mastery=0.0,
        snippet_length=0,
        llm_difficulty=0,
    )
    assert result == 1


def test_high_mastery_does_not_lower_difficulty():
    """current_mastery NO afecta el cálculo (la decisión de regen vive en el nodo)."""
    high = estimate_difficulty(
        num_concepts=2, current_mastery=0.95, snippet_length=200, llm_difficulty=3
    )
    low = estimate_difficulty(
        num_concepts=2, current_mastery=0.05, snippet_length=200, llm_difficulty=3
    )
    assert high == low == 3


def test_defensive_types():
    """None y strings convertibles no rompen el cálculo."""
    result = estimate_difficulty(
        num_concepts=None,           # type: ignore[arg-type]
        current_mastery=None,        # type: ignore[arg-type]
        snippet_length="300",        # type: ignore[arg-type]
        llm_difficulty="2",          # type: ignore[arg-type]
    )
    assert result == 2  # base=2, snippet 300 < 800, num_concepts treated as 0
