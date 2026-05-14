"""Tests F12-T2: validación Pydantic de RubricCriterion y RoutineOutput.

Sin LLM, sin grafo, sin Store: solo pruebas puras del contrato extendido.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.graph.schemas.routine import RoutineOutput, RubricCriterion


# ---- helpers -------------------------------------------------------------

_MIN_DESCRIPTION = "A description with enough characters."
_REFERENCE_SOLUTION = (
    "## Reference\n\n```python\nclass LRU: pass\n```\n\nNotes on the approach."
)


def _valid_rubric_dicts(n: int = 3) -> list[dict]:
    base = {
        "concept": "Caching",
        "description": _MIN_DESCRIPTION,
        "weight": 4,
    }
    return [
        {**base, "concept": f"Concept{i}", "description": f"{_MIN_DESCRIPTION} item {i}"}
        for i in range(n)
    ]


def _valid_routine_output_kwargs() -> dict:
    return {
        "title": "Implement an LRU cache",
        "target_weakness": "Caching",
        "expected_concepts": ["LRU", "eviction"],
        "difficulty": 3,
        "challenge_md": "## Challenge\n\nImplement an LRU cache step by step.",
        "rubric": _valid_rubric_dicts(),
        "reference_solution": _REFERENCE_SOLUTION,
    }


# ==========================================================================
# RubricCriterion
# ==========================================================================

def test_rubric_criterion_rejects_weight_below_1():
    """`weight=0` está fuera del rango 1..5."""
    with pytest.raises(ValidationError):
        RubricCriterion(concept="Caching", description=_MIN_DESCRIPTION, weight=0)


def test_rubric_criterion_rejects_weight_above_5():
    """`weight=6` excede el techo de la rúbrica."""
    with pytest.raises(ValidationError):
        RubricCriterion(concept="Caching", description=_MIN_DESCRIPTION, weight=6)


def test_rubric_criterion_rejects_empty_description():
    """`description` debe tener al menos 10 caracteres."""
    with pytest.raises(ValidationError):
        RubricCriterion(concept="Caching", description="too short", weight=3)


def test_rubric_criterion_accepts_boundary_values():
    """Cuatro casos límite válidos: weight=1, weight=5, descripción de 10 chars
    y concept de 1 char (todos en el límite inferior/superior aceptado)."""
    c1 = RubricCriterion(concept="X", description="0123456789", weight=1)
    assert c1.weight == 1 and c1.concept == "X"

    c5 = RubricCriterion(concept="LRU", description=_MIN_DESCRIPTION, weight=5)
    assert c5.weight == 5


# ==========================================================================
# RoutineOutput
# ==========================================================================

def test_routine_output_rejects_rubric_too_small():
    """Rúbrica con <3 ítems es inválida (min_length=3)."""
    kwargs = _valid_routine_output_kwargs()
    kwargs["rubric"] = _valid_rubric_dicts(2)
    with pytest.raises(ValidationError):
        RoutineOutput(**kwargs)


def test_routine_output_rejects_rubric_too_large():
    """Rúbrica con >5 ítems es inválida (max_length=5)."""
    kwargs = _valid_routine_output_kwargs()
    kwargs["rubric"] = _valid_rubric_dicts(6)
    with pytest.raises(ValidationError):
        RoutineOutput(**kwargs)


def test_routine_output_rejects_short_reference_solution():
    """`reference_solution` con <20 chars es inválida."""
    kwargs = _valid_routine_output_kwargs()
    kwargs["reference_solution"] = "x"
    with pytest.raises(ValidationError):
        RoutineOutput(**kwargs)


def test_routine_output_accepts_minimum_valid_rubric():
    """Happy path con rúbrica de 3 ítems y reference_solution mínima válida."""
    output = RoutineOutput(**_valid_routine_output_kwargs())
    assert len(output.rubric) == 3
    assert all(isinstance(c, RubricCriterion) for c in output.rubric)
    assert output.reference_solution.startswith("## Reference")


def test_routine_output_serializes_rubric_to_dicts():
    """`model_dump()` produce lista de dicts con las claves esperadas."""
    output = RoutineOutput(**_valid_routine_output_kwargs())
    dumped = output.model_dump()
    assert isinstance(dumped["rubric"], list)
    assert len(dumped["rubric"]) == 3
    for item in dumped["rubric"]:
        assert {"concept", "description", "weight"} <= set(item.keys())
        assert isinstance(item["weight"], int) and 1 <= item["weight"] <= 5
    assert isinstance(dumped["reference_solution"], str)


def test_routine_output_rubric_item_validates_recursively():
    """Si un dict del array de rubric viola las reglas, la validación falla."""
    kwargs = _valid_routine_output_kwargs()
    bad_rubric = _valid_rubric_dicts(3)
    bad_rubric[1]["weight"] = 99  # fuera de rango
    kwargs["rubric"] = bad_rubric
    with pytest.raises(ValidationError):
        RoutineOutput(**kwargs)
