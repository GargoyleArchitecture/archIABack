"""Tests F5-T1: subgrafo RoutineGenerator + nodos individuales.

LLM siempre mockeado para que los tests sean deterministas. El test golden
verifica la estructura del output (no el contenido textual).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.nodes.routine.select_weakness import select_weakness_node
from src.graph.nodes.routine.synthesize_challenge import synthesize_challenge_node
from src.graph.nodes.routine.validate_difficulty import (
    validate_difficulty_node,
    validate_difficulty_router,
)
from src.graph.schemas.routine import RoutineOutput, RubricCriterion
from src.graph.routine_graph import build_routine_graph


# Rúbrica y solución de referencia mínimas válidas (F12-T2). Reutilizables
# por cualquier `RoutineOutput` que los tests construyan a mano.
def _sample_rubric() -> list[RubricCriterion]:
    return [
        RubricCriterion(
            concept="LRU",
            description="Implements eviction by least-recently-used order.",
            weight=5,
        ),
        RubricCriterion(
            concept="TTL",
            description="Supports per-entry time-to-live expiration.",
            weight=3,
        ),
        RubricCriterion(
            concept="Thread safety",
            description="Concurrent reads do not corrupt internal state.",
            weight=4,
        ),
    ]


def _sample_rubric_dicts() -> list[dict]:
    return [c.model_dump() for c in _sample_rubric()]


_SAMPLE_REFERENCE_SOLUTION = (
    "## Reference\n\n```python\nclass LRUCache: ...\n```\n\n"
    "Uses OrderedDict to keep eviction O(1)."
)


# ---- helpers -------------------------------------------------------------

def _profile_with_concepts():
    return {
        "user_id": "u1",
        "strengths": ["Modularidad"],
        "weaknesses": ["Caching"],
        "evaluated_concepts": [
            {"name": "Modularidad", "mastery": 0.8},
            {"name": "Caching", "mastery": 0.2},
            {"name": "Patterns", "mastery": 0.5},
        ],
    }


def _make_llm_mock_returning(output: RoutineOutput):
    """Construye un mock que simula `llm.with_structured_output(...).ainvoke(...)`."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=output)
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


# ==========================================================================
# select_weakness_node
# ==========================================================================

def test_select_weakness_picks_lowest_mastery():
    """Sin target explícito → elige el concepto con menor mastery."""
    state = {
        "user_id": "u1",
        "user_profile": _profile_with_concepts(),
        "target_weakness": None,
    }
    out = select_weakness_node(state)
    assert out["target_weakness"] == "Caching"
    assert out["target_mastery"] == pytest.approx(0.2)


def test_select_weakness_respects_explicit():
    """Con target explícito → lo respeta y busca su mastery."""
    state = {
        "user_id": "u1",
        "user_profile": _profile_with_concepts(),
        "target_weakness": "Modularidad",
    }
    out = select_weakness_node(state)
    assert out["target_weakness"] == "Modularidad"
    assert out["target_mastery"] == pytest.approx(0.8)


def test_select_weakness_fallback_when_profile_empty():
    """Sin conceptos en el perfil → fallback genérico con mastery 0.3."""
    state = {
        "user_id": "u1",
        "user_profile": {"evaluated_concepts": []},
        "target_weakness": None,
    }
    out = select_weakness_node(state)
    assert out["target_weakness"] == "general software architecture"
    assert out["target_mastery"] == pytest.approx(0.3)


# ==========================================================================
# synthesize_challenge_node
# ==========================================================================

def test_synthesize_challenge_populates_state_from_llm():
    """LLM devuelve un RoutineOutput → state queda con todos los campos."""
    fake_output = RoutineOutput(
        title="Refactor LRU cache implementation",
        target_weakness="Caching",
        inverse_rag_snippet=None,
        expected_concepts=["LRU", "TTL", "eviction"],
        difficulty=3,
        challenge_md="## Challenge\n\nImplement an LRU cache with TTL support.",
        rubric=_sample_rubric(),
        reference_solution=_SAMPLE_REFERENCE_SOLUTION,
    )
    llm_mock = _make_llm_mock_returning(fake_output)

    state = {
        "user_id": "u1",
        "target_weakness": "Caching",
        "target_mastery": 0.2,
        "raw_snippet": "",
        "regen_count": 0,
    }

    out = asyncio.run(synthesize_challenge_node(state, llm_obj=llm_mock))

    assert out["title"] == "Refactor LRU cache implementation"
    assert out["challenge_md"].startswith("## Challenge")
    assert out["expected_concepts"] == ["LRU", "TTL", "eviction"]
    assert out["difficulty"] == 3
    # F12-T2: rubric llega al state como lista de dicts; reference_solution como str.
    assert isinstance(out["rubric"], list) and len(out["rubric"]) == 3
    assert out["rubric"][0]["concept"] == "LRU"
    assert out["rubric"][0]["weight"] == 5
    assert out["reference_solution"] == _SAMPLE_REFERENCE_SOLUTION
    llm_mock.with_structured_output.assert_called_once()


def test_synthesize_challenge_falls_back_on_llm_error():
    """Si el LLM lanza, devuelve un payload mínimo coherente.

    F12-T2: el fallback DEBE incluir rubric (>=3 ítems) y reference_solution
    no vacía para que el `RoutineOutput` reconstruido por validate_difficulty
    pase la validación Pydantic extendida.
    """
    structured = MagicMock()
    structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 500"))
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)

    state = {
        "user_id": "u1",
        "target_weakness": "Caching",
        "target_mastery": 0.2,
        "raw_snippet": "",
        "regen_count": 0,
    }

    out = asyncio.run(synthesize_challenge_node(state, llm_obj=llm))

    # Payload mínimo coherente: la dificultad existe y es válida.
    assert out["difficulty"] == 3
    assert "Caching" in out["title"] or "general" in out["title"].lower()
    assert out["challenge_md"]
    # F12-T2: rubric y reference_solution presentes en el state tras el fallback.
    assert isinstance(out["rubric"], list) and len(out["rubric"]) >= 3
    for item in out["rubric"]:
        assert {"concept", "description", "weight"} <= set(item.keys())
        assert 1 <= int(item["weight"]) <= 5
    assert out["reference_solution"] and len(out["reference_solution"]) >= 20


# ==========================================================================
# validate_difficulty_node + router
# ==========================================================================

def test_validate_accepts_when_difficulty_within_ceiling():
    """Mastery 0.5 → ceiling=4. Difficulty 3 pasa, sin regen."""
    state = {
        "target_weakness": "Caching",
        "target_mastery": 0.5,
        "raw_snippet": "",
        "title": "Implement LRU cache",
        "challenge_md": "## Challenge\n\nDo it well enough to be tested.",
        "expected_concepts": ["LRU", "TTL"],
        "difficulty": 3,
        "regen_count": 0,
        "rubric": _sample_rubric_dicts(),
        "reference_solution": _SAMPLE_REFERENCE_SOLUTION,
    }
    out = validate_difficulty_node(state)
    assert out["final"] is not None
    assert out["final"].difficulty == 3
    # F12-T2: rubric y reference_solution se propagan al RoutineOutput final.
    assert len(out["final"].rubric) == 3
    assert out["final"].rubric[0].concept == "LRU"
    assert out["final"].reference_solution == _SAMPLE_REFERENCE_SOLUTION
    assert validate_difficulty_router(out) == "accept"


def test_validate_triggers_regen_when_too_hard_and_can_regen():
    """Mastery 0.0 → ceiling=2. Difficulty 4 supera el techo y regen_count=0 < 2."""
    state = {
        "target_weakness": "Caching",
        "target_mastery": 0.0,
        "raw_snippet": "",
        "title": "Hardcore challenge",
        "challenge_md": "## Hard",
        "expected_concepts": ["A", "B", "C"],
        "difficulty": 4,
        "regen_count": 0,
    }
    out = validate_difficulty_node(state)
    assert out.get("final") is None
    assert out["regen_count"] == 1
    assert validate_difficulty_router(out) == "regenerate"


def test_validate_accepts_after_max_regens():
    """regen_count=2 (límite alcanzado) → acepta aunque sea muy difícil.

    F12-T2: este test omite rubric/reference_solution del state a propósito
    para validar el fallback `_DEFAULT_FALLBACK_RUBRIC` y la solución por
    defecto del nodo.
    """
    state = {
        "target_weakness": "Caching",
        "target_mastery": 0.0,
        "raw_snippet": "",
        "title": "Still hard",
        "challenge_md": "## Hard\n\nThis is a sufficiently long challenge body.",
        "expected_concepts": ["A", "B", "C", "D", "E", "F"],  # 6 concepts → +1
        "difficulty": 5,
        "regen_count": 2,
    }
    out = validate_difficulty_node(state)
    assert out["final"] is not None
    assert out["final"].difficulty == 5  # clamp a 5
    # F12-T2: el RoutineOutput final SIEMPRE trae rubric válida y reference_solution.
    assert len(out["final"].rubric) >= 3
    assert all(1 <= c.weight <= 5 for c in out["final"].rubric)
    assert out["final"].reference_solution and len(out["final"].reference_solution) >= 20
    assert validate_difficulty_router(out) == "accept"


# ==========================================================================
# Test golden: subgrafo end-to-end con LLM mockeado
# ==========================================================================

def test_golden_subgraph_produces_valid_output():
    """Perfil sintético + LLM mockeado → state final tiene `final` válido.

    F12-T2: el LLM mockeado emite rubric (3 ítems) y reference_solution.
    El test verifica que ambos campos se propagan al RoutineOutput final.
    """
    fake_output = RoutineOutput(
        title="Implement LRU cache from scratch",
        target_weakness="Caching",
        inverse_rag_snippet=None,
        expected_concepts=["LRU", "eviction"],
        difficulty=3,
        challenge_md=(
            "## Challenge\n\nImplement an LRU cache. Acceptance:\n"
            "1. O(1) get/put.\n2. Capacity eviction.\n3. Tests."
        ),
        rubric=_sample_rubric(),
        reference_solution=_SAMPLE_REFERENCE_SOLUTION,
    )

    # Patch del llm global usado por synthesize_challenge_node
    from src.graph.nodes.routine import synthesize_challenge as synth_mod

    llm_mock = _make_llm_mock_returning(fake_output)
    original_llm = synth_mod.llm
    synth_mod.llm = llm_mock
    try:
        graph = build_routine_graph()
        result = asyncio.run(
            graph.ainvoke(
                {
                    "user_id": "u1",
                    "user_profile": _profile_with_concepts(),
                    "target_weakness": None,
                    "regen_count": 0,
                }
            )
        )
    finally:
        synth_mod.llm = original_llm

    # El subgrafo eligió Caching (menor mastery)
    assert result["target_weakness"] == "Caching"
    # `raw_snippet` puede estar vacío (corpus bad_code_corpus no construido)
    # o contener un snippet real (corpus disponible tras F5-T3). El golden
    # solo asegura que el campo existe en el state — su contenido depende
    # del entorno local del operador.
    assert "raw_snippet" in result
    # Final populado por validate_difficulty
    assert result["final"] is not None
    final: RoutineOutput = result["final"]
    assert final.title == "Implement LRU cache from scratch"
    assert final.target_weakness == "Caching"
    assert 1 <= final.difficulty <= 5
    assert final.expected_concepts == ["LRU", "eviction"]
    # No debe haber loopeado (mastery 0.2 → ceiling 3, llm_difficulty=3 cabe)
    assert result.get("regen_count", 0) == 0
    # F12-T2: rubric y reference_solution end-to-end
    assert len(final.rubric) == 3
    assert final.rubric[0].concept == "LRU"
    assert final.reference_solution == _SAMPLE_REFERENCE_SOLUTION
