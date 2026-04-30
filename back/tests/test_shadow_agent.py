"""Tests del Shadow Agent (F3-T2): merge_profile, cadencia y task entry."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.schemas.profile import ConceptScore, UserProfileEvaluation
from src.graph.nodes.profile_shadow import (
    merge_profile,
    should_trigger,
    shadow_eval_async,
    fire_shadow_eval,
    _format_messages_window,
    SHADOW_EVERY_N,
)


# ---- should_trigger ------------------------------------------------------

def test_should_trigger_below_threshold():
    assert should_trigger(0) is False
    assert should_trigger(SHADOW_EVERY_N - 1) is False


def test_should_trigger_at_threshold():
    assert should_trigger(SHADOW_EVERY_N) is True


def test_should_trigger_above_threshold():
    assert should_trigger(SHADOW_EVERY_N + 5) is True


def test_should_trigger_handles_none():
    # type: ignore - simulamos un state mal inicializado
    assert should_trigger(None) is False  # type: ignore[arg-type]


# ---- merge_profile -------------------------------------------------------

def _eval(concepts, conf=0.7, strengths=None, weaknesses=None):
    return UserProfileEvaluation(
        evaluated_concepts=[ConceptScore(name=n, mastery=m) for n, m in concepts],
        strengths=[ConceptScore(name=n, mastery=m) for n, m in (strengths or [])],
        weaknesses=[ConceptScore(name=n, mastery=m) for n, m in (weaknesses or [])],
        confidence=conf,
    )


def test_merge_profile_with_empty_prev_uses_new_values():
    new = _eval([("Modularidad", 0.8), ("Caching", 0.3)])
    out = merge_profile({}, new, alpha=0.7)
    masterys = {c["name"]: c["mastery"] for c in out["evaluated_concepts"]}
    assert masterys == {"Modularidad": 0.8, "Caching": 0.3}
    assert out["delta_from_previous"] == {"Modularidad": 0.8, "Caching": 0.3}


def test_merge_profile_weighted_average():
    prev = {
        "user_id": "u1",
        "evaluated_concepts": [
            {"name": "Modularidad", "mastery": 0.5, "evidence": ""},
        ],
    }
    new = _eval([("Modularidad", 1.0)])
    out = merge_profile(prev, new, alpha=0.7)
    # 0.7 * 1.0 + 0.3 * 0.5 = 0.85
    masterys = {c["name"]: c["mastery"] for c in out["evaluated_concepts"]}
    assert masterys["Modularidad"] == pytest.approx(0.85, abs=1e-4)
    assert out["delta_from_previous"]["Modularidad"] == pytest.approx(0.35, abs=1e-4)


def test_merge_profile_preserves_concepts_only_in_prev():
    prev = {
        "evaluated_concepts": [
            {"name": "A", "mastery": 0.5},
            {"name": "B", "mastery": 0.6},
        ],
    }
    new = _eval([("A", 0.9)])
    out = merge_profile(prev, new, alpha=0.7)
    names = sorted(c["name"] for c in out["evaluated_concepts"])
    assert names == ["A", "B"]
    # B no se evaluo en este turno -> queda intacto
    b = next(c for c in out["evaluated_concepts"] if c["name"] == "B")
    assert b["mastery"] == 0.6


def test_merge_profile_unions_strengths_case_insensitive():
    prev = {"strengths": ["Modularidad", "Patrones GoF"]}
    new = _eval([("Modularidad", 0.8)], strengths=[("modularidad", 0.8), ("Caching", 0.7)])
    out = merge_profile(prev, new, alpha=0.7)
    # Modularidad ya estaba; Caching se anade. No se duplica con casefold.
    assert sorted(s.casefold() for s in out["strengths"]) == sorted(
        ["modularidad", "patrones gof", "caching"]
    )


def test_merge_profile_writes_updated_at_and_user_id():
    prev = {"user_id": "u1"}
    new = _eval([("X", 0.5)])
    out = merge_profile(prev, new)
    assert out["user_id"] == "u1"
    assert "updated_at" in out and out["updated_at"]


# ---- _format_messages_window --------------------------------------------

def test_format_messages_window_truncates():
    from langchain_core.messages import HumanMessage, AIMessage

    msgs = [HumanMessage(content="hola"), AIMessage(content="x" * 1000)]
    out = _format_messages_window(msgs, window=2)
    assert "[USER] hola" in out
    assert "[ASSISTANT]" in out
    # cada contenido se trunca a 600 chars
    assistant_line = [ln for ln in out.split("\n\n") if ln.startswith("[ASSISTANT]")][0]
    assert len(assistant_line) <= 612  # "[ASSISTANT] " + 600 chars


def test_format_messages_window_empty():
    assert _format_messages_window([], window=4) == "(empty conversation)"


# ---- shadow_eval_async (con LLM y store mockeados) -----------------------

class _FakeStoreItem:
    def __init__(self, value):
        self.value = value


class _FakeStore:
    """InMemoryStore-like minimo para el test."""

    def __init__(self, initial=None):
        self._data = dict(initial or {})

    async def aget(self, ns, key):
        return _FakeStoreItem(self._data.get((ns, key))) if (ns, key) in self._data else None

    async def aput(self, ns, key, value):
        self._data[(ns, key)] = value


def _make_fake_llm(eval_obj: UserProfileEvaluation):
    """LLM con .with_structured_output().ainvoke() mockeado."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=eval_obj)
    fake_llm = MagicMock()
    fake_llm.with_structured_output = MagicMock(return_value=structured)
    return fake_llm


def test_shadow_eval_async_writes_merged_profile_to_store():
    from langchain_core.messages import HumanMessage, AIMessage

    fake_eval = _eval([("Modularidad", 0.9)], strengths=[("Modularidad", 0.9)], conf=0.8)
    fake_llm = _make_fake_llm(fake_eval)
    store = _FakeStore()

    msgs = [HumanMessage(content="explicame modularidad"), AIMessage(content="...")]

    out = asyncio.run(shadow_eval_async("u1", msgs, llm_obj=fake_llm, store=store))

    assert out is not None
    assert out["user_id"] == "u1"
    assert "Modularidad" in [c["name"] for c in out["evaluated_concepts"]]
    # Store recibio el escrito
    persisted = asyncio.run(store.aget(("user", "u1", "profile"), "profile"))
    assert persisted.value["user_id"] == "u1"


def test_shadow_eval_async_skips_when_user_id_empty():
    from langchain_core.messages import HumanMessage, AIMessage

    fake_eval = _eval([("X", 0.5)])
    fake_llm = _make_fake_llm(fake_eval)
    store = _FakeStore()
    msgs = [HumanMessage(content="hi"), AIMessage(content="hi")]

    out = asyncio.run(shadow_eval_async("", msgs, llm_obj=fake_llm, store=store))
    assert out is None


def test_shadow_eval_async_skips_when_messages_too_few():
    from langchain_core.messages import HumanMessage

    fake_eval = _eval([("X", 0.5)])
    fake_llm = _make_fake_llm(fake_eval)
    store = _FakeStore()

    out = asyncio.run(shadow_eval_async("u1", [HumanMessage(content="hi")], llm_obj=fake_llm, store=store))
    assert out is None


def test_shadow_eval_async_returns_none_on_llm_failure():
    """Si el evaluador lanza, NO se escribe al Store (perfil previo intacto)."""
    from langchain_core.messages import HumanMessage, AIMessage

    structured = MagicMock()
    structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
    fake_llm = MagicMock()
    fake_llm.with_structured_output = MagicMock(return_value=structured)

    pre_existing = {"user_id": "u1", "strengths": ["Original"], "evaluated_concepts": []}
    store = _FakeStore(initial={(("user", "u1", "profile"), "profile"): pre_existing})

    msgs = [HumanMessage(content="x"), AIMessage(content="y")]
    out = asyncio.run(shadow_eval_async("u1", msgs, llm_obj=fake_llm, store=store))
    assert out is None
    # Perfil previo intacto
    persisted = asyncio.run(store.aget(("user", "u1", "profile"), "profile"))
    assert persisted.value["strengths"] == ["Original"]


def test_shadow_eval_async_merges_with_existing_profile():
    from langchain_core.messages import HumanMessage, AIMessage

    pre_existing = {
        "user_id": "u1",
        "strengths": ["A"],
        "weaknesses": [],
        "evaluated_concepts": [{"name": "A", "mastery": 0.4}],
    }
    fake_eval = _eval([("A", 1.0), ("B", 0.5)], strengths=[("B", 0.5)], conf=0.8)
    fake_llm = _make_fake_llm(fake_eval)
    store = _FakeStore(initial={(("user", "u1", "profile"), "profile"): pre_existing})

    msgs = [HumanMessage(content="hola"), AIMessage(content="ok")]
    out = asyncio.run(shadow_eval_async("u1", msgs, llm_obj=fake_llm, store=store))
    assert out is not None
    masterys = {c["name"]: c["mastery"] for c in out["evaluated_concepts"]}
    # A: 0.7*1.0 + 0.3*0.4 = 0.82
    assert masterys["A"] == pytest.approx(0.82, abs=1e-4)
    assert masterys["B"] == 0.5
    # Strengths union (A pre-existing + B nuevo)
    assert sorted(s.casefold() for s in out["strengths"]) == ["a", "b"]


# ---- fire_shadow_eval (sync trigger from unifier) ------------------------

def test_fire_shadow_eval_skips_when_below_threshold():
    state = {"turn_count_since_eval": 0, "user_id": "u1", "messages": []}
    # No hay loop activo -> deberia retornar None sin lanzar
    result = fire_shadow_eval(state, final_text="x")
    assert result is None


def test_fire_shadow_eval_skips_when_no_user_id():
    state = {"turn_count_since_eval": 999, "user_id": "", "messages": []}
    result = fire_shadow_eval(state, final_text="x")
    assert result is None
