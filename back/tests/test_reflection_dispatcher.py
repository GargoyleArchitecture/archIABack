"""Tests F12-T4: reflection_dispatcher + integración con evaluate_attempt.

Cobertura:
  - Función pura `_mastery_delta` con thresholds.
  - Refuerzo de concepto existente (delta, clamp, last_seen_at).
  - Insert de concepto nuevo cuando target_weakness no existe.
  - Match case-insensitive del nombre del concepto.
  - Defensivo: Store roto → retorna None, NO lanza.
  - Integración: evaluate_attempt_for_user dispara el task cuando aplica
    (test golden del criterio del backlog: `await asyncio.sleep(0.1)` y el
    Store muestra mastery incrementado).
  - Sin reflection → dispatcher NO se invoca.

Patrones:
  - Fake `Store` en memoria con `aget` / `aput` para tests deterministas
    sin tocar el Store real del lifespan.
  - Patch de `sync_profile` y `telemetry.emit` para evitar I/O HTTP / log.
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.schemas.feedback import (
    EvaluateAttemptInput,
    ReflectionPayload,
    RoutineFeedback,
)
from src.graph.schemas.routine import RubricCriterion
from src.services import attempt_evaluator as orq
from src.services import reflection_dispatcher as rd


# ---------------------------------------------------------------------------
# Fake Store: replica el contrato mínimo de LangGraph Store.
# ---------------------------------------------------------------------------

class _FakeItem:
    def __init__(self, value):
        self.value = value


class _FakeStore:
    """Store async en memoria. `aget` devuelve _FakeItem o None; `aput`
    sobrescribe la entrada."""

    def __init__(self, initial: Optional[dict] = None):
        # mapping (ns_tuple, key) -> value dict
        self._data: dict = {}
        if initial:
            for (ns, key), value in initial.items():
                self._data[(tuple(ns), key)] = value

    async def aget(self, ns, key):
        v = self._data.get((tuple(ns), key))
        return _FakeItem(v) if v is not None else None

    async def aput(self, ns, key, value):
        self._data[(tuple(ns), key)] = value


def _profile_with_caching(mastery: float = 0.40) -> dict:
    return {
        "user_id": "u-1",
        "strengths": [],
        "weaknesses": ["Caching"],
        "evaluated_concepts": [
            {
                "name": "Caching",
                "mastery": mastery,
                "last_seen_at": "2026-05-01T00:00:00+00:00",
            },
            {
                "name": "Modularity",
                "mastery": 0.80,
                "last_seen_at": "2026-05-01T00:00:00+00:00",
            },
        ],
        "confidence": 0.7,
    }


def _good_reflection() -> ReflectionPayload:
    return ReflectionPayload(
        difficult_part="Identificar el invariante del cache LRU.",
        would_do_differently="Empezar por los tests antes que la implementación.",
    )


def _valid_input(**overrides) -> EvaluateAttemptInput:
    base = dict(
        user_id="u-1",
        routine_id="r-1",
        user_response="def lru(): ...",
        rubric=[
            RubricCriterion(concept="LRU", description="x" * 12, weight=5),
            RubricCriterion(concept="Capacity", description="y" * 12, weight=4),
            RubricCriterion(concept="Thread safety", description="z" * 12, weight=3),
        ],
        expected_concepts=["LRU", "eviction"],
        reference_solution="## Ref\n\nUse OrderedDict.",
        target_weakness="Caching",
        reflection=None,
    )
    base.update(overrides)
    return EvaluateAttemptInput(**base)


def _patch_sync_and_telemetry(monkeypatch):
    """Silencia profile_sync y telemetry para que los tests no hagan I/O."""
    from src.services import profile_sync as ps_mod
    from src.services import telemetry as tel_mod

    monkeypatch.setattr(ps_mod, "sync_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(tel_mod, "emit", lambda *a, **kw: None)


# ============================================================================
# (1) _mastery_delta — función pura
# ============================================================================

def test_mastery_delta_thresholds():
    assert rd._mastery_delta(0) == 0.0
    assert rd._mastery_delta(60) == 0.0
    assert rd._mastery_delta(69) == 0.0
    assert rd._mastery_delta(70) == 0.05
    assert rd._mastery_delta(80) == 0.05
    assert rd._mastery_delta(84) == 0.05
    assert rd._mastery_delta(85) == 0.10
    assert rd._mastery_delta(100) == 0.10
    # Defensivo: tipos raros
    assert rd._mastery_delta("foo") == 0.0
    assert rd._mastery_delta(None) == 0.0


# ============================================================================
# (2) Concepto existente: delta + clamp + last_seen_at
# ============================================================================

def test_dispatch_reinforces_existing_concept(monkeypatch):
    _patch_sync_and_telemetry(monkeypatch)
    store = _FakeStore({
        (("user", "u-1", "profile"), "profile"): _profile_with_caching(mastery=0.40),
    })

    result = asyncio.run(
        rd.dispatch_reflection_to_profile(
            user_id="u-1",
            target_weakness="Caching",
            score=80,
            reflection=_good_reflection(),
            trace_id="t-1",
            store=store,
        )
    )
    assert result is not None
    caching = next(c for c in result["evaluated_concepts"] if c["name"] == "Caching")
    # 0.40 + 0.05 (delta para 70..84)
    assert caching["mastery"] == pytest.approx(0.45)
    assert caching["last_seen_at"]  # actualizado
    # Modularity intacto
    modularity = next(c for c in result["evaluated_concepts"] if c["name"] == "Modularity")
    assert modularity["mastery"] == 0.80


# ============================================================================
# (3) Clamp en 1.0 cuando el delta lo desbordaría
# ============================================================================

def test_dispatch_clamps_mastery_to_one(monkeypatch):
    _patch_sync_and_telemetry(monkeypatch)
    store = _FakeStore({
        (("user", "u-1", "profile"), "profile"): _profile_with_caching(mastery=0.95),
    })
    result = asyncio.run(
        rd.dispatch_reflection_to_profile(
            user_id="u-1",
            target_weakness="Caching",
            score=90,
            reflection=_good_reflection(),
            trace_id="t-2",
            store=store,
        )
    )
    caching = next(c for c in result["evaluated_concepts"] if c["name"] == "Caching")
    # 0.95 + 0.10 = 1.05 → clamp a 1.0
    assert caching["mastery"] == pytest.approx(1.0)


# ============================================================================
# (4) Concepto no existe: se añade con seed mastery clampeado [0.5, 0.95]
# ============================================================================

def test_dispatch_adds_new_concept_when_absent(monkeypatch):
    _patch_sync_and_telemetry(monkeypatch)
    store = _FakeStore({
        (("user", "u-1", "profile"), "profile"): {
            "user_id": "u-1",
            "evaluated_concepts": [{"name": "Modularity", "mastery": 0.8}],
        },
    })
    result = asyncio.run(
        rd.dispatch_reflection_to_profile(
            user_id="u-1",
            target_weakness="Distributed Tracing",
            score=75,
            reflection=_good_reflection(),
            trace_id="t-3",
            store=store,
        )
    )
    assert result is not None
    names = [c["name"] for c in result["evaluated_concepts"]]
    assert "Distributed Tracing" in names
    seeded = next(c for c in result["evaluated_concepts"] if c["name"] == "Distributed Tracing")
    # 75/100 = 0.75 (cae dentro de [0.5, 0.95])
    assert seeded["mastery"] == pytest.approx(0.75)
    assert seeded["evidence"] == "Reflexión post-reto"


# ============================================================================
# (5) Match case-insensitive del nombre del concepto
# ============================================================================

def test_dispatch_matches_concept_case_insensitive(monkeypatch):
    _patch_sync_and_telemetry(monkeypatch)
    store = _FakeStore({
        (("user", "u-1", "profile"), "profile"): _profile_with_caching(mastery=0.40),
    })
    # `target_weakness="caching"` debe matchear con `name="Caching"`.
    result = asyncio.run(
        rd.dispatch_reflection_to_profile(
            user_id="u-1",
            target_weakness="caching",
            score=80,
            reflection=_good_reflection(),
            trace_id="t-4",
            store=store,
        )
    )
    names = [c["name"] for c in result["evaluated_concepts"]]
    # NO se debe haber creado un concepto duplicado "caching"
    assert names.count("Caching") == 1
    assert "caching" not in names
    caching = next(c for c in result["evaluated_concepts"] if c["name"] == "Caching")
    assert caching["mastery"] == pytest.approx(0.45)


# ============================================================================
# (6) Defensivo: score < 70 → no-op (sin tocar Store)
# ============================================================================

def test_dispatch_noop_below_threshold(monkeypatch):
    _patch_sync_and_telemetry(monkeypatch)
    store = _FakeStore({
        (("user", "u-1", "profile"), "profile"): _profile_with_caching(mastery=0.40),
    })
    result = asyncio.run(
        rd.dispatch_reflection_to_profile(
            user_id="u-1",
            target_weakness="Caching",
            score=60,
            reflection=_good_reflection(),
            trace_id="t-5",
            store=store,
        )
    )
    assert result is None
    # El Store sigue con el mastery original (no se actualizó).
    raw = store._data[(("user", "u-1", "profile"), "profile")]
    caching = next(c for c in raw["evaluated_concepts"] if c["name"] == "Caching")
    assert caching["mastery"] == 0.40


# ============================================================================
# (7) Defensivo: Store roto (aget/aput lanzan) → None, NO propaga
# ============================================================================

def test_dispatch_defensive_when_store_aget_fails(monkeypatch):
    _patch_sync_and_telemetry(monkeypatch)
    broken = MagicMock()
    broken.aget = AsyncMock(side_effect=RuntimeError("store down"))
    broken.aput = AsyncMock()

    result = asyncio.run(
        rd.dispatch_reflection_to_profile(
            user_id="u-1",
            target_weakness="Caching",
            score=80,
            reflection=_good_reflection(),
            trace_id="t-6",
            store=broken,
        )
    )
    assert result is None
    # aput nunca debió llamarse cuando aget falló.
    broken.aput.assert_not_called()


def test_dispatch_defensive_when_store_aput_fails(monkeypatch):
    _patch_sync_and_telemetry(monkeypatch)
    broken = MagicMock()
    broken.aget = AsyncMock(return_value=None)
    broken.aput = AsyncMock(side_effect=RuntimeError("disk full"))

    result = asyncio.run(
        rd.dispatch_reflection_to_profile(
            user_id="u-1",
            target_weakness="Caching",
            score=80,
            reflection=_good_reflection(),
            trace_id="t-7",
            store=broken,
        )
    )
    assert result is None


# ============================================================================
# (8) Integración: evaluate_attempt_for_user dispara dispatcher en background
# ============================================================================

def test_evaluate_attempt_dispatches_reflection_in_background(monkeypatch):
    """Test golden del criterio del backlog: con reflection no vacía y score=80,
    el Store muestra mastery incrementado tras `await asyncio.sleep(0.1)`."""
    _patch_sync_and_telemetry(monkeypatch)
    # Store fake compartido entre dispatcher y el assert.
    store = _FakeStore({
        (("user", "u-1", "profile"), "profile"): _profile_with_caching(mastery=0.40),
    })
    # Patch get_store en el módulo del dispatcher para que use nuestro fake.
    monkeypatch.setattr(rd, "get_store", lambda: store)

    # Patch del nodo evaluador para que retorne un feedback con score=80
    # sin invocar LLM.
    fake_feedback = RoutineFeedback(
        score=80,
        criteria=[],
        strengths=["x"],
        improvements=["y"],
        socratic_comment="?",
    )
    monkeypatch.setattr(
        orq, "evaluate_attempt_node",
        AsyncMock(return_value=fake_feedback),
    )

    async def _run():
        payload = _valid_input(reflection=_good_reflection())
        feedback = await orq.evaluate_attempt_for_user(payload, trace_id="t-int")
        # La respuesta HTTP ya está fuera. El task corre en background.
        # Esperamos que termine antes de inspeccionar el Store.
        pending = list(orq._pending_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return feedback

    result = asyncio.run(_run())
    assert result.score == 80

    # El Store debe mostrar Caching con mastery incrementado (0.40 + 0.05 = 0.45)
    raw = store._data[(("user", "u-1", "profile"), "profile")]
    caching = next(c for c in raw["evaluated_concepts"] if c["name"] == "Caching")
    assert caching["mastery"] == pytest.approx(0.45)


# ============================================================================
# (9) Integración: sin reflection → dispatcher NO se invoca
# ============================================================================

def test_evaluate_attempt_does_not_dispatch_without_reflection(monkeypatch):
    _patch_sync_and_telemetry(monkeypatch)
    dispatcher_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(rd, "dispatch_reflection_to_profile", dispatcher_mock)
    # Re-importar el módulo wireado por si el patch no llega — usamos el
    # alias `_reflection_dispatcher` que está en attempt_evaluator.
    monkeypatch.setattr(orq._reflection_dispatcher, "dispatch_reflection_to_profile", dispatcher_mock)

    fake_feedback = RoutineFeedback(
        score=90,
        criteria=[],
        strengths=["x"],
        improvements=["y"],
        socratic_comment="?",
    )
    monkeypatch.setattr(
        orq, "evaluate_attempt_node",
        AsyncMock(return_value=fake_feedback),
    )

    async def _run():
        payload = _valid_input(reflection=None)
        await orq.evaluate_attempt_for_user(payload, trace_id="t-int-2")
        # No hay tasks pendientes.
        pending = list(orq._pending_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(_run())
    dispatcher_mock.assert_not_called()
