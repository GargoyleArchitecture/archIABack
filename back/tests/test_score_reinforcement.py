"""F16-T2: tests de score_reinforcement (EWMA bidireccional, idempotente).

Fake Store en memoria (mismo contrato que test_reflection_dispatcher).
`sync_profile` no hace HTTP sin INTERNAL_API_TOKEN; no se mockea.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from src.services import score_reinforcement as sr


class _FakeItem:
    def __init__(self, value):
        self.value = value


class _FakeStore:
    def __init__(self, initial: Optional[dict] = None):
        self._data: dict = {}
        if initial:
            for (ns, key), value in initial.items():
                self._data[(tuple(ns), key)] = value

    async def aget(self, ns, key):
        v = self._data.get((tuple(ns), key))
        return _FakeItem(v) if v is not None else None

    async def aput(self, ns, key, value):
        self._data[(tuple(ns), key)] = value


def _store_with(mastery: float, name: str = "Caching") -> _FakeStore:
    return _FakeStore({
        (("user", "u-1", "profile"), "profile"): {
            "user_id": "u-1",
            "evaluated_concepts": [
                {"name": name, "mastery": mastery, "last_seen_at": "2026-05-01T00:00:00Z"},
            ],
        }
    })


def _mastery(profile: dict, name: str = "Caching") -> float:
    for c in profile["evaluated_concepts"]:
        if c["name"].lower() == name.lower():
            return c["mastery"]
    raise AssertionError("concept not found")


def _run(coro):
    return asyncio.run(coro)


# ── EWMA bidireccional ───────────────────────────────────────────────────────

def test_high_score_raises_mastery(monkeypatch):
    monkeypatch.setenv("SCORE_REINFORCE_ALPHA", "0.3")
    store = _store_with(0.40)
    out = _run(sr.dispatch_score_to_profile(
        "u-1", "Caching", 90, "att-1", trace_id="t", store=store,
    ))
    # 0.3*0.9 + 0.7*0.40 = 0.55
    assert out is not None
    assert abs(_mastery(out) - 0.55) < 1e-6


def test_low_score_lowers_mastery(monkeypatch):
    monkeypatch.setenv("SCORE_REINFORCE_ALPHA", "0.3")
    store = _store_with(0.80)
    out = _run(sr.dispatch_score_to_profile(
        "u-1", "Caching", 20, "att-1", trace_id="t", store=store,
    ))
    # 0.3*0.2 + 0.7*0.80 = 0.62  (baja: bidireccional)
    assert abs(_mastery(out) - 0.62) < 1e-6


def test_seeds_new_concept_with_score(monkeypatch):
    monkeypatch.setenv("SCORE_REINFORCE_ALPHA", "0.3")
    store = _FakeStore({(("user", "u-1", "profile"), "profile"): {"user_id": "u-1"}})
    out = _run(sr.dispatch_score_to_profile(
        "u-1", "Load Balancer", 36, "att-1", trace_id="t", store=store,
    ))
    assert abs(_mastery(out, "Load Balancer") - 0.36) < 1e-6


def test_clamps_to_unit_interval(monkeypatch):
    monkeypatch.setenv("SCORE_REINFORCE_ALPHA", "0.9")
    store = _store_with(1.0)
    out = _run(sr.dispatch_score_to_profile(
        "u-1", "Caching", 100, "att-1", trace_id="t", store=store,
    ))
    assert _mastery(out) <= 1.0


# ── idempotencia por attempt_id ──────────────────────────────────────────────

def test_idempotent_same_attempt_id_is_noop():
    store = _store_with(0.40)
    first = _run(sr.dispatch_score_to_profile(
        "u-1", "Caching", 90, "att-1", trace_id="t", store=store,
    ))
    m_after_first = _mastery(first)
    # Segunda vez con el MISMO attempt_id → no-op (None), mastery sin cambios.
    second = _run(sr.dispatch_score_to_profile(
        "u-1", "Caching", 10, "att-1", trace_id="t", store=store,
    ))
    assert second is None
    persisted = _run(store.aget(("user", "u-1", "profile"), "profile")).value
    assert abs(_mastery(persisted) - m_after_first) < 1e-6


def test_distinct_attempt_ids_both_apply():
    store = _store_with(0.40)
    _run(sr.dispatch_score_to_profile(
        "u-1", "Caching", 90, "att-1", trace_id="t", store=store,
    ))
    out2 = _run(sr.dispatch_score_to_profile(
        "u-1", "Caching", 90, "att-2", trace_id="t", store=store,
    ))
    assert out2 is not None  # distinto attempt_id → sí aplica


# ── guards / defensivo ───────────────────────────────────────────────────────

def test_empty_user_or_target_returns_none():
    store = _store_with(0.40)
    assert _run(sr.dispatch_score_to_profile("", "Caching", 90, "a", trace_id="t", store=store)) is None
    assert _run(sr.dispatch_score_to_profile("u-1", "", 90, "a", trace_id="t", store=store)) is None


def test_broken_store_returns_none_no_raise():
    class _Broken:
        async def aget(self, *a, **k):
            raise RuntimeError("store down")

    out = _run(sr.dispatch_score_to_profile(
        "u-1", "Caching", 90, "att-1", trace_id="t", store=_Broken(),
    ))
    assert out is None
