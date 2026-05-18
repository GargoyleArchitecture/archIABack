"""Tests F5-T2: endpoint POST /generate-routine + orquestador.

Estrategia:
- Tests unitarios del orquestador (`routine_generator.py`) sin levantar
  TestClient: invocan las funciones directamente con mocks.
- Tests HTTP usan un `mini_app` aislado (no el `app` global de `main.py`)
  para evitar el lifespan completo que requiere DB, ChromaDB, OpenAI key, etc.
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.graph import resources as resources_mod
from src.graph.schemas.routine import RoutineOutput, RubricCriterion
from src.services import routine_generator as orq


# ============================================================================
# Helpers
# ============================================================================

class _FakeStoreItem:
    def __init__(self, value):
        self.value = value


class _FakeStore:
    def __init__(self, initial=None):
        self._data = dict(initial or {})

    async def aget(self, ns, key):
        if (ns, key) in self._data:
            return _FakeStoreItem(self._data[(ns, key)])
        return None


def _patch_store(monkeypatch, store):
    monkeypatch.setattr(resources_mod, "_store_holder", {"instance": store})


def _patch_routine_graph(monkeypatch, graph):
    monkeypatch.setattr(
        resources_mod, "_routine_graph_holder", {"instance": graph}
    )


def _make_request(headers: dict | None = None) -> Request:
    """Construye un Request mínimo para invocar funciones del orquestador."""
    from starlette.requests import Request as StarletteRequest

    headers = headers or {}
    raw_headers = [
        (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "headers": raw_headers,
        "path": "/generate-routine",
        "raw_path": b"/generate-routine",
        "query_string": b"",
        "client": ("test", 12345),
    }
    return StarletteRequest(scope)


def _good_routine_output() -> RoutineOutput:
    """RoutineOutput válido tras F12-T2: incluye rubric (3 ítems) y
    reference_solution (≥20 chars), ambos campos requeridos por el schema
    extendido."""
    return RoutineOutput(
        title="Refactor LRU cache",
        target_weakness="Caching",
        inverse_rag_snippet=None,
        expected_concepts=["LRU", "eviction"],
        difficulty=3,
        challenge_md="## Challenge\n\nImplement an LRU cache from scratch.",
        rubric=[
            RubricCriterion(
                concept="LRU",
                description="Implements eviction by least-recently-used order.",
                weight=5,
            ),
            RubricCriterion(
                concept="Capacity",
                description="Respects the configured capacity bound at all times.",
                weight=4,
            ),
            RubricCriterion(
                concept="Correctness",
                description="get/put operate in expected average complexity.",
                weight=3,
            ),
        ],
        reference_solution=(
            "## Reference\n\n```python\nfrom collections import OrderedDict\n"
            "class LRUCache: ...\n```\n\nUses OrderedDict for O(1) operations."
        ),
    )


# ============================================================================
# verify_internal_token
# ============================================================================

def test_verify_internal_token_rejects_missing_header(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-xyz")
    req = _make_request(headers={})
    with pytest.raises(HTTPException) as exc:
        orq.verify_internal_token(req)
    assert exc.value.status_code == 401


def test_verify_internal_token_rejects_wrong_value(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-xyz")
    req = _make_request(headers={"X-Internal-Token": "WRONG"})
    with pytest.raises(HTTPException) as exc:
        orq.verify_internal_token(req)
    assert exc.value.status_code == 401


def test_verify_internal_token_accepts_correct(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-xyz")
    req = _make_request(headers={"X-Internal-Token": "secret-xyz"})
    orq.verify_internal_token(req)  # no raise


def test_verify_internal_token_fail_closed_when_env_empty(monkeypatch):
    """Si INTERNAL_API_TOKEN está vacío, también rechaza (fail-closed)."""
    monkeypatch.setenv("INTERNAL_API_TOKEN", "")
    req = _make_request(headers={"X-Internal-Token": "anything"})
    with pytest.raises(HTTPException) as exc:
        orq.verify_internal_token(req)
    assert exc.value.status_code == 401


# ============================================================================
# make_trace_id
# ============================================================================

def test_make_trace_id_uses_provided_header():
    req = _make_request(headers={"X-Trace-Id": "trace-123"})
    assert orq.make_trace_id(req) == "trace-123"


def test_make_trace_id_generates_uuid_when_absent():
    req = _make_request(headers={})
    tid = orq.make_trace_id(req)
    # UUID4 genera 36 chars con guiones; verificación blanda
    assert len(tid) == 36
    assert tid.count("-") == 4


# ============================================================================
# load_user_profile_for_routine
# ============================================================================

def test_load_profile_returns_empty_when_user_id_blank(monkeypatch):
    _patch_store(monkeypatch, _FakeStore())
    out = asyncio.run(orq.load_user_profile_for_routine(""))
    assert out == {}


def test_load_profile_returns_empty_when_store_miss(monkeypatch):
    _patch_store(monkeypatch, _FakeStore())  # vacío
    out = asyncio.run(orq.load_user_profile_for_routine("u1"))
    assert out == {}


def test_load_profile_applies_decay(monkeypatch):
    """Si el Store tiene perfil con last_seen_at viejo, mastery se decay-ea
    en lectura (mismo comportamiento que boot_node)."""
    from datetime import datetime, timezone, timedelta

    last_seen = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    pre = {
        "user_id": "u1",
        "evaluated_concepts": [
            {"name": "Caching", "mastery": 1.0, "last_seen_at": last_seen}
        ],
    }
    store = _FakeStore(initial={(("user", "u1", "profile"), "profile"): pre})
    _patch_store(monkeypatch, store)

    out = asyncio.run(orq.load_user_profile_for_routine("u1"))
    concept = out["evaluated_concepts"][0]
    assert concept["mastery"] < 1.0  # decay aplicado
    assert concept["mastery_original"] == 1.0  # preservado por apply_decay


# ============================================================================
# generate_routine_for_user
# ============================================================================

def test_generate_routine_invokes_subgraph_and_returns_final(monkeypatch):
    _patch_store(monkeypatch, _FakeStore())  # perfil vacío

    final = _good_routine_output()
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"final": final, "regen_count": 0}
    )
    _patch_routine_graph(monkeypatch, fake_graph)

    out = asyncio.run(
        orq.generate_routine_for_user(
            user_id="u1", target_weakness="Caching", trace_id="trace-xyz"
        )
    )
    assert isinstance(out, RoutineOutput)
    assert out.target_weakness == "Caching"
    fake_graph.ainvoke.assert_awaited_once()
    # Verifica que el initial_state pasado al grafo es el correcto
    initial_state = fake_graph.ainvoke.call_args[0][0]
    assert initial_state["user_id"] == "u1"
    assert initial_state["target_weakness"] == "Caching"
    assert initial_state["regen_count"] == 0


def test_generate_routine_rejects_blank_user_id(monkeypatch):
    _patch_store(monkeypatch, _FakeStore())
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            orq.generate_routine_for_user(
                user_id="", target_weakness=None, trace_id="t"
            )
        )
    assert exc.value.status_code == 400


def test_generate_routine_500_when_subgraph_returns_no_final(monkeypatch):
    _patch_store(monkeypatch, _FakeStore())

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={"final": None})
    _patch_routine_graph(monkeypatch, fake_graph)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            orq.generate_routine_for_user(
                user_id="u1", target_weakness="Caching", trace_id="t"
            )
        )
    assert exc.value.status_code == 500


def test_generate_routine_500_when_subgraph_raises(monkeypatch):
    _patch_store(monkeypatch, _FakeStore())

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 500"))
    _patch_routine_graph(monkeypatch, fake_graph)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            orq.generate_routine_for_user(
                user_id="u1", target_weakness="Caching", trace_id="t"
            )
        )
    assert exc.value.status_code == 500


# ============================================================================
# Smoke HTTP con mini-app aislado
# ============================================================================

class _MiniBody(BaseModel):
    """Body del endpoint en el mini-app de tests. Definido top-level porque
    FastAPI tiene issues resolviendo BaseModel declarados dentro de funciones."""
    user_id: str
    target_weakness: Optional[str] = None


def _build_mini_app() -> FastAPI:
    """Mini-app que monta SOLO el endpoint generate-routine sin lifespan."""
    mini = FastAPI()

    @mini.post("/generate-routine")
    async def _ep(request: Request, body: _MiniBody):
        orq.verify_internal_token(request)
        trace_id = orq.make_trace_id(request)
        final = await orq.generate_routine_for_user(
            user_id=body.user_id,
            target_weakness=body.target_weakness,
            trace_id=trace_id,
        )
        return final.model_dump()

    return mini


def test_endpoint_401_without_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret")
    _patch_store(monkeypatch, _FakeStore())
    _patch_routine_graph(monkeypatch, MagicMock())  # no se llega a invocar

    with TestClient(_build_mini_app()) as client:
        r = client.post(
            "/generate-routine",
            json={"user_id": "u1", "target_weakness": "Caching"},
        )
    assert r.status_code == 401


def test_endpoint_401_with_wrong_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret")
    _patch_store(monkeypatch, _FakeStore())
    _patch_routine_graph(monkeypatch, MagicMock())

    with TestClient(_build_mini_app()) as client:
        r = client.post(
            "/generate-routine",
            headers={"X-Internal-Token": "WRONG"},
            json={"user_id": "u1", "target_weakness": "Caching"},
        )
    assert r.status_code == 401


def test_endpoint_200_with_valid_token_and_mocked_graph(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret")
    _patch_store(monkeypatch, _FakeStore())

    final = _good_routine_output()
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"final": final, "regen_count": 0}
    )
    _patch_routine_graph(monkeypatch, fake_graph)

    with TestClient(_build_mini_app()) as client:
        r = client.post(
            "/generate-routine",
            headers={"X-Internal-Token": "secret"},
            json={"user_id": "u1", "target_weakness": "Caching"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Refactor LRU cache"
    assert body["target_weakness"] == "Caching"
    assert 1 <= body["difficulty"] <= 5


def test_endpoint_200_when_target_weakness_omitted(monkeypatch):
    """Sin target_weakness, el subgrafo lo infiere; el endpoint NO falla."""
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret")
    _patch_store(monkeypatch, _FakeStore())

    final = _good_routine_output()
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(
        return_value={"final": final, "regen_count": 0}
    )
    _patch_routine_graph(monkeypatch, fake_graph)

    with TestClient(_build_mini_app()) as client:
        r = client.post(
            "/generate-routine",
            headers={"X-Internal-Token": "secret"},
            json={"user_id": "u1"},  # sin target_weakness
        )
    assert r.status_code == 200
    initial_state = fake_graph.ainvoke.call_args[0][0]
    assert initial_state["target_weakness"] is None  # se delega al subgrafo


def test_endpoint_422_when_user_id_missing(monkeypatch):
    """Pydantic rechaza body sin user_id (validación estándar de FastAPI)."""
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret")
    with TestClient(_build_mini_app()) as client:
        r = client.post(
            "/generate-routine",
            headers={"X-Internal-Token": "secret"},
            json={},
        )
    assert r.status_code == 422
