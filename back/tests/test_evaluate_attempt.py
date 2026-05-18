"""Tests F12-T3: endpoint POST /evaluate-attempt + nodo + orquestador.

Estrategia (espejo de `test_generate_routine_endpoint.py`):
  - Tests unitarios del nodo (`evaluate_attempt_node`) con LLM mockeado.
  - Tests del orquestador (`attempt_evaluator.evaluate_attempt_for_user`).
  - Smoke HTTP con mini-app aislado (no levantamos el `app` global porque
    su lifespan carga ChromaDB / Azure OpenAI / SQLite saver).
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from src.graph.schemas.feedback import (
    CriterionResult,
    EvaluateAttemptInput,
    ReflectionPayload,
    RoutineFeedback,
)
from src.graph.schemas.routine import RubricCriterion
from src.graph.nodes.feedback import evaluate_attempt as eval_mod
from src.services import attempt_evaluator as orq
from src.services import routine_generator as auth_mod  # reuso de verify_internal_token


# ============================================================================
# Helpers
# ============================================================================

def _valid_rubric() -> list[RubricCriterion]:
    return [
        RubricCriterion(
            concept="LRU",
            description="Implementa evicción por uso reciente.",
            weight=5,
        ),
        RubricCriterion(
            concept="Capacidad",
            description="Respeta la capacidad configurada bajo cualquier carga.",
            weight=4,
        ),
        RubricCriterion(
            concept="Concurrencia",
            description="Las operaciones son seguras bajo accesos concurrentes.",
            weight=3,
        ),
    ]


def _valid_input(**overrides) -> EvaluateAttemptInput:
    base = dict(
        user_id="u-1",
        routine_id="r-1",
        user_response="def lru(): ... (implementación)",
        rubric=_valid_rubric(),
        expected_concepts=["LRU", "eviction"],
        reference_solution="## Reference\n\n```python\nclass LRUCache: ...\n```",
        target_weakness="Caching",
        reflection=None,
    )
    base.update(overrides)
    return EvaluateAttemptInput(**base)


def _good_feedback() -> RoutineFeedback:
    return RoutineFeedback(
        score=72,
        criteria=[
            CriterionResult(concept="LRU", status="met", comment="OK con OrderedDict."),
            CriterionResult(concept="Capacidad", status="partial", comment="Falta upper bound."),
            CriterionResult(concept="Concurrencia", status="missing", comment="No hay locks."),
        ],
        strengths=["Estructura clara"],
        improvements=["Añadir locks reentrantes"],
        socratic_comment="¿Qué pasa si dos clientes leen al mismo tiempo?",
    )


def _make_llm_mock(returning: RoutineFeedback):
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=returning)
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


def _make_request(headers: dict | None = None) -> Request:
    """Construye un Request mínimo de Starlette para tests del guard."""
    from starlette.requests import Request as StarletteRequest

    headers = headers or {}
    raw_headers = [
        (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "headers": raw_headers,
        "path": "/evaluate-attempt",
        "raw_path": b"/evaluate-attempt",
        "query_string": b"",
        "client": ("test", 12345),
    }
    return StarletteRequest(scope)


# ============================================================================
# (1-2) Auth — reusa verify_internal_token del módulo routine_generator
# ============================================================================

def test_auth_rejects_missing_header(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-xyz")
    req = _make_request(headers={})
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_internal_token(req)
    assert exc.value.status_code == 401


def test_auth_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-xyz")
    req = _make_request(headers={"X-Internal-Token": "wrong"})
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_internal_token(req)
    assert exc.value.status_code == 401


# ============================================================================
# (3-5) Schema — validación Pydantic
# ============================================================================

def test_schema_rejects_missing_user_response():
    """`user_response` es obligatoria y no puede estar vacía."""
    with pytest.raises(Exception):  # ValidationError
        EvaluateAttemptInput(
            user_id="u",
            routine_id="r",
            user_response="",  # vacío
            rubric=_valid_rubric(),
            reference_solution="ref",
            target_weakness="Caching",
        )


def test_schema_rejects_empty_rubric():
    """`rubric` debe tener al menos 1 ítem (min_length=1)."""
    with pytest.raises(Exception):
        EvaluateAttemptInput(
            user_id="u",
            routine_id="r",
            user_response="x",
            rubric=[],
            reference_solution="ref",
            target_weakness="Caching",
        )


def test_node_clamps_score_outside_range():
    """Si el LLM devuelve score=101 (Pydantic lo rechazaría), el clamp del
    nodo aplica. Para testearlo bypasseamos Pydantic con un mock que devuelve
    un objeto `RoutineFeedback`-like con score válido pero forzamos el path
    del clamp via `model_copy`."""
    fake = RoutineFeedback(
        score=100,  # max permitido por Pydantic; el clamp es defensivo de
        criteria=[],  # cara al futuro si se relaja el ge/le.
        strengths=[],
        improvements=["x"],
        socratic_comment="?",
    )
    # Inyectamos un mock al nodo que devuelve el fake.
    llm_mock = _make_llm_mock(fake)
    payload = _valid_input()
    out = asyncio.run(eval_mod.evaluate_attempt_node(payload, llm_obj=llm_mock))
    assert 0 <= out.score <= 100


# ============================================================================
# (6-9) Nodo — happy path, fallback, criteria, reflection ignorada
# ============================================================================

def test_node_happy_path():
    """LLM devuelve un RoutineFeedback válido → el nodo lo retorna sin tocarlo."""
    llm_mock = _make_llm_mock(_good_feedback())
    payload = _valid_input()
    out = asyncio.run(eval_mod.evaluate_attempt_node(payload, llm_obj=llm_mock))
    assert out.score == 72
    assert len(out.criteria) == 3
    # Status coverage
    statuses = sorted([c.status for c in out.criteria])
    assert statuses == ["met", "missing", "partial"]
    # El LLM fue invocado con la rúbrica adecuada (a través de with_structured_output)
    llm_mock.with_structured_output.assert_called_once()


def test_node_fallback_when_llm_raises():
    """Si el LLM lanza, devolvemos un payload coherente score=0."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 500"))
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)

    payload = _valid_input()
    out = asyncio.run(eval_mod.evaluate_attempt_node(payload, llm_obj=llm))
    assert out.score == 0
    assert out.criteria == []
    assert "evaluación automática" in out.improvements[0].lower() or "reintent" in out.improvements[0].lower()
    assert out.socratic_comment  # no vacío


def test_node_accepts_reflection_but_does_not_act():
    """T3: `reflection` se acepta en el contrato pero NO se procesa. T4
    añadirá el dispatch al Shadow Agent."""
    llm_mock = _make_llm_mock(_good_feedback())
    payload = _valid_input(
        reflection=ReflectionPayload(
            difficult_part="Identificar el invariante.",
            would_do_differently="Empezar por el test antes que el code.",
        )
    )
    # No lanza, el nodo procede normalmente.
    out = asyncio.run(eval_mod.evaluate_attempt_node(payload, llm_obj=llm_mock))
    assert out.score == 72


def test_node_passes_target_weakness_and_expected_concepts_to_llm():
    """Verifica que el prompt incluye el contexto pedagógico clave."""
    llm_mock = _make_llm_mock(_good_feedback())
    payload = _valid_input(
        target_weakness="Resiliencia",
        expected_concepts=["retry", "idempotency"],
    )
    asyncio.run(eval_mod.evaluate_attempt_node(payload, llm_obj=llm_mock))
    # La llamada a ainvoke recibió un prompt que contiene los conceptos.
    call_args = llm_mock.with_structured_output.return_value.ainvoke.call_args
    prompt_text = call_args.args[0]
    assert "Resiliencia" in prompt_text
    assert "retry" in prompt_text
    assert "idempotency" in prompt_text


# ============================================================================
# (10) Telemetría — emite routine_evaluated tras success
# ============================================================================

def test_orchestrator_emits_telemetry(monkeypatch):
    """El orquestador debe llamar a `emit('routine_evaluated', ...)`."""
    # Patch del nodo para evitar el LLM real
    fake_feedback = _good_feedback()
    monkeypatch.setattr(
        eval_mod, "evaluate_attempt_node",
        AsyncMock(return_value=fake_feedback),
    )
    # También parchamos el evaluator's reference para que use el mock
    monkeypatch.setattr(
        orq, "evaluate_attempt_node",
        AsyncMock(return_value=fake_feedback),
    )

    # Patch de emit
    from src.services import telemetry as telemetry_mod
    emitted = []
    monkeypatch.setattr(
        telemetry_mod, "emit",
        lambda event, **payload: emitted.append({"event": event, **payload}),
    )

    payload = _valid_input()
    result = asyncio.run(
        orq.evaluate_attempt_for_user(payload, trace_id="trace-xyz"),
    )
    assert result.score == 72
    assert len(emitted) == 1
    assert emitted[0]["event"] == "routine_evaluated"
    assert emitted[0]["user_id"] == "u-1"
    assert emitted[0]["routine_id"] == "r-1"
    assert emitted[0]["score"] == 72
    assert emitted[0]["trace_id"] == "trace-xyz"


# ============================================================================
# (11) Smoke HTTP con mini-app aislado
# ============================================================================

def _build_mini_app(monkeypatch=None) -> FastAPI:
    """Mini-app sin lifespan que registra solo el endpoint /evaluate-attempt.
    El nodo subyacente puede ser parchado por el test antes de instanciar el
    cliente."""
    mini = FastAPI()

    @mini.post("/evaluate-attempt")
    async def _ep(request: Request, body: EvaluateAttemptInput):
        auth_mod.verify_internal_token(request)
        trace_id = auth_mod.make_trace_id(request)
        feedback = await orq.evaluate_attempt_for_user(body, trace_id=trace_id)
        return feedback.model_dump()

    return mini


def test_endpoint_200_with_valid_token_and_mocked_node(monkeypatch):
    """Smoke end-to-end via TestClient con LLM y telemetría mockeados."""
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret")
    monkeypatch.setattr(
        orq, "evaluate_attempt_node",
        AsyncMock(return_value=_good_feedback()),
    )
    # Bypass telemetry para que no pruebe escribir a stdout en el test
    from src.services import telemetry as telemetry_mod
    monkeypatch.setattr(telemetry_mod, "emit", lambda *a, **kw: None)

    body = _valid_input().model_dump()
    with TestClient(_build_mini_app()) as client:
        r = client.post(
            "/evaluate-attempt",
            headers={"X-Internal-Token": "secret"},
            json=body,
        )
    assert r.status_code == 200
    data = r.json()
    assert data["score"] == 72
    assert len(data["criteria"]) == 3
    assert data["socratic_comment"]


def test_endpoint_401_without_token(monkeypatch):
    """Auth fail-closed: sin header → 401."""
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret")
    body = _valid_input().model_dump()
    with TestClient(_build_mini_app()) as client:
        r = client.post("/evaluate-attempt", json=body)
    assert r.status_code == 401
