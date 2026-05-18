"""F16-T1: tests de attempt_sync.sync_attempt_feedback (sync-back IA→Negocio).

Mismo enfoque que test_profile_sync/test_profile_fetch: respx para mockear
httpx + monkeypatch de env. La función NUNCA debe lanzar al caller.
"""
import asyncio

import httpx
import respx

from src.services.attempt_sync import sync_attempt_feedback

_FEEDBACK = {
    "score": 72,
    "criteria": [{"concept": "LRU", "status": "met", "comment": "ok"}],
    "strengths": ["s"],
    "improvements": ["i"],
    "socratic_comment": "¿y si...?",
}


def _env(monkeypatch, *, token="tok", enabled="true"):
    monkeypatch.setenv("BUSINESS_API_BASE_URL", "http://negocio-test")
    monkeypatch.setenv("INTERNAL_API_TOKEN", token)
    monkeypatch.setenv("PROFILE_SYNC_ENABLED", enabled)
    monkeypatch.setenv("PROFILE_SYNC_TIMEOUT", "5")


# ── guards de configuración ──────────────────────────────────────────────────

def test_skipped_when_token_empty(monkeypatch):
    _env(monkeypatch, token="")
    assert asyncio.run(sync_attempt_feedback("att-1", _FEEDBACK)) is False


def test_skipped_when_attempt_id_empty(monkeypatch):
    _env(monkeypatch)
    assert asyncio.run(sync_attempt_feedback("", _FEEDBACK)) is False


def test_skipped_when_disabled(monkeypatch):
    _env(monkeypatch, enabled="false")
    assert asyncio.run(sync_attempt_feedback("att-1", _FEEDBACK)) is False


# ── respuestas HTTP ──────────────────────────────────────────────────────────

@respx.mock
def test_posts_payload_and_returns_true_on_2xx(monkeypatch):
    _env(monkeypatch)
    route = respx.post(
        "http://negocio-test/internal/routine-attempts/att-1/feedback"
    ).mock(return_value=httpx.Response(200, json={"id": "att-1"}))

    ok = asyncio.run(sync_attempt_feedback("att-1", _FEEDBACK))

    assert ok is True
    assert route.called
    sent = route.calls.last.request
    assert sent.headers["X-Internal-Token"] == "tok"
    import json as _json

    body = _json.loads(sent.content)
    assert body["score"] == 72
    assert body["feedback"] == _FEEDBACK


@respx.mock
def test_returns_false_and_never_raises_on_non_2xx(monkeypatch):
    _env(monkeypatch)
    respx.post(
        "http://negocio-test/internal/routine-attempts/att-1/feedback"
    ).mock(return_value=httpx.Response(404))

    # max_attempts=1 → sin sleeps de backoff (test rápido).
    ok = asyncio.run(sync_attempt_feedback("att-1", _FEEDBACK, max_attempts=1))
    assert ok is False


@respx.mock
def test_returns_false_on_network_error(monkeypatch):
    _env(monkeypatch)
    respx.post(
        "http://negocio-test/internal/routine-attempts/att-1/feedback"
    ).mock(side_effect=httpx.ConnectError("refused"))

    ok = asyncio.run(sync_attempt_feedback("att-1", _FEEDBACK, max_attempts=1))
    assert ok is False
