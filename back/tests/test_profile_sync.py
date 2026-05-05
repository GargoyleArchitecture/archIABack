"""Tests del cliente HTTP profile_sync (F3-T5)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.profile_sync import (
    sync_profile,
    _to_business_payload,
    _scale_to_business,
)


def _set_env(monkeypatch, **kw):
    for k, v in kw.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, str(v))


# ---- _scale_to_business --------------------------------------------------

def test_scale_to_business_basic():
    assert _scale_to_business(0.0) == 0.0
    assert _scale_to_business(1.0) == 100.0
    assert _scale_to_business(0.5) == 50.0


def test_scale_clamps_out_of_range():
    assert _scale_to_business(1.5) == 100.0
    assert _scale_to_business(-0.5) == 0.0


def test_scale_handles_invalid_input():
    assert _scale_to_business(None) == 0.0
    assert _scale_to_business("garbage") == 0.0


# ---- _to_business_payload ------------------------------------------------

def test_payload_camelcases_and_scales_mastery_original():
    """Si el concepto trae mastery_original (post-decay), se envia ese valor."""
    merged = {
        "user_id": "u1",
        "strengths": ["A"],
        "weaknesses": ["B"],
        "evaluated_concepts": [
            {
                "name": "A",
                "mastery": 0.4,  # decayed (no se envia)
                "mastery_original": 0.8,  # el valor pre-decay (se envia)
                "last_seen_at": "2026-04-29T00:00:00+00:00",
                "evidence": "evi",
            }
        ],
        "confidence": 0.7,
        "delta_from_previous": {"A": 0.1},
        "updated_at": "2026-04-29T00:00:00+00:00",
    }
    body = _to_business_payload(merged)
    assert body["confidence"] == 0.7
    assert body["deltaFromPrevious"] == {"A": 0.1}
    assert body["updatedAt"] == "2026-04-29T00:00:00+00:00"
    assert body["strengths"] == ["A"]
    assert body["weaknesses"] == ["B"]
    c = body["evaluatedConcepts"][0]
    assert c["name"] == "A"
    assert c["mastery"] == 80.0  # 0.8 * 100
    assert c["lastSeenAt"] == "2026-04-29T00:00:00+00:00"
    assert c["evidence"] == "evi"


def test_payload_falls_back_to_mastery_when_no_original():
    """Sin mastery_original, usa mastery directo."""
    merged = {
        "evaluated_concepts": [{"name": "X", "mastery": 0.5}],
    }
    body = _to_business_payload(merged)
    assert body["evaluatedConcepts"][0]["mastery"] == 50.0


def test_payload_includes_decay_rate_when_present():
    merged = {
        "evaluated_concepts": [
            {"name": "X", "mastery": 0.5, "decay_rate": 0.02},
        ],
    }
    body = _to_business_payload(merged)
    assert body["evaluatedConcepts"][0]["decayRate"] == 0.02


def test_payload_omits_optional_fields_when_missing():
    merged = {"evaluated_concepts": [{"name": "X", "mastery": 0.5}]}
    body = _to_business_payload(merged)
    c = body["evaluatedConcepts"][0]
    assert "lastSeenAt" not in c
    assert "decayRate" not in c
    assert "evidence" not in c


def test_payload_handles_empty_profile():
    body = _to_business_payload({})
    assert body["strengths"] == []
    assert body["weaknesses"] == []
    assert body["evaluatedConcepts"] == []
    assert body["confidence"] == 0.0
    assert body["deltaFromPrevious"] == {}


def test_payload_skips_non_dict_concepts():
    merged = {"evaluated_concepts": [{"name": "OK", "mastery": 0.5}, "not-a-dict", None]}
    body = _to_business_payload(merged)
    assert len(body["evaluatedConcepts"]) == 1
    assert body["evaluatedConcepts"][0]["name"] == "OK"


# ---- sync_profile guards (sin httpx) ------------------------------------

def test_sync_skipped_when_disabled(monkeypatch):
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="false", INTERNAL_API_TOKEN="t")
    out = asyncio.run(sync_profile("u1", {"user_id": "u1"}))
    assert out is False


def test_sync_skipped_when_no_token(monkeypatch):
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="true", INTERNAL_API_TOKEN="")
    out = asyncio.run(sync_profile("u1", {"user_id": "u1"}))
    assert out is False


def test_sync_skipped_when_no_user_id(monkeypatch):
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="true", INTERNAL_API_TOKEN="t")
    out = asyncio.run(sync_profile("", {"user_id": ""}))
    assert out is False


# ---- sync_profile con httpx mockeado ------------------------------------

def _resp(status, body=""):
    r = MagicMock()
    r.status_code = status
    r.text = body
    return r


def _mock_client_with_responses(responses):
    """Crea un mock de httpx.AsyncClient cuyo PUT retorna `responses` en orden.

    `responses` puede ser una lista de _resp(...) o de excepciones a lanzar.
    """
    client = MagicMock()
    client.put = AsyncMock(side_effect=responses)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm, client


def test_sync_success_2xx_first_try(monkeypatch):
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="true", INTERNAL_API_TOKEN="tk",
             BUSINESS_API_BASE_URL="http://nego")
    cm, client = _mock_client_with_responses([_resp(200)])
    with patch("src.services.profile_sync.httpx.AsyncClient", return_value=cm):
        out = asyncio.run(sync_profile("u1", {"user_id": "u1"}))
    assert out is True
    assert client.put.await_count == 1
    args, kwargs = client.put.call_args
    assert args[0] == "http://nego/internal/users/u1/profile"
    assert kwargs["headers"]["X-Internal-Token"] == "tk"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    # body camelCase
    assert "evaluatedConcepts" in kwargs["json"]


def test_sync_retries_on_5xx_then_success(monkeypatch):
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="true", INTERNAL_API_TOKEN="tk")
    cm, client = _mock_client_with_responses([_resp(503, "down"), _resp(503, "still"), _resp(200)])
    with patch("src.services.profile_sync.httpx.AsyncClient", return_value=cm), \
         patch("src.services.profile_sync.asyncio.sleep", new=AsyncMock()):
        out = asyncio.run(sync_profile("u1", {"user_id": "u1"}))
    assert out is True
    assert client.put.await_count == 3


def test_sync_gives_up_after_3_attempts(monkeypatch):
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="true", INTERNAL_API_TOKEN="tk")
    cm, client = _mock_client_with_responses([_resp(503), _resp(503), _resp(503)])
    with patch("src.services.profile_sync.httpx.AsyncClient", return_value=cm), \
         patch("src.services.profile_sync.asyncio.sleep", new=AsyncMock()):
        out = asyncio.run(sync_profile("u1", {"user_id": "u1"}))
    assert out is False
    assert client.put.await_count == 3


def test_sync_handles_connection_error(monkeypatch):
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="true", INTERNAL_API_TOKEN="tk")
    cm, client = _mock_client_with_responses([
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
    ])
    with patch("src.services.profile_sync.httpx.AsyncClient", return_value=cm), \
         patch("src.services.profile_sync.asyncio.sleep", new=AsyncMock()):
        out = asyncio.run(sync_profile("u1", {"user_id": "u1"}))
    assert out is False
    assert client.put.await_count == 3


def test_sync_uses_4xx_as_attempt_too(monkeypatch):
    """4xx tampoco es exito, pero igual cuenta como intento (sin lanzar)."""
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="true", INTERNAL_API_TOKEN="tk")
    cm, client = _mock_client_with_responses([_resp(401), _resp(401), _resp(401)])
    with patch("src.services.profile_sync.httpx.AsyncClient", return_value=cm), \
         patch("src.services.profile_sync.asyncio.sleep", new=AsyncMock()):
        out = asyncio.run(sync_profile("u1", {"user_id": "u1"}))
    assert out is False
    assert client.put.await_count == 3


def test_sync_path_template_substitutes_user_id(monkeypatch):
    _set_env(monkeypatch, PROFILE_SYNC_ENABLED="true", INTERNAL_API_TOKEN="tk",
             BUSINESS_API_BASE_URL="http://b",
             BUSINESS_API_PROFILE_PATH="/api/v1/profile/{userId}")
    cm, client = _mock_client_with_responses([_resp(200)])
    with patch("src.services.profile_sync.httpx.AsyncClient", return_value=cm):
        out = asyncio.run(sync_profile("USER-42", {}))
    assert out is True
    args, _ = client.put.call_args
    assert args[0] == "http://b/api/v1/profile/USER-42"
