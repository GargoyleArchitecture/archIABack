"""Tests F3-T6: fetch_profile_from_negocio + _from_business_payload."""
import asyncio

import pytest
import httpx
import respx  # HTTP mock compatible con httpx

from src.services.profile_sync import fetch_profile_from_negocio, _from_business_payload


# ---------------------------------------------------------------------------
# _from_business_payload — conversion camelCase Negocio → snake_case Store
# ---------------------------------------------------------------------------

def test_from_business_payload_converts_mastery_scale():
    """mastery 0-100 (Negocio) → 0-1 (Store interno)."""
    data = {
        "userId": "abc",
        "strengths": ["SOLID"],
        "weaknesses": ["Testing"],
        "evaluatedConcepts": [
            {"name": "SOLID", "mastery": 80.0, "lastSeenAt": "2026-04-01T00:00:00Z"},
        ],
        "updatedAt": "2026-04-01T00:00:00Z",
    }
    result = _from_business_payload(data)

    assert result["user_id"] == "abc"
    assert result["strengths"] == ["SOLID"]
    assert result["weaknesses"] == ["Testing"]
    assert len(result["evaluated_concepts"]) == 1
    concept = result["evaluated_concepts"][0]
    assert concept["name"] == "SOLID"
    assert concept["mastery"] == pytest.approx(0.8, abs=1e-4)
    assert concept["last_seen_at"] == "2026-04-01T00:00:00Z"


def test_from_business_payload_optional_decay_rate():
    """decayRate presente → se incluye como decay_rate; ausente → se omite."""
    data = {
        "evaluatedConcepts": [
            {"name": "A", "mastery": 50.0, "decayRate": 0.03},
            {"name": "B", "mastery": 20.0},
        ],
    }
    result = _from_business_payload(data)
    assert result["evaluated_concepts"][0]["decay_rate"] == pytest.approx(0.03)
    assert "decay_rate" not in result["evaluated_concepts"][1]


def test_from_business_payload_empty_concepts():
    """Sin evaluatedConcepts → lista vacia, sin crash."""
    result = _from_business_payload({"evaluatedConcepts": []})
    assert result["evaluated_concepts"] == []


def test_from_business_payload_initializes_neutral_fields():
    """confidence y delta_from_previous se inicializan neutros (Negocio no los persiste)."""
    result = _from_business_payload({})
    assert result["confidence"] == 0.0
    assert result["delta_from_previous"] == {}


def test_from_business_payload_skips_non_dict_concepts():
    """Conceptos que no son dict se ignoran sin crash."""
    data = {"evaluatedConcepts": [None, "string", {"name": "X", "mastery": 100.0}]}
    result = _from_business_payload(data)
    assert len(result["evaluated_concepts"]) == 1
    assert result["evaluated_concepts"][0]["name"] == "X"


# ---------------------------------------------------------------------------
# fetch_profile_from_negocio — guards de configuracion
# ---------------------------------------------------------------------------

def test_fetch_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("PROFILE_SYNC_ENABLED", "false")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "tok")
    result = asyncio.run(fetch_profile_from_negocio("u1"))
    assert result is None


def test_fetch_returns_none_when_token_empty(monkeypatch):
    monkeypatch.setenv("PROFILE_SYNC_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "")
    result = asyncio.run(fetch_profile_from_negocio("u1"))
    assert result is None


def test_fetch_returns_none_when_user_id_empty(monkeypatch):
    monkeypatch.setenv("PROFILE_SYNC_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "tok")
    result = asyncio.run(fetch_profile_from_negocio(""))
    assert result is None


# ---------------------------------------------------------------------------
# fetch_profile_from_negocio — respuestas HTTP
# ---------------------------------------------------------------------------

@respx.mock
def test_fetch_returns_profile_on_200(monkeypatch):
    monkeypatch.setenv("PROFILE_SYNC_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    monkeypatch.setenv("BUSINESS_API_BASE_URL", "http://negocio-test")
    monkeypatch.setenv("BUSINESS_API_PROFILE_PATH", "/internal/users/{userId}/profile")

    payload = {
        "userId": "u1",
        "strengths": ["DDD"],
        "weaknesses": [],
        "evaluatedConcepts": [
            {"name": "DDD", "mastery": 70.0, "lastSeenAt": "2026-04-28T10:00:00Z"},
        ],
        "updatedAt": "2026-04-28T10:00:00Z",
    }
    respx.get("http://negocio-test/internal/users/u1/profile").mock(
        return_value=httpx.Response(200, json=payload)
    )

    result = asyncio.run(fetch_profile_from_negocio("u1"))

    assert result is not None
    assert result["user_id"] == "u1"
    assert result["strengths"] == ["DDD"]
    assert result["evaluated_concepts"][0]["mastery"] == pytest.approx(0.7, abs=1e-4)


@respx.mock
def test_fetch_returns_none_on_404(monkeypatch):
    monkeypatch.setenv("PROFILE_SYNC_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    monkeypatch.setenv("BUSINESS_API_BASE_URL", "http://negocio-test")
    monkeypatch.setenv("BUSINESS_API_PROFILE_PATH", "/internal/users/{userId}/profile")

    respx.get("http://negocio-test/internal/users/u1/profile").mock(
        return_value=httpx.Response(404)
    )

    result = asyncio.run(fetch_profile_from_negocio("u1"))
    assert result is None


@respx.mock
def test_fetch_returns_none_on_network_error(monkeypatch):
    monkeypatch.setenv("PROFILE_SYNC_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    monkeypatch.setenv("BUSINESS_API_BASE_URL", "http://negocio-test")
    monkeypatch.setenv("BUSINESS_API_PROFILE_PATH", "/internal/users/{userId}/profile")

    respx.get("http://negocio-test/internal/users/u1/profile").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    result = asyncio.run(fetch_profile_from_negocio("u1"))
    assert result is None


@respx.mock
def test_fetch_returns_none_on_500(monkeypatch):
    monkeypatch.setenv("PROFILE_SYNC_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    monkeypatch.setenv("BUSINESS_API_BASE_URL", "http://negocio-test")
    monkeypatch.setenv("BUSINESS_API_PROFILE_PATH", "/internal/users/{userId}/profile")

    respx.get("http://negocio-test/internal/users/u1/profile").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    result = asyncio.run(fetch_profile_from_negocio("u1"))
    assert result is None
