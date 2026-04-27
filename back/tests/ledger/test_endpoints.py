"""FastAPI TestClient tests for the three ledger endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.memory as memory_module
from src.ledger.store import save_ledger
from src.ledger.types import empty_ledger

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def patched_db(tmp_path_factory):
    """Module-scoped temp DB so we don't re-import main for every test."""
    db_path = tmp_path_factory.mktemp("endpt") / "test.db"
    import unittest.mock as mock
    # Patch before importing main
    patcher = mock.patch.object(memory_module, "DB_PATH", db_path)
    patcher.start()
    memory_module.init()
    yield db_path
    patcher.stop()


@pytest.fixture(scope="module")
def client(patched_db):
    from src.main import app
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def seeded_session(patched_db):
    """Insert an asr_only ledger for session 'test-session'."""
    with open(FIXTURES / "asr_only.json", encoding="utf-8") as f:
        ledger = json.load(f)
    user_id    = "test-session"
    project_id = None
    ledger["user_id"]     = user_id
    ledger["project_id"]  = ""
    save_ledger(user_id, ledger, project_id)
    return user_id


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/dossier
# ---------------------------------------------------------------------------

def test_get_dossier_returns_markdown_for_existing_session(client, seeded_session):
    r = client.get(f"/sessions/{seeded_session}/dossier?lang=es")
    assert r.status_code == 200
    assert "text/markdown" in r.headers["content-type"]
    assert "# Design Dossier" in r.text


def test_get_dossier_compact_query_param(client, seeded_session):
    r = client.get(f"/sessions/{seeded_session}/dossier?compact=true")
    assert r.status_code == 200
    # Compact output is shorter and doesn't have full sections
    full_r = client.get(f"/sessions/{seeded_session}/dossier")
    assert len(r.text) < len(full_r.text)


def test_get_dossier_unknown_session_returns_empty_ledger(client):
    r = client.get("/sessions/no-such-session-xyz/dossier")
    assert r.status_code == 200
    assert "# Design Dossier" in r.text


def test_get_dossier_english_lang(client, seeded_session):
    r = client.get(f"/sessions/{seeded_session}/dossier?lang=en")
    assert r.status_code == 200
    assert "Phase:" in r.text


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/ledger
# ---------------------------------------------------------------------------

def test_get_ledger_returns_json(client, seeded_session):
    r = client.get(f"/sessions/{seeded_session}/ledger")
    assert r.status_code == 200
    data = r.json()
    assert "current_phase" in data
    assert "decisions" in data
    assert isinstance(data["decisions"], list)


def test_get_ledger_has_correct_content_type(client, seeded_session):
    r = client.get(f"/sessions/{seeded_session}/ledger")
    assert "application/json" in r.headers["content-type"]


def test_get_ledger_unknown_session_returns_empty(client):
    r = client.get("/sessions/totally-new-session-abc/ledger")
    assert r.status_code == 200
    data = r.json()
    assert data["current_phase"] == "INTAKE"
    assert data["decisions"] == []


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/phase
# ---------------------------------------------------------------------------

def test_get_phase_returns_completion_flags(client, seeded_session):
    r = client.get(f"/sessions/{seeded_session}/phase")
    assert r.status_code == 200
    data = r.json()
    assert "current_phase" in data
    assert "current_iteration" in data
    assert "pending_advance" in data
    assert "completion" in data
    completion = data["completion"]
    # Expect keys for all non-INTAKE phases
    assert set(completion.keys()) == {"asr", "style", "tactics", "diagram", "analysis"}
    # Our seeded ledger has an active ASR
    assert completion["asr"] is True
    assert completion["style"] is False


def test_get_phase_correct_json_types(client, seeded_session):
    r = client.get(f"/sessions/{seeded_session}/phase")
    data = r.json()
    assert isinstance(data["current_iteration"], int)
    assert isinstance(data["completion"]["asr"], bool)
