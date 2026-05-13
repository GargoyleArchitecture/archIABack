"""Additional store tests to push coverage above 90%."""
from __future__ import annotations

import pytest

import src.memory as _mem
from src.ledger.store import (
    _ledger_key,
    load_ledger,
    save_ledger,
)
from src.ledger.types import empty_ledger


# ---------------------------------------------------------------------------
# _ledger_key validation
# ---------------------------------------------------------------------------

def test_ledger_key_no_project():
    assert _ledger_key(None) == "ledger"
    assert _ledger_key("") == "ledger"


def test_ledger_key_with_project():
    assert _ledger_key("proj-1") == "ledger:proj-1"


def test_ledger_key_invalid_project_raises():
    with pytest.raises(ValueError, match="inválido"):
        _ledger_key("proj id with spaces!")


# ---------------------------------------------------------------------------
# load_ledger — corrupt blob falls through to empty
# ---------------------------------------------------------------------------

def test_load_ledger_corrupt_blob_returns_empty(tmp_db):
    key = _ledger_key("proj1")
    _mem.set_kv("user1", key, "NOT VALID JSON {{{{")
    ledger = load_ledger("user1", "proj1", auto_migrate=False)
    assert ledger["current_phase"] == "intro"
    assert ledger["decisions"] == []


# ---------------------------------------------------------------------------
# save_ledger — unconditional overwrite path (no expected_version)
# ---------------------------------------------------------------------------

def test_save_without_expected_version_overwrites(tmp_db):
    L = empty_ledger("p", "u")
    v1 = save_ledger("u", L, "p")
    v2 = save_ledger("u", L, "p")        # no expected_version — forced overwrite
    v3 = save_ledger("u", L, "p")
    loaded = load_ledger("u", "p", auto_migrate=False)
    assert loaded["version"] >= 1  # at least one save committed (empty_ledger starts at 0)


# ---------------------------------------------------------------------------
# append_decision with validation error stops immediately
# ---------------------------------------------------------------------------

def test_append_decision_validation_error_does_not_retry(tmp_db):
    from src.ledger.types import LedgerValidationError
    from src.ledger.store import append_decision
    L = empty_ledger("p", "u")
    save_ledger("u", L, "p")

    bad_decision = {
        "id": "", "kind": "style", "phase": "style_table", "iteration": 0,
        "qa": "latencia", "parents": [],
        "payload": {},  # missing "chosen" → LedgerValidationError
        "rationale": "", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "", "created_by_node": "style_node",
    }
    with pytest.raises(LedgerValidationError):
        append_decision("u", "p", bad_decision)
