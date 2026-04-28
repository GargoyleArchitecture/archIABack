"""Additional store tests to push coverage above 90%."""
from __future__ import annotations

import json
import pytest

import src.memory as _mem
from src.ledger.store import (
    _ledger_key,
    _parse_legacy_asr_text,
    load_ledger,
    migrate_legacy_arch_flow,
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
    assert ledger["current_phase"] == "INTAKE"
    assert ledger["decisions"] == []


# ---------------------------------------------------------------------------
# _parse_legacy_asr_text branches
# ---------------------------------------------------------------------------

def test_parse_asr_from_fenced_block():
    text = "```asr\nSummary: High availability system\nSource: ops\n```"
    result = _parse_legacy_asr_text(text)
    assert "High availability system" in result


def test_parse_asr_from_keywords():
    text = "Summary: This is an ASR\nTrade-offs: some tradeoffs here"
    result = _parse_legacy_asr_text(text)
    assert result == text.strip()


def test_parse_asr_from_asr_heading():
    text = "ASR: Payment latency under 200ms\nDetails follow here"
    result = _parse_legacy_asr_text(text)
    assert "Payment latency" in result


def test_parse_asr_bullet_pattern():
    text = "Some context\n- ASR: the critical requirement"
    result = _parse_legacy_asr_text(text)
    assert "critical requirement" in result


def test_parse_asr_empty_returns_empty():
    assert _parse_legacy_asr_text("") == ""
    assert _parse_legacy_asr_text(None) == ""  # type: ignore


# ---------------------------------------------------------------------------
# Migration with last_diagram
# ---------------------------------------------------------------------------

def test_migrate_with_last_diagram_creates_diagram_decision(tmp_db):
    flow = {
        "stage": "DEPLOYMENT",
        "quality_attribute": "latencia",
        "add_context": "",
        "current_asr": "Summary: latencia",
        "style": "CQRS",
        "tactics": [],
        "deployment_diagram_puml": "",
        "deployment_diagram_svg_b64": "",
        "last_diagram": {
            "dot": "digraph { A -> B }",
            "dot_raw": "digraph { A -> B }",
            "dot_drawio": "",
            "detail_level": "overview",
            "level": 1,
            "overview_mapping": None,
        },
    }
    key = _mem._arch_flow_key("proj2")
    _mem.set_kv("user2", key, json.dumps(flow))

    ledger = migrate_legacy_arch_flow("user2", "proj2")
    assert ledger is not None
    kinds = {d["kind"] for d in ledger["decisions"]}
    assert "diagram" in kinds
    diagram_d = next(d for d in ledger["decisions"] if d["kind"] == "diagram")
    assert "A -> B" in diagram_d["payload"].get("dot", "")


# ---------------------------------------------------------------------------
# save_ledger — unconditional overwrite path (no expected_version)
# ---------------------------------------------------------------------------

def test_save_without_expected_version_overwrites(tmp_db):
    L = empty_ledger("p", "u")
    v1 = save_ledger("u", L, "p")
    v2 = save_ledger("u", L, "p")        # no expected_version — forced overwrite
    v3 = save_ledger("u", L, "p")
    loaded = load_ledger("u", "p", auto_migrate=False)
    assert loaded["version"] >= 2


# ---------------------------------------------------------------------------
# append_decision with validation error stops immediately
# ---------------------------------------------------------------------------

def test_append_decision_validation_error_does_not_retry(tmp_db):
    from src.ledger.types import LedgerValidationError
    from src.ledger.store import append_decision
    L = empty_ledger("p", "u")
    save_ledger("u", L, "p")

    bad_decision = {
        "id": "", "kind": "style", "phase": "STYLE", "iteration": 0,
        "qa": "latencia", "parents": [],
        "payload": {},  # missing "chosen" → LedgerValidationError
        "rationale": "", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "", "created_by_node": "style_node",
    }
    with pytest.raises(LedgerValidationError):
        append_decision("u", "p", bad_decision)
