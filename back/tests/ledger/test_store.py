"""Unit tests for src.ledger.store."""
from __future__ import annotations

import json
import threading
import time

import pytest

from src.ledger.types import (
    LedgerConcurrencyError,
    LedgerValidationError,
    LEDGER_SCHEMA_VERSION,
    Phase,
    empty_ledger,
)
from src.ledger.store import (
    append_decision,
    clear_pending_advance,
    compute_active_view,
    is_phase_complete,
    load_ledger,
    reject_decision,
    save_ledger,
    stage_pending_advance,
    transition_phase,
)
import src.memory as _mem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _asr_decision(qa="latencia", parents=None):
    return {
        "id": "", "kind": "asr", "phase": "ASR", "iteration": 0,
        "qa": qa, "parents": parents or [],
        "payload": {"summary": "test ASR", "source": "", "stimulus": "", "environment": "", "artifact": "", "response": "", "response_measure": "p95<200ms", "domain": ""},
        "rationale": "test", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "", "created_by_node": "asr_node",
    }


def _style_decision(qa="latencia", parents=None):
    return {
        "id": "", "kind": "style", "phase": "STYLE", "iteration": 0,
        "qa": qa, "parents": parents or [],
        "payload": {"chosen": "CQRS", "candidates": [], "tradeoffs": ""},
        "rationale": "test", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "", "created_by_node": "style_latency",
    }


def _make_transition(from_p, to_p, iteration):
    return {
        "from_phase": from_p, "to_phase": to_p, "iteration": iteration,
        "triggered_by": "user_request", "user_message": "", "skipped_phases": [],
        "timestamp": "2024-01-01T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------

def test_load_ledger_returns_empty_when_missing(tmp_db):
    ledger = load_ledger("u1", "p1", auto_migrate=False)
    assert ledger["current_phase"] == Phase.INTAKE.value
    assert ledger["decisions"] == []
    assert ledger["version"] == LEDGER_SCHEMA_VERSION


def test_save_then_load_round_trip(tmp_db):
    L = empty_ledger("proj1", "user1")
    saved = save_ledger("user1", L, "proj1")
    loaded = load_ledger("user1", "proj1", auto_migrate=False)
    assert loaded == saved


def test_save_increments_version(tmp_db):
    L = empty_ledger("p", "u")
    assert L["version"] == 1
    saved = save_ledger("u", L, "p")
    assert saved["version"] == 2


def test_save_with_expected_version_ok(tmp_db):
    L = empty_ledger("p", "u")
    saved = save_ledger("u", L, "p")          # version → 2
    saved2 = save_ledger("u", saved, "p", expected_version=2)  # version → 3
    assert saved2["version"] == 3


def test_optimistic_concurrency_conflict_raises(tmp_db):
    L = empty_ledger("p", "u")
    saved = save_ledger("u", L, "p")      # version → 2
    save_ledger("u", saved, "p")          # version → 3 (another write)
    with pytest.raises(LedgerConcurrencyError):
        save_ledger("u", saved, "p", expected_version=2)   # expects 2, stored 3


def test_load_no_project_id(tmp_db):
    L = empty_ledger("", "u")
    save_ledger("u", L)
    loaded = load_ledger("u", auto_migrate=False)
    assert loaded["project_id"] == ""


# ---------------------------------------------------------------------------
# append_decision
# ---------------------------------------------------------------------------

def test_append_decision_assigns_ulid_and_iteration(tmp_db):
    # Seed ledger in DB
    L = empty_ledger("proj", "user")
    save_ledger("user", L, "proj")

    d = _asr_decision()
    result = append_decision("user", "proj", d)

    assert result["id"] != ""
    assert len(result["id"]) == 26  # ULID length
    assert result["iteration"] == 0  # iteration matches current_iteration at append time
    assert result["created_at"].endswith("Z")


def test_append_supersedes_prior_active_same_kind_same_parents(tmp_db):
    L = empty_ledger("proj", "user")
    save_ledger("user", L, "proj")

    # Append first ASR
    d1 = _asr_decision()
    r1 = append_decision("user", "proj", d1)
    assert r1["status"] == "active"

    # Append second ASR with same parents (should supersede first)
    d2 = _asr_decision()
    r2 = append_decision("user", "proj", d2)

    ledger = load_ledger("user", "proj", auto_migrate=False)
    statuses = {d["id"]: d["status"] for d in ledger["decisions"]}
    assert statuses[r1["id"]] == "superseded"
    assert statuses[r2["id"]] == "active"
    superseded_by = next(d["superseded_by"] for d in ledger["decisions"] if d["id"] == r1["id"])
    assert superseded_by == r2["id"]


def test_reject_walks_descendants_and_flags_orphans(tmp_db):
    L = empty_ledger("proj", "user")
    save_ledger("user", L, "proj")

    # Append ASR + style; then reject style
    asr_d = append_decision("user", "proj", _asr_decision())
    sty_d = append_decision(
        "user", "proj",
        _style_decision(parents=[{"id": asr_d["id"], "kind": "asr", "iteration": asr_d["iteration"]}]),
    )

    # Now add a tactic parented to the style
    tactic_d_raw = {
        "id": "", "kind": "tactic", "phase": "TACTICS", "iteration": 0,
        "qa": "latencia",
        "parents": [
            {"id": asr_d["id"], "kind": "asr",   "iteration": asr_d["iteration"]},
            {"id": sty_d["id"], "kind": "style", "iteration": sty_d["iteration"]},
        ],
        "payload": {"items": [{"name": "Cache", "purpose": "speed", "rationale": "", "risks": "", "tradeoffs": "", "traces_to_asr": "p95<200ms", "expected_effect": "", "success_probability": "0.9", "rank": 1}]},
        "rationale": "", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "", "created_by_node": "tactics_latency",
    }
    tac_d = append_decision("user", "proj", tactic_d_raw)

    reject_decision("user", "proj", sty_d["id"], "no encaja")

    ledger = load_ledger("user", "proj", auto_migrate=False)
    by_id = {d["id"]: d for d in ledger["decisions"]}
    assert by_id[sty_d["id"]]["status"] == "rejected"
    assert by_id[tac_d["id"]]["parent_status"] == "parent_rejected"


def test_reject_nonexistent_raises(tmp_db):
    L = empty_ledger("p", "u")
    save_ledger("u", L, "p")
    with pytest.raises(LedgerValidationError, match="not found"):
        reject_decision("u", "p", "nonexistent_id", "reason")


# ---------------------------------------------------------------------------
# concurrent writes
# ---------------------------------------------------------------------------

def test_concurrent_appends_serialize_via_begin_immediate(tmp_db):
    """Two threads appending to the same ledger should both succeed without data loss."""
    L = empty_ledger("proj", "user")
    save_ledger("user", L, "proj")

    errors: list[Exception] = []
    results: list[str] = []

    def _append():
        try:
            d = _asr_decision()
            r = append_decision("user", "proj", d)
            results.append(r["id"])
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_append)
    t2 = threading.Thread(target=_append)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 2  # both writes committed


# ---------------------------------------------------------------------------
# compute_active_view
# ---------------------------------------------------------------------------

def test_compute_active_view_picks_latest_active_per_kind(sample_ledger):
    ledger = sample_ledger("asr_style_tactic")
    active = compute_active_view(ledger)
    assert active["asr"]["id"]    == "01FIXTURE000000000ASR00001"
    assert active["style"]["id"]  == "01FIXTURE000000000STY00001"
    assert active["tactic"]["id"] == "01FIXTURE000000000TAC00001"


def test_compute_active_view_empty_for_no_decisions(sample_ledger):
    ledger = sample_ledger("empty")
    active = compute_active_view(ledger)
    assert active == {}


# ---------------------------------------------------------------------------
# is_phase_complete
# ---------------------------------------------------------------------------

def test_is_phase_complete_per_phase(sample_ledger):
    ledger = sample_ledger("asr_style_tactic")
    assert is_phase_complete(ledger, Phase.INTAKE)   is False  # no constraint
    assert is_phase_complete(ledger, Phase.ASR)      is True
    assert is_phase_complete(ledger, Phase.STYLE)    is True
    assert is_phase_complete(ledger, Phase.TACTICS)  is True
    assert is_phase_complete(ledger, Phase.DIAGRAM)  is False
    assert is_phase_complete(ledger, Phase.ANALYSIS) is False


def test_is_intake_complete_with_constraint(sample_ledger):
    ledger = sample_ledger("intake_only")
    assert is_phase_complete(ledger, Phase.INTAKE) is True


def test_asr_incomplete_when_empty(sample_ledger):
    ledger = sample_ledger("empty")
    assert is_phase_complete(ledger, Phase.ASR) is False


# ---------------------------------------------------------------------------
# phase transitions
# ---------------------------------------------------------------------------

def test_stage_and_clear_pending_advance(tmp_db):
    L = empty_ledger("p", "u")
    save_ledger("u", L, "p")
    t = _make_transition("INTAKE", "ASR", 1)
    stage_pending_advance("u", "p", t)

    ledger = load_ledger("u", "p", auto_migrate=False)
    assert ledger["pending_advance"] is not None
    assert ledger["pending_advance"]["to_phase"] == "ASR"

    clear_pending_advance("u", "p")
    ledger = load_ledger("u", "p", auto_migrate=False)
    assert ledger["pending_advance"] is None


def test_transition_phase_commits(tmp_db):
    L = empty_ledger("p", "u")
    save_ledger("u", L, "p")
    t = _make_transition("INTAKE", "ASR", 1)
    transition_phase("u", "p", t)

    ledger = load_ledger("u", "p", auto_migrate=False)
    assert ledger["current_phase"] == "ASR"
    assert ledger["current_iteration"] == 1
    assert len(ledger["phase_history"]) == 1
