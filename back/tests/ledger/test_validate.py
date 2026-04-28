"""Unit tests for src.ledger.validate."""
from __future__ import annotations

import copy
import pytest

from src.ledger.types import LedgerValidationError, empty_ledger, Phase
from src.ledger.validate import (
    validate_decision,
    validate_parents,
    validate_qa_match,
    validate_supersede_target,
    validate_transition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _asr(id="asr_1", qa="latencia", status="active"):
    return {
        "id": id, "kind": "asr", "phase": "ASR", "iteration": 1,
        "qa": qa, "parents": [],
        "payload": {"summary": "test ASR"},
        "rationale": "", "sources": [], "status": status,
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "2024-01-01T00:00:00Z", "created_by_node": "asr_node",
    }


def _style(id="sty_1", qa="latencia", asr_id="asr_1", status="active"):
    return {
        "id": id, "kind": "style", "phase": "STYLE", "iteration": 2,
        "qa": qa,
        "parents": [{"id": asr_id, "kind": "asr", "iteration": 1}],
        "payload": {"chosen": "CQRS", "candidates": [], "tradeoffs": ""},
        "rationale": "", "sources": [], "status": status,
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "2024-01-01T01:00:00Z", "created_by_node": "style_latency",
    }


def _tactic(id="tac_1", qa="latencia", asr_id="asr_1", sty_id="sty_1"):
    return {
        "id": id, "kind": "tactic", "phase": "TACTICS", "iteration": 3,
        "qa": qa,
        "parents": [
            {"id": asr_id, "kind": "asr",   "iteration": 1},
            {"id": sty_id, "kind": "style", "iteration": 2},
        ],
        "payload": {"items": [{"name": "Cache", "purpose": "speed", "rationale": "", "risks": "", "tradeoffs": "", "traces_to_asr": "p95<200ms", "expected_effect": "", "success_probability": "0.9", "rank": 1}]},
        "rationale": "", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "2024-01-01T02:00:00Z", "created_by_node": "tactics_latency",
    }


def _ledger_with(*decisions):
    L = empty_ledger("proj", "user")
    L["decisions"] = list(decisions)
    return L


# ---------------------------------------------------------------------------
# validate_decision
# ---------------------------------------------------------------------------

def test_valid_asr_decision_passes():
    ledger = _ledger_with()
    d = _asr()
    validate_decision(ledger, d)  # no exception


def test_missing_created_by_node_raises():
    ledger = _ledger_with()
    d = _asr()
    d["created_by_node"] = ""
    with pytest.raises(LedgerValidationError, match="created_by_node"):
        validate_decision(ledger, d)


def test_unknown_kind_raises():
    ledger = _ledger_with()
    d = _asr()
    d["kind"] = "unknown_kind"
    with pytest.raises(LedgerValidationError, match="Unknown kind"):
        validate_decision(ledger, d)


def test_style_missing_chosen_raises():
    ledger = _ledger_with()
    d = _style()
    del d["payload"]["chosen"]
    with pytest.raises(LedgerValidationError, match="missing required keys"):
        validate_decision(ledger, d)


def test_tactic_missing_items_raises():
    ledger = _ledger_with()
    d = _tactic()
    del d["payload"]["items"]
    with pytest.raises(LedgerValidationError, match="missing required keys"):
        validate_decision(ledger, d)


def test_analysis_missing_target_id_raises():
    ledger = _ledger_with()
    d = {
        "id": "a1", "kind": "analysis", "phase": "ANALYSIS", "iteration": 1,
        "qa": "general", "parents": [], "payload": {},
        "rationale": "", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "2024-01-01T00:00:00Z", "created_by_node": "evaluator",
    }
    with pytest.raises(LedgerValidationError, match="missing required keys"):
        validate_decision(ledger, d)


# ---------------------------------------------------------------------------
# validate_parents
# ---------------------------------------------------------------------------

def test_style_with_wrong_parent_kind_rejected():
    """Style that claims an asr parent but the referenced decision is a tactic."""
    asr = _asr()
    tac = _tactic()
    ledger = _ledger_with(asr, tac)
    d = _style(asr_id="tac_1")  # wrong parent kind
    with pytest.raises(LedgerValidationError):
        validate_parents(ledger, d)


def test_parent_not_found_raises():
    ledger = _ledger_with()
    d = _style(asr_id="nonexistent")
    with pytest.raises(LedgerValidationError, match="not found"):
        validate_parents(ledger, d)


def test_style_with_superseded_parent_rejected():
    asr = _asr(status="superseded")
    ledger = _ledger_with(asr)
    d = _style()
    with pytest.raises(LedgerValidationError, match="must be active"):
        validate_parents(ledger, d)


def test_diagram_with_no_parents_rejected():
    ledger = _ledger_with()
    d = {
        "id": "dgr1", "kind": "diagram", "phase": "DIAGRAM", "iteration": 3,
        "qa": "general", "parents": [],
        "payload": {"level": 1, "dot": "graph {}", "dot_drawio": "", "svg_b64": "", "focus": "", "mapping": {}},
        "rationale": "", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "2024-01-01T00:00:00Z", "created_by_node": "diagram_agent",
    }
    with pytest.raises(LedgerValidationError, match="requires a parent of kind"):
        validate_parents(ledger, d)


def test_analysis_target_id_must_exist():
    """validate_parents on analysis doesn't enforce parent by kind; but if parents listed, they must exist."""
    ledger = _ledger_with()
    # An analysis with a parent ref pointing to a non-existent decision.
    d = {
        "id": "anl1", "kind": "analysis", "phase": "ANALYSIS", "iteration": 4,
        "qa": "general",
        "parents": [{"id": "nonexistent_id", "kind": "asr", "iteration": 1}],
        "payload": {"target_id": "nonexistent_id", "positive": "", "negative": "", "suggestions": ""},
        "rationale": "", "sources": [], "status": "active",
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "2024-01-01T00:00:00Z", "created_by_node": "evaluator",
    }
    with pytest.raises(LedgerValidationError, match="not found"):
        validate_parents(ledger, d)


# ---------------------------------------------------------------------------
# validate_qa_match
# ---------------------------------------------------------------------------

def test_style_with_qa_mismatch_rejected():
    asr   = _asr(qa="latencia")
    ledger = _ledger_with(asr)
    d     = _style(qa="escalabilidad")  # QA doesn't match parent ASR
    with pytest.raises(LedgerValidationError, match="doesn't match"):
        validate_qa_match(ledger, d)


def test_style_with_matching_qa_passes():
    asr   = _asr(qa="latencia")
    ledger = _ledger_with(asr)
    d     = _style(qa="latencia")
    validate_qa_match(ledger, d)  # no exception


# ---------------------------------------------------------------------------
# validate_transition
# ---------------------------------------------------------------------------

def test_forward_transition_passes():
    L = empty_ledger("p", "u")
    t = {
        "from_phase": "INTAKE", "to_phase": "ASR", "iteration": 1,
        "triggered_by": "user_request", "user_message": "", "skipped_phases": [],
        "timestamp": "2024-01-01T00:00:00Z",
    }
    validate_transition(L, t)  # no exception


def test_forward_transition_with_skipped_phases_records_skip():
    """Jumping from INTAKE to STYLE skips ASR; skipped_phases must list it."""
    L = empty_ledger("p", "u")
    t = {
        "from_phase": "INTAKE", "to_phase": "STYLE", "iteration": 1,
        "triggered_by": "user_request", "user_message": "", "skipped_phases": ["ASR"],
        "timestamp": "2024-01-01T00:00:00Z",
    }
    validate_transition(L, t)  # no exception


def test_forward_jump_missing_skipped_raises():
    L = empty_ledger("p", "u")
    t = {
        "from_phase": "INTAKE", "to_phase": "STYLE", "iteration": 1,
        "triggered_by": "user_request", "user_message": "", "skipped_phases": [],
        "timestamp": "2024-01-01T00:00:00Z",
    }
    with pytest.raises(LedgerValidationError, match="skipped_phases"):
        validate_transition(L, t)


def test_backward_transition_increments_iteration():
    """Going from STYLE back to ASR is allowed; iteration must still be current+1."""
    L = empty_ledger("p", "u")
    L["current_phase"] = "STYLE"
    L["current_iteration"] = 2
    t = {
        "from_phase": "STYLE", "to_phase": "ASR", "iteration": 3,
        "triggered_by": "user_jump_back", "user_message": "", "skipped_phases": [],
        "timestamp": "2024-01-01T00:00:00Z",
    }
    validate_transition(L, t)  # no exception


def test_wrong_from_phase_raises():
    L = empty_ledger("p", "u")  # current_phase == INTAKE
    t = {
        "from_phase": "ASR", "to_phase": "STYLE", "iteration": 1,
        "triggered_by": "user_request", "user_message": "", "skipped_phases": [],
        "timestamp": "2024-01-01T00:00:00Z",
    }
    with pytest.raises(LedgerValidationError, match="from_phase"):
        validate_transition(L, t)


def test_wrong_iteration_raises():
    L = empty_ledger("p", "u")
    t = {
        "from_phase": "INTAKE", "to_phase": "ASR", "iteration": 5,  # should be 1
        "triggered_by": "user_request", "user_message": "", "skipped_phases": [],
        "timestamp": "2024-01-01T00:00:00Z",
    }
    with pytest.raises(LedgerValidationError, match="iteration"):
        validate_transition(L, t)


# ---------------------------------------------------------------------------
# validate_supersede_target
# ---------------------------------------------------------------------------

def test_supersede_target_found_same_kind_parents():
    asr    = _asr()
    style1 = _style(id="sty_1")
    ledger = _ledger_with(asr, style1)
    new_style = _style(id="sty_2")  # same kind, same parents
    result = validate_supersede_target(ledger, new_style)
    assert result == "sty_1"


def test_supersede_target_none_when_no_match():
    ledger = _ledger_with()
    result = validate_supersede_target(ledger, _asr())
    assert result is None
