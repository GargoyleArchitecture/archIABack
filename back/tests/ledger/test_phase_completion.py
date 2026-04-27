"""Tests for is_phase_complete across all phase definitions."""
from __future__ import annotations

import pytest

from src.ledger.types import Phase, empty_ledger
from src.ledger.store import compute_active_view, is_phase_complete


def _build_ledger(*decisions):
    L = empty_ledger("p", "u")
    L["decisions"] = list(decisions)
    return L


def _d(id, kind, qa="latencia", parents=None, status="active", payload=None):
    _default_payloads = {
        "asr":        {"summary": "S", "source": "", "stimulus": "", "environment": "", "artifact": "", "response": "", "response_measure": "p95<200ms", "domain": ""},
        "style":      {"chosen": "CQRS", "candidates": [], "tradeoffs": ""},
        "tactic":     {"items": []},
        "diagram":    {"level": 1, "dot": "digraph {}", "dot_drawio": "", "svg_b64": "", "focus": "", "mapping": {}},
        "analysis":   {"target_id": "", "positive": "", "negative": "", "suggestions": ""},
        "constraint": {"tech_stack": [], "business_rules": []},
    }
    return {
        "id": id, "kind": kind, "phase": "ASR", "iteration": 1,
        "qa": qa, "parents": parents or [],
        "payload": payload or _default_payloads.get(kind, {}),
        "rationale": "", "sources": [], "status": status,
        "parent_status": "ok", "superseded_by": None, "rejection_reason": None,
        "created_at": "2024-01-01T00:00:00Z", "created_by_node": "test",
    }


# INTAKE

def test_intake_incomplete_without_constraint():
    L = _build_ledger()
    assert is_phase_complete(L, Phase.INTAKE) is False


def test_intake_complete_with_constraint():
    L = _build_ledger(_d("c1", "constraint"))
    assert is_phase_complete(L, Phase.INTAKE) is True


# ASR

def test_asr_incomplete_without_asr():
    L = _build_ledger()
    assert is_phase_complete(L, Phase.ASR) is False


def test_asr_complete_with_active_asr():
    L = _build_ledger(_d("a1", "asr"))
    assert is_phase_complete(L, Phase.ASR) is True


def test_asr_incomplete_with_rejected_asr():
    L = _build_ledger(_d("a1", "asr", status="rejected"))
    assert is_phase_complete(L, Phase.ASR) is False


# STYLE

def test_style_incomplete_without_style():
    L = _build_ledger(_d("a1", "asr"))
    assert is_phase_complete(L, Phase.STYLE) is False


def test_style_complete_with_matching_parent():
    asr   = _d("a1", "asr")
    style = _d("s1", "style", parents=[{"id": "a1", "kind": "asr", "iteration": 1}])
    L     = _build_ledger(asr, style)
    assert is_phase_complete(L, Phase.STYLE) is True


def test_style_incomplete_if_parent_asr_not_active():
    asr   = _d("a1", "asr", status="superseded")
    style = _d("s1", "style", parents=[{"id": "a1", "kind": "asr", "iteration": 1}])
    L     = _build_ledger(asr, style)
    # active_view won't include superseded asr
    assert is_phase_complete(L, Phase.STYLE) is False


# TACTICS

def test_tactics_complete_with_full_parents():
    asr    = _d("a1", "asr")
    style  = _d("s1", "style", parents=[{"id": "a1", "kind": "asr", "iteration": 1}])
    tactic = _d("t1", "tactic", parents=[
        {"id": "a1", "kind": "asr",   "iteration": 1},
        {"id": "s1", "kind": "style", "iteration": 1},
    ])
    L = _build_ledger(asr, style, tactic)
    assert is_phase_complete(L, Phase.TACTICS) is True


def test_tactics_incomplete_missing_style_parent():
    asr    = _d("a1", "asr")
    tactic = _d("t1", "tactic", parents=[{"id": "a1", "kind": "asr", "iteration": 1}])
    L      = _build_ledger(asr, tactic)
    assert is_phase_complete(L, Phase.TACTICS) is False


# DIAGRAM

def test_diagram_incomplete_without_diagram():
    L = _build_ledger()
    assert is_phase_complete(L, Phase.DIAGRAM) is False


def test_diagram_complete_with_active_diagram():
    dgr = _d("d1", "diagram")
    L   = _build_ledger(dgr)
    assert is_phase_complete(L, Phase.DIAGRAM) is True


# ANALYSIS

def test_analysis_complete_with_targeting_analysis():
    dgr = _d("d1", "diagram")
    anl = _d("anl1", "analysis", payload={"target_id": "d1", "positive": "", "negative": "", "suggestions": ""})
    L   = _build_ledger(dgr, anl)
    assert is_phase_complete(L, Phase.ANALYSIS) is True


def test_analysis_incomplete_when_target_mismatch():
    dgr = _d("d1", "diagram")
    anl = _d("anl1", "analysis", payload={"target_id": "OTHER", "positive": "", "negative": "", "suggestions": ""})
    L   = _build_ledger(dgr, anl)
    assert is_phase_complete(L, Phase.ANALYSIS) is False
