"""Tests for migrate_legacy_arch_flow."""
from __future__ import annotations

import json

import pytest

import src.memory as _mem
from src.ledger.store import load_ledger, migrate_legacy_arch_flow, save_ledger
from src.ledger.types import Phase, empty_ledger


def _save_arch_flow(user_id, project_id, data: dict, monkeypatch=None):
    """Helper: write arch_flow data directly into memory table."""
    key = _mem._arch_flow_key(project_id)
    _mem.set_kv(user_id, key, json.dumps(data))


def _legacy_arch_flow(stage="STYLE", qa="latencia"):
    return {
        "stage": stage,
        "quality_attribute": qa,
        "add_context": "Sistema bancario de pagos",
        "current_asr": "ASR: Sistema de pagos\nSummary: Latencia <200ms\nSource: usuario\nStimulus: pago\nEnvironment: prod\nArtifact: servicio\nResponse: procesar\nResponse Measure: p95 < 200ms",
        "style": "CQRS + Event Sourcing",
        "tactics": [
            {
                "name": "Cache Redis",
                "purpose": "Reducir latencia",
                "rationale": "Redis es rápido",
                "risks": "Consistencia eventual",
                "tradeoffs": "Consistencia vs velocidad",
                "traces_to_asr": "p95 < 200ms",
                "expected_effect": "50% reducción",
                "success_probability": "0.90",
                "rank": 1,
            }
        ],
        "deployment_diagram_puml": "",
        "deployment_diagram_svg_b64": "",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_migrate_empty_arch_flow_yields_none(tmp_db):
    """No arch_flow → migrate returns None."""
    result = migrate_legacy_arch_flow("user1", "proj1")
    assert result is None


def test_migrate_full_arch_flow_synthesizes_all_kinds(tmp_db):
    _save_arch_flow("user1", "proj1", _legacy_arch_flow())
    ledger = migrate_legacy_arch_flow("user1", "proj1")

    assert ledger is not None
    kinds = {d["kind"] for d in ledger["decisions"]}
    assert "constraint" in kinds
    assert "asr" in kinds
    assert "style" in kinds
    assert "tactic" in kinds


def test_migrate_partial_arch_flow_only_asr(tmp_db):
    flow = {
        "stage": "ASR",
        "quality_attribute": "latencia",
        "add_context": "",
        "current_asr": "ASR: pagos rápidos\nSummary: latencia < 200ms",
        "style": "",
        "tactics": [],
        "deployment_diagram_puml": "",
        "deployment_diagram_svg_b64": "",
    }
    _save_arch_flow("user1", "proj1", flow)
    ledger = migrate_legacy_arch_flow("user1", "proj1")

    assert ledger is not None
    kinds = {d["kind"] for d in ledger["decisions"]}
    assert "asr" in kinds
    assert "style" not in kinds
    assert "tactic" not in kinds


def test_migrate_infers_qa_from_asr_text_when_legacy_qa_missing(tmp_db):
    flow = {
        "stage": "ASR",
        "quality_attribute": "",
        "add_context": "",
        "current_asr": "ASR: búsqueda hotelera\nSummary: latencia p95 < 800ms",
        "style": "",
        "tactics": [],
        "deployment_diagram_puml": "",
        "deployment_diagram_svg_b64": "",
    }
    _save_arch_flow("user1", "proj1", flow)

    ledger = migrate_legacy_arch_flow("user1", "proj1")

    assert ledger is not None
    asr = next(d for d in ledger["decisions"] if d["kind"] == "asr")
    assert asr["qa"] == "latencia"


def test_migrate_is_idempotent(tmp_db):
    _save_arch_flow("user1", "proj1", _legacy_arch_flow())
    ledger1 = migrate_legacy_arch_flow("user1", "proj1")
    ledger2 = migrate_legacy_arch_flow("user1", "proj1")  # second call

    assert ledger1 is not None
    assert ledger2 is not None
    # Second call returns the stored ledger without re-creating decisions
    assert len(ledger1["decisions"]) == len(ledger2["decisions"])


def test_migrate_does_not_overwrite_existing_ledger(tmp_db):
    """If a ledger already exists, migrate returns it unchanged."""
    L = empty_ledger("proj1", "user1")
    L["user_style_hint"] = "SENTINEL"
    save_ledger("user1", L, "proj1")

    _save_arch_flow("user1", "proj1", _legacy_arch_flow())
    returned = migrate_legacy_arch_flow("user1", "proj1")

    assert returned is not None
    assert returned["user_style_hint"] == "SENTINEL"


def test_migrate_maps_stage_deployment_to_phase_diagram(tmp_db):
    flow = {
        "stage": "DEPLOYMENT",
        "quality_attribute": "latencia",
        "add_context": "",
        "current_asr": "ASR: pagos\nSummary: latencia",
        "style": "CQRS",
        "tactics": [],
        "deployment_diagram_puml": "",
        "deployment_diagram_svg_b64": "",
        "last_diagram": {"dot": "digraph {A->B}", "dot_raw": "", "dot_drawio": "", "detail_level": "overview", "level": 1, "overview_mapping": None},
    }
    _save_arch_flow("user1", "proj1", flow)
    ledger = migrate_legacy_arch_flow("user1", "proj1")

    assert ledger is not None
    assert ledger["current_phase"] == Phase.DIAGRAM.value
    kinds = {d["kind"] for d in ledger["decisions"]}
    assert "diagram" in kinds


def test_migrate_does_not_touch_arch_flow_row(tmp_db):
    """Migration must not delete or modify the original arch_flow row."""
    flow = _legacy_arch_flow()
    _save_arch_flow("user1", "proj1", flow)

    migrate_legacy_arch_flow("user1", "proj1")

    key = _mem._arch_flow_key("proj1")
    raw = _mem.get("user1", key, "")
    assert raw != "", "arch_flow row should still exist after migration"
    data = json.loads(raw)
    assert data["stage"] == flow["stage"]
    assert data["style"] == flow["style"]
