"""Golden-file tests for src.ledger.render."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ledger.render import render_dossier, render_dossier_compact, render_phase_prompt

SNAPSHOTS = Path(__file__).parent / "snapshots"
FIXTURES  = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    with open(FIXTURES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _snap(name: str) -> str:
    with open(SNAPSHOTS / f"{name}.md", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Golden-file (snapshot) tests
# ---------------------------------------------------------------------------

def test_render_empty_ledger_is_deterministic():
    ledger = _load("empty")
    out1 = render_dossier(ledger, lang="es")
    out2 = render_dossier(ledger, lang="es")
    assert out1 == out2


def test_render_asr_only_matches_snapshot():
    ledger = _load("asr_only")
    assert render_dossier(ledger, lang="es") == _snap("asr_only")


def test_render_full_pipeline_matches_snapshot():
    ledger = _load("asr_style_tactic")
    assert render_dossier(ledger, lang="es") == _snap("asr_style_tactic")


def test_render_with_rejected_and_orphans_matches_snapshot():
    ledger = _load("with_rejected_orphans")
    assert render_dossier(ledger, lang="es") == _snap("with_rejected_orphans")


def test_render_compact_matches_snapshot():
    ledger = _load("asr_style_tactic")
    assert render_dossier_compact(ledger, lang="es") == _snap("asr_style_tactic_compact")


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

def test_render_focus_asr_omits_other_sections():
    ledger = _load("asr_style_tactic")
    out = render_dossier(ledger, lang="es", focus="asr")
    assert "### ASR" in out
    assert "### Estilo" not in out
    assert "### Tácticas" not in out


def test_render_clips_long_rationale_with_marker():
    ledger = _load("asr_only")
    ledger = dict(ledger)
    ledger["decisions"] = [dict(d) for d in ledger["decisions"]]
    ledger["decisions"][0]["rationale"] = "X" * 2000
    out = render_dossier(ledger, lang="es")
    assert "… [truncado]" in out


def test_render_empty_sections_use_none_yet():
    ledger = _load("empty")
    out = render_dossier(ledger, lang="es")
    assert "_(ninguna aún)_" in out


def test_render_english_labels():
    ledger = _load("asr_only")
    out = render_dossier(ledger, lang="en")
    assert "Phase:" in out
    assert "Iteration:" in out
    assert "none yet" in out


def test_render_phase_prompt_returns_next_step():
    ledger = _load("asr_only")
    out = render_phase_prompt(ledger, lang="es")
    assert "Próximo paso" in out


def test_render_phase_prompt_returns_empty_for_done():
    import json
    ledger = _load("empty")
    ledger["current_phase"] = "DONE"
    out = render_phase_prompt(ledger, lang="es")
    assert out == ""


def test_render_dossier_idempotent_all_fixtures():
    """Idempotence: rendering twice yields identical output."""
    for name in ["empty", "intake_only", "asr_only", "asr_style_tactic", "with_rejected_orphans"]:
        ledger = _load(name)
        assert render_dossier(ledger) == render_dossier(ledger), f"Not idempotent for {name}"


def test_render_compact_contains_phase():
    ledger = _load("asr_style_tactic")
    out = render_dossier_compact(ledger)
    assert "TACTICS" in out


def test_render_compact_pending_advance():
    ledger = _load("empty")
    ledger["pending_advance"] = {
        "from_phase": "INTAKE", "to_phase": "ASR", "iteration": 1,
        "triggered_by": "agent_suggestion_accepted", "user_message": "",
        "skipped_phases": [], "timestamp": "2024-01-01T00:00:00Z",
    }
    out = render_dossier_compact(ledger)
    assert "Avance pendiente" in out
