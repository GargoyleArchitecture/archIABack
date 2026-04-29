"""Phase 2 — context_loader ledger hydration tests."""
import pytest
from unittest.mock import patch, MagicMock, call

from src.graph.nodes.context_loader import context_loader_node, _mirror_legacy
from src.ledger.types import empty_ledger
from src.ledger.store import compute_active_view


# ── helpers ──────────────────────────────────────────────────────────────────

_PATCH_LOAD   = "src.graph.nodes.context_loader.load_ledger"
_PATCH_ACTIVE = "src.graph.nodes.context_loader.compute_active_view"
_PATCH_RENDER = "src.graph.nodes.context_loader.render_dossier"
_PATCH_COMPACT = "src.graph.nodes.context_loader.render_dossier_compact"
_PATCH_PROMPT  = "src.graph.nodes.context_loader.render_phase_prompt"


def _run(state, ledger_return=None, load_raises=None):
    """Run context_loader_node with mocked ledger layer."""
    empty = empty_ledger("proj-test", "user-test")
    ledger_val = ledger_return if ledger_return is not None else empty

    with patch(_PATCH_LOAD, return_value=ledger_val) as mock_load, \
         patch(_PATCH_RENDER, return_value="# dossier") as mock_render, \
         patch(_PATCH_COMPACT, return_value="compact") as _, \
         patch(_PATCH_PROMPT, return_value="phase_prompt") as _:

        if load_raises:
            mock_load.side_effect = load_raises

        result = context_loader_node(state, config=None)
        return result, mock_load, mock_render


# ── test cases ────────────────────────────────────────────────────────────────

def test_hydrates_ledger_fields_on_empty_ledger(base_state, mock_ledger_empty):
    result, _, _ = _run(base_state, ledger_return=mock_ledger_empty)

    assert result["ledger"] == mock_ledger_empty
    assert result["ledger_active"] == {}
    assert result["current_phase"] == "INTAKE"
    assert result["ledger_pending_advance"] == {}
    # Legacy scalars untouched (empty ledger has no active decisions)
    assert result["current_asr"] == ""
    assert result["quality_attribute"] == ""


def test_mirrors_asr_to_legacy_scalars(base_state, mock_ledger_with_asr):
    result, _, _ = _run(base_state, ledger_return=mock_ledger_with_asr)

    assert result["current_asr"] == "p95 < 200ms at 5k RPS"
    assert result["quality_attribute"] == "latencia"


def test_infers_asr_qa_from_summary_when_ledger_has_general(base_state, mock_ledger_with_asr):
    ledger = dict(mock_ledger_with_asr)
    ledger["decisions"] = [dict(mock_ledger_with_asr["decisions"][0], qa="general")]

    result, _, _ = _run(base_state, ledger_return=ledger)

    assert result["quality_attribute"] == "latencia"


def test_general_ledger_qa_does_not_overwrite_existing_specific_qa(base_state, mock_ledger_with_asr):
    ledger = dict(mock_ledger_with_asr)
    ledger["decisions"] = [dict(mock_ledger_with_asr["decisions"][0], qa="general", payload={"summary": "ASR vigente"})]
    state = {**base_state, "quality_attribute": "seguridad"}

    result, _, _ = _run(state, ledger_return=ledger)

    assert result["quality_attribute"] == "seguridad"


def test_mirrors_style_to_legacy_scalars(base_state, mock_ledger_full):
    result, _, _ = _run(base_state, ledger_return=mock_ledger_full)

    assert result["style"] == "Event-Driven"
    assert result["selected_style"] == "Event-Driven"
    assert result["last_style"] == "Event-Driven"


def test_mirrors_tactic_to_legacy_scalars(base_state, mock_ledger_full):
    result, _, _ = _run(base_state, ledger_return=mock_ledger_full)

    assert len(result["tactics_struct"]) == 1
    assert result["tactics_struct"][0]["name"] == "CQRS"
    assert result["tactics_list"] == ["CQRS"]


def test_empty_ledger_does_not_overwrite_existing_legacy_scalars(base_state, mock_ledger_empty):
    state = {**base_state, "current_asr": "old value", "quality_attribute": "seguridad"}
    result, _, _ = _run(state, ledger_return=mock_ledger_empty)

    assert result["current_asr"] == "old value"
    assert result["quality_attribute"] == "seguridad"


def test_ledger_load_failure_is_nonfatal(base_state, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="context_loader"):
        result, _, _ = _run(base_state, load_raises=Exception("DB unavailable"))

    # Should not raise; ledger fields not updated; prior state preserved
    assert result["ledger"] == {}
    assert result["ledger_active"] == {}
    assert result["current_phase"] == ""
    assert any("ledger hydration failed" in r.message for r in caplog.records)


def test_early_return_when_no_user_id(base_state):
    state = {**base_state, "user_id_for_prefs": ""}

    with patch(_PATCH_LOAD) as mock_load:
        result = context_loader_node(state, config=None)

    mock_load.assert_not_called()
    assert result is state  # exact same object — early return


def test_ledger_loaded_every_turn_no_session_guard(base_state, mock_ledger_empty):
    with patch(_PATCH_LOAD, return_value=mock_ledger_empty) as mock_load, \
         patch(_PATCH_RENDER, return_value=""), \
         patch(_PATCH_COMPACT, return_value=""), \
         patch(_PATCH_PROMPT, return_value=""):

        context_loader_node(base_state, config=None)
        context_loader_node(base_state, config=None)

    assert mock_load.call_count == 2


def test_language_defaults_to_es_on_first_turn(base_state, mock_ledger_empty):
    state = {**base_state, "language": None}

    with patch(_PATCH_LOAD, return_value=mock_ledger_empty), \
         patch(_PATCH_RENDER, return_value="") as mock_render, \
         patch(_PATCH_COMPACT, return_value=""), \
         patch(_PATCH_PROMPT, return_value=""):

        context_loader_node(state, config=None)

    mock_render.assert_called_once()
    _, kwargs = mock_render.call_args
    assert kwargs.get("lang") == "es"


def test_pending_advance_stored_as_empty_dict_when_none(base_state, mock_ledger_empty):
    ledger = dict(mock_ledger_empty)
    ledger["pending_advance"] = None
    result, _, _ = _run(base_state, ledger_return=ledger)

    assert result["ledger_pending_advance"] == {}


def test_pending_advance_stored_when_present(base_state, mock_ledger_empty):
    advance = {"from_phase": "INTAKE", "to_phase": "ASR", "iteration": 2, "skipped_phases": []}
    ledger = dict(mock_ledger_empty)
    ledger["pending_advance"] = advance
    result, _, _ = _run(base_state, ledger_return=ledger)

    assert result["ledger_pending_advance"] == advance
