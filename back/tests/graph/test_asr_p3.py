"""Phase 3 — asr_node ledger write-back & prompt injection tests."""
from unittest.mock import MagicMock, patch

import pytest

from src.graph.nodes.asr import (
    _build_asr_payload,
    _build_sources_from_docs,
    _coerce_single_asr_markdown,
    _extract_dossier_history,
    asr_node,
)
from src.ledger.types import LedgerConcurrencyError, LedgerValidationError

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_PATCH_LLM       = "src.graph.nodes.asr.llm"
_PATCH_APPEND    = "src.graph.nodes.asr.append_decision"
_PATCH_LOAD      = "src.graph.nodes.asr.load_ledger"
_PATCH_RENDER    = "src.graph.nodes.asr.render_dossier"
_PATCH_RENDER_C  = "src.graph.nodes.asr.render_dossier_compact"
_PATCH_PHASE_P   = "src.graph.nodes.asr.render_phase_prompt"
_PATCH_RETRIEVER = "src.graph.nodes.asr.get_indexed_retriever"

# ---------------------------------------------------------------------------
# Standard LLM mock output
# ---------------------------------------------------------------------------
_MOCK_CONTENT = """\
## ASR

**ASR complete:** Users making checkout requests during peak traffic see p95 latency under 200ms.

### Scenario

- **Source:** End user browser
- **Stimulus:** Checkout button click during flash sale
- **Environment:** Production, 10k concurrent users
- **Artifact:** Payment service API
- **Response:** Process payment and return confirmation
- **Response Measure:** p95 < 200ms at 10k RPS
"""

_DEFAULT_SAVED = {"id": "01NEWDECISION00000000001", "kind": "asr", "qa": "latencia"}

_MULTI_ASR_CONTENT = """\
## ASR 1

**ASR complete:** Primer ASR de latencia.

### Scenario

- **Source:** User
- **Stimulus:** Search request
- **Environment:** Normal load
- **Artifact:** Search API
- **Response:** Return results
- **Response Measure:** p95 < 800 ms

## ASR 2

**ASR complete:** Segundo ASR de latencia.

### Scenario

- **Source:** User
- **Stimulus:** Peak search request
- **Environment:** Peak load
- **Artifact:** Search API
- **Response:** Return results under stress
- **Response Measure:** p95 < 800 ms at 3000 RPS
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**kw):
    base = {
        "user_id_for_prefs": "user-test",
        "project_id":        "proj-test",
        "userQuestion":      "I need a latency ASR for my checkout service",
        "language":          "es",
        "resolved_index":    "latencia",
        "force_rag":         False,
        "doc_only":          False,
        "doc_context":       "",
        "add_context":       "",
        "project_context_text": "",
        "user_style_hint":   "",
        "design_dossier_md": "",
        "memory_text":       "",
        "messages":          [],
        "turn_messages":     [],
        "ledger":            {},
        "ledger_active":     {},
        "current_phase":     "INTAKE",
        "ledger_dossier_compact": "",
        "ledger_phase_prompt":    "",
        "ledger_pending_advance": {},
    }
    base.update(kw)
    return base


def _mock_llm_result(content=_MOCK_CONTENT):
    r = MagicMock()
    r.content = content
    return r


def _run(state, *, append_return=None, append_raises=None,
         load_raises=None, load_return=None):
    """Run asr_node with all ledger + LLM + retriever calls mocked."""
    from src.ledger.types import empty_ledger

    fresh = load_return or empty_ledger("proj-test", "user-test")

    with patch(_PATCH_LLM) as ml, \
         patch(_PATCH_APPEND) as ma, \
         patch(_PATCH_LOAD, side_effect=load_raises or (lambda *a, **k: fresh)) as mload, \
         patch(_PATCH_RENDER, return_value="# dossier"), \
         patch(_PATCH_RENDER_C, return_value="compact"), \
         patch(_PATCH_PHASE_P, return_value="phase_prompt"), \
         patch(_PATCH_RETRIEVER):

        ml.invoke.return_value = _mock_llm_result()
        if append_raises:
            ma.side_effect = append_raises
        else:
            ma.return_value = append_return or _DEFAULT_SAVED

        result = asr_node(state)
        return result, ml, ma, mload


# ===========================================================================
# Pure-function tests (no LLM, no DB)
# ===========================================================================

def test_build_asr_payload_extracts_all_fields():
    payload = _build_asr_payload(_MOCK_CONTENT, "e-commerce")
    assert payload["summary"] == "Users making checkout requests during peak traffic see p95 latency under 200ms."
    assert payload["source"]           == "End user browser"
    assert payload["stimulus"]         == "Checkout button click during flash sale"
    assert payload["environment"]      == "Production, 10k concurrent users"
    assert payload["artifact"]         == "Payment service API"
    assert payload["response"]         == "Process payment and return confirmation"
    assert payload["response_measure"] == "p95 < 200ms at 10k RPS"
    assert payload["domain"]           == "e-commerce"


def test_build_asr_payload_fallback_on_malformed():
    payload = _build_asr_payload("just plain text with no structure", "fintech")
    assert payload["summary"] != ""
    assert payload["source"]           == ""
    assert payload["stimulus"]         == ""
    assert payload["environment"]      == ""
    assert payload["artifact"]         == ""
    assert payload["response"]         == ""
    assert payload["response_measure"] == ""
    assert payload["domain"]           == "fintech"


def test_extract_dossier_history_empty_for_none_yet_only():
    dossier = (
        "# Design Dossier\n\n"
        "## Historial (reemplazadas / rechazadas)\n\n"
        "_(ninguna aún)_\n"
    )
    assert _extract_dossier_history(dossier) == ""


def test_extract_dossier_history_empty_for_blank_input():
    assert _extract_dossier_history("") == ""
    assert _extract_dossier_history(None) == ""  # type: ignore[arg-type]


def test_extract_dossier_history_returns_section_when_superseded():
    dossier = (
        "# Design Dossier\n\n"
        "## Historial (reemplazadas / rechazadas)\n\n"
        "### Reemplazadas\n"
        "- ASR 01ABC → reemplazado por 01XYZ el 2024-01-01\n"
    )
    result = _extract_dossier_history(dossier)
    assert result != ""
    assert "01ABC" in result


def test_extract_dossier_history_works_for_english_heading():
    dossier = (
        "## History (superseded / rejected)\n\n"
        "### Superseded\n"
        "- ASR 01ABC → replaced by 01XYZ on 2024-01-01\n"
    )
    result = _extract_dossier_history(dossier)
    assert result != ""
    assert "01ABC" in result


def test_build_sources_from_docs_dedupes_and_caps():
    doc = MagicMock()
    doc.metadata = {"source_title": "Bass", "page_label": "12", "source_path": "/a/b"}
    docs = [doc] * 6
    result = _build_sources_from_docs(docs)
    assert len(result) == 1  # deduped
    assert result[0]["title"] == "Bass"
    assert result[0]["page"]  == "12"


def test_build_sources_from_docs_caps_at_four():
    docs = []
    for i in range(6):
        d = MagicMock()
        d.metadata = {"source_title": f"Book{i}", "page_label": str(i), "source_path": f"/p{i}"}
        docs.append(d)
    result = _build_sources_from_docs(docs)
    assert len(result) == 4


def test_coerce_single_asr_markdown_keeps_only_first_asr():
    result = _coerce_single_asr_markdown(_MULTI_ASR_CONTENT)
    assert result.startswith("## ASR")
    assert "## ASR 2" not in result
    assert "Segundo ASR de latencia" not in result
    assert "Primer ASR de latencia" in result


# ===========================================================================
# Unit tests — mocked LLM + mocked ledger
# ===========================================================================

def test_scalar_writes_always_set():
    result, _, _, _ = _run(_state())
    assert result["current_asr"]        != ""
    assert result["quality_attribute"]  == "latencia"
    assert result["arch_stage"]         == "ASR"
    assert result["hasVisitedASR"]      is True
    assert result["nextNode"]           == "unifier"
    assert result["memory_text"]        != ""


def test_asr_node_coerces_multi_asr_output_to_single():
    multi = _mock_llm_result(_MULTI_ASR_CONTENT)
    with patch(_PATCH_LLM) as ml, \
         patch(_PATCH_APPEND, return_value=_DEFAULT_SAVED), \
         patch(_PATCH_LOAD), \
         patch(_PATCH_RENDER, return_value="# dossier"), \
         patch(_PATCH_RENDER_C, return_value="compact"), \
         patch(_PATCH_PHASE_P, return_value="phase_prompt"), \
         patch(_PATCH_RETRIEVER):

        ml.invoke.return_value = multi
        result = asr_node(_state())

    assert result["current_asr"].startswith("## ASR")
    assert "## ASR 2" not in result["current_asr"]
    assert "Segundo ASR de latencia" not in result["current_asr"]
    assert "Primer ASR de latencia" in result["current_asr"]
    assert "## ASR 2" not in result["last_asr"]


def test_no_history_injection_when_dossier_empty():
    captured = []
    with patch(_PATCH_LLM) as ml, \
         patch(_PATCH_APPEND, return_value=_DEFAULT_SAVED), \
         patch(_PATCH_LOAD), \
         patch(_PATCH_RENDER, return_value="# dossier"), \
         patch(_PATCH_RENDER_C, return_value="compact"), \
         patch(_PATCH_PHASE_P, return_value="phase_prompt"), \
         patch(_PATCH_RETRIEVER):

        ml.invoke.side_effect = lambda p: (captured.append(p), _mock_llm_result())[1]
        asr_node(_state(design_dossier_md=""))

    assert len(captured) == 1
    assert "PRIOR ASR HISTORY" not in captured[0]


def test_history_injected_when_dossier_has_superseded():
    dossier_with_history = (
        "# Design Dossier\n\n"
        "## Historial (reemplazadas / rechazadas)\n\n"
        "### Reemplazadas\n"
        "- ASR 01OLDASR → reemplazado por 01NEWASR el 2024-01-01\n"
    )
    captured = []
    with patch(_PATCH_LLM) as ml, \
         patch(_PATCH_APPEND, return_value=_DEFAULT_SAVED), \
         patch(_PATCH_LOAD), \
         patch(_PATCH_RENDER, return_value="# dossier"), \
         patch(_PATCH_RENDER_C, return_value="compact"), \
         patch(_PATCH_PHASE_P, return_value="phase_prompt"), \
         patch(_PATCH_RETRIEVER):

        ml.invoke.side_effect = lambda p: (captured.append(p), _mock_llm_result())[1]
        asr_node(_state(design_dossier_md=dossier_with_history))

    assert len(captured) == 1
    # Spanish default: "HISTORIAL DE ASR PREVIOS"; English: "PRIOR ASR HISTORY"
    assert ("HISTORIAL DE ASR PREVIOS" in captured[0] or "PRIOR ASR HISTORY" in captured[0])
    assert "01OLDASR" in captured[0]


def test_append_called_with_correct_shape():
    _, _, ma, _ = _run(_state())
    assert ma.call_count == 1
    _, _project_id, decision = ma.call_args[0]
    assert decision["kind"]            == "asr"
    assert decision["qa"]              == "latencia"
    assert decision["created_by_node"] == "asr_node"
    assert decision["parents"]         == []
    assert "summary" in decision["payload"]


def test_append_called_with_correct_user_and_project():
    _, _, ma, _ = _run(_state(user_id_for_prefs="alice", project_id="proj-x"))
    _user_id, _project_id, _ = ma.call_args[0]
    assert _user_id    == "alice"
    assert _project_id == "proj-x"


def test_validation_error_is_nonfatal():
    result, _, _, _ = _run(_state(), append_raises=LedgerValidationError("bad"))
    assert result["current_asr"]       != ""
    assert result["quality_attribute"] == "latencia"
    assert result["arch_stage"]        == "ASR"


def test_concurrency_error_is_nonfatal():
    result, _, _, _ = _run(_state(), append_raises=LedgerConcurrencyError("conflict"))
    assert result["current_asr"]       != ""
    assert result["quality_attribute"] == "latencia"


def test_generic_exception_is_nonfatal():
    result, _, _, _ = _run(_state(), append_raises=RuntimeError("db down"))
    assert result["current_asr"]       != ""
    assert result["quality_attribute"] == "latencia"
    assert result["arch_stage"]        == "ASR"


def test_no_user_id_skips_ledger_write():
    result, _, ma, _ = _run(_state(user_id_for_prefs=""))
    ma.assert_not_called()
    assert result["current_asr"]       != ""
    assert result["quality_attribute"] == "latencia"


def test_state_ledger_fields_refreshed_after_success():
    from src.ledger.types import empty_ledger
    saved_id = "01REFRESHED000000000001"
    saved    = {"id": saved_id, "kind": "asr", "qa": "latencia",
                "status": "active", "payload": {"summary": "x"}}
    fresh    = empty_ledger("proj-test", "user-test")
    fresh["decisions"] = [saved]

    result, _, _, _ = _run(
        _state(),
        append_return=saved,
        load_return=fresh,
    )
    # _refresh_ledger_state is called after append; render_dossier mock returns "# dossier"
    assert result["design_dossier_md"]      == "# dossier"
    assert result["ledger_dossier_compact"] == "compact"
    assert result["ledger_phase_prompt"]    == "phase_prompt"


def test_state_refresh_failure_is_nonfatal():
    result, _, _, _ = _run(
        _state(),
        load_raises=RuntimeError("db gone"),
    )
    # scalar writes must still be present despite refresh failure
    assert result["current_asr"]       != ""
    assert result["quality_attribute"] == "latencia"
    assert result["arch_stage"]        == "ASR"


# ===========================================================================
# Integration test — real SQLite via tmp_db
# ===========================================================================

def test_second_asr_supersedes_first(tmp_db):
    from src.ledger.store import load_ledger as _load, save_ledger
    from src.ledger.types import empty_ledger

    save_ledger("user-test", empty_ledger("proj-test", "user-test"), "proj-test")

    mock_result = _mock_llm_result()
    with patch(_PATCH_LLM) as ml, patch(_PATCH_RETRIEVER):
        ml.invoke.return_value = mock_result
        asr_node(_state(resolved_index="latencia"))
        asr_node(_state(resolved_index="latencia"))

    ledger  = _load("user-test", "proj-test", auto_migrate=False)
    asr_ds  = [d for d in ledger["decisions"] if d["kind"] == "asr"]
    assert len(asr_ds) == 2

    statuses = {d["status"] for d in asr_ds}
    assert statuses == {"active", "superseded"}

    active_d = next(d for d in asr_ds if d["status"] == "active")
    supr_d   = next(d for d in asr_ds if d["status"] == "superseded")
    assert supr_d["superseded_by"] == active_d["id"]
    assert active_d["qa"] == "latencia"
