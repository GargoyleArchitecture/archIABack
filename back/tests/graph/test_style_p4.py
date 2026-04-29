"""Phase 4 — style_node_impl ledger write-back & prompt injection tests."""
from unittest.mock import MagicMock, patch

import pytest

from src.graph.nodes.styles.common import (
    _build_asr_parent_ref,
    _build_dossier_asr_binding,
    _build_style_payload,
    style_node_impl,
)
from src.ledger.types import LedgerConcurrencyError, LedgerValidationError

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_PATCH_LLM       = "src.graph.nodes.styles.common.llm"
_PATCH_APPEND    = "src.graph.nodes.styles.common.append_decision"
_PATCH_LOAD      = "src.graph.nodes.styles.common.load_ledger"
_PATCH_RENDER    = "src.graph.nodes.styles.common.render_dossier"
_PATCH_RENDER_C  = "src.graph.nodes.styles.common.render_dossier_compact"
_PATCH_PHASE_P   = "src.graph.nodes.styles.common.render_phase_prompt"
_PATCH_RETRIEVER = "src.graph.nodes.styles.common.get_indexed_retriever"

# ---------------------------------------------------------------------------
# Standard mock LLM output
# ---------------------------------------------------------------------------
_MOCK_STYLE_JSON = (
    '{"style_1": {"name": "Event-Driven", "impact": "async processing reduces p95 latency, '
    'decouples producers from consumers"}, "style_2": {"name": "Layered", "impact": "simple '
    'but synchronous bottleneck under high load"}, "best_style": "style_1", "rationale": '
    '"Event-Driven directly improves latency by decoupling producers from consumers, enabling '
    'async processing that satisfies p95<200ms@5kRPS"}'
)

_SAVED_STYLE = {
    "id": "01SAVEDSTYLE00000000001",
    "kind": "style",
    "qa": "latencia",
    "iteration": 1,
    "payload": {"chosen": "Event-Driven"},
}


def _base_state(**kw):
    base = {
        "user_id_for_prefs": "user-test",
        "project_id": "proj-test",
        "language": "es",
        "current_asr": "p95 < 200ms at 5k RPS",
        "quality_attribute": "latencia",
        "ledger": {},
        "ledger_active": {},
        "design_dossier_md": "",
        "ledger_dossier_compact": "",
        "ledger_phase_prompt": "",
        "current_phase": "ASR",
        "ledger_pending_advance": {},
        "memory_text": "",
        "turn_messages": [],
    }
    base.update(kw)
    return base


def _mock_llm(content: str = _MOCK_STYLE_JSON):
    mock_result = MagicMock()
    mock_result.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_result
    return mock_llm


def _mock_retriever():
    mock_ret = MagicMock()
    mock_ret.invoke.return_value = []
    mock_factory = MagicMock(return_value=mock_ret)
    return mock_factory


# ---------------------------------------------------------------------------
# 1. Pure-function tests (no mocked LLM)
# ---------------------------------------------------------------------------

def test_build_dossier_asr_binding_empty_when_no_asr():
    assert _build_dossier_asr_binding({}) == ""
    assert _build_dossier_asr_binding({"style": {"id": "x"}}) == ""


def test_build_dossier_asr_binding_contains_qa_and_rm():
    ledger_active = {
        "asr": {
            "id": "01TEST000000000000ASR0001",
            "qa": "latencia",
            "payload": {"response_measure": "p95<200ms@5kRPS", "domain": "e-commerce", "summary": "test"},
        }
    }
    block = _build_dossier_asr_binding(ledger_active, lang="en")
    assert "ACTIVE ASR BINDING" in block
    assert "latencia" in block
    assert "p95<200ms@5kRPS" in block


def test_build_style_payload_structure():
    style1 = {"name": "Event-Driven", "impact": "async"}
    style2 = {"name": "Layered", "impact": "simple"}
    payload = _build_style_payload({}, "Event-Driven", style1, style2, "rationale text")
    assert payload["chosen"] == "Event-Driven"
    assert payload["name"] == "Event-Driven"
    assert len(payload["candidates"]) == 2
    assert payload["tradeoffs"] == "rationale text"


def test_build_asr_parent_ref_empty_when_no_asr():
    assert _build_asr_parent_ref({}) == []
    assert _build_asr_parent_ref({"style": {"id": "x"}}) == []


def test_build_asr_parent_ref_returns_correct_ref():
    ledger_active = {
        "asr": {"id": "01TEST000000000000ASR0001", "qa": "latencia", "iteration": 2}
    }
    refs = _build_asr_parent_ref(ledger_active)
    assert len(refs) == 1
    assert refs[0]["id"] == "01TEST000000000000ASR0001"
    assert refs[0]["kind"] == "asr"
    assert refs[0]["iteration"] == 2


# ---------------------------------------------------------------------------
# 2. Unit tests (mocked LLM + mocked ledger)
# ---------------------------------------------------------------------------

def _run_style(state, append_side_effect=None, saved=None):
    """Helper: run style_node_impl with standard mocks."""
    _saved = saved or _SAVED_STYLE
    with (
        patch(_PATCH_LLM, _mock_llm()),
        patch(_PATCH_RETRIEVER, _mock_retriever()),
        patch(_PATCH_APPEND, return_value=_saved) as mock_append,
        patch(_PATCH_LOAD, return_value={"decisions": [_saved], "current_phase": "STYLE", "pending_advance": {}}) as mock_load,
        patch(_PATCH_RENDER, return_value="dossier") as mock_render,
        patch(_PATCH_RENDER_C, return_value="compact") as mock_render_c,
        patch(_PATCH_PHASE_P, return_value="phase") as mock_phase_p,
    ):
        if append_side_effect is not None:
            mock_append.side_effect = append_side_effect
        result = style_node_impl(state)
        return result, mock_append


def test_scalar_writes_always_set():
    state = _base_state()
    result, _ = _run_style(state)
    assert result["style"] == "Event-Driven"
    assert result["selected_style"] == "Event-Driven"
    assert result["last_style"] == "Event-Driven"
    assert result["arch_stage"] == "STYLE"
    assert result["quality_attribute"] == "latencia"
    assert result["endMessage"]
    assert result["nextNode"] == "unifier"


def test_no_binding_block_when_ledger_empty():
    state = _base_state(ledger_active={})
    captured_prompt = []

    class CaptureLLM:
        def invoke(self, prompt):
            captured_prompt.append(prompt)
            m = MagicMock()
            m.content = _MOCK_STYLE_JSON
            return m

    with (
        patch(_PATCH_LLM, CaptureLLM()),
        patch(_PATCH_RETRIEVER, _mock_retriever()),
        patch(_PATCH_APPEND, return_value=_SAVED_STYLE),
        patch(_PATCH_LOAD, return_value={"decisions": [], "current_phase": "STYLE", "pending_advance": {}}),
        patch(_PATCH_RENDER, return_value=""),
        patch(_PATCH_RENDER_C, return_value=""),
        patch(_PATCH_PHASE_P, return_value=""),
    ):
        style_node_impl(state)

    assert captured_prompt
    assert "ACTIVE ASR BINDING" not in captured_prompt[0]


def test_binding_block_injected_when_active_asr(mock_ledger_with_asr):
    from src.ledger.store import compute_active_view
    active = compute_active_view(mock_ledger_with_asr)
    state = _base_state(ledger_active=active, ledger=mock_ledger_with_asr)
    captured_prompt = []

    class CaptureLLM:
        def invoke(self, prompt):
            captured_prompt.append(prompt)
            m = MagicMock()
            m.content = _MOCK_STYLE_JSON
            return m

    with (
        patch(_PATCH_LLM, CaptureLLM()),
        patch(_PATCH_RETRIEVER, _mock_retriever()),
        patch(_PATCH_APPEND, return_value=_SAVED_STYLE),
        patch(_PATCH_LOAD, return_value={"decisions": [], "current_phase": "STYLE", "pending_advance": {}}),
        patch(_PATCH_RENDER, return_value=""),
        patch(_PATCH_RENDER_C, return_value=""),
        patch(_PATCH_PHASE_P, return_value=""),
    ):
        style_node_impl(state)

    assert captured_prompt
    assert "ACTIVE ASR BINDING" in captured_prompt[0] or "VINCULACIÓN CON EL ASR ACTIVO" in captured_prompt[0]
    assert "p95<200ms@5kRPS" in captured_prompt[0]


def test_append_called_with_kind_style():
    state = _base_state()
    result, mock_append = _run_style(state)
    mock_append.assert_called_once()
    call_args = mock_append.call_args[0]
    decision = call_args[2]
    assert decision["kind"] == "style"
    assert decision["qa"] == "latencia"
    assert decision["created_by_node"] == "style_node"


def test_append_called_with_asr_parent(mock_ledger_with_asr):
    from src.ledger.store import compute_active_view
    active = compute_active_view(mock_ledger_with_asr)
    state = _base_state(ledger_active=active, ledger=mock_ledger_with_asr)
    result, mock_append = _run_style(state)
    mock_append.assert_called_once()
    decision = mock_append.call_args[0][2]
    assert len(decision["parents"]) == 1
    assert decision["parents"][0]["kind"] == "asr"
    assert decision["parents"][0]["id"] == "01TEST000000000000ASR0001"


def test_append_not_called_when_no_user_id():
    state = _base_state(user_id_for_prefs="")
    result, mock_append = _run_style(state)
    mock_append.assert_not_called()
    # Scalar writes still present
    assert result["style"] == "Event-Driven"
    assert result["nextNode"] == "unifier"


def test_validation_error_is_nonfatal():
    state = _base_state()
    result, _ = _run_style(state, append_side_effect=LedgerValidationError("test validation"))
    assert result["style"] == "Event-Driven"
    assert result["nextNode"] == "unifier"


def test_concurrency_error_is_nonfatal():
    state = _base_state()
    result, _ = _run_style(state, append_side_effect=LedgerConcurrencyError("test concurrency"))
    assert result["style"] == "Event-Driven"
    assert result["nextNode"] == "unifier"


def test_bare_exception_is_nonfatal():
    state = _base_state()
    result, _ = _run_style(state, append_side_effect=RuntimeError("unexpected"))
    assert result["style"] == "Event-Driven"
    assert result["nextNode"] == "unifier"


def test_state_ledger_fields_refreshed_after_success():
    """After append_decision succeeds, ledger_active reflects the saved style."""
    saved = {
        "id": "01SAVEDSTYLE00000000001",
        "kind": "style", "qa": "latencia", "iteration": 1,
        "payload": {"chosen": "Event-Driven"},
        "status": "active",
    }
    fresh_ledger = {
        "decisions": [saved],
        "current_phase": "STYLE",
        "pending_advance": {},
    }
    state = _base_state()
    with (
        patch(_PATCH_LLM, _mock_llm()),
        patch(_PATCH_RETRIEVER, _mock_retriever()),
        patch(_PATCH_APPEND, return_value=saved),
        patch(_PATCH_LOAD, return_value=fresh_ledger),
        patch(_PATCH_RENDER, return_value="dossier"),
        patch(_PATCH_RENDER_C, return_value="compact"),
        patch(_PATCH_PHASE_P, return_value="phase"),
    ):
        result = style_node_impl(state)

    assert result["current_phase"] == "STYLE"
    assert result["design_dossier_md"] == "dossier"


def test_qa_continuity_from_ledger(mock_ledger_with_asr):
    """QA sourced from ledger active ASR is passed to append_decision."""
    from src.ledger.store import compute_active_view
    active = compute_active_view(mock_ledger_with_asr)
    # quality_attribute already set by P2's _mirror_legacy to ledger QA
    state = _base_state(
        quality_attribute="latencia",
        ledger_active=active,
        ledger=mock_ledger_with_asr,
    )
    result, mock_append = _run_style(state)
    mock_append.assert_called_once()
    decision = mock_append.call_args[0][2]
    assert decision["qa"] == "latencia"


# ---------------------------------------------------------------------------
# 3. Integration test — second style supersedes first (real SQLite via tmp_db)
# ---------------------------------------------------------------------------

def test_second_style_supersedes_first(tmp_db):
    import src.memory as _memory_mod
    from src.ledger.store import (
        save_ledger,
        append_decision as _append,
        load_ledger as _load,
        compute_active_view,
    )
    from src.ledger.types import empty_ledger

    # Seed: empty ledger + one ASR
    ledger = empty_ledger("proj-test", "user-test")
    save_ledger("user-test", ledger, "proj-test")
    asr_dec = _append("user-test", "proj-test", {
        "id": "", "kind": "asr", "phase": "ASR", "iteration": 0,
        "qa": "latencia", "parents": [],
        "payload": {
            "summary": "p95<200ms", "source": "", "stimulus": "",
            "environment": "", "artifact": "", "response": "",
            "response_measure": "p95<200ms@5kRPS", "domain": "e-commerce",
        },
        "rationale": "", "sources": [], "status": "active", "parent_status": "ok",
        "superseded_by": None, "rejection_reason": None,
        "created_at": "", "created_by_node": "asr_node",
    })

    fresh = _load("user-test", "proj-test")
    active = compute_active_view(fresh)

    base = {
        "user_id_for_prefs": "user-test",
        "project_id": "proj-test",
        "language": "es",
        "current_asr": "p95 < 200ms at 5k RPS",
        "quality_attribute": "latencia",
        "ledger": fresh,
        "ledger_active": active,
        "design_dossier_md": "",
        "ledger_dossier_compact": "",
        "ledger_phase_prompt": "",
        "current_phase": "ASR",
        "ledger_pending_advance": {},
        "memory_text": "",
        "turn_messages": [],
    }

    # First style call
    with patch(_PATCH_LLM, _mock_llm()), patch(_PATCH_RETRIEVER, _mock_retriever()):
        r1 = style_node_impl(dict(base))

    # Reload and pass refreshed state for second call
    fresh2 = _load("user-test", "proj-test")
    active2 = compute_active_view(fresh2)
    base2 = {**r1, "ledger": fresh2, "ledger_active": active2}

    # Second style call
    with patch(_PATCH_LLM, _mock_llm()), patch(_PATCH_RETRIEVER, _mock_retriever()):
        r2 = style_node_impl(dict(base2))

    final = _load("user-test", "proj-test")
    style_decisions = [d for d in final["decisions"] if d["kind"] == "style"]
    assert len(style_decisions) == 2, f"Expected 2 style decisions, got {len(style_decisions)}"

    statuses = {d["status"] for d in style_decisions}
    assert "active" in statuses
    assert "superseded" in statuses

    superseded = next(d for d in style_decisions if d["status"] == "superseded")
    active_dec = next(d for d in style_decisions if d["status"] == "active")
    assert superseded["superseded_by"] == active_dec["id"]
