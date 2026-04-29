"""Phase 4 — tactics_node_impl ledger write-back & prompt injection tests."""
from unittest.mock import MagicMock, patch

import pytest

from src.graph.nodes.tactics.common import (
    _build_dossier_design_binding,
    _build_parent_refs,
    _build_tactic_payload,
    _validate_tactic_traces,
    tactics_node_impl,
)
from src.ledger.types import LedgerConcurrencyError, LedgerValidationError

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_PATCH_LLM       = "src.graph.nodes.tactics.common.llm"
_PATCH_APPEND    = "src.graph.nodes.tactics.common.append_decision"
_PATCH_LOAD      = "src.graph.nodes.tactics.common.load_ledger"
_PATCH_RENDER    = "src.graph.nodes.tactics.common.render_dossier"
_PATCH_RENDER_C  = "src.graph.nodes.tactics.common.render_dossier_compact"
_PATCH_PHASE_P   = "src.graph.nodes.tactics.common.render_phase_prompt"
_PATCH_RETRIEVER = "src.graph.nodes.tactics.common.get_indexed_retriever"

# ---------------------------------------------------------------------------
# Standard mock LLM output
# ---------------------------------------------------------------------------
_MOCK_TACTICS_CONTENT = """\
## ASR & Style Context
Latency ASR: p95<200ms at 5k RPS. Style: Event-Driven.

## Tactics (TOP-3)

### Event Queue
**Rationale**: Decouples producers from consumers.

```json
[
  {
    "name": "Event Queue",
    "purpose": "Decouple producers",
    "rationale": "Reduces latency by async processing",
    "risks": ["ordering"],
    "tradeoffs": ["complexity"],
    "categories": ["latency"],
    "traces_to_asr": "Satisfies p95<200ms@5kRPS",
    "expected_effect": "p95 reduced",
    "success_probability": 0.85,
    "rank": 1
  },
  {
    "name": "Cache Aside",
    "purpose": "Reduce DB calls",
    "rationale": "Caches hot paths",
    "risks": ["staleness"],
    "tradeoffs": ["consistency"],
    "categories": ["latency"],
    "traces_to_asr": "Satisfies p95<200ms@5kRPS",
    "expected_effect": "DB load reduced",
    "success_probability": 0.80,
    "rank": 2
  },
  {
    "name": "Connection Pooling",
    "purpose": "Reuse connections",
    "rationale": "Reduces connection overhead",
    "risks": ["pool exhaustion"],
    "tradeoffs": ["config complexity"],
    "categories": ["latency"],
    "traces_to_asr": "Satisfies p95<200ms@5kRPS",
    "expected_effect": "Overhead reduced",
    "success_probability": 0.75,
    "rank": 3
  }
]
```
"""

_SAVED_TACTIC = {
    "id": "01SAVEDTACTIC0000000001",
    "kind": "tactic",
    "qa": "latencia",
    "iteration": 1,
    "payload": {"items": []},
}


def _base_state(**kw):
    base = {
        "user_id_for_prefs": "user-test",
        "project_id": "proj-test",
        "language": "es",
        "current_asr": "p95 < 200ms at 5k RPS",
        "quality_attribute": "latencia",
        "style": "Event-Driven",
        "selected_style": "Event-Driven",
        "last_style": "Event-Driven",
        "ledger": {},
        "ledger_active": {},
        "design_dossier_md": "",
        "ledger_dossier_compact": "",
        "ledger_phase_prompt": "",
        "current_phase": "STYLE",
        "ledger_pending_advance": {},
        "memory_text": "",
        "turn_messages": [],
        "messages": [],
    }
    base.update(kw)
    return base


def _mock_llm(content: str = _MOCK_TACTICS_CONTENT):
    mock_result = MagicMock()
    mock_result.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_result
    return mock_llm


def _mock_retriever():
    mock_ret = MagicMock()
    mock_ret.invoke.return_value = []
    return MagicMock(return_value=mock_ret)


# ---------------------------------------------------------------------------
# 1. Pure-function tests (no mocked LLM)
# ---------------------------------------------------------------------------

def test_build_dossier_design_binding_empty_when_no_style():
    ledger_active = {
        "asr": {"id": "x", "qa": "latencia", "payload": {"response_measure": "p95<200ms@5kRPS"}},
    }
    assert _build_dossier_design_binding(ledger_active) == ""


def test_build_dossier_design_binding_empty_when_no_asr():
    ledger_active = {
        "style": {"id": "y", "payload": {"chosen": "Event-Driven", "tradeoffs": "async"}},
    }
    assert _build_dossier_design_binding(ledger_active) == ""


def test_build_dossier_design_binding_contains_rm_and_style():
    ledger_active = {
        "asr": {
            "id": "01ASR0001",
            "qa": "latencia",
            "payload": {"response_measure": "p95<200ms@5kRPS", "domain": "e-commerce"},
        },
        "style": {
            "id": "01STY0001",
            "payload": {"chosen": "Event-Driven", "tradeoffs": "async decoupling"},
        },
    }
    block = _build_dossier_design_binding(ledger_active, lang="en")
    assert "ACTIVE DESIGN DECISIONS" in block
    assert "p95<200ms@5kRPS" in block
    assert "Event-Driven" in block


def test_validate_tactic_traces_fills_empty():
    items = [
        {"name": "T1", "traces_to_asr": ""},
        {"name": "T2", "traces_to_asr": None},
        {"name": "T3"},
    ]
    result = _validate_tactic_traces(items, "p95<200ms@5kRPS")
    for item in result:
        assert item.get("traces_to_asr") == "Satisfies Response Measure: p95<200ms@5kRPS"


def test_validate_tactic_traces_preserves_existing():
    items = [{"name": "T1", "traces_to_asr": "Already filled"}]
    result = _validate_tactic_traces(items, "p95<200ms@5kRPS")
    assert result[0]["traces_to_asr"] == "Already filled"


def test_build_parent_refs_returns_both_parents(mock_ledger_with_style):
    from src.ledger.store import compute_active_view
    active = compute_active_view(mock_ledger_with_style)
    refs = _build_parent_refs(active)
    kinds = {r["kind"] for r in refs}
    assert "asr" in kinds
    assert "style" in kinds
    assert len(refs) == 2


def test_build_parent_refs_asr_only_when_no_style(mock_ledger_with_asr):
    from src.ledger.store import compute_active_view
    active = compute_active_view(mock_ledger_with_asr)
    refs = _build_parent_refs(active)
    assert len(refs) == 1
    assert refs[0]["kind"] == "asr"


# ---------------------------------------------------------------------------
# 2. Unit tests (mocked LLM + mocked ledger)
# ---------------------------------------------------------------------------

def _run_tactics(state, append_side_effect=None, saved=None):
    _saved = saved or _SAVED_TACTIC
    with (
        patch(_PATCH_LLM, _mock_llm()),
        patch(_PATCH_RETRIEVER, _mock_retriever()),
        patch(_PATCH_APPEND, return_value=_saved) as mock_append,
        patch(_PATCH_LOAD, return_value={"decisions": [_saved], "current_phase": "TACTICS", "pending_advance": {}}),
        patch(_PATCH_RENDER, return_value="dossier"),
        patch(_PATCH_RENDER_C, return_value="compact"),
        patch(_PATCH_PHASE_P, return_value="phase"),
    ):
        if append_side_effect is not None:
            mock_append.side_effect = append_side_effect
        result = tactics_node_impl(state)
        return result, mock_append


def test_scalar_writes_always_set():
    state = _base_state()
    result, _ = _run_tactics(state)
    assert result["tactics_struct"] is not None
    assert isinstance(result["tactics_list"], list)
    assert result["arch_stage"] == "TACTICS"
    assert result["endMessage"] is not None
    assert result["nextNode"] == "unifier"


def test_no_binding_block_when_ledger_empty():
    state = _base_state(ledger_active={})
    captured_prompt = []

    class CaptureLLM:
        def invoke(self, prompt):
            captured_prompt.append(prompt)
            m = MagicMock()
            m.content = _MOCK_TACTICS_CONTENT
            return m

    with (
        patch(_PATCH_LLM, CaptureLLM()),
        patch(_PATCH_RETRIEVER, _mock_retriever()),
        patch(_PATCH_APPEND, return_value=_SAVED_TACTIC),
        patch(_PATCH_LOAD, return_value={"decisions": [], "current_phase": "TACTICS", "pending_advance": {}}),
        patch(_PATCH_RENDER, return_value=""),
        patch(_PATCH_RENDER_C, return_value=""),
        patch(_PATCH_PHASE_P, return_value=""),
    ):
        tactics_node_impl(state)

    assert captured_prompt
    assert "ACTIVE DESIGN DECISIONS" not in captured_prompt[0]
    assert "DECISIONES DE DISEÑO ACTIVAS" not in captured_prompt[0]


def test_binding_block_injected_when_full_pipeline(mock_ledger_with_style):
    from src.ledger.store import compute_active_view
    active = compute_active_view(mock_ledger_with_style)
    state = _base_state(ledger_active=active, ledger=mock_ledger_with_style)
    captured_prompt = []

    class CaptureLLM:
        def invoke(self, prompt):
            captured_prompt.append(prompt)
            m = MagicMock()
            m.content = _MOCK_TACTICS_CONTENT
            return m

    with (
        patch(_PATCH_LLM, CaptureLLM()),
        patch(_PATCH_RETRIEVER, _mock_retriever()),
        patch(_PATCH_APPEND, return_value=_SAVED_TACTIC),
        patch(_PATCH_LOAD, return_value={"decisions": [], "current_phase": "TACTICS", "pending_advance": {}}),
        patch(_PATCH_RENDER, return_value=""),
        patch(_PATCH_RENDER_C, return_value=""),
        patch(_PATCH_PHASE_P, return_value=""),
    ):
        tactics_node_impl(state)

    assert captured_prompt
    prompt = captured_prompt[0]
    assert ("ACTIVE DESIGN DECISIONS" in prompt or "DECISIONES DE DISEÑO ACTIVAS" in prompt)
    assert "p95<200ms@5kRPS" in prompt
    assert "Event-Driven" in prompt


def test_append_called_with_kind_tactic(mock_ledger_with_style):
    from src.ledger.store import compute_active_view
    active = compute_active_view(mock_ledger_with_style)
    state = _base_state(ledger_active=active, ledger=mock_ledger_with_style)
    result, mock_append = _run_tactics(state)
    mock_append.assert_called_once()
    decision = mock_append.call_args[0][2]
    assert decision["kind"] == "tactic"
    # Two parents: asr + style
    kinds = {p["kind"] for p in decision["parents"]}
    assert "asr" in kinds
    assert "style" in kinds


def test_validation_error_is_nonfatal():
    state = _base_state()
    result, _ = _run_tactics(state, append_side_effect=LedgerValidationError("test"))
    assert result["tactics_struct"] is not None
    assert result["nextNode"] == "unifier"


def test_no_user_id_skips_ledger_write():
    state = _base_state(user_id_for_prefs="")
    result, mock_append = _run_tactics(state)
    mock_append.assert_not_called()
    assert result["tactics_struct"] is not None
    assert result["nextNode"] == "unifier"


def test_state_ledger_fields_refreshed_after_success():
    saved = {
        "id": "01SAVEDTACTIC0000000001",
        "kind": "tactic", "qa": "latencia", "iteration": 1,
        "payload": {"items": []}, "status": "active",
    }
    fresh_ledger = {
        "decisions": [saved],
        "current_phase": "TACTICS",
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
        result = tactics_node_impl(state)

    assert result["current_phase"] == "TACTICS"
    assert result["design_dossier_md"] == "dossier"


def test_bare_exception_is_nonfatal():
    state = _base_state()
    result, _ = _run_tactics(state, append_side_effect=RuntimeError("unexpected"))
    assert result["tactics_struct"] is not None
    assert result["nextNode"] == "unifier"


# ---------------------------------------------------------------------------
# 3. Integration test — tactic rejected when parent style is rejected
# ---------------------------------------------------------------------------

def test_tactic_rejected_when_parent_style_is_rejected(tmp_db):
    """Core traceability guarantee: if the parent style is rejected, validate_parents()
    must raise LedgerValidationError, caught nonfatally by the node.
    The invalid tactic MUST NOT be persisted in the database.
    """
    from src.ledger.store import (
        save_ledger,
        append_decision as _append,
        reject_decision,
        load_ledger as _load,
        compute_active_view,
    )
    from src.ledger.types import empty_ledger

    # 1. Build ASR + style decisions in the real DB
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
    style_dec = _append("user-test", "proj-test", {
        "id": "", "kind": "style", "phase": "STYLE", "iteration": 0,
        "qa": "latencia",
        "parents": [{"id": asr_dec["id"], "kind": "asr", "iteration": asr_dec["iteration"]}],
        "payload": {"name": "Layered", "candidates": [], "chosen": "Layered", "tradeoffs": "test"},
        "rationale": "", "sources": [], "status": "active", "parent_status": "ok",
        "superseded_by": None, "rejection_reason": None,
        "created_at": "", "created_by_node": "style_node",
    })

    # 2. Reject the style
    reject_decision("user-test", "proj-test", style_dec["id"], "rejected in test")

    # 3. Build ledger_active reflecting rejected style (compute_active_view returns no style)
    current_ledger = _load("user-test", "proj-test")
    # Manually set ledger_active to include the rejected style as parent ref
    # so that _build_parent_refs includes it and validate_parents rejects it
    ledger_active_with_rejected = {
        "asr": {
            "id": asr_dec["id"],
            "kind": "asr",
            "qa": "latencia",
            "iteration": asr_dec["iteration"],
            "payload": {"response_measure": "p95<200ms@5kRPS"},
            "status": "active",
        },
        "style": {
            "id": style_dec["id"],
            "kind": "style",
            "qa": "latencia",
            "iteration": style_dec["iteration"],
            "payload": {"chosen": "Layered", "tradeoffs": "test"},
            "status": "rejected",
        },
    }

    state = _base_state(
        ledger=current_ledger,
        ledger_active=ledger_active_with_rejected,
    )

    with patch(_PATCH_LLM, _mock_llm()), patch(_PATCH_RETRIEVER, _mock_retriever()):
        result = tactics_node_impl(state)

    # 4. Node returned without crashing; scalar writes present
    assert result["tactics_struct"] is not None
    assert result["nextNode"] == "unifier"

    # 5. NO tactic decision in the DB
    final_ledger = _load("user-test", "proj-test", auto_migrate=False)
    tactic_decisions = [d for d in final_ledger["decisions"] if d["kind"] == "tactic"]
    assert len(tactic_decisions) == 0, \
        f"Tactic MUST NOT be persisted when parent style is rejected, got {tactic_decisions}"
