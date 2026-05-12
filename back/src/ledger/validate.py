from __future__ import annotations

from src.ledger.types import (
    Decision,
    DesignLedger,
    LedgerValidationError,
    PHASE_ORDER,
    Phase,
    PhaseTransition,
)

_VALID_KINDS = {"asr", "style", "tactic", "diagram", "analysis", "constraint"}

_REQUIRED_PAYLOAD_KEYS: dict[str, set[str]] = {
    "asr":        {"summary"},
    "style":      {"chosen"},
    "tactic":     {"items"},
    "diagram":    set(),
    "analysis":   {"target_id"},
    "constraint": set(),
}

# For each kind, which parent kinds are mandatory.
_REQUIRED_PARENT_KINDS: dict[str, list[str]] = {
    "asr":        [],
    "style":      ["asr"],
    "tactic":     ["asr", "style"],
    "diagram":    ["style"],
    "analysis":   [],
    "constraint": [],
}


def validate_decision(ledger: DesignLedger, decision: Decision) -> None:
    """Check shape, qa canonical form, created_by_node, and payload key presence."""
    if not decision.get("created_by_node"):
        raise LedgerValidationError("created_by_node must be non-empty")

    kind = decision.get("kind")
    if kind not in _VALID_KINDS:
        raise LedgerValidationError(f"Unknown kind: {kind!r}")

    # Validate qa using qa_registry if available; otherwise accept any non-empty string.
    try:
        from src.graph.qa_registry import normalize_qa
        qa = decision.get("qa", "")
        if qa and normalize_qa(qa) != qa:
            raise LedgerValidationError(
                f"qa {qa!r} is not a canonical QA id; use normalize_qa() first"
            )
    except ImportError:
        pass  # qa_registry not available in isolated tests

    payload = decision.get("payload") or {}
    required = _REQUIRED_PAYLOAD_KEYS.get(kind, set())
    missing = required - set(payload.keys())
    if missing:
        raise LedgerValidationError(
            f"payload for kind={kind!r} missing required keys: {missing}"
        )


def validate_parents(ledger: DesignLedger, decision: Decision) -> None:
    """Each parent ref must point to an existing ACTIVE decision of the right kind."""
    decision_index = {d["id"]: d for d in ledger["decisions"]}
    kind = decision["kind"]
    parents = decision.get("parents") or []

    for ref in parents:
        parent = decision_index.get(ref["id"])
        if parent is None:
            raise LedgerValidationError(
                f"Parent {ref['id']!r} not found in ledger"
            )
        if parent["status"] != "active":
            raise LedgerValidationError(
                f"Parent {ref['id']!r} (kind={parent['kind']!r}) "
                f"has status {parent['status']!r}; must be active"
            )
        if parent["kind"] != ref["kind"]:
            raise LedgerValidationError(
                f"Parent {ref['id']!r} has actual kind={parent['kind']!r} "
                f"but ref claims kind={ref['kind']!r}"
            )

    required_kinds = _REQUIRED_PARENT_KINDS.get(kind, [])
    parent_kinds = [ref["kind"] for ref in parents]
    for req in required_kinds:
        if req not in parent_kinds:
            raise LedgerValidationError(
                f"Decision of kind={kind!r} requires a parent of kind={req!r}; "
                f"got parent kinds: {parent_kinds}"
            )


def validate_qa_match(ledger: DesignLedger, decision: Decision) -> None:
    """Style must share QA with its parent ASR; tactic must share QA with parents."""
    decision_index = {d["id"]: d for d in ledger["decisions"]}
    kind = decision["kind"]
    parents = decision.get("parents") or []

    if kind not in ("style", "tactic"):
        return

    for ref in parents:
        parent = decision_index.get(ref["id"])
        if parent and parent["kind"] in ("asr", "style"):
            if parent.get("qa") != decision.get("qa"):
                raise LedgerValidationError(
                    f"Decision qa={decision.get('qa')!r} doesn't match "
                    f"parent {parent['kind']} (id={parent['id']!r}) qa={parent.get('qa')!r}"
                )


def validate_transition(ledger: DesignLedger, transition: PhaseTransition) -> None:
    """Validate that a phase transition is structurally consistent with the ledger."""
    from_phase = transition["from_phase"]
    to_phase   = transition["to_phase"]
    iteration  = transition["iteration"]

    if from_phase != ledger["current_phase"]:
        raise LedgerValidationError(
            f"transition.from_phase={from_phase!r} doesn't match "
            f"ledger.current_phase={ledger['current_phase']!r}"
        )

    valid_phases = {p.value for p in Phase}
    if to_phase not in valid_phases:
        raise LedgerValidationError(f"Invalid to_phase: {to_phase!r}")

    expected_iter = ledger["current_iteration"] + 1
    if iteration != expected_iter:
        raise LedgerValidationError(
            f"transition.iteration={iteration} must equal "
            f"current_iteration+1={expected_iter}"
        )

    # Populate skipped_phases for forward jumps (caller responsibility, but we verify).
    try:
        from_idx = PHASE_ORDER.index(Phase(from_phase))
        to_idx   = PHASE_ORDER.index(Phase(to_phase))
    except ValueError:
        raise LedgerValidationError(f"Unknown phase in transition: {from_phase!r} → {to_phase!r}")

    if to_idx > from_idx + 1:
        expected_skipped = [p.value for p in PHASE_ORDER[from_idx + 1:to_idx]]
        actual_skipped   = transition.get("skipped_phases") or []
        if sorted(actual_skipped) != sorted(expected_skipped):
            raise LedgerValidationError(
                f"skipped_phases={actual_skipped!r} doesn't match "
                f"expected skips={expected_skipped!r} for jump {from_phase}→{to_phase}"
            )


def validate_supersede_target(ledger: DesignLedger, decision: Decision) -> str | None:
    """Return the id of an existing active same-kind, same-parents decision, or None."""
    kind = decision["kind"]
    new_parent_ids = frozenset(r["id"] for r in (decision.get("parents") or []))

    for d in ledger["decisions"]:
        if d["kind"] == kind and d["status"] == "active":
            existing_parent_ids = frozenset(r["id"] for r in (d.get("parents") or []))
            if existing_parent_ids == new_parent_ids:
                return d["id"]
    return None
