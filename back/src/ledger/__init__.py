"""Design Ledger package — public API."""
from src.ledger.types import (
    LEDGER_SCHEMA_VERSION,
    PHASE_ORDER,
    Decision,
    DecisionKind,
    DecisionRef,
    DesignLedger,
    LedgerConcurrencyError,
    LedgerValidationError,
    Phase,
    PhaseTransition,
    empty_ledger,
)
from src.ledger.validate import (
    validate_decision,
    validate_parents,
    validate_qa_match,
    validate_supersede_target,
    validate_transition,
)
from src.ledger.store import (
    append_decision,
    clear_pending_advance,
    compute_active_view,
    is_phase_complete,
    load_ledger,
    migrate_legacy_arch_flow,
    reject_decision,
    save_ledger,
    stage_pending_advance,
    transition_phase,
)
from src.ledger.render import (
    render_dossier,
    render_dossier_compact,
    render_phase_prompt,
)

__all__ = [
    # types
    "Phase", "DecisionKind", "DesignLedger", "Decision", "DecisionRef",
    "PhaseTransition", "PHASE_ORDER", "LEDGER_SCHEMA_VERSION", "empty_ledger",
    "LedgerValidationError", "LedgerConcurrencyError",
    # validate
    "validate_decision", "validate_parents", "validate_qa_match",
    "validate_supersede_target", "validate_transition",
    # store
    "load_ledger", "save_ledger", "append_decision", "reject_decision",
    "transition_phase", "stage_pending_advance", "clear_pending_advance",
    "compute_active_view", "is_phase_complete", "migrate_legacy_arch_flow",
    # render
    "render_dossier", "render_dossier_compact", "render_phase_prompt",
]
