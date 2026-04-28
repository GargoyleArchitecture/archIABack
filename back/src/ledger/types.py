from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, TypedDict


class Phase(str, Enum):
    INTAKE   = "INTAKE"
    ASR      = "ASR"
    STYLE    = "STYLE"
    TACTICS  = "TACTICS"
    DIAGRAM  = "DIAGRAM"
    ANALYSIS = "ANALYSIS"
    DONE     = "DONE"


PHASE_ORDER: list[Phase] = [
    Phase.INTAKE, Phase.ASR, Phase.STYLE, Phase.TACTICS,
    Phase.DIAGRAM, Phase.ANALYSIS, Phase.DONE,
]

LEDGER_SCHEMA_VERSION = 1

DecisionKind = Literal["asr", "style", "tactic", "diagram", "analysis", "constraint"]
DecisionStatus = Literal["active", "superseded", "rejected", "orphaned"]
ParentStatus = Literal["ok", "parent_rejected", "parent_superseded"]


class LedgerValidationError(Exception):
    pass


class LedgerConcurrencyError(Exception):
    pass


class DecisionRef(TypedDict):
    id: str
    kind: str
    iteration: int


class Decision(TypedDict):
    id: str
    kind: str
    phase: str
    iteration: int
    qa: str
    parents: list[DecisionRef]
    payload: dict[str, Any]
    rationale: str
    sources: list[dict]
    status: str
    parent_status: str
    superseded_by: Optional[str]
    rejection_reason: Optional[str]
    created_at: str
    created_by_node: str


class PhaseTransition(TypedDict):
    from_phase: str
    to_phase: str
    iteration: int
    triggered_by: str
    user_message: str
    skipped_phases: list[str]
    timestamp: str


class DesignLedger(TypedDict):
    version: int
    project_id: str
    user_id: str
    current_phase: str
    current_iteration: int
    phase_history: list[PhaseTransition]
    pending_advance: Optional[PhaseTransition]
    decisions: list[Decision]
    project_context: dict[str, Any]
    user_style_hint: str


def empty_ledger(project_id: str, user_id: str) -> DesignLedger:
    return DesignLedger(
        version=LEDGER_SCHEMA_VERSION,
        project_id=project_id,
        user_id=user_id,
        current_phase=Phase.INTAKE.value,
        current_iteration=0,
        phase_history=[],
        pending_advance=None,
        decisions=[],
        project_context={},
        user_style_hint="",
    )
