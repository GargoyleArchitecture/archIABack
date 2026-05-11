from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, TypedDict


class Phase(str, Enum):
    INTRO          = "intro"
    DIAGNOSIS      = "diagnosis"
    ASR_TABLE      = "asr_table"
    STYLE_TABLE    = "style_table"
    TACTICS_TABLE  = "tactics_table"
    TECH_PROPOSALS = "tech_proposals"
    DIAGRAM        = "diagram"
    ANALYSIS       = "analysis"
    DONE           = "done"


PHASE_ORDER: list[Phase] = [
    Phase.INTRO, Phase.DIAGNOSIS, Phase.ASR_TABLE, Phase.STYLE_TABLE,
    Phase.TACTICS_TABLE, Phase.TECH_PROPOSALS, Phase.DIAGRAM,
    Phase.ANALYSIS, Phase.DONE,
]

PhaseLiteral = Literal[
    "intro", "diagnosis", "asr_table", "style_table",
    "tactics_table", "tech_proposals", "diagram", "analysis", "done",
]

LEDGER_SCHEMA_VERSION = 1

DecisionKind = Literal["asr", "style", "tactic", "tech", "diagram", "analysis", "constraint"]
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
    phase: PhaseLiteral
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
    from_phase: PhaseLiteral
    to_phase: PhaseLiteral
    iteration: int
    triggered_by: str
    user_message: str
    skipped_phases: list[str]
    timestamp: str


class DesignLedger(TypedDict):
    version: int
    project_id: str
    user_id: str
    current_phase: PhaseLiteral
    current_iteration: int
    phase_history: list[PhaseTransition]
    pending_advance: Optional[PhaseTransition]
    decisions: list[Decision]
    project_context: dict[str, Any]
    user_style_hint: str


def empty_ledger(project_id: str, user_id: str) -> DesignLedger:
    return DesignLedger(
        version=0,  # 0 = never persisted; save_ledger increments to 1 on first write
        project_id=project_id,
        user_id=user_id,
        current_phase=Phase.INTRO.value,
        current_iteration=0,
        phase_history=[],
        pending_advance=None,
        decisions=[],
        project_context={},
        user_style_hint="",
    )
