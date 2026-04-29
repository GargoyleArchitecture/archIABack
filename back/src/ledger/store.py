from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

import ulid

import src.memory as _mem
from src.ledger.types import (
    Decision,
    DesignLedger,
    LedgerConcurrencyError,
    LedgerValidationError,
    LEDGER_SCHEMA_VERSION,
    PHASE_ORDER,
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ledger_key(project_id: str | None) -> str:
    pid = (project_id or "").strip()
    if pid and not re.match(r"^[\w\-.:]+$", pid):
        raise ValueError(f"project_id inválido: {pid!r}")
    return f"ledger:{pid}" if pid else "ledger"


def _new_decision_id() -> str:
    # Support both the lightweight `ulid` module (which exposes `ulid()`)
    # and libraries that offer object-oriented constructors.
    if hasattr(ulid, "ulid"):
        return str(ulid.ulid())
    if hasattr(ulid, "new"):
        return str(ulid.new())
    if hasattr(ulid, "ULID"):
        return str(ulid.ULID())
    raise RuntimeError("No compatible ULID generator found")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _conn_rw() -> sqlite3.Connection:
    """Autocommit connection for explicit BEGIN IMMEDIATE control."""
    return sqlite3.connect(str(_mem.DB_PATH), isolation_level=None, check_same_thread=False)


def _apply_supersession(ledger: DesignLedger, new_decision: Decision) -> str | None:
    """Mark any prior active same-kind same-parents decision as superseded. Returns its id."""
    kind = new_decision["kind"]
    new_parent_ids = frozenset(r["id"] for r in (new_decision.get("parents") or []))

    for d in ledger["decisions"]:
        if d["kind"] == kind and d["status"] == "active":
            existing_parent_ids = frozenset(r["id"] for r in (d.get("parents") or []))
            if existing_parent_ids == new_parent_ids:
                d["status"] = "superseded"
                d["superseded_by"] = new_decision["id"]
                return d["id"]
    return None


# ---------------------------------------------------------------------------
# Public API — read
# ---------------------------------------------------------------------------

def load_ledger(
    user_id: str,
    project_id: str | None = None,
    *,
    auto_migrate: bool = True,
) -> DesignLedger:
    key = _ledger_key(project_id)
    with _mem._conn() as conn:
        row = conn.execute(
            "SELECT value FROM memory WHERE user_id=? AND key=?", (user_id, key)
        ).fetchone()

    if row:
        try:
            return json.loads(row[0])
        except Exception:
            pass  # corrupt blob — fall through to empty

    if auto_migrate:
        migrated = migrate_legacy_arch_flow(user_id, project_id)
        if migrated is not None:
            return migrated

    return empty_ledger(project_id or "", user_id)


# ---------------------------------------------------------------------------
# Public API — write
# ---------------------------------------------------------------------------

def save_ledger(
    user_id: str,
    ledger: DesignLedger,
    project_id: str | None = None,
    *,
    expected_version: int | None = None,
) -> DesignLedger:
    """Atomically write the ledger. Increments version. Raises LedgerConcurrencyError on mismatch."""
    key = _ledger_key(project_id)
    conn = _conn_rw()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value FROM memory WHERE user_id=? AND key=?", (user_id, key)
        ).fetchone()

        if expected_version is not None:
            stored_version = 0
            if row:
                try:
                    stored_version = json.loads(row[0]).get("version", 0)
                except Exception:
                    pass
            if stored_version != expected_version:
                conn.execute("ROLLBACK")
                raise LedgerConcurrencyError(
                    f"Concurrency conflict: expected version {expected_version}, "
                    f"found {stored_version}"
                )

        new_ledger: dict = dict(ledger)
        new_ledger["version"] = ledger.get("version", 0) + 1

        conn.execute(
            """INSERT INTO memory(user_id, key, value) VALUES(?,?,?)
               ON CONFLICT(user_id,key)
               DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
            (user_id, key, json.dumps(new_ledger, default=str)),
        )
        conn.execute("COMMIT")
        return new_ledger  # type: ignore[return-value]
    except LedgerConcurrencyError:
        raise
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def append_decision(
    user_id: str,
    project_id: str | None,
    decision: Decision,
) -> Decision:
    """Validate, assign id+iteration, supersede any prior same-kind decision, and persist."""
    for attempt in range(2):
        ledger = load_ledger(user_id, project_id, auto_migrate=False)

        validate_decision(ledger, decision)
        validate_parents(ledger, decision)
        validate_qa_match(ledger, decision)

        # Assign generated fields
        new_decision: dict = dict(decision)
        new_decision["id"]         = _new_decision_id()
        new_decision["iteration"]  = ledger["current_iteration"]
        new_decision["created_at"] = _now_iso()
        new_decision.setdefault("status",        "active")
        new_decision.setdefault("parent_status", "ok")
        new_decision.setdefault("superseded_by",   None)
        new_decision.setdefault("rejection_reason", None)
        new_decision.setdefault("sources",  [])
        new_decision.setdefault("rationale", "")

        _apply_supersession(ledger, new_decision)  # type: ignore[arg-type]
        ledger["decisions"].append(new_decision)   # type: ignore[arg-type]

        try:
            save_ledger(user_id, ledger, project_id, expected_version=ledger["version"])
            return new_decision  # type: ignore[return-value]
        except LedgerConcurrencyError:
            if attempt == 1:
                raise
            continue

    raise LedgerConcurrencyError("append_decision: exceeded retry limit")


def reject_decision(
    user_id: str,
    project_id: str | None,
    decision_id: str,
    reason: str,
) -> Decision:
    """Mark decision as rejected and flag dependent active children as orphaned."""
    for attempt in range(2):
        ledger = load_ledger(user_id, project_id, auto_migrate=False)
        decision_index = {d["id"]: d for d in ledger["decisions"]}

        target = decision_index.get(decision_id)
        if target is None:
            raise LedgerValidationError(f"Decision {decision_id!r} not found")

        target["status"]           = "rejected"
        target["rejection_reason"] = reason

        for d in ledger["decisions"]:
            if d["status"] == "active":
                parent_ids = {r["id"] for r in (d.get("parents") or [])}
                if decision_id in parent_ids:
                    d["parent_status"] = "parent_rejected"

        try:
            save_ledger(user_id, ledger, project_id, expected_version=ledger["version"])
            return target  # type: ignore[return-value]
        except LedgerConcurrencyError:
            if attempt == 1:
                raise
            continue

    raise LedgerConcurrencyError("reject_decision: exceeded retry limit")


def transition_phase(
    user_id: str,
    project_id: str | None,
    transition: PhaseTransition,
) -> DesignLedger:
    """Commit a phase transition: update current_phase, current_iteration, phase_history."""
    for attempt in range(2):
        ledger = load_ledger(user_id, project_id, auto_migrate=False)
        validate_transition(ledger, transition)

        ledger["phase_history"].append(transition)
        ledger["current_phase"]     = transition["to_phase"]
        ledger["current_iteration"] = transition["iteration"]

        # Clear pending_advance if it matches this transition
        pending = ledger.get("pending_advance")
        if pending and pending.get("to_phase") == transition["to_phase"]:
            ledger["pending_advance"] = None

        try:
            return save_ledger(user_id, ledger, project_id, expected_version=ledger["version"])
        except LedgerConcurrencyError:
            if attempt == 1:
                raise
            continue

    raise LedgerConcurrencyError("transition_phase: exceeded retry limit")


def stage_pending_advance(
    user_id: str,
    project_id: str | None,
    transition: PhaseTransition,
) -> DesignLedger:
    """Set pending_advance without committing the transition."""
    for attempt in range(2):
        ledger = load_ledger(user_id, project_id, auto_migrate=False)
        ledger["pending_advance"] = transition
        try:
            return save_ledger(user_id, ledger, project_id, expected_version=ledger["version"])
        except LedgerConcurrencyError:
            if attempt == 1:
                raise
            continue

    raise LedgerConcurrencyError("stage_pending_advance: exceeded retry limit")


def clear_pending_advance(
    user_id: str,
    project_id: str | None,
) -> DesignLedger:
    for attempt in range(2):
        ledger = load_ledger(user_id, project_id, auto_migrate=False)
        ledger["pending_advance"] = None
        try:
            return save_ledger(user_id, ledger, project_id, expected_version=ledger["version"])
        except LedgerConcurrencyError:
            if attempt == 1:
                raise
            continue

    raise LedgerConcurrencyError("clear_pending_advance: exceeded retry limit")


# ---------------------------------------------------------------------------
# Views and helpers
# ---------------------------------------------------------------------------

def compute_active_view(ledger: DesignLedger) -> dict[str, Any]:
    """Return the latest active decision per kind."""
    active: dict[str, Any] = {}
    for d in ledger["decisions"]:
        if d["status"] == "active":
            active[d["kind"]] = d  # last in append-only list wins
    return active


def is_phase_complete(ledger: DesignLedger, phase: Phase) -> bool:
    """Check whether the given phase has been completed in the ledger."""
    active = compute_active_view(ledger)

    if phase == Phase.INTAKE:
        return active.get("constraint") is not None

    if phase == Phase.ASR:
        return active.get("asr") is not None

    if phase == Phase.STYLE:
        asr   = active.get("asr")
        style = active.get("style")
        if not asr or not style:
            return False
        style_asr_parents = {r["id"] for r in (style.get("parents") or []) if r["kind"] == "asr"}
        return asr["id"] in style_asr_parents

    if phase == Phase.TACTICS:
        asr    = active.get("asr")
        style  = active.get("style")
        tactic = active.get("tactic")
        if not asr or not style or not tactic:
            return False
        tactic_parent_ids = {r["id"] for r in (tactic.get("parents") or [])}
        return asr["id"] in tactic_parent_ids and style["id"] in tactic_parent_ids

    if phase == Phase.DIAGRAM:
        return active.get("diagram") is not None

    if phase == Phase.ANALYSIS:
        diagram = active.get("diagram")
        if not diagram:
            return False
        return any(
            d["kind"] == "analysis"
            and d["status"] == "active"
            and d.get("payload", {}).get("target_id") == diagram["id"]
            for d in ledger["decisions"]
        )

    return False


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _parse_legacy_asr_text(text: str) -> str:
    """
    Best-effort extraction of ASR text from a result blob.
    Duplicated from main.py:_extract_asr_from_result_text to avoid import cycle.
    """
    import re as _re
    if not text:
        return ""
    m = _re.search(r"```asr\s*([\s\S]*?)```", text, _re.I)
    if m:
        return m.group(1).strip()
    if _re.search(
        r"\b(Summary|Context|Design\s+tactics|Trade[-\s]?offs|"
        r"Acceptance\s+criteria|Validation\s+plan)\b",
        text,
        _re.I,
    ):
        return text.strip()
    head_re = _re.compile(
        r"\b(ASR|Architecture[-\s]?Significant[-\s]?Requirement|"
        r"Requisit[oa]\s+Significativ[oa]\s+de\s+Arquitectura)\b[:：]?",
        _re.I,
    )
    m = head_re.search(text)
    if m:
        asr = text[m.start():]
        asr = _re.split(
            r"\n\s*#{1,6}\s|\n\s*(?:Rationale|Razonamiento|Conclusiones)\s*[:：]",
            asr,
            maxsplit=1,
        )[0]
        return asr.strip()
    m = _re.search(r"(?:^|\n)\s*[-*]\s*ASR\s*[:：]\s*(.+)", text, _re.I)
    if m:
        return m.group(1).strip()
    return ""


_STAGE_TO_PHASE: dict[str, str] = {
    "":           Phase.INTAKE.value,
    "ASR":        Phase.ASR.value,
    "STYLE":      Phase.STYLE.value,
    "TACTICS":    Phase.TACTICS.value,
    "DEPLOYMENT": Phase.DIAGRAM.value,
}


def migrate_legacy_arch_flow(
    user_id: str,
    project_id: str | None,
) -> DesignLedger | None:
    """
    One-shot, idempotent migration from arch_flow to a ledger.

    Returns:
      - Existing ledger if already migrated.
      - Newly synthesised ledger if arch_flow existed.
      - None if nothing to migrate.
    """
    ledger_key   = _ledger_key(project_id)
    arch_key     = _mem._arch_flow_key(project_id)

    # If ledger already exists, return it (idempotent).
    with _mem._conn() as conn:
        ledger_row = conn.execute(
            "SELECT value FROM memory WHERE user_id=? AND key=?", (user_id, ledger_key)
        ).fetchone()
        if ledger_row:
            try:
                return json.loads(ledger_row[0])
            except Exception:
                pass

        arch_row = conn.execute(
            "SELECT value, updated_at FROM memory WHERE user_id=? AND key=?",
            (user_id, arch_key),
        ).fetchone()

    if not arch_row:
        return None

    try:
        arch_flow: dict = json.loads(arch_row[0])
    except Exception:
        arch_flow = {}

    if not arch_flow:
        return None

    arch_updated_at = arch_row[1] or _now_iso()
    # Normalise to ISO 8601 Z format
    if arch_updated_at and "T" not in arch_updated_at:
        arch_updated_at = arch_updated_at.replace(" ", "T") + "Z"

    stage = (arch_flow.get("stage") or "").strip().upper()
    current_phase = _STAGE_TO_PHASE.get(stage, Phase.INTAKE.value)

    ledger = empty_ledger(project_id or "", user_id)
    ledger["current_phase"] = current_phase
    ledger["user_style_hint"] = arch_flow.get("user_style_hint", "")

    # Synthesise a single phase transition so current_iteration=1 is consistent.
    if current_phase != Phase.INTAKE.value:
        ledger["phase_history"] = [
            PhaseTransition(
                from_phase=Phase.INTAKE.value,
                to_phase=current_phase,
                iteration=1,
                triggered_by="user_request",
                user_message="(migrated from legacy arch_flow)",
                skipped_phases=[],
                timestamp=arch_updated_at,
            )
        ]
        ledger["current_iteration"] = 1

    decisions: list[Decision] = []
    now_ts = arch_updated_at

    # Constraint decision from add_context / project_context_text
    add_ctx = (arch_flow.get("add_context") or arch_flow.get("project_context_text") or "").strip()
    if add_ctx:
        decisions.append(
            _make_decision(
                kind="constraint",
                phase=Phase.INTAKE.value,
                iteration=1,
                qa="general",
                parents=[],
                payload={"tech_stack": [], "business_rules": [add_ctx]},
                rationale="Migrado desde arch_flow.add_context",
                created_at=now_ts,
                created_by_node="migrate_legacy_arch_flow",
            )
        )

    # ASR decision
    asr_text = arch_flow.get("current_asr", "").strip()
    asr_decision: Decision | None = None
    if asr_text:
        qa = arch_flow.get("quality_attribute", "general") or "general"
        asr_decision = _make_decision(
            kind="asr",
            phase=Phase.ASR.value,
            iteration=1,
            qa=qa,
            parents=[],
            payload={"summary": _parse_legacy_asr_text(asr_text) or asr_text, "source": "", "stimulus": "", "environment": "", "artifact": "", "response": "", "response_measure": "", "domain": ""},
            rationale="Migrado desde arch_flow.current_asr",
            created_at=now_ts,
            created_by_node="migrate_legacy_arch_flow",
        )
        decisions.append(asr_decision)

    # Style decision
    style_text = arch_flow.get("style", "").strip()
    style_decision: Decision | None = None
    if style_text and asr_decision:
        style_decision = _make_decision(
            kind="style",
            phase=Phase.STYLE.value,
            iteration=1,
            qa=asr_decision["qa"],
            parents=[{"id": asr_decision["id"], "kind": "asr", "iteration": 1}],
            payload={"chosen": style_text, "candidates": [], "tradeoffs": ""},
            rationale="Migrado desde arch_flow.style",
            created_at=now_ts,
            created_by_node="migrate_legacy_arch_flow",
        )
        decisions.append(style_decision)

    # Tactic decision
    tactics_raw = arch_flow.get("tactics") or []
    if tactics_raw and asr_decision and style_decision:
        if isinstance(tactics_raw, list):
            items = tactics_raw
        else:
            items = [{"name": str(tactics_raw), "purpose": "", "rationale": "", "risks": "", "tradeoffs": "", "traces_to_asr": "", "expected_effect": "", "success_probability": "", "rank": 1}]
        tactic_parents = [
            {"id": asr_decision["id"],   "kind": "asr",   "iteration": 1},
            {"id": style_decision["id"], "kind": "style", "iteration": 1},
        ]
        decisions.append(
            _make_decision(
                kind="tactic",
                phase=Phase.TACTICS.value,
                iteration=1,
                qa=asr_decision["qa"],
                parents=tactic_parents,
                payload={"items": items},
                rationale="Migrado desde arch_flow.tactics",
                created_at=now_ts,
                created_by_node="migrate_legacy_arch_flow",
            )
        )

    # Diagram decision
    last_diagram = arch_flow.get("last_diagram") or {}
    dot = last_diagram.get("dot", "") or arch_flow.get("deployment_diagram_puml", "")
    if dot and style_decision:
        decisions.append(
            _make_decision(
                kind="diagram",
                phase=Phase.DIAGRAM.value,
                iteration=1,
                qa="general",
                parents=[{"id": style_decision["id"], "kind": "style", "iteration": 1}],
                payload={
                    "level": last_diagram.get("level", 1),
                    "dot":   dot,
                    "dot_drawio": last_diagram.get("dot_drawio", ""),
                    "svg_b64": "",
                    "focus": "",
                    "mapping": {},
                },
                rationale="Migrado desde arch_flow.last_diagram",
                created_at=now_ts,
                created_by_node="migrate_legacy_arch_flow",
            )
        )

    ledger["decisions"] = decisions

    return save_ledger(user_id, ledger, project_id)


def _make_decision(
    *,
    kind: str,
    phase: str,
    iteration: int,
    qa: str,
    parents: list,
    payload: dict,
    rationale: str,
    created_at: str,
    created_by_node: str,
) -> Decision:
    return Decision(
        id=_new_decision_id(),
        kind=kind,
        phase=phase,
        iteration=iteration,
        qa=qa,
        parents=parents,
        payload=payload,
        rationale=rationale,
        sources=[],
        status="active",
        parent_status="ok",
        superseded_by=None,
        rejection_reason=None,
        created_at=created_at,
        created_by_node=created_by_node,
    )
