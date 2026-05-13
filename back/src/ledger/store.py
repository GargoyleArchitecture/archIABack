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
    """Mark any prior active same-kind same-parents decision as superseded. Returns its id.

    BUG-003 fix: after marking the old decision superseded, propagate
    parent_status="parent_superseded" to every active child decision that
    references the superseded id in its parents list.  Previously only the
    superseded node itself was mutated; children kept parent_status="ok" even
    though their parent was gone, causing compute_active_view to return stale
    style/tactic decisions linked to a superseded ASR.
    """
    kind = new_decision["kind"]
    new_parent_ids = frozenset(r["id"] for r in (new_decision.get("parents") or []))

    superseded_id: str | None = None
    for d in ledger["decisions"]:
        if d["kind"] == kind and d["status"] == "active":
            existing_parent_ids = frozenset(r["id"] for r in (d.get("parents") or []))
            if existing_parent_ids == new_parent_ids:
                d["status"] = "superseded"
                d["superseded_by"] = new_decision["id"]
                superseded_id = d["id"]
                break

    if superseded_id:
        for d in ledger["decisions"]:
            if d["status"] == "active":
                parent_ids = {r["id"] for r in (d.get("parents") or [])}
                if superseded_id in parent_ids:
                    d["parent_status"] = "parent_superseded"

    return superseded_id


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


def get_all_active_asrs(ledger: DesignLedger) -> list[dict]:
    """Return all active ASR decisions, ordered by creation (earliest first)."""
    return [
        d for d in ledger.get("decisions", [])
        if d.get("kind") == "asr" and d.get("status") == "active"
    ]


def is_phase_complete(ledger: DesignLedger, phase: Phase) -> bool:
    """Check whether the given phase has been completed in the ledger."""
    active = compute_active_view(ledger)

    if phase == Phase.INTRO:
        return True  # single-step phase; M6 will refine

    if phase == Phase.DIAGNOSIS:
        return active.get("constraint") is not None

    if phase == Phase.ASR_TABLE:
        return active.get("asr") is not None

    if phase == Phase.STYLE_TABLE:
        asr   = active.get("asr")
        style = active.get("style")
        if not asr or not style:
            return False
        style_asr_parents = {r["id"] for r in (style.get("parents") or []) if r["kind"] == "asr"}
        return asr["id"] in style_asr_parents

    if phase == Phase.TACTICS_TABLE:
        asr    = active.get("asr")
        style  = active.get("style")
        tactic = active.get("tactic")
        if not asr or not style or not tactic:
            return False
        tactic_parent_ids = {r["id"] for r in (tactic.get("parents") or [])}
        return asr["id"] in tactic_parent_ids and style["id"] in tactic_parent_ids

    if phase == Phase.TECH_PROPOSALS:
        return False  # completed by M5 (tech kind not introduced in M2)

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
