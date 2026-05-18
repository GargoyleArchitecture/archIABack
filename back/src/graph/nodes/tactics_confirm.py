# -*- coding: utf-8 -*-
"""tactics_confirm_node — el usuario confirma las tácticas arquitectónicas.

Resuelve la selección del usuario (IDs T1/T2/… o "acepto las tácticas") contra
`state["tactics_candidates"]`, persiste los IDs confirmados en `selected_tactics`,
escribe la decisión en el ledger, avanza la fase tactics_table → tech_proposals
y enruta a unifier.

BUG-012/007/013: antes `T1` caía en "Bienvenido de vuelta" porque el classifier
no tenía un bloque para `tactics_confirm` ni el supervisor tenía el branch.
"""

import logging
from datetime import datetime, timezone

from src.graph.state import GraphState
from src.graph.nodes._ledger_helpers import _refresh_ledger_state
from src.graph.qa_registry import normalize_qa
from src.ledger import (
    append_decision,
    compute_active_view,
    transition_phase,
    LedgerValidationError,
    LedgerConcurrencyError,
)
from src.ledger.types import Phase

log = logging.getLogger("tactics_confirm_node")


def _pick_chosen_tactics(state: GraphState) -> list[dict]:
    """Resolve user-selected tactic IDs to their full candidate dicts.

    If `state["selected_tactics"]` is populated (e.g. ["T1", "T2"]) match
    against `tactics_candidates`. If no IDs given, return ALL candidates
    (treating "acepto las tácticas" as blanket acceptance).
    """
    candidates: list[dict] = list(state.get("tactics_candidates") or [])
    if not candidates:
        return []

    selected_raw = [str(x).strip().upper() for x in (state.get("selected_tactics") or [])]
    if not selected_raw:
        # Blanket acceptance: confirm all proposed tactics.
        return candidates

    matched = []
    for sid in selected_raw:
        for c in candidates:
            cid = str(c.get("id") or "").strip().upper()
            if cid and cid == sid:
                matched.append(c)
                break
        else:
            # ID not found in candidates — include a stub so the user's
            # selection is preserved even if the candidate list was trimmed.
            matched.append({"id": sid, "name": sid, "rationale": ""})

    return matched or candidates


def _build_style_parent_ref(ledger_active: dict) -> list:
    """Return parent refs for the new tactics decision (asr + style chain)."""
    refs = []
    for kind in ("asr", "style"):
        entry = (ledger_active or {}).get(kind)
        if entry and entry.get("id"):
            refs.append({"id": entry["id"], "kind": kind, "iteration": entry.get("iteration", 0)})
    return refs


def tactics_confirm_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    user_id = (state.get("user_id_for_prefs") or "").strip()
    project_id = (state.get("project_id") or "").strip() or None
    uq = state.get("userQuestion", "") or ""

    chosen_tactics = _pick_chosen_tactics(state)
    if not chosen_tactics:
        active = compute_active_view(state.get("ledger") or {})
        existing_tactic = active.get("tactic")
        if not existing_tactic:
            msg = (
                "No encuentro tácticas candidatas para confirmar. Pide primero la "
                "propuesta de tácticas."
                if lang == "es"
                else "No tactic candidates available to confirm. Please ask for the "
                "tactic proposal first."
            )
            return {
                **state,
                "endMessage": msg,
                "nextNode": "unifier",
                "intent": "general",
            }
        # Fallback: use the single active tactic from ledger.
        chosen_tactics = [{
            "id":        (existing_tactic.get("payload") or {}).get("id", "T1"),
            "name":      (existing_tactic.get("payload") or {}).get("name", ""),
            "rationale": (existing_tactic.get("payload") or {}).get("rationale", ""),
        }]

    confirmed_ids = [str(t.get("id") or "").upper() for t in chosen_tactics]

    qa = normalize_qa(state.get("quality_attribute", "")) or "general"

    if user_id:
        try:
            _parents = _build_style_parent_ref(state.get("ledger_active") or {})
            _payload = {
                "items": [
                    {
                        "id":       str(t.get("id") or ""),
                        "name":     (t.get("name") or "").strip(),
                        "rationale": (t.get("rationale") or "")[:200],
                    }
                    for t in chosen_tactics
                ],
                "count": len(chosen_tactics),
            }
            _new_decision = {
                "id":               "",
                "kind":             "tactic",
                "phase":            Phase.TACTICS_TABLE.value,
                "iteration":        0,
                "qa":               qa,
                "parents":          _parents,
                "payload":          _payload,
                "rationale":        f"Confirmed {len(chosen_tactics)} tactic(s) by user",
                "sources":          [],
                "status":           "active",
                "parent_status":    "ok",
                "superseded_by":    None,
                "rejection_reason": None,
                "created_at":       "",
                "created_by_node":  "tactics_confirm_node",
            }
            append_decision(user_id, project_id, _new_decision)
            log.info(
                "tactics_confirm: ledger ok ids=%s qa=%s project=%s",
                confirmed_ids, qa, project_id,
            )
        except (LedgerValidationError, LedgerConcurrencyError) as exc:
            log.warning("tactics_confirm: append_decision failed (nonfatal): %s", exc)
        except Exception as exc:
            log.warning("tactics_confirm: unexpected ledger error (nonfatal): %s", exc)

    transitioned = False
    if user_id:
        try:
            ledger = state.get("ledger") or {}
            transition = {
                "from_phase":     Phase.TACTICS_TABLE.value,
                "to_phase":       Phase.TECH_PROPOSALS.value,
                "iteration":      int(ledger.get("current_iteration", 0)) + 1,
                "triggered_by":   "tactics_confirmed_by_user",
                "user_message":   uq,
                "skipped_phases": [],
                "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            transition_phase(user_id, project_id, transition)
            _refresh_ledger_state(state, user_id, project_id, lang)
            transitioned = True
            log.info(
                "tactics_confirm: phase advanced tactics_table→tech_proposals user=%s",
                user_id,
            )
        except (LedgerValidationError, LedgerConcurrencyError) as exc:
            log.warning("tactics_confirm: phase transition failed (nonfatal): %s", exc)
        except Exception as exc:
            log.warning("tactics_confirm: unexpected ledger error (nonfatal): %s", exc)

    # Set confirmed IDs after ledger refresh so refresh can't overwrite them.
    state["selected_tactics"] = confirmed_ids
    # Clear stale markdown so the unifier doesn't echo the previous tactics table.
    state["tactics_md"] = ""
    # BUG-015: clear candidates so the tech node uses only selected_tactics.
    state["tactics_candidates"] = []

    state["routing_phase"] = "tech"
    if not transitioned:
        state["current_phase"] = "tech_proposals"

    names = ", ".join(
        (t.get("name") or t.get("id") or "?") for t in chosen_tactics
    )
    ids_str = ", ".join(confirmed_ids)

    if lang == "es":
        body = (
            f"✅ **Tácticas confirmadas:** {names} ({ids_str}).\n\n"
            f"_Siguiente paso: seleccionar las **tecnologías** que implementan estas tácticas._"
        )
        suggestions = [
            "¿Qué tecnologías recomiendas para implementar estas tácticas?",
            "Propón tecnologías concretas para las tácticas confirmadas.",
        ]
    else:
        body = (
            f"✅ **Tactics confirmed:** {names} ({ids_str}).\n\n"
            f"_Next step: select the **technologies** that implement these tactics._"
        )
        suggestions = [
            "What technologies do you recommend for these tactics?",
            "Propose concrete technologies for the confirmed tactics.",
        ]

    state["endMessage"]  = body
    state["suggestions"] = suggestions
    state["intent"]      = "tactics_confirm"
    state["nextNode"]    = "unifier"
    return state
