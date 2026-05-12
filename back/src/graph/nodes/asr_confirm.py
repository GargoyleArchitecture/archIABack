# -*- coding: utf-8 -*-
"""asr_confirm_node — el usuario confirma el ASR vigente.

Pobla `selected_asrs` y commitea la transición de fase asr_table → style_table
en el ledger. Genera el endMessage final del turno y enruta directo a unifier.
"""

import logging
from datetime import datetime, timezone

from src.graph.state import GraphState
from src.graph.nodes._ledger_helpers import _refresh_ledger_state
from src.ledger import (
    compute_active_view,
    transition_phase,
    LedgerValidationError,
    LedgerConcurrencyError,
)
from src.ledger.types import Phase

log = logging.getLogger("asr_confirm_node")


def asr_confirm_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    user_id = (state.get("user_id_for_prefs") or "").strip()
    project_id = (state.get("project_id") or "").strip() or None
    uq = state.get("userQuestion", "") or ""

    asr_id = ""
    candidates = state.get("asr_candidates") or []
    if candidates:
        asr_id = (candidates[-1] or {}).get("id", "") or ""
    if not asr_id:
        active = compute_active_view(state.get("ledger") or {})
        asr_id = (active.get("asr") or {}).get("id", "") or ""

    if not asr_id:
        msg = (
            "No encuentro un ASR previo para confirmar. ¿Quieres generar uno?"
            if lang == "es"
            else "I can't find a prior ASR to confirm. Want to generate one?"
        )
        return {
            **state,
            "endMessage": msg,
            "nextNode": "unifier",
            "intent": "general",
        }

    state["selected_asrs"] = [asr_id]

    transitioned = False
    if user_id:
        try:
            ledger = state.get("ledger") or {}
            transition = {
                "from_phase":      Phase.ASR_TABLE.value,
                "to_phase":        Phase.STYLE_TABLE.value,
                "iteration":       int(ledger.get("current_iteration", 0)) + 1,
                "triggered_by":    "asr_confirmed_by_user",
                "user_message":    uq,
                "skipped_phases":  [],
                "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            transition_phase(user_id, project_id, transition)
            _refresh_ledger_state(state, user_id, project_id, lang)
            transitioned = True
            log.info("asr_confirm: phase advanced asr_table→style_table for user=%s", user_id)
        except (LedgerValidationError, LedgerConcurrencyError) as exc:
            log.warning("asr_confirm: phase transition failed (nonfatal): %s", exc)
        except Exception as exc:
            log.warning("asr_confirm: unexpected ledger error (nonfatal): %s", exc)

    state["routing_phase"] = "style"
    if not transitioned:
        state["current_phase"] = "style_table"

    if lang == "es":
        end_text = (
            "✅ **ASR confirmado.** Quedó como driver arquitectónico activo.\n\n"
            "El siguiente paso es seleccionar el **estilo arquitectónico** que mejor "
            "soporte este ASR."
        )
        suggestions = [
            "Propón estilos arquitectónicos para este ASR.",
            "Compara estilos para este ASR.",
        ]
    else:
        end_text = (
            "✅ **ASR confirmed.** It is now the active architectural driver.\n\n"
            "The next step is to select the **architecture style** that best "
            "supports this ASR."
        )
        suggestions = [
            "Propose architecture styles for this ASR.",
            "Compare styles for this ASR.",
        ]

    state["endMessage"] = end_text
    state["suggestions"] = suggestions
    state["intent"] = "asr_confirm"
    state["nextNode"] = "unifier"
    return state
