# -*- coding: utf-8 -*-
"""style_confirm_node — el usuario confirma el estilo arquitectónico vigente.

Resuelve la selección del usuario (mapeando el ID humano S1/S2 al estilo de
`state["style_candidates"]`), persiste el estilo elegido como decisión activa
del ledger (auto-supersede de cualquier estilo previo), commitea la transición
style_table → tactics_table y enruta a unifier. Este nodo es la analogía a
`asr_confirm_node`.
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

log = logging.getLogger("style_confirm_node")


def _pick_chosen_style(state: GraphState) -> dict | None:
    """Find the style candidate the user picked.

    Resolves `state["selected_style"]` (e.g. "S1") against
    `state["style_candidates"]`. Falls back to the first candidate if nothing
    matches (defensive — should not happen because the classifier only sets
    the intent when `style_candidates` is populated).
    """
    candidates = state.get("style_candidates") or []
    if not candidates:
        return None

    raw_sid = str(state.get("selected_style") or "").strip().upper()
    if raw_sid:
        for c in candidates:
            cid = str(c.get("id") or "").strip().upper()
            if cid and cid == raw_sid:
                return c
            cname = str(c.get("name") or "").strip()
            if cname and cname.upper() == raw_sid:
                return c
    return candidates[0]


def _build_asr_parent_ref(ledger_active: dict) -> list:
    """Mirror of styles/common.py:_build_asr_parent_ref — returns a parent
    reference list for the new style decision so the ledger chain stays intact.
    """
    asr = (ledger_active or {}).get("asr")
    if not asr or not asr.get("id"):
        return []
    return [{"id": asr["id"], "kind": "asr"}]


def style_confirm_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    user_id = (state.get("user_id_for_prefs") or "").strip()
    project_id = (state.get("project_id") or "").strip() or None
    uq = state.get("userQuestion", "") or ""

    chosen = _pick_chosen_style(state)
    if chosen is None:
        # Fall back to the active style in the ledger if the candidates list
        # is gone (e.g. checkpoint restore without style_candidates).
        active = compute_active_view(state.get("ledger") or {})
        existing = active.get("style")
        if not existing:
            msg = (
                "No encuentro estilos candidatos para confirmar. Pide primero la "
                "propuesta de estilos."
                if lang == "es"
                else "No style candidates available to confirm. Please ask for the "
                "style proposal first."
            )
            return {
                **state,
                "endMessage": msg,
                "nextNode": "unifier",
                "intent": "general",
            }
        chosen = {
            "id":            (existing.get("payload") or {}).get("chosen", "S1"),
            "name":          (existing.get("payload") or {}).get("name", ""),
            "justification": "",
            "tradeoff":      (existing.get("payload") or {}).get("tradeoffs", ""),
        }

    chosen_id   = str(chosen.get("id") or "").strip().upper() or "S1"
    chosen_name = (chosen.get("name") or "").strip() or chosen_id
    justification = (chosen.get("justification") or "").strip()
    tradeoff      = (chosen.get("tradeoff") or "").strip()

    # Scalars update — keeps downstream nodes (tactics) in sync with the
    # confirmed choice without waiting for a re-render.
    state["style"]          = chosen_name
    state["selected_style"] = chosen_name
    state["last_style"]     = chosen_name

    qa = normalize_qa(state.get("quality_attribute", "")) or "general"
    if user_id:
        try:
            _payload = {
                "name":       chosen_name,
                "candidates": [
                    {"name": (c.get("name") or "").strip(), "impact": ""}
                    for c in (state.get("style_candidates") or [])
                    if (c.get("name") or "").strip()
                ],
                "chosen":     chosen_name,
                "tradeoffs":  tradeoff or justification,
            }
            _parents = _build_asr_parent_ref(state.get("ledger_active") or {})
            _new_decision = {
                "id":               "",
                "kind":             "style",
                "phase":            Phase.STYLE_TABLE.value,
                "iteration":        0,
                "qa":               qa,
                "parents":          _parents,
                "payload":          _payload,
                "rationale":        tradeoff or justification,
                "sources":          [],
                "status":           "active",
                "parent_status":    "ok",
                "superseded_by":    None,
                "rejection_reason": None,
                "created_at":       "",
                "created_by_node":  "style_confirm_node",
            }
            append_decision(user_id, project_id, _new_decision)
            log.info("style_confirm: ledger ok chosen=%s qa=%s", chosen_name, qa)
        except (LedgerValidationError, LedgerConcurrencyError) as exc:
            log.warning("style_confirm: append_decision failed (nonfatal): %s", exc)
        except Exception as exc:
            log.warning("style_confirm: unexpected ledger error (nonfatal): %s", exc)

    transitioned = False
    if user_id:
        try:
            ledger = state.get("ledger") or {}
            transition = {
                "from_phase":     Phase.STYLE_TABLE.value,
                "to_phase":       Phase.TACTICS_TABLE.value,
                "iteration":      int(ledger.get("current_iteration", 0)) + 1,
                "triggered_by":   "style_confirmed_by_user",
                "user_message":   uq,
                "skipped_phases": [],
                "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            transition_phase(user_id, project_id, transition)
            _refresh_ledger_state(state, user_id, project_id, lang)
            transitioned = True
            log.info("style_confirm: phase advanced style_table→tactics_table for user=%s", user_id)
        except (LedgerValidationError, LedgerConcurrencyError) as exc:
            log.warning("style_confirm: phase transition failed (nonfatal): %s", exc)
        except Exception as exc:
            log.warning("style_confirm: unexpected ledger error (nonfatal): %s", exc)

    state["routing_phase"] = "tactics"
    if not transitioned:
        state["current_phase"] = "tactics_table"

    if lang == "es":
        body = (
            f"✅ **Estilo confirmado:** {chosen_name} ({chosen_id}).\n\n"
            f"{('**Trade-off:** ' + tradeoff) if tradeoff else ''}\n\n"
            f"_Siguiente paso: seleccionar las **tácticas** que implementan este estilo "
            f"para el ASR activo._\n\n"
            f"¿Deseas que genere un **diagrama de arquitectura** con el estilo seleccionado, "
            f"o prefieres continuar directamente a las **tácticas**?"
        ).strip()
        suggestions = [
            "Genera el diagrama de arquitectura.",
            "Propón tácticas para este estilo.",
        ]
    else:
        body = (
            f"✅ **Style confirmed:** {chosen_name} ({chosen_id}).\n\n"
            f"{('**Trade-off:** ' + tradeoff) if tradeoff else ''}\n\n"
            f"_Next step: select the **tactics** that implement this style for the "
            f"active ASR._\n\n"
            f"Would you like me to generate an **architecture diagram** with the "
            f"selected style, or do you prefer to continue directly to **tactics**?"
        ).strip()
        suggestions = [
            "Generate the architecture diagram.",
            "Propose tactics for this style.",
        ]

    state["endMessage"]  = body
    state["suggestions"] = suggestions
    state["intent"]      = "style_confirm"
    state["nextNode"]    = "unifier"
    return state
