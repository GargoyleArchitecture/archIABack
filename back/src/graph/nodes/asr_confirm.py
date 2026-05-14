# -*- coding: utf-8 -*-
"""asr_confirm_node — el usuario confirma el ASR vigente.

Resuelve la selección del usuario (mapeando el ID humano A1/A2/… al ULID del
ledger), expande el candidato a sus 6 partes canónicas vía LLM, escribe el
ASR expandido como una nueva decisión activa (lo que supersede a A4/etc. en
el ledger), commitea la transición asr_table → style_table y enruta a unifier.

BUG-052 / BUG-053: antes este nodo seleccionaba `candidates[-1]` (siempre A4
en una tabla de 4 candidatos) y emitía solo "✅ ASR confirmado" sin expandir
el detalle 6-partes. Ahora resuelve por `state["selected_asrs"]`, llama a
`_expand_asr_to_six_part` y persiste el detalle estructurado en el ledger
para que los nodos downstream (style, tactics) usen el QA correcto.
"""

import logging
from datetime import datetime, timezone

from src.graph.state import GraphState
from src.graph.nodes._ledger_helpers import _refresh_ledger_state
from src.graph.nodes.asr import _expand_asr_to_six_part
from src.graph.qa_registry import normalize_qa
from src.ledger import (
    append_decision,
    compute_active_view,
    transition_phase,
    LedgerValidationError,
    LedgerConcurrencyError,
)
from src.ledger.types import Phase

log = logging.getLogger("asr_confirm_node")


def _pick_chosen_candidate(state: GraphState) -> dict | None:
    """Find the candidate the user picked.

    Maps the human-readable ID list `state["selected_asrs"]` (e.g. ["A1"]) to
    the matching row in `state["asr_candidates"]`. Falls back to the last
    candidate if nothing matches (preserves prior behaviour as a safety net).
    """
    candidates = state.get("asr_candidates") or []
    if not candidates:
        return None

    selected_raw = [str(x).strip().upper() for x in (state.get("selected_asrs") or [])]
    for sid in selected_raw:
        for c in candidates:
            cid = str(c.get("candidate_id") or "").strip().upper()
            if cid and cid == sid:
                return c
            # Also match by ULID — selected_asrs may already be normalised.
            if sid and sid == str(c.get("id") or "").strip().upper():
                return c
    return candidates[-1]


def asr_confirm_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    user_id = (state.get("user_id_for_prefs") or "").strip()
    project_id = (state.get("project_id") or "").strip() or None
    uq = state.get("userQuestion", "") or ""

    chosen = _pick_chosen_candidate(state)
    if chosen is None:
        # Fall back to the active ASR already in the ledger if no candidates
        # are in state (e.g. session restored from checkpoint without the
        # asr_candidates list).
        active = compute_active_view(state.get("ledger") or {})
        existing = active.get("asr")
        if not existing:
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
        # Synthesize a candidate shape from the active ledger entry so the
        # downstream expansion still runs.
        chosen = {
            "id":            existing.get("id", ""),
            "candidate_id":  (existing.get("payload") or {}).get("candidate_id", "A1"),
            "qa":            (existing.get("payload") or {}).get("qa", ""),
            "scenario":      (existing.get("payload") or {}).get("scenario", ""),
            "business":      (existing.get("payload") or {}).get("business", "M"),
            "risk":          (existing.get("payload") or {}).get("risk", "M"),
            "payload":       existing.get("payload") or {},
        }

    base_payload = dict(chosen.get("payload") or {})
    if "candidate_id" not in base_payload:
        base_payload["candidate_id"] = chosen.get("candidate_id", "")
    if "scenario" not in base_payload:
        base_payload["scenario"] = chosen.get("scenario", "")
    if "qa" not in base_payload:
        base_payload["qa"] = chosen.get("qa", "")
    if "business" not in base_payload:
        base_payload["business"] = chosen.get("business", "M")
    if "risk" not in base_payload:
        base_payload["risk"] = chosen.get("risk", "M")

    expanded_payload, six_part_md = _expand_asr_to_six_part(base_payload, lang)
    chosen_qa = normalize_qa(expanded_payload.get("qa", "")) or "general"

    new_ulid = ""
    sources = []
    try:
        # Pull sources from the original candidate decision if available, so
        # the expanded entry keeps its RAG provenance.
        for d in (state.get("ledger") or {}).get("decisions", []) or []:
            if d.get("id") == chosen.get("id") and d.get("kind") == "asr":
                sources = list(d.get("sources") or [])
                break
    except Exception:
        sources = []

    if user_id:
        new_decision = {
            "id":               "",
            "kind":             "asr",
            "phase":            Phase.ASR_TABLE.value,
            "iteration":        0,
            "qa":               chosen_qa,
            "parents":          [],
            "payload":          expanded_payload,
            "rationale":        "Expanded after user selection",
            "sources":          sources,
            "status":           "active",
            "parent_status":    "ok",
            "superseded_by":    None,
            "rejection_reason": None,
            "created_at":       "",
            "created_by_node":  "asr_confirm_node",
        }
        try:
            # append_decision auto-supersedes any prior active ASR with the
            # same parents — so the previous "A4" winner becomes superseded
            # and this expanded selection becomes the new ledger_active.asr.
            saved = append_decision(user_id, project_id, new_decision)
            new_ulid = saved.get("id", "") or ""
        except (LedgerValidationError, LedgerConcurrencyError) as exc:
            log.warning("asr_confirm: append_decision failed (nonfatal): %s", exc)
        except Exception as exc:
            log.warning("asr_confirm: unexpected ledger error (nonfatal): %s", exc)

    # Normalise selected_asrs to the ULID so style/tactics nodes can resolve
    # the active ASR unambiguously. Keep the human ID as a secondary entry so
    # the styles/common.py defense-in-depth check still matches.
    human_id = (expanded_payload.get("candidate_id") or chosen.get("candidate_id") or "").upper()
    state["selected_asrs"] = [new_ulid or chosen.get("id", "") or human_id]
    if human_id and human_id not in state["selected_asrs"]:
        state["selected_asrs"].append(human_id)

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
    state["quality_attribute"] = chosen_qa
    state["resolved_index"] = chosen_qa

    if lang == "es":
        suggestions = [
            "Propón estilos arquitectónicos para este ASR.",
            "Compara estilos para este ASR.",
        ]
    else:
        suggestions = [
            "Propose architecture styles for this ASR.",
            "Compare styles for this ASR.",
        ]

    state["endMessage"] = six_part_md
    state["suggestions"] = suggestions
    state["intent"] = "asr_confirm"
    state["nextNode"] = "unifier"
    return state
