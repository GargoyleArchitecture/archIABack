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

BUG-056 (multi-selección): cuando el usuario confirma varios IDs ("confirmo
A1, A2, A3"), el nodo ahora expande y escribe al ledger TODOS los candidatos
seleccionados. El primario (A1) determina quality_attribute y endMessage;
los secundarios se añaden al ledger y quedan en state["selected_asrs"] para
que tech_node tenga contexto de todos los ASRs activos.
"""

import logging
import re
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

    # ── Step 1: Read all human IDs before any mutation ───────────────────────
    candidates = state.get("asr_candidates") or []
    raw_selected = [str(x).strip().upper() for x in (state.get("selected_asrs") or [])]
    human_ids_requested = [s for s in raw_selected if re.match(r"^A\d+$", s)]

    # ── Step 2: O(1) lookup map by candidate_id ──────────────────────────────
    candidate_map: dict[str, dict] = {
        str(c.get("candidate_id") or "").strip().upper(): c
        for c in candidates
        if c.get("candidate_id")
    }

    # ── Step 3: Resolve all requested IDs; fallback when nothing matches ─────
    resolved_candidates: list[dict] = []
    for hid in human_ids_requested:
        if hid in candidate_map:
            resolved_candidates.append(candidate_map[hid])
        else:
            log.warning("asr_confirm: no candidate found for %s — skipped", hid)

    if not resolved_candidates:
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
        resolved_candidates = [chosen]

    # ── Step 4: Pull RAG sources for the primary candidate ───────────────────
    primary_sources: list = []
    try:
        for d in (state.get("ledger") or {}).get("decisions", []) or []:
            if d.get("id") == resolved_candidates[0].get("id") and d.get("kind") == "asr":
                primary_sources = list(d.get("sources") or [])
                break
    except Exception:
        primary_sources = []

    # ── Step 5: Expand + ledger-write every resolved candidate ───────────────
    confirmed_pairs: list[tuple[str, str]] = []  # (ulid_or_fallback, human_id)
    primary_expanded_payload: dict = {}
    chosen_qa: str = "general"
    all_exp_mds: list[str] = []    # Issue 2b: full detail for every confirmed ASR
    all_exp_qas: list[str] = []    # Issue 2-bis: QA queue initialization

    for idx, cand in enumerate(resolved_candidates):
        base_payload = dict(cand.get("payload") or {})
        for field, key in [
            ("candidate_id", "candidate_id"),
            ("scenario",     "scenario"),
            ("qa",           "qa"),
            ("business",     "business"),
            ("risk",         "risk"),
        ]:
            if field not in base_payload:
                base_payload[field] = cand.get(key, "M" if key in ("business", "risk") else "")

        exp_payload, exp_md = _expand_asr_to_six_part(base_payload, lang)
        exp_qa = normalize_qa(exp_payload.get("qa", "")) or "general"
        exp_human_id = (exp_payload.get("candidate_id") or
                        cand.get("candidate_id") or "").upper()

        if idx == 0:
            primary_expanded_payload = exp_payload
            chosen_qa = exp_qa
        all_exp_mds.append(exp_md)   # Issue 2b: keep full detail for all confirmed ASRs
        all_exp_qas.append(exp_qa)   # Issue 2-bis: collect QAs for per-QA queue

        new_ulid = ""
        if user_id:
            new_decision = {
                "id":               "",
                "kind":             "asr",
                "phase":            Phase.ASR_TABLE.value,
                "iteration":        0,
                "qa":               exp_qa,
                "parents":          [],
                "payload":          exp_payload,
                "rationale":        "Expanded after user selection",
                "sources":          primary_sources if idx == 0 else [],
                "status":           "active",
                "parent_status":    "ok",
                "superseded_by":    None,
                "rejection_reason": None,
                "created_at":       "",
                "created_by_node":  "asr_confirm_node",
            }
            try:
                saved = append_decision(user_id, project_id, new_decision)
                new_ulid = saved.get("id", "") or ""
            except (LedgerValidationError, LedgerConcurrencyError) as exc:
                log.warning("asr_confirm: append_decision failed for %s (nonfatal): %s",
                            exp_human_id, exc)
            except Exception as exc:
                log.warning("asr_confirm: unexpected ledger error for %s (nonfatal): %s",
                            exp_human_id, exc)

        confirmed_pairs.append((
            new_ulid or cand.get("id", "") or exp_human_id,
            exp_human_id,
        ))

    # ── Step 6: Build state["selected_asrs"] preserving all confirmed IDs ────
    # Structure: [ULID_A1, "A1", ULID_A2, "A2", ...] so both styles/common.py
    # (needs an A\d+ entry for the display label) and tech/common.py (iterates
    # all IDs for context) work correctly. Single-confirm path produces
    # [ULID, "A1"] — identical to the pre-fix behaviour.
    new_selected_asrs: list[str] = []
    for ulid_or_fallback, human_id in confirmed_pairs:
        if ulid_or_fallback and ulid_or_fallback not in new_selected_asrs:
            new_selected_asrs.append(ulid_or_fallback)
        if human_id and human_id not in new_selected_asrs:
            new_selected_asrs.append(human_id)
    state["selected_asrs"] = new_selected_asrs

    # Issue 2-bis: unique ordered QA queue for per-QA design loop iteration
    seen_qas: set[str] = set()
    qa_queue: list[str] = []
    for q in all_exp_qas:
        if q and q not in seen_qas:
            seen_qas.add(q)
            qa_queue.append(q)
    state["selected_qa_queue"] = qa_queue

    # ── Step 7: Build endMessage — full 6-part detail for every confirmed ASR ─
    end_msg = "\n\n---\n\n".join(all_exp_mds)

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

    # BUG-005: clear candidates so the table is not re-echoed in the style phase.
    state["asr_candidates"] = []

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

    state["endMessage"] = end_msg
    state["suggestions"] = suggestions
    state["intent"] = "asr_confirm"
    state["nextNode"] = "unifier"
    return state
