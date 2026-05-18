# -*- coding: utf-8 -*-
"""asr_detail_node — re-renders the 6-part detail of an already-confirmed ASR.

Fires when the user requests the detail of a previously confirmed ASR (e.g.
"muestrame el detalle de A1") from any phase. Does NOT advance the phase,
does NOT write to the ledger — it is a read-only presentation node.
"""

import logging
import re

from src.graph.state import GraphState
from src.graph.nodes.asr import _expand_asr_to_six_part
from src.ledger import get_all_active_asrs

log = logging.getLogger("asr_detail_node")

_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def asr_detail_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    detail_ids = [str(x).strip().upper() for x in (state.get("asr_detail_ids") or [])]

    if not detail_ids:
        msg = (
            "No especificaste qué ASR quieres ver. Escribe el ID (p.ej. 'A1')."
            if lang == "es"
            else "No ASR ID specified. Type the ID (e.g. 'A1')."
        )
        return {**state, "endMessage": msg, "nextNode": "unifier", "intent": "asr_detail"}

    # Find matching ASR decisions from ledger
    ledger = state.get("ledger") or {}
    active_asrs = get_all_active_asrs(ledger)

    # Build lookup: human_id → payload
    asr_by_id: dict[str, dict] = {}
    for d in active_asrs:
        payload = d.get("payload") or {}
        cid = str(payload.get("candidate_id") or "").strip().upper()
        if cid:
            asr_by_id[cid] = payload

    mds: list[str] = []
    not_found: list[str] = []

    for hid in detail_ids:
        payload = asr_by_id.get(hid)
        if payload is None:
            not_found.append(hid)
            continue
        try:
            _, md = _expand_asr_to_six_part(payload, lang)
            mds.append(md)
        except Exception as exc:
            log.warning("asr_detail: expansion failed for %s: %s", hid, exc)
            not_found.append(hid)

    if not mds:
        ids_str = ", ".join(detail_ids)
        msg = (
            f"No encontré los ASRs {ids_str} en el ledger activo."
            if lang == "es"
            else f"Could not find ASRs {ids_str} in the active ledger."
        )
        return {**state, "endMessage": msg, "nextNode": "unifier", "intent": "asr_detail"}

    end_msg = "\n\n---\n\n".join(mds)
    if not_found:
        nf_str = ", ".join(not_found)
        suffix = (
            f"\n\n_(No se encontraron: {nf_str})_"
            if lang == "es"
            else f"\n\n_(Not found: {nf_str})_"
        )
        end_msg += suffix

    return {
        **state,
        "endMessage": end_msg,
        "nextNode": "unifier",
        "intent": "asr_detail",
    }
