# -*- coding: utf-8 -*-
"""Shared ledger-state refresh helper used by asr_node and asr_confirm_node."""

import logging

from src.ledger import (
    compute_active_view,
    load_ledger,
    render_dossier,
    render_dossier_compact,
    render_phase_prompt,
)

log = logging.getLogger("ledger_helpers")


def _refresh_ledger_state(
    state: dict,
    user_id: str,
    project_id: str | None,
    lang: str,
) -> None:
    """Refresh ledger-derived state fields in-place after a successful ledger write."""
    try:
        fresh  = load_ledger(user_id, project_id, auto_migrate=False)
        active = compute_active_view(fresh)
        state["ledger"]                 = fresh
        state["ledger_active"]          = active
        state["design_dossier_md"]      = render_dossier(fresh, lang=lang)
        state["ledger_dossier_compact"] = render_dossier_compact(fresh, lang=lang)
        state["ledger_phase_prompt"]    = render_phase_prompt(fresh, lang=lang)
        state["current_phase"]          = fresh.get("current_phase") or "intro"
        state["ledger_pending_advance"] = fresh.get("pending_advance") or {}
        log.debug("ledger state refreshed phase=%s", state["current_phase"])
    except Exception as exc:
        log.warning("ledger state refresh failed (nonfatal): %s", exc)
