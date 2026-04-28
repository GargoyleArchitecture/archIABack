# -*- coding: utf-8 -*-
import logging
from langchain_core.runnables import RunnableConfig
from src.graph.state import GraphState
from src.services.context_service import (
    fetch_project_context,
    fetch_user_preferences,
    format_project_context_text,
    format_user_style_hint,
)
from src.ledger import (
    load_ledger,
    compute_active_view,
    render_dossier,
    render_dossier_compact,
    render_phase_prompt,
)

log = logging.getLogger("context_loader")


def _mirror_legacy(active: dict, updates: dict) -> None:
    """Copy active ledger decisions into legacy scalar fields.

    Only overwrites a scalar when the corresponding ledger decision exists
    and has a non-empty value. An empty ledger must not clear state already
    populated by worker nodes in earlier turns.
    """
    asr = active.get("asr")
    if asr:
        summary = (asr.get("payload") or {}).get("summary", "")
        if summary:
            updates["current_asr"] = summary
        qa = asr.get("qa", "")
        if qa:
            updates["quality_attribute"] = qa

    style_dec = active.get("style")
    if style_dec:
        chosen = (style_dec.get("payload") or {}).get("chosen", "")
        if chosen:
            updates["style"]          = chosen
            updates["selected_style"] = chosen
            updates["last_style"]     = chosen

    tactic = active.get("tactic")
    if tactic:
        items = (tactic.get("payload") or {}).get("items") or []
        if items:
            updates["tactics_struct"] = items
            updates["tactics_list"]   = [
                t.get("name", "") for t in items if isinstance(t, dict)
            ]


def context_loader_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Nodo de carga de contexto dual. Se ejecuta en cada turno.

    - ProjectContext / UserPreferences: cargados UNA VEZ por sesión
      (guardados por flags project_context_loaded / user_style_loaded).
    - Ledger: recargado en CADA turno — no tiene flag de sesión porque
      los nodos de decisión pueden escribir al ledger entre turnos.

    El api_token se lee desde config["configurable"]["api_token"] para
    evitar persistirlo en el checkpoint (MemorySaver).
    """
    project_id = (state.get("project_id") or "").strip()
    user_id    = (state.get("user_id_for_prefs") or "").strip()
    api_token  = ((config or {}).get("configurable") or {}).get("api_token", "")

    need_project = bool(project_id) and not state.get("project_context_loaded")
    need_prefs   = bool(user_id)    and not state.get("user_style_loaded")
    need_ledger  = bool(user_id)    # always reload; no session guard

    if not need_project and not need_prefs and not need_ledger:
        return state

    updates: dict = {}

    if need_project:
        try:
            raw  = fetch_project_context(project_id, api_token)
            text = format_project_context_text(raw)
            updates["project_context_text"]   = text
            updates["project_context_loaded"] = True
            log.info("context_loader: project context cargado project_id=%s", project_id)
        except Exception as exc:
            log.warning("context_loader: fallo al cargar project context: %s", exc)
            updates["project_context_text"]   = ""
            updates["project_context_loaded"] = True

    if need_prefs:
        try:
            raw  = fetch_user_preferences(user_id, api_token)
            hint = format_user_style_hint(raw)
            updates["user_style_hint"]   = hint
            updates["user_style_loaded"] = True
            log.info("context_loader: user preferences cargadas user_id=%s", user_id)
        except Exception as exc:
            log.warning("context_loader: fallo al cargar user preferences: %s", exc)
            updates["user_style_hint"]   = ""
            updates["user_style_loaded"] = True

    if need_ledger:
        try:
            ledger = load_ledger(user_id, project_id or None)
            lang   = (state.get("language") or "es") or "es"
            active = compute_active_view(ledger)

            updates["ledger"]                 = ledger
            updates["ledger_active"]          = active
            updates["design_dossier_md"]      = render_dossier(ledger, lang=lang)
            updates["current_phase"]          = ledger.get("current_phase", "INTAKE")
            updates["ledger_dossier_compact"] = render_dossier_compact(ledger, lang=lang)
            updates["ledger_phase_prompt"]    = render_phase_prompt(ledger, lang=lang)
            updates["ledger_pending_advance"] = ledger.get("pending_advance") or {}

            _mirror_legacy(active, updates)

            log.info(
                "context_loader: ledger hydrated user=%s project=%s phase=%s decisions=%d",
                user_id, project_id,
                ledger.get("current_phase"),
                len(ledger.get("decisions", [])),
            )
        except Exception as exc:
            log.warning("context_loader: ledger hydration failed: %s", exc)
            # Do NOT touch ledger fields on failure.
            # State retains whatever was checkpointed from the previous turn.

    return {**state, **updates}
