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

log = logging.getLogger("context_loader")


def context_loader_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Nodo de carga de contexto dual. Se ejecuta UNA VEZ por sesión.

    Obtiene del Backend API:
    - ProjectContext (techStack + businessRules) → project_context_text
    - UserPreference (explanationStyle + verbosity) → user_style_hint

    Guard de sesión: usa flags project_context_loaded / user_style_loaded (True incluso
    cuando el resultado fue vacío o falló) para evitar llamadas HTTP repetidas en cada turno.

    El api_token se lee desde config["configurable"]["api_token"] para evitar
    persistirlo en el checkpoint (MemorySaver).

    Degradación silenciosa: cualquier fallo en el fetch deja el campo como ""
    sin interrumpir el flujo del grafo.
    """
    project_id = (state.get("project_id") or "").strip()
    user_id    = (state.get("user_id_for_prefs") or "").strip()
    api_token  = ((config or {}).get("configurable") or {}).get("api_token", "")

    need_project = bool(project_id) and not state.get("project_context_loaded")
    need_prefs   = bool(user_id)    and not state.get("user_style_loaded")

    if not need_project and not need_prefs:
        return state

    updates = {}

    if need_project:
        try:
            raw = fetch_project_context(project_id, api_token)
            text = format_project_context_text(raw)
            updates["project_context_text"] = text
            updates["project_context_loaded"] = True
            if text:
                log.info("context_loader: project context cargado para project_id=%s", project_id)
            else:
                log.info("context_loader: project context vacío para project_id=%s", project_id)
        except Exception as exc:
            log.warning("context_loader: fallo al cargar project context: %s", exc)
            updates["project_context_text"] = ""
            updates["project_context_loaded"] = True

    if need_prefs:
        try:
            raw = fetch_user_preferences(user_id, api_token)
            hint = format_user_style_hint(raw)
            updates["user_style_hint"] = hint
            updates["user_style_loaded"] = True
            if hint:
                log.info("context_loader: user preferences cargadas para user_id=%s", user_id)
            else:
                log.info("context_loader: user preferences vacías para user_id=%s", user_id)
        except Exception as exc:
            log.warning("context_loader: fallo al cargar user preferences: %s", exc)
            updates["user_style_hint"] = ""
            updates["user_style_loaded"] = True

    return {**state, **updates}
