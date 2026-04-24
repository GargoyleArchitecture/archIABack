# -*- coding: utf-8 -*-
import os
import logging
import requests
from urllib.parse import quote

log = logging.getLogger("context_service")

_HTTP = requests.Session()

_BASE_URL = os.getenv("ARCHIA_API_BASE_URL", "").rstrip("/")


def fetch_project_context(project_id: str, api_token: str) -> dict:
    """
    Llama GET /api/v1/projects/:projectId/context en el Backend API.
    Retorna {"techStack": [...], "businessRules": "..."} o {} en caso de error.
    Degradación silenciosa: cualquier excepción retorna {}.
    """
    if not _BASE_URL or not project_id or not api_token:
        return {}
    url = f"{_BASE_URL}/api/v1/projects/{quote(project_id, safe='')}/context"
    try:
        resp = _HTTP.get(
            url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5,
        )
        if resp.status_code == 200:
            body = resp.json()
            # Backend API wraps responses as { statusCode, message, data: {...} }
            return body.get("data") if isinstance(body.get("data"), dict) else body
        log.warning("fetch_project_context: status %s para project_id=%s", resp.status_code, project_id)
    except Exception as exc:
        log.warning("fetch_project_context: %s", exc)
    return {}


def fetch_user_preferences(user_id: str, api_token: str) -> dict:
    """
    Llama GET /api/v1/users/:userId/preferences en el Backend API.
    Retorna {"explanationStyle": "ANALOGY|FORMAL|CONCISE", "verbosity": "LOW|MEDIUM|HIGH"} o {}.
    Degradación silenciosa: cualquier excepción retorna {}.
    """
    if not _BASE_URL or not user_id or not api_token:
        return {}
    url = f"{_BASE_URL}/api/v1/users/{quote(user_id, safe='')}/preferences"
    try:
        resp = _HTTP.get(
            url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=5,
        )
        if resp.status_code == 200:
            body = resp.json()
            # Backend API wraps responses as { statusCode, message, data: {...} }
            return body.get("data") if isinstance(body.get("data"), dict) else body
        log.warning("fetch_user_preferences: status %s para user_id=%s", resp.status_code, user_id)
    except Exception as exc:
        log.warning("fetch_user_preferences: %s", exc)
    return {}


def format_project_context_text(data: dict) -> str:
    """
    Convierte el ProjectContext del Backend API en un bloque de texto inyectable en prompts.
    Retorna "" si data está vacío (degradación silenciosa = sin cambio en prompts).
    """
    if not data:
        return ""
    tech = data.get("techStack") or []
    rules = (data.get("businessRules") or "").strip()
    if not tech and not rules:
        return ""
    lines = ["## PROJECT CONTEXT"]
    if tech:
        tech_str = ", ".join(tech) if isinstance(tech, list) else str(tech)
        lines.append(f"Tech stack: {tech_str}")
    if rules:
        lines.append(f"Business rules: {rules}")
    return "\n".join(lines)


def format_user_style_hint(data: dict) -> str:
    """
    Convierte UserPreference del Backend API en una directiva de una línea para el prompt final.
    Retorna "" si data está vacío (degradación silenciosa = sin cambio en prompts).
    """
    if not data:
        return ""
    style_map = {
        "ANALOGY": "use ANALOGIES and concrete real-world comparisons",
        "FORMAL": "use FORMAL academic-style language with precise terminology",
        "CONCISE": "be CONCISE — short direct answers, no elaboration",
    }
    verbosity_map = {
        "LOW": "LOW verbosity — minimal text, key points only",
        "MEDIUM": "MEDIUM verbosity — balanced detail",
        "HIGH": "HIGH verbosity — elaborate with examples and context",
    }
    parts = []
    s = (data.get("explanationStyle") or "").upper()
    v = (data.get("verbosity") or "").upper()
    if s in style_map:
        parts.append(style_map[s])
    if v in verbosity_map:
        parts.append(verbosity_map[v])
    if not parts:
        return ""
    return "Communication style: " + "; ".join(parts) + "."
