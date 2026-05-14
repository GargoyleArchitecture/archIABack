"""Cliente HTTP que persiste mensajes del agente IA en el Backend Negocio.

Disparado en /message (FastAPI) justo ANTES de emitir el evento SSE 'complete'.
Garantiza que cuando el cliente recibe el endMessage, la respuesta ya esta
persistida en PostgreSQL. Asi el cliente puede desconectarse con seguridad
(cambio de vista, cierre de pestana, navegacion entre rutas) sin perder la
respuesta del agente.

A diferencia de profile_sync.py (X-Internal-Token), este cliente forwarda el
JWT del usuario que vino en /message. Preserva la semantica de ownership del
endpoint publico de Negocio:
    POST /api/v1/chats/:chatId/messages   (JwtAuthGuard global)

Configuracion via env:
- BUSINESS_API_BASE_URL       (default http://localhost:3000/api/v1)
- BUSINESS_API_MESSAGES_PATH  (default /chats/{chatId}/messages)
- MESSAGE_SYNC_ENABLED        (default true; false para tests locales sin Negocio)
- MESSAGE_SYNC_TIMEOUT        (segundos por intento, default 10)
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

log = logging.getLogger("message_sync")


def _env_base_url() -> str:
    return os.getenv("BUSINESS_API_BASE_URL", "http://localhost:3000/api/v1").rstrip("/")


def _env_path_template() -> str:
    return os.getenv("BUSINESS_API_MESSAGES_PATH", "/chats/{chatId}/messages")


def _env_enabled() -> bool:
    return (os.getenv("MESSAGE_SYNC_ENABLED", "true") or "true").lower() == "true"


def _env_timeout() -> float:
    try:
        return float(os.getenv("MESSAGE_SYNC_TIMEOUT", "10"))
    except (TypeError, ValueError):
        return 10.0


async def persist_ai_message(
    chat_id: str,
    content: str,
    authorization: str | None,
    *,
    max_attempts: int = 3,
) -> bool:
    """POST de la respuesta IA al endpoint publico de Negocio.

    Args:
        chat_id: equivale al session_id del grafo. UUID del chat en Negocio.
        content: el endMessage final del unifier (texto plano markdown).
        authorization: header Authorization completo tal cual llego a /message,
                       e.g. "Bearer eyJhbGci...". Si vacio, el sync se omite.
        max_attempts: intentos con backoff exponencial 1s/2s/4s.

    Retorna:
        True si algun intento fue 2xx.
        False si todos fallaron, si la config esta deshabilitada, si faltan
        argumentos requeridos, o si Negocio respondio 401/403/404 (errores
        de configuracion, no transitorios — no se reintenta).

    NUNCA lanza al caller. El stream del agente no se rompe si esto falla.
    """
    if not _env_enabled():
        log.debug("message_sync skipped: MESSAGE_SYNC_ENABLED=false")
        return False
    chat_id = (chat_id or "").strip()
    if not chat_id:
        log.warning("message_sync skipped: empty chat_id")
        return False
    content = content or ""
    if not content.strip():
        log.debug("message_sync skipped: empty content for chat_id=%s", chat_id)
        return False
    authorization = (authorization or "").strip()
    if not authorization:
        log.warning(
            "message_sync skipped: missing Authorization header for chat_id=%s",
            chat_id,
        )
        return False

    url = _env_base_url() + _env_path_template().replace("{chatId}", chat_id)
    payload = {"content": content, "role": "AI"}
    timeout = _env_timeout()

    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": authorization,
                        "Content-Type": "application/json",
                    },
                )
            if 200 <= resp.status_code < 300:
                log.info(
                    "message_sync ok chat_id=%s attempt=%d status=%d",
                    chat_id, attempt, resp.status_code,
                )
                return True
            # 401/403/404 son errores de configuracion o de dominio (chat
            # inexistente, JWT expirado, ownership fallo). Reintentar no
            # ayuda y solo agrega latencia al cierre del stream.
            if resp.status_code in (401, 403, 404):
                log.warning(
                    "message_sync giving up early chat_id=%s status=%d body=%s",
                    chat_id, resp.status_code, (resp.text or "")[:200],
                )
                return False
            log.warning(
                "message_sync attempt=%d status=%d body=%s",
                attempt, resp.status_code, (resp.text or "")[:200],
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            log.warning("message_sync attempt=%d error=%s", attempt, exc)

        if attempt < max_attempts:
            await asyncio.sleep(backoff)
            backoff *= 2

    log.error(
        "message_sync gave up after %d attempts for chat_id=%s url=%s",
        max_attempts, chat_id, url,
    )
    return False
