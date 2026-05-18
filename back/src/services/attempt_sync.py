"""F16-T1: sync-back idempotente del feedback de un intento hacia Negocio.

Problema que resuelve: `/evaluate-attempt` es síncrono y Negocio aborta a los
60s (antes 30s). Si el LLM se demora, Negocio pierde el resultado aunque IA
termine. Este módulo hace un push best-effort del feedback a un endpoint
interno idempotente de Negocio, disparado fire-and-forget tras computar el
feedback, de modo que el resultado se persiste aunque el HTTP de Negocio ya
haya expirado.

Reusa la configuración/credenciales de `profile_sync` (mismo `X-Internal-Token`
y `BUSINESS_API_BASE_URL`). Reintenta con backoff exponencial (3 intentos:
1s, 2s, 4s). NUNCA lanza al caller.

Config via env:
- BUSINESS_API_BASE_URL          (default http://localhost:3000)
- BUSINESS_API_ATTEMPT_FEEDBACK_PATH
      (default /internal/routine-attempts/{attemptId}/feedback)
- INTERNAL_API_TOKEN             (vacío -> sync deshabilitado)
- PROFILE_SYNC_ENABLED           (reusa el mismo flag global; default true)
- PROFILE_SYNC_TIMEOUT           (segundos por intento, default 15)
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

from src.services.profile_sync import (
    _env_base_url,
    _env_enabled,
    _env_timeout,
    _env_token,
)

log = logging.getLogger("attempt_sync")


def _env_path_template() -> str:
    return os.getenv(
        "BUSINESS_API_ATTEMPT_FEEDBACK_PATH",
        "/internal/routine-attempts/{attemptId}/feedback",
    )


async def sync_attempt_feedback(
    attempt_id: str,
    feedback: dict,
    *,
    max_attempts: int = 3,
) -> bool:
    """POST idempotente del feedback al Backend Negocio con retry exponencial.

    `feedback` es el `RoutineFeedback.model_dump()` (incluye `score`). El
    endpoint de Negocio es idempotente (no-op si el intento ya fue evaluado),
    así que reintentar o que llegue también por la vía síncrona es seguro.

    Devuelve True si algún intento fue 2xx; False si todos fallaron o se omitió.
    NUNCA lanza al caller (criterio del backlog: la app IA sigue corriendo).
    """
    if not _env_enabled():
        log.debug("attempt_sync skipped: PROFILE_SYNC_ENABLED=false")
        return False
    attempt_id = (attempt_id or "").strip()
    if not attempt_id:
        log.warning("attempt_sync skipped: empty attempt_id")
        return False
    token = _env_token()
    if not token:
        log.warning("attempt_sync skipped: INTERNAL_API_TOKEN empty")
        return False

    score = feedback.get("score") if isinstance(feedback, dict) else None
    url = _env_base_url() + _env_path_template().replace("{attemptId}", str(attempt_id))
    payload = {"score": score, "feedback": feedback}
    timeout = _env_timeout()

    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "X-Internal-Token": token,
                        "Content-Type": "application/json",
                    },
                )
            if 200 <= resp.status_code < 300:
                log.info(
                    "attempt_sync ok attempt_id=%s attempt=%d status=%d",
                    attempt_id, attempt, resp.status_code,
                )
                return True
            log.warning(
                "attempt_sync attempt=%d status=%d body=%s",
                attempt, resp.status_code, (resp.text or "")[:200],
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            log.warning("attempt_sync attempt=%d error=%s", attempt, exc)

        if attempt < max_attempts:
            await asyncio.sleep(backoff)
            backoff *= 2

    log.error(
        "attempt_sync gave up after %d attempts for attempt_id=%s url=%s",
        max_attempts, attempt_id, url,
    )
    return False
