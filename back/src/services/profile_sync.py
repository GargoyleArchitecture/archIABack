"""Cliente HTTP que sincroniza el perfil de usuario al Backend Negocio (F3-T5).

Disparado tras un `store.aput(...)` exitoso en el Shadow Agent. Reintenta
con backoff exponencial (3 intentos: 1s, 2s, 4s). Si Negocio esta caido,
la app IA sigue funcionando con el perfil del Store local intacto.

Configuracion via env:
- BUSINESS_API_BASE_URL     (default http://localhost:3000)
- BUSINESS_API_PROFILE_PATH (default /internal/users/{userId}/profile)
- INTERNAL_API_TOKEN        (sin default; vacio -> sync deshabilitado)
- PROFILE_SYNC_ENABLED      (default true)
- PROFILE_SYNC_TIMEOUT      (segundos por intento, default 15)
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

log = logging.getLogger("profile_sync")


def _env_base_url() -> str:
    return os.getenv("BUSINESS_API_BASE_URL", "http://localhost:3000").rstrip("/")


def _env_path_template() -> str:
    return os.getenv("BUSINESS_API_PROFILE_PATH", "/internal/users/{userId}/profile")


def _env_token() -> str:
    return (os.getenv("INTERNAL_API_TOKEN") or "").strip()


def _env_enabled() -> bool:
    return (os.getenv("PROFILE_SYNC_ENABLED", "true") or "true").lower() == "true"


def _env_timeout() -> float:
    try:
        return float(os.getenv("PROFILE_SYNC_TIMEOUT", "15"))
    except (TypeError, ValueError):
        return 15.0


def _scale_to_business(mastery_0_to_1) -> float:
    """0-1 (interno) -> 0-100 (contrato Negocio EvaluatedConcept)."""
    try:
        v = float(mastery_0_to_1)
    except (TypeError, ValueError):
        v = 0.0
    return round(max(0.0, min(1.0, v)) * 100.0, 2)


def _to_business_payload(merged: dict) -> dict:
    """Transforma el dict mergeado del Store al body camelCase para Negocio.

    Mapeo:
    - user_id            -> userId (path arg, NO en body)
    - strengths          -> strengths (lista de strings, igual)
    - weaknesses         -> weaknesses
    - evaluated_concepts -> evaluatedConcepts (mastery 0-100; lastSeenAt camelCase)
    - confidence         -> confidence
    - delta_from_previous -> deltaFromPrevious
    - updated_at         -> updatedAt

    `mastery_original` (si existe) tiene prioridad sobre `mastery` para
    evitar enviar el valor decayed-en-lectura.
    """
    merged = merged or {}
    concepts_out = []
    for c in (merged.get("evaluated_concepts") or []):
        if not isinstance(c, dict):
            continue
        mastery_raw = c.get("mastery_original")
        if mastery_raw is None:
            mastery_raw = c.get("mastery")
        item = {
            "name": c.get("name", ""),
            "mastery": _scale_to_business(mastery_raw or 0.0),
        }
        if c.get("last_seen_at"):
            item["lastSeenAt"] = c["last_seen_at"]
        if c.get("decay_rate") is not None:
            try:
                item["decayRate"] = float(c["decay_rate"])
            except (TypeError, ValueError):
                pass
        if c.get("evidence"):
            item["evidence"] = c["evidence"]
        concepts_out.append(item)

    return {
        "strengths": list(merged.get("strengths") or []),
        "weaknesses": list(merged.get("weaknesses") or []),
        "evaluatedConcepts": concepts_out,
        "confidence": float(merged.get("confidence") or 0.0),
        "deltaFromPrevious": dict(merged.get("delta_from_previous") or {}),
        "updatedAt": merged.get("updated_at") or "",
    }


async def sync_profile(user_id: str, merged_profile: dict, *, max_attempts: int = 3) -> bool:
    """PUT del perfil al Backend Negocio con retry exponencial.

    Devuelve True si algun intento fue 2xx; False si todos fallaron o se omitio.
    NUNCA lanza al caller (criterio del backlog: app IA sigue corriendo).
    """
    if not _env_enabled():
        log.debug("profile_sync skipped: PROFILE_SYNC_ENABLED=false")
        return False
    user_id = (user_id or "").strip()
    if not user_id:
        log.warning("profile_sync skipped: empty user_id")
        return False
    token = _env_token()
    if not token:
        log.warning("profile_sync skipped: INTERNAL_API_TOKEN empty")
        return False

    url = _env_base_url() + _env_path_template().replace("{userId}", str(user_id))
    payload = _to_business_payload(merged_profile)
    timeout = _env_timeout()

    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.put(
                    url,
                    json=payload,
                    headers={
                        "X-Internal-Token": token,
                        "Content-Type": "application/json",
                    },
                )
            if 200 <= resp.status_code < 300:
                log.info(
                    "profile_sync ok user_id=%s attempt=%d status=%d",
                    user_id, attempt, resp.status_code,
                )
                return True
            log.warning(
                "profile_sync attempt=%d status=%d body=%s",
                attempt, resp.status_code, (resp.text or "")[:200],
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            log.warning("profile_sync attempt=%d error=%s", attempt, exc)

        if attempt < max_attempts:
            await asyncio.sleep(backoff)
            backoff *= 2

    log.error(
        "profile_sync gave up after %d attempts for user_id=%s url=%s",
        max_attempts, user_id, url,
    )
    return False
