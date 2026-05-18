"""F16-T2: refuerzo de mastery por SCORE del intento (EWMA, bidireccional).

Complementa a `reflection_dispatcher` (F12-T4):
  - F12-T4 reflexión: bonus aditivo (+0.05/+0.10) SOLO si hay reflexión y
    score >= 70. Nunca baja. Se mantiene tal cual, aparte.
  - F16-T2 (este módulo): se aplica en CADA evaluación de intento, con o sin
    reflexión, mezclando el score con el mastery previo por media móvil
    exponencial (EWMA):

        mastery_new = alpha * (score / 100) + (1 - alpha) * mastery_prev

    Bidireccional por construcción: un score alto sube el mastery, uno bajo
    lo baja. `alpha` (default 0.3) controla cuánto pesa el último intento.

Idempotente por `attempt_id`: si un intento ya reforzó el perfil (reintento
de `/evaluate-attempt` tras timeout, o doble disparo), es no-op. El registro
de ids aplicados vive en el propio dict de perfil del Store (clave
`score_reinforced_attempts`, acotada) — NO se sincroniza a Negocio
(`profile_sync._to_business_payload` sólo mapea campos conocidos).

Convenciones del Store idénticas a `reflection_dispatcher` / `profile_shadow`:
  - namespace: ("user", user_id, "profile"), key "profile"
  - mastery escala 0..1 (Negocio convierte a 0..100 vía profile_sync)

NUNCA lanza al caller. Disparado fire-and-forget por
`attempt_evaluator.evaluate_attempt_for_user`.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from src.graph.resources import get_store

log = logging.getLogger("score_reinforcement")

_MAX_TRACKED_ATTEMPTS = 500


def _alpha() -> float:
    """Peso EWMA del último intento. env SCORE_REINFORCE_ALPHA (default 0.3),
    clampeado a (0, 1) — fuera de rango cae al default."""
    try:
        a = float(os.getenv("SCORE_REINFORCE_ALPHA", "0.3"))
    except (TypeError, ValueError):
        return 0.3
    if not (0.0 < a < 1.0):
        return 0.3
    return a


def _find_concept_index(concepts: list, target_norm: str) -> Optional[int]:
    """Match case-insensitive + strip de `name`. None si no existe."""
    for i, c in enumerate(concepts):
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "") or "").strip().lower()
        if name and name == target_norm:
            return i
    return None


async def dispatch_score_to_profile(
    user_id: str,
    target_weakness: str,
    score: int,
    attempt_id: Optional[str],
    *,
    trace_id: str,
    store=None,
) -> Optional[dict]:
    """Aplica EWMA del score al mastery del `target_weakness` en el Store.

    Returns el dict de perfil actualizado, o None si no se persistió nada
    (guards, idempotente, o Store roto). NUNCA lanza.
    """
    user_id = (user_id or "").strip()
    target = (target_weakness or "").strip()
    if not user_id or not target:
        log.debug(
            "score_reinforce skipped: empty user_id/target (trace_id=%s)",
            trace_id,
        )
        return None

    try:
        s = int(score)
    except (TypeError, ValueError):
        log.warning("score_reinforce skipped: score no numérico=%r", score)
        return None
    s = max(0, min(100, s))

    ns = ("user", user_id, "profile")
    try:
        store_obj = store if store is not None else get_store()
        existing = await store_obj.aget(ns, key="profile")
    except Exception as exc:
        log.exception(
            "score_reinforce: store resolve/aget failed user_id=%s: %s",
            user_id, exc,
        )
        return None

    prev = (existing.value if existing else {}) or {}
    if not isinstance(prev, dict):
        prev = {}

    # Idempotencia por attempt_id.
    applied = list(prev.get("score_reinforced_attempts") or [])
    aid = (attempt_id or "").strip()
    if aid and aid in applied:
        log.info(
            "score_reinforce idempotent no-op attempt_id=%s user_id=%s (trace_id=%s)",
            aid, user_id, trace_id,
        )
        return None

    alpha = _alpha()
    score_unit = s / 100.0
    concepts = list(prev.get("evaluated_concepts") or [])
    target_norm = target.lower()
    found_idx = _find_concept_index(concepts, target_norm)
    now_iso = datetime.now(timezone.utc).isoformat()

    if found_idx is not None:
        concept = dict(concepts[found_idx])
        try:
            prev_m = float(concept.get("mastery") or 0.0)
        except (TypeError, ValueError):
            prev_m = 0.0
        new_m = alpha * score_unit + (1.0 - alpha) * prev_m
        new_m = max(0.0, min(1.0, round(new_m, 4)))
        concept["mastery"] = new_m
        concept["last_seen_at"] = now_iso
        concepts[found_idx] = concept
        log.info(
            "score_reinforce EWMA name=%r score=%d mastery %.4f → %.4f "
            "(alpha=%.2f trace_id=%s)",
            target, s, prev_m, new_m, alpha, trace_id,
        )
    else:
        # Sin prior: el límite EWMA es el propio score normalizado.
        new_m = max(0.0, min(1.0, round(score_unit, 4)))
        prev_m = 0.0
        concepts.append({
            "name": target,
            "mastery": new_m,
            "last_seen_at": now_iso,
            "evidence": "Intento evaluado",
        })
        log.info(
            "score_reinforce seeded name=%r score=%d mastery=%.4f "
            "(trace_id=%s)",
            target, s, new_m, trace_id,
        )

    if aid:
        applied.append(aid)
        prev["score_reinforced_attempts"] = applied[-_MAX_TRACKED_ATTEMPTS:]
    prev["evaluated_concepts"] = concepts
    prev["user_id"] = user_id
    prev["updated_at"] = now_iso

    try:
        await store_obj.aput(ns, key="profile", value=prev)
    except Exception as exc:
        log.exception(
            "score_reinforce: store.aput failed user_id=%s: %s", user_id, exc
        )
        return None

    # Sync best-effort a Negocio (0..1 → 0..100 vía profile_sync).
    try:
        from src.services.profile_sync import sync_profile  # noqa: E402

        await sync_profile(user_id, prev)
    except Exception:
        log.exception(
            "score_reinforce: profile_sync raised unexpectedly user_id=%s",
            user_id,
        )

    try:
        from src.services.telemetry import emit as _emit_telemetry  # noqa: E402

        _emit_telemetry(
            "attempt_mastery_reinforced",
            user_id=user_id,
            target_weakness=target,
            score=s,
            mastery_prev=round(prev_m, 4),
            mastery_new=new_m,
            alpha=alpha,
            attempt_id=aid or None,
            trace_id=trace_id,
        )
    except Exception:
        pass

    return prev
