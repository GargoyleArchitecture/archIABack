"""F12-T4: Dispatcher de reflexión metacognitiva → refuerzo del perfil.

Cierra el bucle pedagógico de la Fase 12: cuando el alumno envía su reflexión
tras un attempt evaluado, este módulo refuerza directamente el mastery del
concepto trabajado (`target_weakness`) en el Store del LangGraph.

NOTA terminológica: el backlog dice "se envía al Shadow Agent". En la práctica
NO invocamos `shadow_eval_async` (que es LLM-bound). Lo que hacemos es escribir
directamente en el mismo namespace del Store que el Shadow Agent administra,
con una regla determinista de delta de mastery. Esto:
  - es testeable sin LLM,
  - mantiene coste cero,
  - cumple el criterio del backlog ("delta positivo de mastery cuando score >= 70"),
  - converge con el Shadow Agent en la próxima evaluación periódica (el Shadow
    Agent leerá el perfil ya reforzado como punto de partida).

Disparado por `attempt_evaluator.evaluate_attempt_for_user` (F12-T3) tras
calcular el feedback, fire-and-forget. NUNCA bloquea la respuesta HTTP.

Convenciones del Store consistentes con `profile_shadow` (F3-T2):
  - namespace: ("user", user_id, "profile")
  - key:       "profile"
  - mastery:   escala 0..1 (Negocio convierte a 0..100 si aplica)
  - field key: `evaluated_concepts`, `name`, `mastery`, `last_seen_at`
    (snake_case del lado IA; Negocio usa camelCase y traduce con `profile_sync`).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.graph.resources import get_store
from src.graph.schemas.feedback import ReflectionPayload

log = logging.getLogger("reflection_dispatcher")


def _mastery_delta(score: int) -> float:
    """Regla de refuerzo basada en score 0..100.

    - score < 70:  0.00 (sin refuerzo: el aprendizaje no consolidó).
    - 70..84:      0.05 (refuerzo moderado).
    - >= 85:       0.10 (refuerzo fuerte: consolidación profunda).
    """
    try:
        s = int(score)
    except (TypeError, ValueError):
        return 0.0
    if s >= 85:
        return 0.10
    if s >= 70:
        return 0.05
    return 0.0


def _find_concept_index(concepts: list, target_norm: str) -> Optional[int]:
    """Match case-insensitive + strip de `name`. Devuelve None si no existe."""
    for i, c in enumerate(concepts):
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "") or "").strip().lower()
        if name and name == target_norm:
            return i
    return None


async def dispatch_reflection_to_profile(
    user_id: str,
    target_weakness: str,
    score: int,
    reflection: ReflectionPayload,
    *,
    trace_id: str,
    store=None,
) -> Optional[dict]:
    """Best-effort: refuerza `mastery` del `target_weakness` en el Store.

    Diseño:
      - NUNCA lanza. Cualquier excepción se loggea y retorna None.
      - Si `score < 70`, retorna None sin tocar el Store (umbral pedagógico).
      - Si el concepto existe, suma el delta (clampeado a [0, 1]).
      - Si NO existe, lo añade con un mastery inicial derivado del score
        (clampeado a [0.5, 0.95]).
      - Emite telemetría `reflection_submitted` (best-effort).

    Args:
        user_id:         id estable del usuario.
        target_weakness: nombre del concepto a reforzar.
        score:           0..100, del feedback ya calculado.
        reflection:      el payload metacognitivo (lo aceptamos para validar
                         que efectivamente vino, pero no persistimos el texto
                         acá — Negocio lo guarda en su tabla).
        trace_id:        propagación de trazabilidad para logs/telemetría.
        store:           inyección opcional para tests.

    Returns:
        El dict del perfil actualizado, o None si no se persistió nada.
    """
    user_id = (user_id or "").strip()
    target = (target_weakness or "").strip()
    if not user_id or not target:
        log.debug(
            "reflection_dispatch skipped: empty user_id or target_weakness "
            "(trace_id=%s)", trace_id,
        )
        return None
    if reflection is None:
        log.debug("reflection_dispatch skipped: no reflection payload (trace_id=%s)", trace_id)
        return None

    delta = _mastery_delta(score)
    if delta <= 0:
        log.info(
            "reflection_dispatch noop: score=%d below threshold (trace_id=%s user_id=%s)",
            score, trace_id, user_id,
        )
        return None

    # 1) Leer perfil RAW del Store (mastery_0 sin decay aplicado).
    store_obj = store if store is not None else get_store()
    ns = ("user", user_id, "profile")
    try:
        existing = await store_obj.aget(ns, key="profile")
    except Exception as exc:
        log.exception(
            "reflection_dispatch: store.aget failed for user_id=%s: %s",
            user_id, exc,
        )
        return None

    prev = (existing.value if existing else {}) or {}
    if not isinstance(prev, dict):
        prev = {}
    concepts = list(prev.get("evaluated_concepts") or [])

    # 2) Buscar el concepto (case-insensitive).
    target_norm = target.lower()
    found_idx = _find_concept_index(concepts, target_norm)
    now_iso = datetime.now(timezone.utc).isoformat()

    if found_idx is not None:
        existing_concept = dict(concepts[found_idx])
        try:
            cur_mastery = float(existing_concept.get("mastery") or 0.0)
        except (TypeError, ValueError):
            cur_mastery = 0.0
        new_mastery = max(0.0, min(1.0, cur_mastery + delta))
        existing_concept["mastery"] = new_mastery
        existing_concept["last_seen_at"] = now_iso
        concepts[found_idx] = existing_concept
        log.info(
            "reflection_dispatch reinforced existing concept name=%r mastery %.2f → %.2f (trace_id=%s)",
            target, cur_mastery, new_mastery, trace_id,
        )
    else:
        seed_mastery = max(0.5, min(0.95, float(score) / 100.0))
        concepts.append({
            "name": target,
            "mastery": seed_mastery,
            "last_seen_at": now_iso,
            "evidence": "Reflexión post-reto",
        })
        log.info(
            "reflection_dispatch seeded new concept name=%r mastery=%.2f (trace_id=%s)",
            target, seed_mastery, trace_id,
        )

    prev["evaluated_concepts"] = concepts
    prev["user_id"] = user_id
    prev["updated_at"] = now_iso

    # 3) Persistir en el Store.
    try:
        await store_obj.aput(ns, key="profile", value=prev)
    except Exception as exc:
        log.exception(
            "reflection_dispatch: store.aput failed for user_id=%s: %s",
            user_id, exc,
        )
        return None

    # 4) Sync best-effort a Negocio (F3-T5). Si falla, el Store local ya está OK.
    try:
        from src.services.profile_sync import sync_profile  # noqa: E402

        await sync_profile(user_id, prev)
    except Exception:
        log.exception(
            "reflection_dispatch: profile_sync raised unexpectedly for user_id=%s",
            user_id,
        )

    # 5) Telemetría F11-T6 / F12-T10.
    try:
        from src.services.telemetry import emit as _emit_telemetry  # noqa: E402

        _emit_telemetry(
            "reflection_submitted",
            user_id=user_id,
            target_weakness=target,
            score=score,
            mastery_delta=delta,
            trace_id=trace_id,
        )
    except Exception:
        pass

    return prev
