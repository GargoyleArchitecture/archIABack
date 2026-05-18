"""F12-T3 + F12-T4: Orquestador del endpoint POST /evaluate-attempt.

Sigue el patrón establecido por `routine_generator.py` (F5-T2 / Ciclo 2.5):
  - Auth: reutiliza `verify_internal_token` del módulo `routine_generator`
    para que el header `X-Internal-Token` se valide en un único lugar.
  - Trace: idem con `make_trace_id`.
  - Lógica: invoca el nodo single-shot `evaluate_attempt_node` y devuelve
    el `RoutineFeedback` que serializa el handler HTTP.
  - Telemetría: emite `routine_evaluated` con score, routine_id, user_id,
    trace_id — alineado con `docs/observability.md`.

Decisión deliberada de NO crear un grafo nuevo:
  El nodo es atómico (una llamada al LLM), no necesita state ni checkpointer.
  Mantenerlo como función plain async facilita los tests directos y el
  mocking del LLM, igual que en el patrón F3-T2 Shadow Agent.

F12-T4 (reflexión metacognitiva):
  Si `EvaluateAttemptInput.reflection` viene en el body, tras calcular el
  feedback lanzamos `dispatch_reflection_to_profile(...)` con
  `asyncio.create_task(...)` para reforzar el mastery del `target_weakness`
  en el Store del LangGraph. NUNCA bloquea la respuesta HTTP; si falla, el
  feedback ya está fuera y el flujo del alumno no se ve afectado.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Set

from fastapi import HTTPException

from src.graph.nodes.feedback import evaluate_attempt_node
from src.graph.schemas.feedback import EvaluateAttemptInput, RoutineFeedback
from src.services import reflection_dispatcher as _reflection_dispatcher
from src.services import attempt_sync as _attempt_sync
from src.services import score_reinforcement as _score_reinforcement

log = logging.getLogger("attempt_evaluator")

# F12-T4: tracking de tasks fire-and-forget para prevenir GC prematuro.
# Mismo patrón validado en `profile_shadow._pending_tasks` (F3-T2).
_pending_tasks: Set[asyncio.Task] = set()


async def evaluate_attempt_for_user(
    payload: EvaluateAttemptInput,
    *,
    trace_id: str,
) -> RoutineFeedback:
    """Invoca el nodo evaluador y emite telemetría. Nunca lanza salvo errores
    realmente inesperados (HTTPException 500); el nodo internamente ya cae a
    un fallback graceful si el LLM falla."""
    log.info(
        "evaluate_attempt started trace_id=%s user_id=%s routine_id=%s response_chars=%d rubric_size=%d",
        trace_id,
        payload.user_id,
        payload.routine_id,
        len(payload.user_response),
        len(payload.rubric),
    )

    try:
        feedback = await evaluate_attempt_node(payload)
    except Exception as exc:
        log.exception(
            "evaluate_attempt orchestrator raised trace_id=%s routine_id=%s: %s",
            trace_id,
            payload.routine_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Attempt evaluation failed unexpectedly.",
        )

    log.info(
        "evaluate_attempt completed trace_id=%s user_id=%s routine_id=%s score=%d criteria=%d",
        trace_id,
        payload.user_id,
        payload.routine_id,
        feedback.score,
        len(feedback.criteria),
    )

    # F11-T6 / F12-T10: telemetría estructurada. Import lazy + try/except
    # defensivo para que la telemetría jamás rompa el endpoint principal.
    try:
        from src.services.telemetry import emit as _emit_telemetry  # noqa: E402

        _emit_telemetry(
            "routine_evaluated",
            user_id=payload.user_id,
            routine_id=payload.routine_id,
            score=feedback.score,
            criteria_count=len(feedback.criteria),
            target_weakness=payload.target_weakness,
            trace_id=trace_id,
        )
    except Exception:
        pass

    # F16-T1: sync-back idempotente del feedback a Negocio (fire-and-forget).
    # Garantiza que el score/feedback se persista aunque el HTTP síncrono de
    # Negocio haya expirado (el endpoint de Negocio es idempotente por
    # attemptId). Solo si el caller envió attempt_id.
    if payload.attempt_id:
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                _attempt_sync.sync_attempt_feedback(
                    payload.attempt_id,
                    feedback.model_dump(),
                )
            )
            _pending_tasks.add(task)
            task.add_done_callback(_pending_tasks.discard)
        except RuntimeError:
            log.warning(
                "attempt sync-back skipped: no running event loop "
                "(trace_id=%s)", trace_id,
            )

    # F16-T2 + F12-T4: refuerzo de mastery (fire-and-forget, NUNCA bloquea
    # la respuesta HTTP). Se ejecutan SECUENCIADOS en una sola task para
    # serializar el read-modify-write sobre el mismo perfil del Store y
    # evitar clobber entre ambos dispatchers:
    #   1) EWMA por score (F16-T2, SIEMPRE): mastery_new =
    #      α·(score/100) + (1-α)·mastery_prev. Bidireccional, idempotente
    #      por attempt_id.
    #   2) Bonus de reflexión (F12-T4, sólo si vino reflexión): se aplica
    #      ENCIMA del valor ya EWMA-do (stacking correcto; no es doble conteo
    #      del mismo señal — uno mezcla el score, el otro premia el acto
    #      metacognitivo de reflexionar). Ambos dispatchers nunca lanzan.
    async def _apply_mastery() -> None:
        await _score_reinforcement.dispatch_score_to_profile(
            user_id=payload.user_id,
            target_weakness=payload.target_weakness,
            score=feedback.score,
            attempt_id=payload.attempt_id,
            trace_id=trace_id,
        )
        if payload.reflection is not None:
            await _reflection_dispatcher.dispatch_reflection_to_profile(
                user_id=payload.user_id,
                target_weakness=payload.target_weakness,
                score=feedback.score,
                reflection=payload.reflection,
                trace_id=trace_id,
            )

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_apply_mastery())
        _pending_tasks.add(task)
        task.add_done_callback(_pending_tasks.discard)
    except RuntimeError:
        log.warning(
            "mastery reinforcement skipped: no running event loop "
            "(trace_id=%s)", trace_id,
        )

    return feedback
