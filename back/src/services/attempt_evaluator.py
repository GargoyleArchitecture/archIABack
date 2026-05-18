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

    # F12-T4: reflexión metacognitiva → refuerzo del perfil (fire-and-forget).
    # El dispatcher decide internamente si aplicar delta según score >= 70;
    # de cualquier modo nunca lanza, por eso el try/except aquí cubre sólo el
    # caso "no event loop" (tests síncronos que invocan esta función sin
    # `asyncio.run`).
    if payload.reflection is not None:
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                _reflection_dispatcher.dispatch_reflection_to_profile(
                    user_id=payload.user_id,
                    target_weakness=payload.target_weakness,
                    score=feedback.score,
                    reflection=payload.reflection,
                    trace_id=trace_id,
                )
            )
            _pending_tasks.add(task)
            task.add_done_callback(_pending_tasks.discard)
        except RuntimeError:
            log.warning(
                "reflection dispatch skipped: no running event loop "
                "(trace_id=%s)", trace_id,
            )

    return feedback
