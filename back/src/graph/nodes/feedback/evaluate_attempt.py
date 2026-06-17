"""F12-T3: Nodo `evaluate_attempt` — single-shot LLM contra la rúbrica.

Diferencias clave vs. el subgrafo de retos (F5-T1):
  - Una sola llamada al LLM; sin loop de regen.
  - Sin estado LangGraph: la función toma un Pydantic `EvaluateAttemptInput`
    y devuelve un Pydantic `RoutineFeedback`. El orquestador (F12-T3
    `attempt_evaluator.py`) se encarga del flujo HTTP.
  - Fallback graceful: si el LLM lanza, devolvemos un payload coherente con
    `score=0` y un `socratic_comment` explicando la degradación. El alumno
    ve un mensaje útil en lugar de un 500 opaco.

`method="function_calling"` por consistencia con `synthesize_challenge` y
el Shadow Agent — evita el modo `json_schema` que exige todos los campos
required y rompe con `default_factory`.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.graph.resources import llm
from src.graph.schemas.feedback import EvaluateAttemptInput, RoutineFeedback

log = logging.getLogger("evaluate_attempt")


_SYSTEM_PROMPT = """\
You are a software architecture coach evaluating a learner's submission
against an explicit rubric. Your output MUST conform to the RoutineFeedback
schema exactly.

Inputs you receive:
- target_weakness, expected_concepts: what the challenge was meant to teach.
- rubric: 3-5 criteria; for each one, decide met / partial / missing based
  ONLY on the user_response.
    * `met`     — the response clearly demonstrates the criterion.
    * `partial` — there is evidence but it is incomplete or imprecise.
    * `missing` — the response does not address it.
- user_response: the learner's attempt.
- reference_solution: a model solution. Use it as a comparison anchor, BUT
  do not penalize stylistic differences; reward conceptual coverage.

How to compute `score` (integer 0..100):
  weighted = Σ (weight_i × factor_i)
    where factor = 1.0 for met, 0.5 for partial, 0.0 for missing.
  score = round( weighted / Σ(weight_i) × 100 )
  Clamp to [0, 100].

For `criteria`: emit one `CriterionResult` per rubric entry, in the same
order. The `concept` should match the rubric `concept` verbatim.

For `strengths`: 1-3 short bullet points naming what the learner did well.
For `improvements`: 1-4 specific, actionable suggestions.
For `socratic_comment`: ONE open-ended question (no answer) that nudges
the learner toward deeper thinking. Do NOT reveal the reference solution.
"""


def _format_rubric_for_prompt(rubric_items) -> str:
    """Serializa la rúbrica como lista de objetos JSON-like para el LLM."""
    parts = []
    for c in rubric_items:
        parts.append(
            f'  {{"concept": "{c.concept}", '
            f'"description": "{c.description}", '
            f'"weight": {c.weight}}}'
        )
    return "[\n" + ",\n".join(parts) + "\n]"


def _fallback_feedback(payload: EvaluateAttemptInput) -> RoutineFeedback:
    """Payload mínimo coherente cuando el LLM falla.

    El alumno verá `score=0`, una mejora útil y un comentario socrático que
    explica la degradación sin sonar como un error técnico opaco.
    """
    return RoutineFeedback(
        score=0,
        criteria=[],
        strengths=[],
        improvements=[
            "La evaluación automática no pudo procesarse. Reinténtalo en unos segundos.",
        ],
        socratic_comment=(
            "El evaluador no respondió a tiempo. ¿Puedes releer tu solución y "
            "marcar mentalmente qué criterios de la rúbrica crees que cumple, "
            "antes de reintentar el envío?"
        ),
    )


async def evaluate_attempt_node(
    payload: EvaluateAttemptInput,
    llm_obj: Optional[object] = None,
) -> RoutineFeedback:
    """Evalúa el intento del alumno contra la rúbrica.

    Args:
        payload: input validado (Pydantic ya pasó).
        llm_obj: permite inyectar un mock en tests sin tocar el módulo global.

    Returns:
        Un `RoutineFeedback` con score y criterios. Nunca lanza; en error
        devuelve el fallback degradado.
    """
    target_llm = llm_obj if llm_obj is not None else llm
    structured = target_llm.with_structured_output(
        RoutineFeedback, method="function_calling"
    )

    expected = ", ".join(payload.expected_concepts) if payload.expected_concepts else "(none)"
    user_prompt = (
        f"TARGET_WEAKNESS: {payload.target_weakness}\n\n"
        f"EXPECTED_CONCEPTS: {expected}\n\n"
        f"RUBRIC:\n{_format_rubric_for_prompt(payload.rubric)}\n\n"
        f"REFERENCE_SOLUTION:\n{payload.reference_solution}\n\n"
        f"USER_RESPONSE:\n{payload.user_response}"
    )

    log.info(
        "evaluate_attempt invoking LLM routine_id=%s user_id=%s rubric_size=%d response_chars=%d",
        payload.routine_id,
        payload.user_id,
        len(payload.rubric),
        len(payload.user_response),
    )

    try:
        result: RoutineFeedback = await structured.ainvoke(
            f"{_SYSTEM_PROMPT}\n\n{user_prompt}"
        )
    except Exception as exc:
        log.exception("evaluate_attempt LLM call failed: %s", exc)
        return _fallback_feedback(payload)

    # Defensa: clamp score fuera de rango. Pydantic ya valida [0,100] vía
    # ge/le; este clamp se aplica si en algún momento se relaja el constraint
    # o si el LLM negocia un workaround inesperado.
    if result.score < 0 or result.score > 100:
        clamped = max(0, min(100, result.score))
        log.warning(
            "evaluate_attempt clamped out-of-range score from %d to %d",
            result.score, clamped,
        )
        result = result.model_copy(update={"score": clamped})

    return result
