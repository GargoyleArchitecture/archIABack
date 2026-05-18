"""Nodo `validate_difficulty` del subgrafo RoutineGenerator (F5-T4).

Decide si el reto generado es coherente con el perfil del usuario o si hay
que regenerar con menos conceptos. Loop limitado a 2 reintentos para evitar
bucles infinitos (criterio explícito de F5-T4).

Reglas:
- Calcula la dificultad estimada vía `estimate_difficulty` (función pura).
- Compara contra el "techo permitido" para el usuario:
    ceiling = round(current_mastery * 5) + 2
  Es decir, un usuario con mastery 0.4 puede recibir hasta dificultad 4
  (round(0.4 * 5) = 2, ceiling = 4). Mastery 0.0 → ceiling = 2.
- Si `final_difficulty > ceiling` y `regen_count < 2` → router devuelve
  "regenerate" y `synthesize_challenge` se ejecuta otra vez con
  `regen_count` incrementado y `reduce_scope` activo.
- Si `regen_count >= 2` o `final_difficulty <= ceiling` → router devuelve
  "accept" y este nodo construye el `RoutineOutput` final en `state["final"]`.

`final_difficulty` se persiste en el output siempre clamp-eada a [1,5].
"""
from __future__ import annotations

import logging

from src.graph.schemas.routine import RoutineOutput, RoutineState, RubricCriterion
from src.graph.services.difficulty import estimate_difficulty

log = logging.getLogger("routine.validate_difficulty")

_MAX_REGEN = 2


# Rúbrica de último recurso: garantiza que `RoutineOutput.rubric` (F12-T2)
# nunca caiga bajo `min_length=3` aunque el state llegue híbrido o vacío
# (ej. tests legacy que no propagan rubric tras synthesize_challenge).
_DEFAULT_FALLBACK_RUBRIC = [
    RubricCriterion(
        concept="Architectural intent",
        description=(
            "The submission addresses the target weakness with a concrete and "
            "explainable change."
        ),
        weight=5,
    ),
    RubricCriterion(
        concept="Clarity",
        description=(
            "The code is readable and the rationale is briefly documented "
            "either inline or in commit-style notes."
        ),
        weight=3,
    ),
    RubricCriterion(
        concept="Correctness",
        description=(
            "The refactor compiles or runs in the target language and does "
            "not introduce obvious regressions."
        ),
        weight=4,
    ),
]


_FALLBACK_REFERENCE_SOLUTION = (
    "## Reference\n\nNo reference solution was emitted by the upstream node. "
    "Apply a minimal viable refactor that resolves the named weakness."
)


def _ceiling_for_mastery(current_mastery: float) -> int:
    """Techo de dificultad aceptable dado el mastery actual del usuario."""
    try:
        m = float(current_mastery)
    except (TypeError, ValueError):
        m = 0.0
    m = max(0.0, min(1.0, m))
    return int(round(m * 5)) + 2  # rango: 2 (m=0) .. 7 (m=1)


def _hydrate_rubric(state: RoutineState) -> list[RubricCriterion]:
    """Convierte `state['rubric']` (lista de dicts) a `List[RubricCriterion]`.

    Tolerante: si el state no trae rubric, está vacía, o trae dicts inválidos,
    cae al `_DEFAULT_FALLBACK_RUBRIC` para no romper la validación Pydantic
    del `RoutineOutput`.
    """
    raw = state.get("rubric")
    if not raw or not isinstance(raw, list):
        return list(_DEFAULT_FALLBACK_RUBRIC)
    try:
        hydrated = [RubricCriterion(**(d or {})) for d in raw]
    except Exception:
        log.warning(
            "validate_difficulty: state['rubric'] failed Pydantic hydration; "
            "falling back to default rubric"
        )
        return list(_DEFAULT_FALLBACK_RUBRIC)
    return hydrated if len(hydrated) >= 3 else list(_DEFAULT_FALLBACK_RUBRIC)


def validate_difficulty_node(state: RoutineState) -> RoutineState:
    """Calcula final_difficulty, decide si regenera, y guarda `final` si OK."""
    llm_difficulty = int(state.get("difficulty") or 1)
    expected = list(state.get("expected_concepts") or [])
    snippet = state.get("raw_snippet") or ""
    current_mastery = float(state.get("target_mastery") or 0.0)
    regen_count = int(state.get("regen_count", 0) or 0)

    final_difficulty = estimate_difficulty(
        num_concepts=len(expected),
        current_mastery=current_mastery,
        snippet_length=len(snippet),
        llm_difficulty=llm_difficulty,
    )

    ceiling = _ceiling_for_mastery(current_mastery)
    too_hard = final_difficulty > ceiling
    can_regen = regen_count < _MAX_REGEN

    log.info(
        "validate_difficulty: llm=%d est=%d ceiling=%d regen_count=%d too_hard=%s can_regen=%s",
        llm_difficulty, final_difficulty, ceiling, regen_count, too_hard, can_regen,
    )

    if too_hard and can_regen:
        # Programa una regeneración: el router enviará el flujo de vuelta a
        # synthesize_challenge. Aumentamos `regen_count` y NO seteamos `final`
        # para que el router detecte el modo "regenerate".
        return {
            **state,
            "regen_count": regen_count + 1,
            "difficulty": final_difficulty,
            # No pisamos title/challenge_md: synthesize los reescribirá.
        }

    # Acepta: clamp final, construye RoutineOutput, persiste en state["final"].
    final_difficulty = max(1, min(5, final_difficulty))
    rubric_objs = _hydrate_rubric(state)
    reference_solution = state.get("reference_solution") or _FALLBACK_REFERENCE_SOLUTION

    output = RoutineOutput(
        title=state.get("title") or f"Routine for {state.get('target_weakness', 'arch')}",
        target_weakness=state.get("target_weakness") or "general software architecture",
        inverse_rag_snippet=snippet or None,
        expected_concepts=expected,
        difficulty=final_difficulty,
        challenge_md=state.get("challenge_md")
        or "## Challenge\n\nNo content was generated.",
        rubric=rubric_objs,
        reference_solution=reference_solution,
    )
    return {
        **state,
        "difficulty": final_difficulty,
        "final": output,
    }


def validate_difficulty_router(state: RoutineState) -> str:
    """Router de la edge condicional posterior a validate_difficulty.

    - "regenerate" → vuelve a synthesize_challenge.
    - "accept"     → END (final ya está en state["final"]).
    """
    if state.get("final") is None:
        return "regenerate"
    return "accept"
