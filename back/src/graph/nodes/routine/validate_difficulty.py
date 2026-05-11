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

from src.graph.schemas.routine import RoutineOutput, RoutineState
from src.graph.services.difficulty import estimate_difficulty

log = logging.getLogger("routine.validate_difficulty")

_MAX_REGEN = 2


def _ceiling_for_mastery(current_mastery: float) -> int:
    """Techo de dificultad aceptable dado el mastery actual del usuario."""
    try:
        m = float(current_mastery)
    except (TypeError, ValueError):
        m = 0.0
    m = max(0.0, min(1.0, m))
    return int(round(m * 5)) + 2  # rango: 2 (m=0) .. 7 (m=1)


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
    output = RoutineOutput(
        title=state.get("title") or f"Routine for {state.get('target_weakness', 'arch')}",
        target_weakness=state.get("target_weakness") or "general software architecture",
        inverse_rag_snippet=snippet or None,
        expected_concepts=expected,
        difficulty=final_difficulty,
        challenge_md=state.get("challenge_md")
        or "## Challenge\n\nNo content was generated.",
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
