"""Nodo `synthesize_challenge` del subgrafo RoutineGenerator (F5-T1).

Llama al LLM con `with_structured_output(RoutineOutput)` para producir el
reto. Recibe la weakness, el snippet del RAG inverso (puede ser vacío) y
una directriz opcional de "menos conceptos" cuando viene un regen del
validator.

`method="function_calling"` por consistencia con el Shadow Agent (F3-T2):
evita el modo `json_schema` de OpenAI que exige todos los campos requeridos
y rompe con `default_factory`.

F12-T2 amplía el contrato:
- El prompt instruye al LLM a producir `rubric` (3..5 criterios) y
  `reference_solution` (Markdown).
- El fallback de error construye una rúbrica genérica de 3 ítems y una
  solución de referencia neutra, garantizando que cualquier `RoutineOutput`
  emitido cumpla la validación Pydantic extendida.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.graph.resources import llm
from src.graph.schemas.routine import RoutineOutput, RoutineState, RubricCriterion

log = logging.getLogger("routine.synthesize_challenge")


_SYSTEM_PROMPT = """\
You are a software architecture coach. Generate a coding challenge that
helps a learner overcome a specific weakness. Your output MUST follow the
RoutineOutput schema exactly.

Requirements for the challenge:
- `title`: short, action-oriented (max 200 chars).
- `target_weakness`: echo the input weakness verbatim.
- `expected_concepts`: 2-5 architectural concepts the learner must apply.
- `difficulty`: integer 1..5. Self-assessed by you; the validator may adjust.
- `challenge_md`: markdown body of the exercise. Include:
  * Brief context (2-3 sentences).
  * Concrete deliverable (what to implement / refactor).
  * 2-3 acceptance criteria.
- `inverse_rag_snippet`: COPY the snippet from the input verbatim if non-empty;
  otherwise leave null.

- `rubric`: 3 to 5 evaluation criteria. Each entry MUST include:
    * `concept`: which expected_concept the criterion targets (use one of them
      or a closely related sub-concept).
    * `description`: one sentence (10+ chars) explaining what the learner
      must demonstrate to satisfy the criterion.
    * `weight`: 1 (nice-to-have) .. 5 (must-have).
  The sum of weights across the rubric is not required to total any
  specific number; the downstream evaluator (F12-T3) handles weighting.

  Example for a Caching challenge:
    [
      {"concept": "LRU", "description": "Implements eviction by least-recently-used order.", "weight": 5},
      {"concept": "TTL", "description": "Supports per-entry time-to-live expiration.", "weight": 3},
      {"concept": "Thread safety", "description": "Concurrent reads do not corrupt internal state.", "weight": 4}
    ]

- `reference_solution`: Markdown showing a model solution (code block + 1-2
  short paragraphs of commentary). MUST be at least 20 characters. The
  platform hides this until the learner submits their first attempt, so
  DO NOT reference its existence inside `challenge_md`.

If a "REDUCE_SCOPE" instruction is present, restrict to <= 3 expected_concepts
AND emit exactly 3 rubric criteria (the most essential ones).
"""


# Fallback genérico para el caso de error del LLM. Tres criterios siempre
# válidos (cumple `min_length=3` de `RoutineOutput.rubric`).
def _fallback_rubric(weakness: str) -> list[RubricCriterion]:
    safe_weakness = weakness or "software architecture"
    return [
        RubricCriterion(
            concept=safe_weakness,
            description=(
                "The submission addresses the named weakness with a concrete "
                "and explainable change."
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
                "The refactor compiles or runs in the target language and "
                "does not introduce obvious regressions."
            ),
            weight=4,
        ),
    ]


_FALLBACK_REFERENCE = (
    "## Reference\n\nA tailored reference solution was not generated for "
    "this exercise. Apply a minimal viable refactor that resolves the named "
    "weakness and explain your reasoning in a short comment."
)


def _build_user_prompt(
    weakness: str,
    raw_snippet: str,
    current_mastery: float,
    reduce_scope: bool,
) -> str:
    parts = [
        f"WEAKNESS: {weakness}",
        f"LEARNER_MASTERY (0..1): {current_mastery:.2f}",
    ]
    if raw_snippet:
        parts.append(f"BAD_CODE_SNIPPET (RAG):\n{raw_snippet}")
    else:
        parts.append("BAD_CODE_SNIPPET (RAG): (empty)")
    if reduce_scope:
        parts.append(
            "REDUCE_SCOPE: previous attempt was too hard; restrict to "
            "<= 3 expected_concepts, emit exactly 3 rubric criteria, and "
            "pick a simpler difficulty."
        )
    return "\n\n".join(parts)


async def synthesize_challenge_node(
    state: RoutineState, llm_obj: Optional[object] = None
) -> RoutineState:
    """Llama al LLM y guarda los campos en el state.

    `llm_obj` permite inyectar un mock en tests.
    """
    target_llm = llm_obj if llm_obj is not None else llm
    structured = target_llm.with_structured_output(
        RoutineOutput, method="function_calling"
    )

    weakness = (state.get("target_weakness") or "").strip()
    raw_snippet = state.get("raw_snippet") or ""
    current_mastery = float(state.get("target_mastery") or 0.0)
    regen_count = int(state.get("regen_count", 0) or 0)
    reduce_scope = regen_count >= 1  # cualquier regen debe reducir

    user_prompt = _build_user_prompt(
        weakness, raw_snippet, current_mastery, reduce_scope
    )

    log.info(
        "synthesize_challenge invoking LLM weakness='%s' regen=%d reduce_scope=%s",
        weakness, regen_count, reduce_scope,
    )

    try:
        result: RoutineOutput = await structured.ainvoke(
            f"{_SYSTEM_PROMPT}\n\n{user_prompt}"
        )
    except Exception as exc:
        # Si el LLM falla, devolvemos un payload mínimo para que el grafo
        # no se rompa. El validator decidirá si lo acepta.
        # El fallback DEBE cumplir la validación Pydantic extendida en F12-T2.
        log.exception("synthesize_challenge LLM call failed: %s", exc)
        result = RoutineOutput(
            title=f"Refactor exercise on {weakness or 'software architecture'}",
            target_weakness=weakness or "general software architecture",
            inverse_rag_snippet=raw_snippet or None,
            expected_concepts=[weakness] if weakness else ["software architecture"],
            difficulty=3,
            challenge_md=(
                "## Challenge\n\nThe AI couldn't generate a tailored exercise. "
                "Try producing a small refactor that addresses the named weakness."
            ),
            rubric=_fallback_rubric(weakness),
            reference_solution=_FALLBACK_REFERENCE,
        )

    return {
        **state,
        "title": result.title,
        "challenge_md": result.challenge_md,
        "expected_concepts": list(result.expected_concepts or []),
        "difficulty": int(result.difficulty),
        "raw_snippet": result.inverse_rag_snippet or raw_snippet,
        "rubric": [c.model_dump() for c in result.rubric],
        "reference_solution": result.reference_solution,
    }
