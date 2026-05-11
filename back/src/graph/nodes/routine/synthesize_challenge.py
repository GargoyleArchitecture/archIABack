"""Nodo `synthesize_challenge` del subgrafo RoutineGenerator (F5-T1).

Llama al LLM con `with_structured_output(RoutineOutput)` para producir el
reto. Recibe la weakness, el snippet del RAG inverso (puede ser vacío) y
una directriz opcional de "menos conceptos" cuando viene un regen del
validator.

`method="function_calling"` por consistencia con el Shadow Agent (F3-T2):
evita el modo `json_schema` de OpenAI que exige todos los campos requeridos
y rompe con `default_factory`.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.graph.resources import llm
from src.graph.schemas.routine import RoutineOutput, RoutineState

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

If a "REDUCE_SCOPE" instruction is present, restrict to <= 3 concepts.
"""


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
            "<= 3 expected_concepts and pick a simpler difficulty."
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
        log.exception("synthesize_challenge LLM call failed: %s", exc)
        result = RoutineOutput(
            title=f"Refactor exercise on {weakness}",
            target_weakness=weakness or "general software architecture",
            inverse_rag_snippet=raw_snippet or None,
            expected_concepts=[weakness] if weakness else ["software architecture"],
            difficulty=3,
            challenge_md=(
                "## Challenge\n\nThe AI couldn't generate a tailored exercise. "
                "Try producing a small refactor that addresses the named weakness."
            ),
        )

    return {
        **state,
        "title": result.title,
        "challenge_md": result.challenge_md,
        "expected_concepts": list(result.expected_concepts or []),
        "difficulty": int(result.difficulty),
        "raw_snippet": result.inverse_rag_snippet or raw_snippet,
    }
