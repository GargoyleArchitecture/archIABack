"""Esquemas para el subgrafo RoutineGenerator (F5-T1, extendido en F12-T2).

Artefactos:
- `RubricCriterion` (Pydantic): criterio individual de la rúbrica pedagógica.
- `RoutineOutput` (Pydantic): output estructurado del subgrafo (lo produce el
  LLM en `synthesize_challenge` vía `with_structured_output`). Es también el
  shape final que el endpoint `POST /generate-routine` (F5-T2) devuelve
  serializado al Backend Negocio.
- `RoutineState` (TypedDict): estado interno que LangGraph propaga entre nodos.

F12-T2 añade dos campos pedagógicos a `RoutineOutput`:
- `rubric`: 3..5 criterios estructurados que el evaluador (F12-T3) usa para
  producir `RoutineFeedback`.
- `reference_solution`: solución modelo en Markdown. El Backend Negocio la
  ocultará hasta el primer attempt evaluado (anti-spoiler, F12-T5).
"""
from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class RubricCriterion(BaseModel):
    """Criterio individual de la rúbrica pedagógica (F12-T2).

    El evaluador (F12-T3) consume estos criterios y emite un `CriterionResult`
    por cada uno con status `met | partial | missing`. La suma de `weight` NO
    está normalizada; el evaluador hace el cálculo de score ponderado.
    """

    concept: str = Field(..., min_length=1, max_length=80)
    description: str = Field(..., min_length=10, max_length=400)
    weight: int = Field(..., ge=1, le=5)


class RoutineOutput(BaseModel):
    """Output estructurado y validable del subgrafo de generación.

    Campos alineados con el contrato OpenAPI `Routine` (los que produce el
    Backend IA; `id`/`userId` los pone Negocio al persistir).

    `difficulty` se entrega siempre clamp-eada al rango 1..5.

    Campos pedagógicos F12-T2:
    - `rubric`: lista de 3..5 `RubricCriterion`.
    - `reference_solution`: Markdown con la solución modelo (min 20 chars).
    """

    title: str = Field(..., min_length=5, max_length=200)
    target_weakness: str = Field(..., min_length=1)
    inverse_rag_snippet: Optional[str] = Field(
        None,
        description="Snippet de mal código del RAG inverso (F5-T3); puede ser None si la búsqueda no produjo hits.",
    )
    expected_concepts: List[str] = Field(default_factory=list)
    difficulty: int = Field(..., ge=1, le=5)
    challenge_md: str = Field(..., min_length=20)

    # ---- F12-T2: campos pedagógicos ----
    rubric: List[RubricCriterion] = Field(
        ...,
        min_length=3,
        max_length=5,
        description=(
            "Rúbrica explícita: 3-5 criterios que el evaluador usa para "
            "producir RoutineFeedback (F12-T3)."
        ),
    )
    reference_solution: str = Field(
        ...,
        min_length=20,
        description=(
            "Solución modelo en Markdown. Negocio la oculta hasta el primer "
            "attempt evaluado (anti-spoiler F12-T5)."
        ),
    )


class RoutineState(TypedDict, total=False):
    """Estado interno del subgrafo. `total=False` permite construir parcialmente
    el state durante el flujo (cada nodo agrega sus campos).

    Campos de entrada (los populamos antes de invocar el grafo):
    - user_id: identidad estable del usuario.
    - user_profile: perfil hidratado por el endpoint (decay aplicado).
    - target_weakness: opcional; si viene, `select_weakness` lo respeta.

    Campos producidos por los nodos:
    - target_mastery: escala 0..1 del concepto seleccionado (lo usa validate_difficulty).
    - raw_snippet: salida de inverse_rag_search.
    - title, challenge_md, expected_concepts, difficulty: salida de synthesize_challenge.
    - rubric: lista de dicts {concept, description, weight} (serializable en LangGraph state).
      `validate_difficulty` la convierte a `List[RubricCriterion]` al construir el `final`.
    - reference_solution: Markdown con la solución modelo.
    - regen_count: contador del loop limitado en validate_difficulty (max 2).
    - final: RoutineOutput completo cuando el flujo concluye OK.
    """

    user_id: str
    user_profile: dict
    target_weakness: Optional[str]
    target_mastery: float
    raw_snippet: str
    title: str
    challenge_md: str
    expected_concepts: List[str]
    difficulty: int
    rubric: List[dict]
    reference_solution: str
    regen_count: int
    final: Optional[RoutineOutput]
