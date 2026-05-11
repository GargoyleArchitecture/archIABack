"""Esquemas para el subgrafo RoutineGenerator (F5-T1).

Dos artefactos:
- `RoutineState`: TypedDict del estado interno que LangGraph propaga entre nodos.
- `RoutineOutput`: Pydantic Model que produce el LLM en `synthesize_challenge`
  via `with_structured_output`. Es también el shape final que el endpoint
  `POST /generate-routine` (F5-T2) devolverá serializado al Backend Negocio.
"""
from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class RoutineOutput(BaseModel):
    """Output estructurado y validable del subgrafo de generación.

    Campos alineados con el contrato OpenAPI `Routine` (los que produce el
    Backend IA; `id`/`userId` los pone Negocio al persistir).

    `difficulty` se entrega siempre clamp-eada al rango 1..5.
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
    regen_count: int
    final: Optional[RoutineOutput]
