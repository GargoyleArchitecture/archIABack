"""F12-T3: Esquemas Pydantic para la evaluación de intentos de retos.

Pieza del ciclo pedagógico completo (F12):
  Reto generado (F5+F12-T2) → Intento del alumno → **Evaluación (este módulo)**
  → Solución de referencia → Reflexión metacognitiva (F12-T4).

El endpoint `POST /evaluate-attempt` (F12-T3) consume `EvaluateAttemptInput`
y devuelve `RoutineFeedback`. El Backend Negocio (F12-T5, pendiente) será el
único caller en producción; en mocks/Frontend (F12-T6) ya se asume el mismo
shape de respuesta.

Reuso de `RubricCriterion`:
  Importado desde `routine.py` para mantener single source of truth — la
  rúbrica que entrega el subgrafo de generación es la misma que recibe el
  evaluador.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.graph.schemas.routine import RubricCriterion


class ReflectionPayload(BaseModel):
    """Reflexión metacognitiva del alumno (F12-T4).

    El endpoint F12-T3 acepta este campo en `EvaluateAttemptInput` pero NO
    actúa sobre él: T4 añadirá el dispatch al Shadow Agent. Aceptarlo desde
    ya estabiliza el contrato para que T4 sea aditivo.
    """

    difficult_part: str = Field(..., min_length=1, max_length=2000)
    would_do_differently: str = Field(..., min_length=1, max_length=2000)


class EvaluateAttemptInput(BaseModel):
    """Body del endpoint POST /evaluate-attempt.

    Llamado por Backend Negocio (F12-T5) con todos los campos necesarios para
    que IA evalúe sin necesidad de consultar la BD relacional. Esto mantiene
    al evaluador IA stateless respecto a Negocio (excepto por el LLM).
    """

    user_id: str = Field(..., min_length=1)
    routine_id: str = Field(..., min_length=1)
    user_response: str = Field(..., min_length=1, max_length=20000)
    rubric: List[RubricCriterion] = Field(..., min_length=1, max_length=10)
    expected_concepts: List[str] = Field(default_factory=list)
    reference_solution: str = Field(..., min_length=1)
    target_weakness: str = Field(..., min_length=1)
    reflection: Optional[ReflectionPayload] = None
    # F16-T1: id del routine_attempt en Negocio. Opcional (clientes previos
    # no lo envían). Si viene, IA hace sync-back idempotente del feedback a
    # Negocio aunque el HTTP síncrono de Negocio haya expirado.
    attempt_id: Optional[str] = Field(default=None, min_length=1)


class CriterionResult(BaseModel):
    """Resultado del LLM sobre un criterio individual de la rúbrica.

    `concept` debería matchear con el `concept` del `RubricCriterion`
    correspondiente. Si no matchea (LLM ocasionalmente lo desnormaliza), el
    consumidor (RubricCard del Frontend) hace fusión por nombre o lo marca
    como pendiente — no penalizamos al LLM con un 422 por divergencia menor.
    """

    concept: str = Field(..., min_length=1)
    status: Literal["met", "partial", "missing"]
    comment: str = Field(..., min_length=1, max_length=500)


class RoutineFeedback(BaseModel):
    """Output del nodo evaluador. Es lo que devuelve el endpoint serializado.

    El cliente Frontend (RubricCard + FeedbackPanel) consume este shape tal
    cual; el mock determinista de F12-T6 ya replica este contrato.

    Política de defaults:
      - `criteria` puede venir vacía (caso de fallback degradado).
      - `strengths` / `improvements` con `default_factory=list` por la misma
        razón.
      - `score` SIEMPRE entre 0 y 100. El nodo aplica clamp defensivo si el
        LLM devuelve fuera de rango.
    """

    score: int = Field(..., ge=0, le=100)
    criteria: List[CriterionResult] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    socratic_comment: str = Field(..., min_length=1, max_length=1000)
