"""Esquemas Pydantic para perfilado de usuario.

Reutilizado por:
- GraphState.user_profile (F2-T1) - tipado documental de la estructura esperada.
- Shadow Agent / Profile Evaluator (F3-T2) - serializacion con structured_output.
- Cliente HTTP profile_sync (F3-T5) - payload hacia el Backend Negocio.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ConceptScore(BaseModel):
    """Concepto evaluado con su nivel de dominio (0..1)."""

    name: str = Field(..., description="Nombre canonico del concepto.")
    mastery: float = Field(..., ge=0.0, le=1.0, description="Dominio en [0,1].")
    evidence: Optional[str] = Field(
        None, description="Cita corta del turno que respalda el score."
    )


class UserProfileSchema(BaseModel):
    """Perfil tecnico del usuario, snapshot del ultimo turno evaluado."""

    user_id: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    evaluated_concepts: List[ConceptScore] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("strengths", "weaknesses")
    @classmethod
    def _strip_empties(cls, v: List[str]) -> List[str]:
        return [s for s in (v or []) if s and s.strip()]


class UserProfileEvaluation(BaseModel):
    """Salida de una evaluacion del Shadow Agent en un turno (F3-T1).

    Diferente de `UserProfileSchema` (que es el perfil persistido). Esta
    estructura es la que produce el evaluador via `llm.with_structured_output`
    y se mergea contra el perfil previo antes de escribir al Store / Negocio.

    Ejemplo:
        UserProfileEvaluation(
            strengths=[ConceptScore(name="Modularidad", mastery=0.85)],
            weaknesses=[ConceptScore(name="Caching", mastery=0.30)],
            evaluated_concepts=[
                ConceptScore(name="Modularidad", mastery=0.85, evidence="..."),
                ConceptScore(name="Caching", mastery=0.30, evidence="..."),
            ],
            confidence=0.7,
            delta_from_previous={"Modularidad": 0.15, "Caching": -0.05},
        )
    """

    strengths: List[ConceptScore] = Field(default_factory=list)
    weaknesses: List[ConceptScore] = Field(default_factory=list)
    evaluated_concepts: List[ConceptScore] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    delta_from_previous: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_empty_when_confident(self):
        """Si la confianza es >= 0.5, al menos una de las listas debe traer
        contenido. Una eval con alta confianza pero todo vacio es ruido."""
        if self.confidence >= 0.5:
            if not (self.strengths or self.weaknesses or self.evaluated_concepts):
                raise ValueError(
                    "When confidence >= 0.5, at least one of "
                    "strengths/weaknesses/evaluated_concepts must be non-empty."
                )
        return self
