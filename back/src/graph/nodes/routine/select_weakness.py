"""Nodo `select_weakness` del subgrafo RoutineGenerator (F5-T1).

Decide cuál es la weakness sobre la que se construirá el reto:
1. Si el caller proveyó `state["target_weakness"]`, lo respeta y solo busca
   el `mastery` correspondiente en el perfil (para validate_difficulty).
2. Si no, elige el concepto evaluado con menor `mastery` del perfil.
3. Si el perfil no tiene conceptos, usa fallback genérico
   ("general software architecture") con mastery 0.3.

Los conceptos del perfil traen `mastery` en escala 0..1 (formato interno del
Store; el escalado a 0..100 lo hace `_to_business_payload` solo al sincronizar).
"""
from __future__ import annotations

import logging
from typing import Tuple

from src.graph.schemas.routine import RoutineState

log = logging.getLogger("routine.select_weakness")

_FALLBACK_WEAKNESS = "general software architecture"
_FALLBACK_MASTERY = 0.3


def _pick_weakest(profile: dict) -> Tuple[str, float]:
    """Retorna (name, mastery) del concepto evaluado con menor mastery.

    Si no hay conceptos válidos (lista vacía, o todos malformados), retorna
    el fallback genérico con mastery moderada para no inflar la dificultad.
    """
    concepts = (profile or {}).get("evaluated_concepts") or []
    candidates: list[tuple[str, float]] = []
    for c in concepts:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip()
        if not name:
            continue
        try:
            mastery = float(c.get("mastery") or 0.0)
        except (TypeError, ValueError):
            mastery = 0.0
        candidates.append((name, mastery))

    if not candidates:
        return _FALLBACK_WEAKNESS, _FALLBACK_MASTERY

    # Menor mastery = mayor debilidad → mejor candidato para un reto.
    candidates.sort(key=lambda t: t[1])
    return candidates[0]


def _find_mastery(profile: dict, name: str) -> float:
    """Retorna el `mastery` del concepto `name` en el perfil, o 0.3 si no
    existe (default neutral para que validate_difficulty no penalice a usuarios
    con weakness explícita pero sin perfil aún)."""
    target = (name or "").strip().casefold()
    for c in (profile or {}).get("evaluated_concepts") or []:
        if not isinstance(c, dict):
            continue
        if (c.get("name") or "").strip().casefold() == target:
            try:
                return float(c.get("mastery") or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return _FALLBACK_MASTERY


def select_weakness_node(state: RoutineState) -> RoutineState:
    """Determina target_weakness y target_mastery (escala 0..1)."""
    profile = state.get("user_profile") or {}
    explicit = (state.get("target_weakness") or "").strip()

    if explicit:
        mastery = _find_mastery(profile, explicit)
        log.info(
            "select_weakness: explicit target='%s' mastery=%.2f", explicit, mastery
        )
        return {
            **state,
            "target_weakness": explicit,
            "target_mastery": mastery,
            "regen_count": state.get("regen_count", 0) or 0,
        }

    name, mastery = _pick_weakest(profile)
    log.info(
        "select_weakness: inferred target='%s' mastery=%.2f", name, mastery
    )
    return {
        **state,
        "target_weakness": name,
        "target_mastery": mastery,
        "regen_count": state.get("regen_count", 0) or 0,
    }
