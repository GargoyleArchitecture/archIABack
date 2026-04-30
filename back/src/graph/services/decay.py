"""Curva de olvido (Ebbinghaus simplificada) para perfilado (F3-T4).

Formula: mastery_t = mastery_0 * exp(-decay_rate * delta_days)

- El Store guarda `mastery_0` (observacion mas reciente del Shadow Agent).
- La hidratacion en `boot_node` aplica decay sobre la copia que va al
  `state.user_profile`. El valor original NUNCA se sobrescribe en el Store.
- `decay_rate` por concepto override > env DECAY_RATE_DEFAULT (default 0.05).
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Optional

DECAY_RATE_DEFAULT = float(os.getenv("DECAY_RATE_DEFAULT", "0.05"))


def _parse_iso(value) -> Optional[datetime]:
    """Parsea ISO 8601 a datetime UTC. Devuelve None si invalido."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        s = str(value).rstrip("Z")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def apply_forgetting_curve(
    concept: dict,
    now: Optional[datetime] = None,
    default_decay_rate: float = DECAY_RATE_DEFAULT,
) -> dict:
    """Funcion pura: devuelve un dict NUEVO con `mastery` decayed.

    - Conserva todos los campos originales (incluyendo `last_seen_at`).
    - Anade `mastery_original` (referencia inmutable a mastery_0).
    - Si `last_seen_at` falta o es invalido, no aplica decay.
    - `delta_days` se trunca a 0 si es negativo (lecturas en el pasado / clock skew).
    """
    out = dict(concept or {})
    try:
        mastery_0 = float(out.get("mastery") or 0.0)
    except (TypeError, ValueError):
        mastery_0 = 0.0
    out["mastery_original"] = mastery_0

    last_seen = _parse_iso(out.get("last_seen_at") or out.get("lastSeenAt"))
    if last_seen is None:
        return out

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    delta_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
    try:
        rate = float(out.get("decay_rate") or default_decay_rate)
    except (TypeError, ValueError):
        rate = default_decay_rate

    decayed = mastery_0 * math.exp(-rate * delta_days)
    out["mastery"] = round(decayed, 4)
    out["delta_days"] = round(delta_days, 4)
    return out


def apply_decay_to_profile(
    profile: dict,
    now: Optional[datetime] = None,
    default_decay_rate: float = DECAY_RATE_DEFAULT,
) -> dict:
    """Aplica `apply_forgetting_curve` a cada elemento de `evaluated_concepts`.

    Devuelve una COPIA del perfil con conceptos decayed; no muta el original.
    """
    if not profile:
        return profile or {}
    out = dict(profile)
    concepts = profile.get("evaluated_concepts") or []
    out["evaluated_concepts"] = [
        apply_forgetting_curve(c, now=now, default_decay_rate=default_decay_rate)
        for c in concepts
    ]
    return out
