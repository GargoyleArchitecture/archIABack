"""F11-T6: Emisor de eventos de telemetría — Backend IA

Loggea eventos estructurados a stdout en formato JSON. La cosecha en
infra real (Loki, CloudWatch, Datadog) recoge esos logs y los presenta
en dashboards (descrito en `docs/observability.md`).

Diseño:
    - No realiza I/O HTTP. El backend de Negocio recibe la copia que
      manda el Frontend cuando aplica; nuestro lado simplemente loggea
      con un schema consistente.
    - Cero acoplamiento con módulos del grafo: importable desde
      cualquier nodo sin riesgo circular.
    - `emit(event, **payload)` produce una línea JSON con keys
      reservadas `event`, `timestamp_utc` y todo lo demás merged.

Convenciones de schema:
    event:          str   — nombre canónico (snake_case).
    user_id:        str?  — id del usuario (si aplica).
    timestamp_utc:  str   — ISO 8601 con sufijo Z (UTC).
    cualquier otra key del payload se incluye como sibling.

Ejemplos de uso:
    >>> from src.services.telemetry import emit
    >>> emit("mode_suggested", user_id="u-1", suggestion="tutor", confidence=0.82)
    >>> emit("routine_generated", user_id="u-1", routine_id="r-9", difficulty=3)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

_TELEMETRY_LOGGER_NAME = "archia.telemetry"
_logger = logging.getLogger(_TELEMETRY_LOGGER_NAME)


def _now_iso() -> str:
    """Timestamp ISO-8601 en UTC con sufijo Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def emit(event: str, **payload: Any) -> dict[str, Any]:
    """Emite un evento de telemetría como log estructurado JSON.

    Args:
        event: Nombre canónico del evento (snake_case). Si no es str o
            está vacío, no se loggea y retorna un dict vacío.
        **payload: Campos adicionales del evento (user_id, etc.). Solo
            valores JSON-serializables; los que no lo sean se convierten
            con `str(value)`.

    Returns:
        El dict del registro emitido (útil para tests y debugging).
        Retorna `{}` si el `event` es inválido.
    """
    if not isinstance(event, str) or not event.strip():
        return {}

    record: dict[str, Any] = {
        "event": event.strip(),
        "timestamp_utc": _now_iso(),
    }

    for key, value in payload.items():
        if value is None:
            continue
        try:
            json.dumps(value)
            record[key] = value
        except (TypeError, ValueError):
            record[key] = str(value)

    # Log como JSON estructurado en una sola línea — los recolectores
    # estándar (fluentd, vector, loki) lo parsean nativamente.
    try:
        _logger.info("telemetry %s", json.dumps(record, ensure_ascii=False))
    except (TypeError, ValueError):
        # Defensa extra ante valores inesperados.
        _logger.info("telemetry %s", str(record))

    return record
