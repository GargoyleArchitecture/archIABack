"""Tests F11-T6: src.services.telemetry"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

# Permite ejecutar `pytest back/tests/test_telemetry.py` desde la raíz del repo
BACK_DIR = Path(__file__).resolve().parents[1]
if str(BACK_DIR) not in sys.path:
    sys.path.insert(0, str(BACK_DIR))

from src.services.telemetry import emit  # noqa: E402


def _capture_logs(caplog):
    """Devuelve la lista de records del logger de telemetría."""
    return [
        r for r in caplog.records if r.name == "archia.telemetry"
    ]


def test_emit_loggea_json_con_event_y_timestamp(caplog):
    caplog.set_level(logging.INFO, logger="archia.telemetry")
    rec = emit("mode_suggested", user_id="u-1", suggestion="tutor", confidence=0.82)

    assert rec["event"] == "mode_suggested"
    assert "timestamp_utc" in rec
    # El timestamp ISO debe terminar en Z (UTC) y tener forma estándar.
    assert rec["timestamp_utc"].endswith("Z")
    assert "T" in rec["timestamp_utc"]

    logs = _capture_logs(caplog)
    assert len(logs) == 1
    # El mensaje contiene la representación JSON serializable.
    assert "mode_suggested" in logs[0].getMessage()
    # Round-trip JSON: el segundo segmento del mensaje es JSON parseable.
    json_segment = logs[0].getMessage().split("telemetry ", 1)[1]
    parsed = json.loads(json_segment)
    assert parsed["event"] == "mode_suggested"
    assert parsed["user_id"] == "u-1"
    assert parsed["suggestion"] == "tutor"
    assert parsed["confidence"] == 0.82


def test_emit_event_invalido_retorna_vacio_y_no_loggea(caplog):
    caplog.set_level(logging.INFO, logger="archia.telemetry")
    assert emit("") == {}
    assert emit("   ") == {}
    assert emit(None) == {}  # type: ignore[arg-type]
    assert emit(42) == {}  # type: ignore[arg-type]
    logs = _capture_logs(caplog)
    assert logs == []


def test_emit_descarta_payload_none_y_serializa_no_json(caplog):
    caplog.set_level(logging.INFO, logger="archia.telemetry")

    class _NotJsonable:
        def __repr__(self) -> str:
            return "<custom-object>"

    rec = emit(
        "weird_event",
        user_id=None,         # descartado
        extra=_NotJsonable(),  # convertido a string
        ok_field=123,
    )
    # user_id descartado.
    assert "user_id" not in rec
    # extra coerced a string.
    assert rec["extra"] == "<custom-object>"
    assert rec["ok_field"] == 123


def test_emit_strip_whitespace_del_event_name():
    rec = emit("   mode_changed   ")
    assert rec["event"] == "mode_changed"
