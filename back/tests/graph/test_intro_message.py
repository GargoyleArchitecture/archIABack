import asyncio
from unittest.mock import AsyncMock, patch

from src.graph.nodes.intake_node import intake_node
from src.graph.nodes.intake_validators import INTAKE_SCRIPT


def _state(**overrides):
    base = {
        "userQuestion": "hola",
        "language": "es",
        "intake_fields": {},
        "intake_current_field": 0,
        "intake_complete": False,
        "current_phase": "intro",
        "user_id_for_prefs": "",   # evita llamada al ledger
        "project_id": "",
        "project_context_text": "",
    }
    return {**base, **overrides}


# Test 1: primer turno con current_phase="intro" → intro ADD 3.0 + primera pregunta
def test_intro_emitted_on_first_turn():
    result = asyncio.run(intake_node(_state()))
    msg = result["endMessage"]
    assert "ADD 3.0" in msg
    assert INTAKE_SCRIPT[0]["question_es"] in msg
    assert result["intake_current_field"] == 0
    assert result["intake_complete"] is False


# Test 2: turno siguiente con current_phase="diagnosis" → NO contiene intro
def test_intro_not_repeated_on_diagnosis():
    # None triggers fail-open path → saved=[], failed=[] → welcome message without intro
    with patch(
        "src.graph.nodes.intake_node.extract_and_validate_fields",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = asyncio.run(intake_node(_state(current_phase="diagnosis")))
    assert "ADD 3.0" not in result["endMessage"]
    assert "SEI" not in result["endMessage"]


# Test 3: primer turno con language="en" → respuesta en inglés
def test_intro_english_when_language_en():
    result = asyncio.run(intake_node(_state(language="en")))
    msg = result["endMessage"]
    assert "ADD 3.0" in msg
    assert INTAKE_SCRIPT[0]["question_en"] in msg
    assert "Soy ArchIA" not in msg
