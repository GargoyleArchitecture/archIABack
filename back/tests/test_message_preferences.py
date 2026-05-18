"""F13-T1 — preferencias por-turno en /message.

Cubre el contrato de precedencia sin levantar el grafo completo:
  1. `format_user_style_hint` (enabler de precedencia): combo válido → hint
     no vacío; inválido/ausente → "" (→ turn_style_from_form=False → fallback).
  2. `context_loader_node`: con `user_style_loaded=True` (sembrado por el
     path Form en main.py) NO se llama a `fetch_user_preferences` (Form gana,
     sin round-trip); con `user_style_loaded=False` SÍ (fallback a Negocio).
"""
from unittest.mock import patch

from src.graph.nodes.context_loader import context_loader_node
from src.services.context_service import format_user_style_hint
from src.ledger.types import empty_ledger


# ── format_user_style_hint — enabler de la precedencia ───────────────────────

def test_valid_combo_produces_non_empty_hint():
    hint = format_user_style_hint({"explanationStyle": "FORMAL", "verbosity": "HIGH"})
    assert hint.startswith("Communication style: ")
    assert hint.endswith(".")


def test_invalid_or_empty_yields_blank_hint_triggers_fallback():
    # main.py normaliza inválidos a "" antes de llamar; el resultado "" es lo
    # que hace turn_style_from_form=False y por tanto se cae al fetch Negocio.
    assert format_user_style_hint({"explanationStyle": "", "verbosity": ""}) == ""
    assert format_user_style_hint({"explanationStyle": "BOGUS", "verbosity": "NOPE"}) == ""
    assert format_user_style_hint({}) == ""


def test_partial_preference_still_produces_hint():
    assert "LOW verbosity" in format_user_style_hint({"explanationStyle": "", "verbosity": "LOW"})


# ── context_loader_node — precedencia Form vs fetch Negocio ──────────────────

_PATCH_FETCH_PREFS = "src.graph.nodes.context_loader.fetch_user_preferences"
_PATCH_FETCH_PROJ  = "src.graph.nodes.context_loader.fetch_project_context"
_PATCH_LOAD        = "src.graph.nodes.context_loader.load_ledger"
_PATCH_RENDER      = "src.graph.nodes.context_loader.render_dossier"
_PATCH_COMPACT     = "src.graph.nodes.context_loader.render_dossier_compact"
_PATCH_PROMPT      = "src.graph.nodes.context_loader.render_phase_prompt"


def _state(**kw):
    base = {
        "project_id": "proj-test",
        "user_id_for_prefs": "user-real-uuid",
        "project_context_loaded": True,   # aísla: no nos interesa el project fetch
        "user_style_loaded": True,
        "user_style_hint": "",
        "language": "es",
        "current_phase": "intro",
    }
    base.update(kw)
    return base


def _run(state):
    ledger_val = empty_ledger("proj-test", "user-real-uuid")
    with patch(_PATCH_FETCH_PREFS, return_value={}) as mock_prefs, \
         patch(_PATCH_FETCH_PROJ, return_value={}), \
         patch(_PATCH_LOAD, return_value=ledger_val), \
         patch(_PATCH_RENDER, return_value="# dossier"), \
         patch(_PATCH_COMPACT, return_value="compact"), \
         patch(_PATCH_PROMPT, return_value="phase_prompt"):
        result = context_loader_node(state, config=None)
    return result, mock_prefs


def test_form_seeded_hint_skips_negocio_fetch():
    """user_style_loaded=True (sembrado por main.py cuando vino por Form):
    el fetch a Negocio NO ocurre y el hint por-turno se preserva."""
    seeded = format_user_style_hint({"explanationStyle": "FORMAL", "verbosity": "HIGH"})
    result, mock_prefs = _run(_state(user_style_loaded=True, user_style_hint=seeded))

    mock_prefs.assert_not_called()
    assert result["user_style_hint"] == seeded


def test_fallback_fetches_negocio_when_not_form_seeded():
    """Sin hint por Form (user_style_loaded=False) con user_id real: se cae
    al fetch a Negocio (fallback intacto, cliente antiguo sin regresión)."""
    result, mock_prefs = _run(_state(user_style_loaded=False, user_style_hint=""))

    mock_prefs.assert_called_once()
    # degradación silenciosa: fetch devolvió {} → hint vacío, loaded marcado.
    assert result["user_style_hint"] == ""
    assert result["user_style_loaded"] is True
