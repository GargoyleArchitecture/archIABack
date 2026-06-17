"""F20-T2 — context_loader_node debe permitir retry automático cuando el
fetch de project context / user preferences devuelve vacío o falla.

Antes de F20-T2 se marcaba `project_context_loaded=True` y
`user_style_loaded=True` SIEMPRE, lo que cementaba un fallo transitorio
(network blip, 404, race del api_token) hasta que el usuario hiciera F5.
Tras F20-T2, esos flags se ponen a True solo si la respuesta trajo contenido
útil; los turnos posteriores reintentan automáticamente.
"""
from unittest.mock import patch

from src.graph.nodes.context_loader import context_loader_node
from src.ledger.types import empty_ledger


_PATCH_FETCH_PROJ  = "src.graph.nodes.context_loader.fetch_project_context"
_PATCH_FETCH_PREFS = "src.graph.nodes.context_loader.fetch_user_preferences"
_PATCH_LOAD_LEDGER = "src.graph.nodes.context_loader.load_ledger"
_PATCH_RENDER       = "src.graph.nodes.context_loader.render_dossier"
_PATCH_COMPACT      = "src.graph.nodes.context_loader.render_dossier_compact"
_PATCH_PROMPT       = "src.graph.nodes.context_loader.render_phase_prompt"


def _state_needing_fetch():
    """Estado mínimo donde ambos flags están en False — fuerza el fetch."""
    return {
        "project_id": "proj-1",
        "user_id_for_prefs": "user-1",
        "project_context_loaded": False,
        "user_style_loaded": False,
        "project_context_text": "",
        "user_style_hint": "",
        "language": "es",
        "current_asr": "",
        "quality_attribute": "",
        "style": "",
        "selected_style": "",
        "last_style": "",
        "tactics_struct": [],
        "tactics_list": [],
        "ledger": {},
        "ledger_active": {},
        "design_dossier_md": "",
        "current_phase": "intro",
        "ledger_dossier_compact": "",
        "ledger_phase_prompt": "",
        "ledger_pending_advance": {},
    }


def _run(state):
    empty = empty_ledger("proj-1", "user-1")
    with patch(_PATCH_LOAD_LEDGER, return_value=empty), \
         patch(_PATCH_RENDER, return_value=""), \
         patch(_PATCH_COMPACT, return_value=""), \
         patch(_PATCH_PROMPT, return_value=""):
        return context_loader_node(state, config={"configurable": {"api_token": "tok"}})


def test_project_context_loaded_stays_false_when_fetch_returns_empty():
    """Empty dict from Negocio (404 o sin context) → loaded=False, retry next turn."""
    with patch(_PATCH_FETCH_PROJ, return_value={}), \
         patch(_PATCH_FETCH_PREFS, return_value={"explanationStyle": "ANALOGY", "verbosity": "MEDIUM"}):
        result = _run(_state_needing_fetch())

    assert result["project_context_text"] == ""
    assert result["project_context_loaded"] is False, (
        "Cuando el fetch devuelve vacío, el flag NO debe quedar en True; "
        "si quedara en True, el siguiente turno no reintentaría y el "
        "usuario tendría que hacer F5 (F20-T2)."
    )


def test_project_context_loaded_true_when_fetch_returns_content():
    """Happy path: backend devuelve techStack + businessRules → loaded=True, no más fetches."""
    payload = {"techStack": ["FastAPI", "PostgreSQL"], "businessRules": "Reglas X"}
    with patch(_PATCH_FETCH_PROJ, return_value=payload), \
         patch(_PATCH_FETCH_PREFS, return_value={"explanationStyle": "ANALOGY", "verbosity": "MEDIUM"}):
        result = _run(_state_needing_fetch())

    assert "FastAPI" in result["project_context_text"]
    assert "Reglas X" in result["project_context_text"]
    assert result["project_context_loaded"] is True


def test_project_context_loaded_false_when_fetch_raises():
    """Exception del fetch → loaded=False; siguiente turno reintenta."""
    with patch(_PATCH_FETCH_PROJ, side_effect=Exception("boom")), \
         patch(_PATCH_FETCH_PREFS, return_value={"explanationStyle": "ANALOGY", "verbosity": "MEDIUM"}):
        result = _run(_state_needing_fetch())

    assert result["project_context_text"] == ""
    assert result["project_context_loaded"] is False


def test_user_style_loaded_stays_false_when_fetch_returns_empty():
    """Empty dict from Negocio → user_style_loaded=False, retry next turn."""
    with patch(_PATCH_FETCH_PROJ, return_value={"techStack": ["X"]}), \
         patch(_PATCH_FETCH_PREFS, return_value={}):
        result = _run(_state_needing_fetch())

    assert result["user_style_hint"] == ""
    assert result["user_style_loaded"] is False


def test_user_style_loaded_true_with_defaults_after_f20_t4():
    """Con F20-T4 Negocio devuelve ANALOGY/MEDIUM por defecto → hint no vacío → loaded=True."""
    with patch(_PATCH_FETCH_PROJ, return_value={"techStack": ["X"]}), \
         patch(_PATCH_FETCH_PREFS, return_value={"explanationStyle": "ANALOGY", "verbosity": "MEDIUM"}):
        result = _run(_state_needing_fetch())

    assert "ANALOG" in result["user_style_hint"].upper()
    assert "MEDIUM" in result["user_style_hint"].upper()
    assert result["user_style_loaded"] is True


def test_user_style_loaded_false_when_fetch_raises():
    """Exception del fetch user_preferences → user_style_loaded=False; retry."""
    with patch(_PATCH_FETCH_PROJ, return_value={"techStack": ["X"]}), \
         patch(_PATCH_FETCH_PREFS, side_effect=Exception("net")):
        result = _run(_state_needing_fetch())

    assert result["user_style_hint"] == ""
    assert result["user_style_loaded"] is False
