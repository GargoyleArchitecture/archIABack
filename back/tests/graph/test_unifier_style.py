"""F13-T1 — unifier: la verbosidad gobierna la longitud y el estilo es
bloque OBLIGATORIO. Garantiza que `verbosity=HIGH` ya NO es anulado por el
viejo tope hardcodeado "6-12 lines" y que sin preferencia el comportamiento
es idéntico al histórico."""
import asyncio
from unittest.mock import AsyncMock, patch

from src.graph.nodes.unifier import unifier_node, _length_directive
from src.services.context_service import format_user_style_hint

_PATCH_LLM      = "src.graph.nodes.unifier.llm"
_PATCH_MODE     = "src.graph.nodes.unifier.apply_mode_prompt"
_PATCH_FINALIZE = "src.graph.nodes.unifier._finalize_turn"


# ── _length_directive (función pura) ─────────────────────────────────────────

def test_length_directive_high_removes_cap():
    out = _length_directive("Communication style: HIGH verbosity — elaborate.")
    assert "NO length cap" in out
    assert "6-12 lines" not in out


def test_length_directive_low_is_brief():
    out = _length_directive("Communication style: LOW verbosity — minimal.")
    assert "3-5 lines" in out
    assert "6-12 lines" not in out


def test_length_directive_medium_is_historical_text():
    # MEDIUM → texto histórico EXACTO (sin regresión para la mayoría).
    assert _length_directive("x MEDIUM verbosity y") == "- Keep it concise (6-12 lines of content)."


def test_length_directive_no_preference_is_historical_text():
    assert _length_directive("") == "- Keep it concise (6-12 lines of content)."
    assert _length_directive(None) == "- Keep it concise (6-12 lines of content)."


# ── unifier_node — ruta de síntesis por defecto ──────────────────────────────

def _state(**kw):
    base = {
        "language": "en",
        "intent": "general",
        "requested_nodes": [],
        "messages": [],
        "turn_messages": [],
        "user_style_hint": "",
        "project_context_text": "",
        "memory_text": "",
        "userQuestion": "How do I scale my checkout API?",
        "diagram": {},
        "suggestions": [],
        "turn_count_since_eval": 0,
    }
    base.update(kw)
    return base


def _run(state):
    with patch(_PATCH_LLM) as ml, \
         patch(_PATCH_MODE, side_effect=lambda s, p: p), \
         patch(_PATCH_FINALIZE, side_effect=lambda s, t: s):
        ml.ainvoke = AsyncMock(return_value=type("R", (), {"content": "## A\n\nok"})())
        out = asyncio.run(unifier_node(state))
        prompt = ml.ainvoke.call_args[0][0]
    return out, prompt


def test_high_verbosity_removes_cap_and_adds_mandatory_block():
    hint = format_user_style_hint({"explanationStyle": "ANALOGY", "verbosity": "HIGH"})
    _, prompt = _run(_state(user_style_hint=hint))
    assert "6-12 lines of content" not in prompt
    assert "NO length cap" in prompt
    assert "=== COMMUNICATION STYLE (MANDATORY) ===" in prompt
    assert hint in prompt


def test_medium_verbosity_keeps_balanced_rule():
    hint = format_user_style_hint({"explanationStyle": "FORMAL", "verbosity": "MEDIUM"})
    _, prompt = _run(_state(user_style_hint=hint))
    assert "Keep it concise (6-12 lines of content)." in prompt
    assert "=== COMMUNICATION STYLE (MANDATORY) ===" in prompt


def test_no_preference_is_unchanged_default():
    _, prompt = _run(_state(user_style_hint=""))
    assert "Keep it concise (6-12 lines of content)." in prompt
    assert "COMMUNICATION STYLE" not in prompt


def test_low_verbosity_sets_very_brief():
    hint = format_user_style_hint({"explanationStyle": "CONCISE", "verbosity": "LOW"})
    _, prompt = _run(_state(user_style_hint=hint))
    assert "3-5 lines" in prompt
    assert "6-12 lines of content" not in prompt


def test_es_mandatory_block_is_localized():
    hint = format_user_style_hint({"explanationStyle": "ANALOGY", "verbosity": "HIGH"})
    _, prompt = _run(_state(language="es", user_style_hint=hint))
    assert "=== ESTILO DE COMUNICACIÓN (OBLIGATORIO) ===" in prompt
