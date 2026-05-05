"""Tests del helper apply_mode_prompt (F2-T3) sin LLM."""
from src.graph.prompts.mode_prompts import (
    TUTOR_SYSTEM_PROMPT,
    PROFESSIONAL_SYSTEM_PROMPT,
    apply_mode_prompt,
)


def test_tutor_prompt_has_socratic_marker():
    assert "?" in TUTOR_SYSTEM_PROMPT
    assert "socratic" in TUTOR_SYSTEM_PROMPT.lower()


def test_professional_prompt_has_action_marker():
    txt = PROFESSIONAL_SYSTEM_PROMPT.lower()
    assert "directa" in txt or "directo" in txt
    assert "codigo" in txt or "comando" in txt or "diagrama" in txt


def test_apply_mode_prefixes_tutor():
    base = "BASE_NODE_PROMPT"
    out = apply_mode_prompt({"mode": "tutor"}, base)
    assert out.startswith(TUTOR_SYSTEM_PROMPT)
    assert out.endswith(base)


def test_apply_mode_prefixes_professional():
    base = "BASE_NODE_PROMPT"
    out = apply_mode_prompt({"mode": "professional"}, base)
    assert out.startswith(PROFESSIONAL_SYSTEM_PROMPT)
    assert out.endswith(base)


def test_apply_mode_default_safe_when_missing():
    base = "BASE_NODE_PROMPT"
    assert apply_mode_prompt({}, base) == base
    assert apply_mode_prompt({"mode": None}, base) == base
    assert apply_mode_prompt({"mode": "weird"}, base) == base


def test_apply_mode_does_not_mutate_state():
    state = {"mode": "tutor"}
    apply_mode_prompt(state, "X")
    assert state == {"mode": "tutor"}


def test_apply_mode_only_one_prefix_per_invocation():
    """Verifica que invocar dos veces NO acumula el prefijo en el mismo string."""
    base = "BASE"
    once = apply_mode_prompt({"mode": "tutor"}, base)
    twice = apply_mode_prompt({"mode": "tutor"}, base)
    # Cada invocacion produce el mismo string; el prefijo no se acumula
    # porque siempre opera sobre `base`, no sobre el resultado anterior.
    assert once == twice
    assert once.count(TUTOR_SYSTEM_PROMPT) == 1
