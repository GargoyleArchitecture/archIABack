"""M1 — Phase gate tests for supervisor_node.

All cases return before any LLM call, so no API key is required.
supervisor_node is synchronous — no asyncio.run needed.
"""

from src.graph.nodes.supervisor import supervisor_node


def _base_state(**overrides):
    defaults = {
        "userQuestion": "test",
        "current_phase": "asr_table",
        "intent": "general",
        "mode": "professional",
        "language": "es",
        "hasVisitedInvestigator": False,
        "hasVisitedEvaluator": False,
        "hasVisitedASR": False,
        "hasVisitedDiagram": False,
        "hasVisitedTech": False,
        "diagram": {},
        "endMessage": "",
        "turn_messages": [],
        "requested_nodes": [],
        "pending_nodes": [],
        "completed_nodes": [],
        "current_asr": "",
        "last_asr": "",
        "messages": [],
        "project_context_text": "",
        "phase_redirect_hint": "",
        "doc_only": False,
        "quality_attribute": "",
        "resolved_index": "",
    }
    defaults.update(overrides)
    return defaults


def test_block_style_when_in_asr_table():
    state = _base_state(current_phase="asr_table", intent="style")
    result = supervisor_node(state)
    assert result["nextNode"] == "unifier"
    assert result["intent"] == "intake"
    assert "selección de estilo" in result["endMessage"]


def test_block_tactics_when_in_asr_table():
    state = _base_state(current_phase="asr_table", intent="tactics")
    result = supervisor_node(state)
    assert result["nextNode"] == "unifier"
    assert result["intent"] == "intake"
    assert "selección de tácticas" in result["endMessage"]


def test_block_tech_when_in_style_table():
    state = _base_state(current_phase="style_table", intent="tech")
    result = supervisor_node(state)
    assert result["nextNode"] == "unifier"
    assert result["intent"] == "intake"
    assert "propuesta de tecnologías" in result["endMessage"]


def test_block_diagram_when_in_asr_table():
    state = _base_state(current_phase="asr_table", intent="diagram")
    result = supervisor_node(state)
    assert result["nextNode"] == "unifier"
    assert result["intent"] == "intake"


def test_funnel_intent_allowed_at_correct_phase():
    state = _base_state(current_phase="style_table", intent="style")
    result = supervisor_node(state)
    assert not (result["nextNode"] == "unifier" and result["intent"] == "intake")


def test_evaluator_free_pass_any_phase():
    state = _base_state(
        current_phase="asr_table",
        intent="architecture",
        userQuestion="evalúa este asr",
    )
    result = supervisor_node(state)
    assert result["nextNode"] == "evaluator"


def test_smalltalk_sets_redirect_hint():
    state = _base_state(current_phase="asr_table", intent="smalltalk")
    result = supervisor_node(state)
    hint = result.get("phase_redirect_hint", "")
    assert hint, "phase_redirect_hint debe ser no vacío para smalltalk con fase activa"
    assert "ASR" in hint or "asr" in hint.lower()
