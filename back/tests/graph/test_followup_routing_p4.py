from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.graph.nodes.asr import asr_node
from src.graph.nodes.investigator import researcher_node
from src.graph.nodes.supervisor import _infer_requested_nodes
from src.graph.utils import is_explicit_asr_request


def test_followup_mentioning_existing_asr_is_not_explicit_asr_request():
    assert not is_explicit_asr_request(
        "Propón estilos de arquitectura para cumplir con el ASR"
    )


def test_realign_request_counts_as_explicit_asr_request():
    assert is_explicit_asr_request(
        "Replantea el ASR porque no está alineado con mi problema"
    )


def test_style_followup_with_existing_asr_only_requests_style(base_state):
    state = {
        **base_state,
        "current_asr": "ASR vigente",
        "last_asr": "ASR vigente",
    }

    plan = _infer_requested_nodes(
        "Propón estilos de arquitectura para cumplir con el ASR",
        state,
        forced="style",
    )

    assert plan == ["style"]


def test_style_followup_without_existing_asr_requests_asr_then_style(base_state):
    plan = _infer_requested_nodes(
        "Propón estilos de arquitectura para cumplir con el ASR",
        base_state,
        forced="style",
    )

    assert plan == ["asr", "style"]


def test_explicit_realign_request_keeps_asr_in_plan(base_state):
    state = {
        **base_state,
        "current_asr": "ASR vigente",
        "last_asr": "ASR vigente",
    }

    plan = _infer_requested_nodes(
        "Replantea el ASR porque no está alineado con mi problema",
        state,
        forced="asr",
    )

    assert plan == ["asr"]


def test_asr_node_preserves_existing_asr_for_non_asr_followup():
    state = {
        "language": "es",
        "userQuestion": "Propón estilos de arquitectura para cumplir con el ASR",
        "doc_only": False,
        "doc_context": "",
        "current_asr": "ASR vigente",
        "last_asr": "ASR vigente",
        "requested_nodes": ["asr", "style"],
        "pending_nodes": ["style"],
        "endMessage": "",
        "nextNode": "asr",
    }

    result = asr_node(state)

    assert result["current_asr"] == "ASR vigente"
    assert result["last_asr"] == "ASR vigente"
    assert result["requested_nodes"] == ["style"]
    assert result["pending_nodes"] == ["style"]


def test_researcher_node_keeps_only_final_useful_answer():
    class DummyAgent:
        def invoke(self, payload, config=None):
            return {
                "messages": [
                    SystemMessage(content="system prompt"),
                    HumanMessage(content="Start by calling the tool `local_RAG` with the user's question."),
                    AIMessage(content="", tool_calls=[]),
                    ToolMessage(
                        content="[1] snippet\n\nSOURCES:\n- Doc A\n- Doc B",
                        tool_call_id="tool-1",
                    ),
                    AIMessage(content="Respuesta final útil para el usuario."),
                ]
            }

    state = {
        "language": "es",
        "intent": "architecture",
        "force_rag": True,
        "doc_only": False,
        "doc_context": "",
        "project_context_text": "",
        "user_style_hint": "",
        "add_context": "",
        "messages": [],
        "turn_messages": [],
        "userQuestion": "Explícame este patrón",
        "localQuestion": "",
        "imagePath1": "",
        "imagePath2": "",
        "diagram": {},
    }

    with patch("src.graph.nodes.investigator.create_react_agent", return_value=DummyAgent()):
        result = researcher_node(state)

    researcher_msgs = [
        m for m in result["messages"]
        if isinstance(m, AIMessage) and getattr(m, "name", None) == "researcher"
    ]
    assert len(researcher_msgs) == 1
    assert "Respuesta final útil para el usuario." in researcher_msgs[0].content
    assert "Start by calling the tool" not in researcher_msgs[0].content
    assert result["add_context"] == "Respuesta final útil para el usuario."
