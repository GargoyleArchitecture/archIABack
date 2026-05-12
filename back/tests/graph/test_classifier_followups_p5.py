from unittest.mock import patch

from src.graph.nodes.classifier import classifier_node


def test_style_followup_preserves_previous_qa_when_message_has_no_new_qa(base_state):
    state = {
        **base_state,
        "userQuestion": "Propón estilos de arquitectura para cumplir con el ASR",
        "current_asr": "ASR de latencia vigente",
        "last_asr": "ASR de latencia vigente",
        "quality_attribute": "latencia",
    }

    with patch(
        "src.graph.nodes.classifier._classify_cached",
        return_value=("es", "style", True, "disponibilidad"),
    ):
        result = classifier_node(state)

    assert result["intent"] == "style"
    assert result["quality_attribute"] == "latencia"
    assert result["resolved_index"] == "latencia"


def test_style_followup_allows_explicit_qa_change(base_state):
    state = {
        **base_state,
        "userQuestion": "Propón estilos de arquitectura enfocados en disponibilidad para cumplir con el ASR",
        "current_asr": "ASR de latencia vigente",
        "last_asr": "ASR de latencia vigente",
        "quality_attribute": "latencia",
    }

    with patch(
        "src.graph.nodes.classifier._classify_cached",
        return_value=("es", "style", True, "disponibilidad"),
    ):
        result = classifier_node(state)

    assert result["intent"] == "style"
    assert result["quality_attribute"] == "disponibilidad"
    assert result["resolved_index"] == "disponibilidad"
