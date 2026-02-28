from src.graph.state import GraphState
from src.graph.qa_registry import normalize_qa
from src.graph.nodes.tactics.common import tactics_node_impl


def tactics_node(state: GraphState) -> GraphState:
    """Nodo general de tácticas, sin forzar QA específico."""
    return tactics_node_impl(state)


def make_tactics_qa_node(qa_id: str):
    """Factory para crear nodos de tácticas especializados por QA."""
    qa_norm = normalize_qa(qa_id)

    def _node(state: GraphState) -> GraphState:
        return tactics_node_impl(state, qa_override=qa_norm)

    _node.__name__ = f"tactics_{qa_norm}_node"
    return _node
