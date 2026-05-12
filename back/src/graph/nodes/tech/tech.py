from src.graph.state import GraphState
from src.graph.qa_registry import normalize_qa
from src.graph.nodes.tech.common import tech_node_impl


def tech_node(state: GraphState) -> GraphState:
    """Nodo general de tecnologías, sin forzar QA específico."""
    return tech_node_impl(state)


def make_tech_qa_node(qa_id: str):
    """Factory para crear nodos de tecnologías especializados por QA."""
    qa_norm = normalize_qa(qa_id)

    def _node(state: GraphState) -> GraphState:
        return tech_node_impl(state, qa_override=qa_norm)

    _node.__name__ = f"tech_{qa_norm}_node"
    return _node
