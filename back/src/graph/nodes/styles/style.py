from src.graph.state import GraphState
from src.graph.qa_registry import normalize_qa
from src.graph.nodes.styles.common import style_node_impl


def style_node(state: GraphState) -> GraphState:
    """Nodo general de estilos, sin forzar QA específico."""
    return style_node_impl(state)


def make_style_qa_node(qa_id: str):
    """Factory para crear nodos de estilo especializados por QA.

    Se usa por el workflow para registrar nodos dinámicamente desde config.
    """
    qa_norm = normalize_qa(qa_id)

    def _node(state: GraphState) -> GraphState:
        return style_node_impl(state, qa_override=qa_norm)

    _node.__name__ = f"style_{qa_norm}_node"
    return _node
