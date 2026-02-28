from src.graph.state import GraphState
from src.graph.nodes.styles.common import style_node_impl


def style_scalability_node(state: GraphState) -> GraphState:
    """Nodo de estilos especializado para QA de escalabilidad."""
    return style_node_impl(state, qa_override="escalabilidad")
