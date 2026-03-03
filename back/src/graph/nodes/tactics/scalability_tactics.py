from src.graph.state import GraphState
from src.graph.nodes.tactics.common import tactics_node_impl


def tactics_scalability_node(state: GraphState) -> GraphState:
    """Nodo de tácticas especializado para QA de escalabilidad."""
    return tactics_node_impl(state, qa_override="escalabilidad")
