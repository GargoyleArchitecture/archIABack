# -*- coding: utf-8 -*-

from src.graph.state import GraphState
from src.graph.nodes.styles.common import style_node_impl


def style_availability_node(state: GraphState) -> GraphState:
    """Nodo de estilos especializado para QA de disponibilidad (availability).
    
    Propone estilos arquitectónicos que maximizen la disponibilidad del sistema,
    considerando redundancia, recuperación de fallos, y tácticas de resilencia.
    """
    return style_node_impl(state, qa_override="disponibilidad")
