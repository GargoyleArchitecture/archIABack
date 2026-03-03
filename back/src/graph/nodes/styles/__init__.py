"""Nodos de estilos arquitectónicos.

Este paquete centraliza todos los nodos relacionados con estilos,
incluyendo el nodo base y la factoría de nodos especializados por QA.
"""

from src.graph.nodes.styles.style import (
    style_node,
    make_style_qa_node,
)
from src.graph.nodes.styles.latency_style import style_latency_node
from src.graph.nodes.styles.scalability_style import style_scalability_node

__all__ = [
    "style_node",
    "make_style_qa_node",
    "style_latency_node",
    "style_scalability_node",
]
