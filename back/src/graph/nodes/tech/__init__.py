"""Nodos de tecnologías arquitectónicas (ADD 3.0 — Point 5).

Este paquete centraliza todos los nodos relacionados con propuestas tecnológicas,
incluyendo el nodo base y la factoría de nodos especializados por QA.
"""

from src.graph.nodes.tech.tech import (
    tech_node,
    make_tech_qa_node,
)

__all__ = [
    "tech_node",
    "make_tech_qa_node",
]
