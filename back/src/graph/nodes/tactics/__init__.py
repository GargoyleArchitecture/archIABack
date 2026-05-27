"""Nodos de tácticas arquitectónicas.

Este paquete centraliza todos los nodos relacionados con tácticas,
incluyendo el nodo base y la factoría de nodos especializados por QA.
"""

from src.graph.nodes.tactics.tactics import (
    tactics_node,
    make_tactics_qa_node,
)
from src.graph.nodes.tactics.latency_tactics import tactics_latency_node
from src.graph.nodes.tactics.scalability_tactics import tactics_scalability_node
from src.graph.nodes.tactics.availability_tactics import tactics_availability_node

__all__ = [
    "tactics_node",
    "make_tactics_qa_node",
    "tactics_latency_node",
    "tactics_scalability_node",
    "tactics_availability_node",
]
