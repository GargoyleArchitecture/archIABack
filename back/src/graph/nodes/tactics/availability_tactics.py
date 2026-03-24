# -*- coding: utf-8 -*-

from src.graph.state import GraphState
from src.graph.nodes.tactics.common import tactics_node_impl


##def tactics_availability_node(state: GraphState) -> GraphState:
##    """Nodo de tácticas especializado para QA de disponibilidad (availability).
##    
##    Propone tácticas de diseño concretas para mejorar disponibilidad:
##    - Detección de fallos (heartbeat, ping, exception handling)
##    - Recuperación de fallos (retry, failover, replicación)
##    - Prevención de fallos (redundancia, graceful degradation)
##    """
##    return tactics_node_impl(state, qa_override="disponibilidad")


"""Tácticas de disponibilidad con foco en detección de fallos (vía `tactics_node_impl`)."""


from src.graph.state import GraphState
from src.graph.nodes.tactics.common import tactics_node_impl

_FAULT_DETECTION_CATALOG: tuple[tuple[str, str], ...] = (
    ("Ping / Echo", "Comprobar que un componente está activo y responde."),
    ("Heartbeat", "Señales periódicas que indican que un proceso sigue vivo."),
    ("Monitor (watchdog)", "Supervisión activa del estado de componentes del sistema."),
    ("Timestamp", "Detectar errores por orden incorrecto o inconsistencia temporal de eventos."),
    ("Sanity checking", "Validar que datos o resultados sean plausibles o coherentes."),
    ("Condition monitoring", "Observar condiciones internas (p. ej. memoria, CPU, colas)."),
    ("Voting (redundancia modular)", "Comparar salidas de réplicas para detectar discrepancias."),
    ("Exception detection", "Detectar condiciones anómalas: errores, timeouts, violaciones de contrato."),
    ("Self-test", "Ejecutar comprobaciones internas del sistema o módulos."),
)


def _fault_detection_lines_for_prompt() -> list[str]:
    return [f"{name} — {desc}" for name, desc in _FAULT_DETECTION_CATALOG]


def tactics_availability_node(state: GraphState) -> GraphState:
    return tactics_node_impl(
        state,
        qa_override="availability",
        preferred_tactics=_fault_detection_lines_for_prompt(),
        preferred_group_label="Fault detection (disponibilidad)",
        restrict_to_preferred_tactics=True,
    )