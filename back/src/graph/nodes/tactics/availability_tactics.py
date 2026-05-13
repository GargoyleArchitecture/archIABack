# -*- coding: utf-8 -*-

"""Tácticas de disponibilidad — catálogo completo ADD 3.0 (Detect / Recover / Prevent)."""
from src.graph.state import GraphState
from src.graph.nodes.tactics.common import tactics_node_impl

# Catálogo según Bass, Clements & Kazman (Software Architecture in Practice).
# Entradas con '#' son encabezados de grupo: se renderizan como secciones en el
# prompt y se omiten al extraer nombres canónicos.
_AVAILABILITY_TACTICS_CATALOG: tuple[tuple[str, str], ...] = (
    # ── Detect Faults ─────────────────────────────────────────────────────────
    ("# Detect Faults", ""),
    ("Ping / Echo", "Comprobar que un componente está activo y responde."),
    ("Heartbeat", "Señales periódicas que indican que un proceso sigue vivo."),
    ("Monitor (watchdog)", "Supervisión activa del estado de componentes del sistema."),
    ("Timestamp", "Detectar errores por orden incorrecto o inconsistencia temporal de eventos."),
    ("Sanity checking", "Validar que datos o resultados sean plausibles o coherentes."),
    ("Condition monitoring", "Observar condiciones internas (p. ej. memoria, CPU, colas)."),
    ("Voting (redundancia modular)", "Comparar salidas de réplicas para detectar discrepancias."),
    ("Exception detection", "Detectar condiciones anómalas: errores, timeouts, violaciones de contrato."),
    ("Self-test", "Ejecutar comprobaciones internas del sistema o módulos."),
    # ── Recover from Faults — Preparation and Repair ──────────────────────────
    ("# Recover from Faults — Preparation and Repair", ""),
    ("Active Redundancy",      "Mantener réplicas activas en caliente listas para asumir carga al instante (hot standby)."),
    ("Passive Redundancy",     "Mantener réplicas en espera que se activan ante fallos (warm/cold standby)."),
    ("Spare",                  "Disponer de componentes de repuesto que sustituyen a los fallidos bajo demanda."),
    ("Exception Handling",     "Capturar y gestionar excepciones para evitar que los fallos se propaguen."),
    ("Rollback",               "Revertir el sistema a un estado consistente previo al fallo."),
    ("Software Upgrade",       "Reemplazar versiones defectuosas en caliente sin interrumpir el servicio."),
    ("Retry",                  "Reintentar operaciones fallidas asumiendo que el fallo puede ser transitorio."),
    ("Ignore Faulty Behavior", "Ignorar mensajes o respuestas de fuentes identificadas como defectuosas."),
    ("Degradation",            "Reducir funcionalidad no esencial para preservar servicios críticos ante fallos."),
    ("Reconfiguration",        "Reasignar responsabilidades entre componentes operativos tras un fallo."),
    # ── Recover from Faults — Reintroduction ──────────────────────────────────
    ("# Recover from Faults — Reintroduction", ""),
    ("Shadow",                  "Ejecutar un componente nuevo en paralelo antes de reemplazar el original para validarlo."),
    ("State Resynchronization", "Sincronizar el estado entre réplicas antes de reintroducir un componente reparado."),
    ("Escalating Restart",      "Reiniciar componentes en niveles progresivos, del más específico al más general."),
    ("Non-Stop Forwarding",     "Continuar el enrutamiento mientras el plano de control se recupera."),
    # ── Prevent Faults ────────────────────────────────────────────────────────
    ("# Prevent Faults", ""),
    ("Removal from Service",    "Retirar componentes proactivamente antes de que fallen para aplicar mantenimiento."),
    ("Transactions",            "Agrupar operaciones en unidades atómicas para garantizar consistencia ante fallos parciales."),
    ("Predictive Model",        "Monitorear indicadores para anticipar fallos antes de que ocurran."),
    ("Exception Prevention",    "Eliminar las causas raíz de excepciones mediante abstracciones o validaciones preventivas."),
    ("Increase Competence Set", "Ampliar la capacidad del sistema para manejar entradas inválidas sin fallar."),
)

_AVAILABILITY_RAG_QUERIES: tuple[str, ...] = (
    "fault detection tactics availability heartbeat ping echo monitor voting",
    "recovery tactics availability redundancy rollback retry reconfiguration restart",
    "fault prevention tactics availability predictive model transactions exception prevention",
    "Bass Clements Kazman availability tactics detect recover prevent faults",
)


def _availability_tactics_for_prompt() -> list[str]:
    result: list[str] = []
    for name, desc in _AVAILABILITY_TACTICS_CATALOG:
        if name.startswith("#"):
            result.append(name)
        else:
            result.append(f"{name} — {desc}")
    return result


def tactics_availability_node(state: GraphState) -> GraphState:
    return tactics_node_impl(
        state,
        qa_override="disponibilidad",
        preferred_tactics=_availability_tactics_for_prompt(),
        preferred_group_label="Availability Tactics — Detect / Recover / Prevent (ADD 3.0)",
        restrict_to_preferred_tactics=True,
        rag_queries_override=list(_AVAILABILITY_RAG_QUERIES),
    )
