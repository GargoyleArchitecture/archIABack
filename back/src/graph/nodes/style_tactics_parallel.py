# -*- coding: utf-8 -*-
"""
Ejecución paralela de los nodos `style` y `tactics` en el mismo turno.

Ambos nodos se lanzan concurrentemente (ThreadPoolExecutor) una vez que el ASR
ya está disponible en el estado. Esto elimina ~8-18 s de LLM calls secuenciales.

Trade-off: tactics no ve el estilo recién generado en este turno; usa el
`last_style` de un turno anterior (si existe). El prompt de tactics ya maneja
esto con "Selected architecture style (if any): ...".
"""

import copy
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable

from src.graph.nodes.styles.common import style_node_impl
from src.graph.nodes.tactics.availability_tactics import _fault_detection_lines_for_prompt
from src.graph.nodes.tactics.common import tactics_node_impl
from src.graph.qa_registry import normalize_qa
from src.graph.state import GraphState

log = logging.getLogger("graph")


# ---------------------------------------------------------------------------
# Builders QA-aware
# ---------------------------------------------------------------------------


def _build_style_fn(qa: str) -> Callable[[GraphState], GraphState]:
    qa_norm = normalize_qa(qa)

    def _call(state: GraphState) -> GraphState:
        return style_node_impl(state, qa_override=qa_norm if qa_norm != "general" else None)

    return _call


def _build_tactics_fn(qa: str) -> Callable[[GraphState], GraphState]:
    """Replica la lógica de los factories QA-específicos.

    NOTE: Si se añade un nuevo QA con preferred_tactics, sincronizar aquí.
    """
    qa_norm = normalize_qa(qa)

    if qa_norm == "disponibilidad":
        def _call(state: GraphState) -> GraphState:
            return tactics_node_impl(
                state,
                qa_override="disponibilidad",
                preferred_tactics=_fault_detection_lines_for_prompt(),
                preferred_group_label="Fault detection (disponibilidad)",
                restrict_to_preferred_tactics=True,
            )
    else:
        def _call(state: GraphState) -> GraphState:
            return tactics_node_impl(
                state,
                qa_override=qa_norm if qa_norm != "general" else None,
            )

    return _call


# ---------------------------------------------------------------------------
# Merge de estado
# ---------------------------------------------------------------------------


def _merge_states(
    input_state: GraphState,
    style_result: GraphState | None,
    tactics_result: GraphState | None,
    style_error: Exception | None,
    tactics_error: Exception | None,
) -> GraphState:
    """Combina los resultados de style y tactics en un único GraphState.

    Reglas de merge por campo:
    - style / selected_style / last_style / suggestions / memory_text → style_result
    - tactics_md / tactics_struct / tactics_list / messages           → tactics_result
    - turn_messages  → input + delta_style + delta_tactics (concatenar en orden)
    - arch_stage / intent / endMessage / quality_attribute / current_asr → tactics_result
    - completed_nodes → añadir "style" y/o "tactics" según éxito

    Manejo de errores:
    - Un nodo falla: usar solo el resultado exitoso; logear WARNING.
    - Ambos fallan: re-raise style_error para que LangGraph lo propague.
    """
    if style_error and tactics_error:
        log.error(
            "style_tactics_parallel: AMBOS nodos fallaron. style_error=%s tactics_error=%s",
            style_error,
            tactics_error,
        )
        raise style_error

    merged: GraphState = dict(input_state)  # type: ignore[assignment]

    input_turn_msgs: list = list(input_state.get("turn_messages") or [])
    input_msgs: list = list(input_state.get("messages") or [])
    extra_turn_msgs: list = []
    completed: list = list(input_state.get("completed_nodes") or [])

    # ── Style ────────────────────────────────────────────────────────────────
    if style_result is not None and style_error is None:
        merged["style"] = style_result.get("style", merged.get("style"))
        merged["selected_style"] = style_result.get("selected_style", merged.get("selected_style"))
        merged["last_style"] = style_result.get("last_style", merged.get("last_style"))
        merged["memory_text"] = style_result.get("memory_text", merged.get("memory_text", ""))
        merged["suggestions"] = style_result.get("suggestions", merged.get("suggestions", []))

        style_turn_msgs: list = list(style_result.get("turn_messages") or [])
        extra_turn_msgs.extend(style_turn_msgs[len(input_turn_msgs):])

        if "style" not in completed:
            completed.append("style")
        log.info("style_tactics_parallel: style OK, style=%s", merged.get("style"))
    else:
        log.warning(
            "style_tactics_parallel: style FALLÓ, manteniendo campos de entrada. error=%s",
            style_error,
        )

    # ── Tactics ──────────────────────────────────────────────────────────────
    if tactics_result is not None and tactics_error is None:
        merged["tactics_md"] = tactics_result.get("tactics_md", merged.get("tactics_md"))
        merged["tactics_struct"] = tactics_result.get("tactics_struct", merged.get("tactics_struct", []))
        merged["tactics_list"] = tactics_result.get("tactics_list", merged.get("tactics_list", []))
        merged["arch_stage"] = tactics_result.get("arch_stage", "TACTICS")
        merged["quality_attribute"] = tactics_result.get("quality_attribute", merged.get("quality_attribute"))
        if tactics_result.get("current_asr"):
            merged["current_asr"] = tactics_result["current_asr"]
        merged["endMessage"] = tactics_result.get("endMessage", merged.get("endMessage", ""))
        merged["intent"] = tactics_result.get("intent", "tactics")
        merged["nextNode"] = "unifier"

        # messages: solo tactics añade AIMessages; tomar delta
        tactics_msgs: list = list(tactics_result.get("messages") or [])
        input_msgs = input_msgs + tactics_msgs[len(input_msgs):]

        tactics_turn_msgs: list = list(tactics_result.get("turn_messages") or [])
        extra_turn_msgs.extend(tactics_turn_msgs[len(input_turn_msgs):])

        if "tactics" not in completed:
            completed.append("tactics")
        log.info("style_tactics_parallel: tactics OK, arch_stage=TACTICS")
    else:
        log.warning(
            "style_tactics_parallel: tactics FALLÓ, manteniendo campos de entrada. error=%s",
            tactics_error,
        )
        if style_result is not None and style_error is None:
            merged["arch_stage"] = style_result.get("arch_stage", "STYLE")
            merged["intent"] = "style"
            merged["endMessage"] = style_result.get("endMessage", merged.get("endMessage", ""))
            merged["nextNode"] = "unifier"

    merged["turn_messages"] = input_turn_msgs + extra_turn_msgs
    merged["messages"] = input_msgs
    merged["completed_nodes"] = completed
    merged["nextNode"] = "unifier"

    return merged  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Nodo principal
# ---------------------------------------------------------------------------


def style_tactics_parallel_node(state: GraphState) -> GraphState:
    """Ejecuta style y tactics concurrentemente y combina sus resultados."""
    qa = normalize_qa(
        state.get("quality_attribute") or state.get("resolved_index") or ""
    )

    style_fn = _build_style_fn(qa)
    tactics_fn = _build_tactics_fn(qa)

    # Cada hilo recibe una copia profunda para que las mutaciones directas
    # sobre el dict (state["field"] = value) no interfieran entre sí.
    state_for_style = copy.deepcopy(dict(state))
    state_for_tactics = copy.deepcopy(dict(state))

    style_result: GraphState | None = None
    tactics_result: GraphState | None = None
    style_error: Exception | None = None
    tactics_error: Exception | None = None

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="st_par") as pool:
        future_style: Future = pool.submit(style_fn, state_for_style)
        future_tactics: Future = pool.submit(tactics_fn, state_for_tactics)

        for future in as_completed([future_style, future_tactics]):
            if future is future_style:
                try:
                    style_result = future.result()
                except Exception as exc:
                    style_error = exc
                    log.exception("style_tactics_parallel: hilo style lanzó excepción")
            else:
                try:
                    tactics_result = future.result()
                except Exception as exc:
                    tactics_error = exc
                    log.exception("style_tactics_parallel: hilo tactics lanzó excepción")

    return _merge_states(
        input_state=state,
        style_result=style_result,
        tactics_result=tactics_result,
        style_error=style_error,
        tactics_error=tactics_error,
    )
