"""Subgrafo independiente RoutineGenerator (F5-T1).

Compila un grafo separado del grafo principal de conversación. Razones:
- Aislamiento: la lógica de generación de retos no comparte estado con los
  nodos del grafo de chat (boot, classifier, supervisor, …).
- Reutilización: el endpoint `POST /generate-routine` (F5-T2) lo invocará
  directamente sin pasar por el grafo de conversación.
- Test golden: se puede ejecutar contra un perfil sintético sin levantar
  todo el stack del agente.

Flujo:
    START
      → select_weakness        (F5-T1: elige weakness por menor mastery)
      → inverse_rag_search     (F5-T1: stub | F5-T3: ChromaDB bad_code_corpus)
      → synthesize_challenge   (F5-T1: LLM con structured output RoutineOutput)
      → validate_difficulty    (F5-T4: regen loop max 2)
            ├── "regenerate" → synthesize_challenge
            └── "accept"     → END

`build_routine_graph()` retorna la instancia compilada. El caller la guarda
en un singleton (`set_routine_graph` en `resources.py`) para que el
endpoint la recupere con `get_routine_graph()`.
"""
from __future__ import annotations

from langgraph.graph import START, END, StateGraph

from src.graph.schemas.routine import RoutineState
from src.graph.nodes.routine import (
    select_weakness_node,
    inverse_rag_search_node,
    synthesize_challenge_node,
    validate_difficulty_node,
    validate_difficulty_router,
)


def build_routine_graph():
    """Compila el subgrafo. Sin checkpointer (las generaciones son one-shot)."""
    builder = StateGraph(RoutineState)

    builder.add_node("select_weakness", select_weakness_node)
    builder.add_node("inverse_rag_search", inverse_rag_search_node)
    builder.add_node("synthesize_challenge", synthesize_challenge_node)
    builder.add_node("validate_difficulty", validate_difficulty_node)

    builder.add_edge(START, "select_weakness")
    builder.add_edge("select_weakness", "inverse_rag_search")
    builder.add_edge("inverse_rag_search", "synthesize_challenge")
    builder.add_edge("synthesize_challenge", "validate_difficulty")

    builder.add_conditional_edges(
        "validate_difficulty",
        validate_difficulty_router,
        {
            "regenerate": "synthesize_challenge",
            "accept": END,
        },
    )

    return builder.compile()
