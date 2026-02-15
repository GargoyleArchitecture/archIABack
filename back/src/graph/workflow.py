
from typing import Literal

from langgraph.graph import StateGraph, START, END

from src.graph.state import GraphState
from src.graph.resources import sqlite_saver, builder

from src.graph.nodes.classifier import classifier_node
from src.graph.nodes.supervisor import supervisor_node
from src.graph.nodes.investigator import researcher_node
from src.graph.nodes.creator import creator_node
from src.graph.nodes.diagram import diagram_orchestrator_node
from src.graph.nodes.evaluator import evaluator_node
from src.graph.nodes.unifier import unifier_node
from src.graph.nodes.asr import asr_node
from src.graph.nodes.style import style_node
from src.graph.nodes.tactics import tactics_node

def boot_node(state: GraphState) -> GraphState:
    """Resetea banderas y buffers al inicio de cada turno (sin borrar last_asr)."""
    return {
        **state,
        "hasVisitedInvestigator": False,
        "hasVisitedCreator": False,
        "hasVisitedEvaluator": False,
        "hasVisitedASR": False,
        "hasVisitedDiagram": False,
        "mermaidCode": "",
        "diagram": {},
        "endMessage": "",
        "requested_nodes": [],
        "pending_nodes": [],
        "completed_nodes": [],
    }

def router(state: GraphState) -> Literal["investigator","creator","evaluator","diagram_agent","tactics","asr","style","unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"

    # NEW: para peticiones de ASR con RAG, pasa primero por el investigador
    if state["nextNode"] == "asr" and not state.get("hasVisitedASR", False):
        if (
            not state.get("hasVisitedInvestigator", False)
            and not state.get("doc_only", False)
            and state.get("force_rag", False)
        ):
            return "investigator"
        return "asr"

    if state["nextNode"] == "style":
        return "style"
    elif state["nextNode"] == "tactics":
        return "tactics"
    elif state["nextNode"] == "investigator" and not state["hasVisitedInvestigator"]:
        return "investigator"
    elif state["nextNode"] == "creator" and not state["hasVisitedCreator"]:
        return "creator"
    elif state["nextNode"] == "diagram_agent" and not state.get("hasVisitedDiagram", False):
        return "diagram_agent"
    elif state["nextNode"] == "evaluator" and not state["hasVisitedEvaluator"]:
        return "evaluator"
    else:
        return "unifier"

# ========== Wiring

builder.add_node("classifier", classifier_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("investigator", researcher_node)
builder.add_node("creator", creator_node)
builder.add_node("diagram_agent", diagram_orchestrator_node)  # Orquestador
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)
builder.add_node("asr", asr_node)
builder.add_node("style", style_node) 
builder.add_node("tactics", tactics_node)


builder.add_node("boot", boot_node)
builder.add_edge(START, "boot")
builder.add_edge("boot", "classifier")
builder.add_edge("classifier", "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
builder.add_edge("creator", "supervisor")
builder.add_edge("diagram_agent", "supervisor")
builder.add_edge("evaluator", "supervisor")
builder.add_edge("asr", "supervisor")
builder.add_edge("style", "supervisor")
builder.add_edge("tactics", "supervisor")
builder.add_edge("unifier", END)

graph = builder.compile(checkpointer=sqlite_saver)
