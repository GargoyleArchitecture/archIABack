
from langgraph.graph import START, END

from src.graph.state import GraphState
from src.graph.resources import builder

from src.graph.nodes.context_loader import context_loader_node
from src.graph.nodes.classifier import classifier_node
from src.graph.nodes.supervisor import supervisor_node
from src.graph.nodes.investigator import researcher_node
from src.graph.nodes.diagram import diagram_orchestrator_node
from src.graph.nodes.evaluator import evaluator_node
from src.graph.nodes.unifier import unifier_node
from src.graph.nodes.asr import asr_node
from src.graph.nodes.styles import style_node, make_style_qa_node
from src.graph.nodes.tactics import tactics_node, make_tactics_qa_node
from src.graph.nodes.style_tactics_parallel import style_tactics_parallel_node
from src.graph.qa_registry import (
    normalize_qa,
    supported_qas,
    style_node_name_for_qa,
    tactics_node_name_for_qa,
)


# Catálogo QA cargado desde config.
# Se usa para registrar nodos especializados y validar rutas del router.
_SUPPORTED_QAS = supported_qas()
_STYLE_QA_NODE_NAMES = {style_node_name_for_qa(qa) for qa in _SUPPORTED_QAS}
_TACTICS_QA_NODE_NAMES = {tactics_node_name_for_qa(qa) for qa in _SUPPORTED_QAS}

def boot_node(state: GraphState) -> GraphState:
    """Resetea banderas y buffers al inicio de cada turno (sin borrar last_asr)."""
    return {
        **state,
        # F3-T2: incrementa contador para el Shadow Agent (cadencia).
        "turn_count_since_eval": (state.get("turn_count_since_eval") or 0) + 1,
        "hasVisitedInvestigator": False,
        "hasVisitedEvaluator": False,
        "hasVisitedASR": False,
        "hasVisitedDiagram": False,
        "diagram": {},
        "endMessage": "",
        "requested_nodes": [],
        "pending_nodes": [],
        "completed_nodes": [],
    }

def router(state: GraphState) -> str:
    if state["nextNode"] == "unifier":
        return "unifier"

    if state["nextNode"] == "style_tactics_parallel":
        return "style_tactics_parallel"

    # NEW: para peticiones de ASR con RAG, pasa primero por el investigador
    if state["nextNode"] == "asr" and not state.get("hasVisitedASR", False):
        if (
            not state.get("hasVisitedInvestigator", False)
            and not state.get("doc_only", False)
            and state.get("force_rag", False)
        ):
            return "investigator"
        return "asr"

    # Enrutado QA-aware para estilos y tácticas.
    # Se mantiene nextNode lógico (style/tactics) en supervisor,
    # pero aquí se selecciona el nodo especializado por atributo.
    qa = normalize_qa(state.get("quality_attribute") or state.get("resolved_index") or "")

    if state["nextNode"] == "style":
        style_target = style_node_name_for_qa(qa)
        if style_target in _STYLE_QA_NODE_NAMES:
            return style_target
        return "style"
    elif state["nextNode"] == "tactics":
        tactics_target = tactics_node_name_for_qa(qa)
        if tactics_target in _TACTICS_QA_NODE_NAMES:
            return tactics_target
        return "tactics"
    elif state["nextNode"] == "investigator" and not state["hasVisitedInvestigator"]:
        return "investigator"
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
builder.add_node("diagram_agent", diagram_orchestrator_node)  # Orquestador
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)
builder.add_node("asr", asr_node)
builder.add_node("style", style_node)
builder.add_node("tactics", tactics_node)
builder.add_node("style_tactics_parallel", style_tactics_parallel_node)

# Registro dinámico de nodos style/tactics por QA.
# Al agregar un QA en config/indices.json, el grafo registra sus nodos
# especializados automáticamente (sin nuevos cambios en workflow.py).
for qa_id in _SUPPORTED_QAS:
    style_name = style_node_name_for_qa(qa_id)
    tactics_name = tactics_node_name_for_qa(qa_id)

    if style_name != "style":
        builder.add_node(style_name, make_style_qa_node(qa_id))
    if tactics_name != "tactics":
        builder.add_node(tactics_name, make_tactics_qa_node(qa_id))


builder.add_node("boot", boot_node)
builder.add_node("context_loader", context_loader_node)
builder.add_edge(START, "boot")
builder.add_edge("boot", "context_loader")
builder.add_edge("context_loader", "classifier")
builder.add_edge("classifier", "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
builder.add_edge("diagram_agent", "supervisor")
builder.add_edge("evaluator", "supervisor")
builder.add_edge("asr", "supervisor")
builder.add_edge("style", "supervisor")
builder.add_edge("tactics", "supervisor")
builder.add_edge("style_tactics_parallel", "supervisor")

# Edges de retorno de nodos QA-específicos.
for node_name in sorted(_STYLE_QA_NODE_NAMES):
    if node_name != "style":
        builder.add_edge(node_name, "supervisor")
for node_name in sorted(_TACTICS_QA_NODE_NAMES):
    if node_name != "tactics":
        builder.add_edge(node_name, "supervisor")

builder.add_edge("unifier", END)


def build_graph(checkpointer, store=None):
    """Compila el grafo con el checkpointer (AsyncSqliteSaver) y el store
    (InMemoryStore) inyectados desde el lifespan de FastAPI.

    Llamar una sola vez al startup. La instancia resultante se publica via
    set_graph() en src/graph/resources.py para que get_graph() la exponga.
    """
    return builder.compile(checkpointer=checkpointer, store=store)
