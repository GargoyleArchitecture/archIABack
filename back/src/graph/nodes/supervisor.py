
import re
from langchain_core.messages import SystemMessage
from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, supervisorSchema
from src.graph.nodes.classifier import FOLLOWUP_PATTERNS
import logging

log = logging.getLogger("graph")
llm = get_chat_model(temperature=0.0)

# ========== Heurísticas helper ==========

EVAL_TRIGGERS = [
    "evaluate this asr", "check this asr", "review this asr",
    "evalúa este asr", "evalua este asr", "revisa este asr",
    "es bueno este asr", "mejorar este asr", "mejorar asr",
    "critique this asr", "assess this asr"
]

def _looks_like_eval(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in EVAL_TRIGGERS)

def detect_lang(text: str) -> str:
    t = (text or "").lower()
    es_hits = sum(w in t for w in ["qué","que","cómo","como","por qué","porque","cuál","cual","hola","táctica","tactica","vista","despliegue"])
    en_hits = sum(w in t for w in ["what","how","why","which","hello","tactic","view","deployment","component"])
    if es_hits > en_hits: return "es"
    if en_hits > es_hits: return "en"
    return "en"

def classify_followup(question: str) -> str | None:
    q = (question or "").lower().strip()
    for intent, pat in FOLLOWUP_PATTERNS:
        if re.search(pat, q):
            return intent
    return None

def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)

def _augment_completed_nodes(state: GraphState, completed: list[str]) -> list[str]:
    out = list(completed or [])
    turn_names = {
        (m.get("name") if isinstance(m, dict) else "")
        for m in (state.get("turn_messages") or [])
    }
    if state.get("hasVisitedASR"):
        _append_unique(out, "asr")
    if "style_recommender" in turn_names:
        _append_unique(out, "style")
    if "tactics_advisor" in turn_names:
        _append_unique(out, "tactics")
    if state.get("hasVisitedDiagram") or (state.get("mermaidCode") or "").strip():
        _append_unique(out, "diagram_agent")
    return out

def _infer_requested_nodes(uq: str, state: GraphState, forced: str | None) -> list[str]:
    low = (uq or "").lower()
    fu_intent = classify_followup(uq) or ""

    style_terms = [
        "style", "styles",
        "architecture style", "architectural style",
        "estilo", "estilos", "estilo arquitectónico", "estilos arquitectónicos"
    ]
    wants_style = any(t in low for t in style_terms) or forced == "style"

    tactics_terms = [
        "táctica", "tácticas", "tactic", "tactics",
        "estrategia", "estrategias", "strategy", "strategies",
        "cómo cumplir", "como cumplir", "how to satisfy",
        "how to meet", "how to achieve"
    ]
    wants_tactics = any(t in low for t in tactics_terms) or forced == "tactics" or fu_intent in ("explain_tactics", "tactics")

    diagram_terms = [
        "diagrama", "diagrama de componentes", "diagrama de arquitectura",
        "diagram", "component diagram", "architecture diagram",
        "mermaid", "plantuml", "c4", "bpmn", "uml", "despliegue", "deployment"
    ]
    has_diagram_terms = any(t in low for t in diagram_terms)
    wants_diagram = has_diagram_terms or fu_intent in ("component_view", "deployment_view", "functional_view")
    # Solo respetar forced=diagram si el texto realmente menciona diagrama/despliegue.
    if forced == "diagram" and has_diagram_terms:
        wants_diagram = True

    wants_asr = (
        forced == "asr"
        or fu_intent == "make_asr"
        or any(x in low for x in [" asr", "asr ", "qas", "quality attribute scenario", "architecture significant requirement"])
        or bool(re.search(r"\b(genera|generate|crea|create|haz|make)\b.*\b(asr|qas)\b", low))
    )

    explicit_chain = wants_asr or wants_style or wants_tactics or wants_diagram
    if not explicit_chain:
        return []

    has_existing_asr = bool((state.get("current_asr") or state.get("last_asr") or "").strip())
    plan: list[str] = []

    if wants_asr:
        _append_unique(plan, "asr")

    if wants_style:
        if (not has_existing_asr) and ("asr" not in plan):
            _append_unique(plan, "asr")
        _append_unique(plan, "style")

    if wants_tactics:
        if (not has_existing_asr) and ("asr" not in plan):
            _append_unique(plan, "asr")
        _append_unique(plan, "tactics")

    if wants_diagram:
        _append_unique(plan, "diagram_agent")

    return plan

def makeSupervisorPrompt(state: GraphState) -> str:
    visited_nodes = []
    if state["hasVisitedInvestigator"]: visited_nodes.append("investigator")
    if state["hasVisitedCreator"]:      visited_nodes.append("creator")
    if state["hasVisitedEvaluator"]:    visited_nodes.append("evaluator")
    if state.get("hasVisitedASR", False): visited_nodes.append("asr")
    visited_nodes_str = ", ".join(visited_nodes) if visited_nodes else "none"
    doc_flag = "ON" if state.get("doc_only") else "OFF"
    return f"""You are a supervisor orchestrating: investigator, creator (diagrams), evaluator, and ASR advisor.
Choose the next worker and craft a specific sub-question.

Rules:
- DOC-ONLY mode is {doc_flag}.
- If DOC-ONLY is ON: DO NOT call or suggest any retrieval tool (no local_RAG). Answers MUST rely only on the PROJECT DOCUMENT context provided.
- If DOC-ONLY is OFF and user asks about ADD/architecture, prefer investigator (and it may call local_RAG).
- If user asks for a diagram, route to creator.
- If user asks for an ASR or a QAS, route to asr.
- If two images are provided, evaluator may compare/analyze.
- Do not go directly to unifier unless at least one worker has produced output.

Visited so far: {visited_nodes_str}.
User question: {state["userQuestion"]}
Outputs: ['investigator','creator','evaluator','asr','unifier'].
"""

def supervisor_node(state: GraphState):
    uq = (state.get("userQuestion") or "")

    # si ya hay un SVG listo en este turno, vamos directo al unifier
    d = state.get("diagram") or {}
    if d.get("ok") and d.get("svg_b64"):
        return {**state, "nextNode": "unifier", "intent": "diagram"}

    # idioma: usa el detectado en el último mensaje del usuario
    state_lang = state.get("language") or detect_lang(uq)
    state_lang = "es" if state_lang == "es" else "en"

    # Estado multi-intent del turno
    completed_nodes = _augment_completed_nodes(state, list(state.get("completed_nodes", []) or []))
    pending_nodes = list(state.get("pending_nodes", []) or [])
    requested_nodes = list(state.get("requested_nodes", []) or [])

    forced = state.get("intent")
    if not requested_nodes and not pending_nodes:
        requested_nodes = _infer_requested_nodes(uq, state, forced)

    if _looks_like_eval(uq) and not requested_nodes and not pending_nodes:
        return {**state,
                "localQuestion": uq,
                "nextNode": "evaluator",
                "intent": "architecture",
                "language": state_lang,
                "requested_nodes": [],
                "pending_nodes": [],
                "completed_nodes": completed_nodes}

    # Scheduler multi-intent
    if pending_nodes:
        next_node = pending_nodes.pop(0)
    elif requested_nodes:
        remaining = [n for n in requested_nodes if n not in completed_nodes]
        if remaining:
            next_node = remaining[0]
            pending_nodes = remaining[1:]
        else:
            next_node = "unifier"
    else:
        next_node = "investigator"

    if next_node in ("asr", "style", "tactics", "diagram_agent"):
        intent_val = "diagram" if next_node == "diagram_agent" else next_node
    elif next_node == "evaluator":
        intent_val = "architecture"
    else:
        intent_val = state.get("intent", "general")

    if next_node == "asr":
        local_q = f"Create a concrete QAS (ASR) for: {uq}"
    elif next_node == "style":
        local_q = uq or (
            "Selecciona el estilo arquitectónico más adecuado para el ASR actual."
            if state_lang == "es"
            else "Select the most appropriate architecture style for the current ASR."
        )
    elif next_node == "tactics":
        local_q = (
            "Propose architecture tactics to satisfy the previous ASR. "
            "Explain why each tactic helps and how it ties to the ASR response/measure."
        )
    else:
        local_q = uq

    # fallback para arquitectura general sin plan explícito: usa LLM del supervisor
    if not requested_nodes and not pending_nodes and next_node == "investigator":
        sys_messages = [SystemMessage(content=makeSupervisorPrompt(state))]
        try:
            resp = llm.with_structured_output(supervisorSchema).invoke(sys_messages)
            next_node = resp.get("nextNode", "investigator")
            local_q = resp.get("localQuestion", uq)
            if next_node in ("asr", "style", "tactics", "diagram_agent"):
                intent_val = "diagram" if next_node == "diagram_agent" else next_node
            elif next_node == "evaluator":
                intent_val = "architecture"
        except Exception:
            pass

    # evita unifier si no se visitó nada este turno
    if next_node == "unifier" and not (
        state.get("hasVisitedInvestigator") or state.get("hasVisitedCreator") or
        state.get("hasVisitedEvaluator") or state.get("hasVisitedASR") or
        state.get("hasVisitedDiagram") or completed_nodes
    ):
        next_node = "investigator"
        intent_val = "architecture"

    return {
        **state,
        "localQuestion": local_q,
        "nextNode": next_node,
        "intent": intent_val,
        "language": state_lang,
        "requested_nodes": requested_nodes,
        "pending_nodes": pending_nodes,
        "completed_nodes": completed_nodes,
    }
