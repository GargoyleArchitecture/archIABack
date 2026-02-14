
import re
from langchain_core.messages import SystemMessage
from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, supervisorSchema
from src.graph.nodes.classifier import FOLLOWUP_PATTERNS
from src.graph.utils import _resolve_response_language
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

    # idioma
    state_lang = _resolve_response_language(state, uq)

    # CORTE DE CIRCUITO: respeta la intención forzada desde main.py
    forced = state.get("intent")
    if forced == "asr":
        return {**state,
                "localQuestion": f"Create a concrete QAS (ASR) for: {uq}",
                "nextNode": "asr",
                "intent": "asr",
                "language": state_lang,
                "response_language": state_lang}
    
    if forced == "style":
        return {
            **state,
            "localQuestion": uq or (
                "Selecciona el estilo arquitectónico más adecuado para el ASR actual."
                if state_lang == "es"
                else "Select the most appropriate architecture style for the current ASR."
            ),
            "nextNode": "style",
            "intent": "style",
            "language": state_lang,
            "response_language": state_lang,
        }

    if forced == "tactics":
        if not (state.get("current_asr") or (state.get("active_decisions") or {}).get("current_asr")):
            pq = state.get("pending_questions", []) + [
                "Necesito un ASR actual antes de proponer tacticas." if state_lang == "es" else "I need the current ASR before proposing tactics."
            ]
            return {
                **state,
                "nextNode": "unifier",
                "intent": "tactics",
                "language": state_lang,
                "response_language": state_lang,
                "pending_questions": pq,
            }
        return {**state,
                "localQuestion": ("Propose architecture tactics to satisfy the previous ASR. "
                                  "Explain why each tactic helps and ties to the ASR response/measure."),
                "nextNode": "tactics",
                "intent": "tactics",
                "language": state_lang,
                "response_language": state_lang}
    if forced == "diagram":
        return {**state,
                "localQuestion": uq,
                "nextNode": "diagram_agent",
                "intent": "diagram",
                "language": state_lang,
                "response_language": state_lang}

    # (a partir de aquí, SOLO si no vino intención forzada)
    fu_intent = classify_followup(uq)

        # --- NEW: detectar petición de estilos arquitectónicos ---
    style_terms = [
        "style", "styles",
        "architecture style", "architectural style",
        "estilo", "estilos", "estilo arquitectónico", "estilos arquitectónicos"
    ]
    wants_style = any(t in uq.lower() for t in style_terms)


    if _looks_like_eval(uq):
        return {**state,
                "localQuestion": uq,
                "nextNode": "evaluator",
                "intent": "architecture",
                "language": state_lang,
                "response_language": state_lang}

    # keywords para DIAGRAMAS (ES/EN)
    diagram_terms = [
        "diagrama", "diagrama de componentes", "diagrama de arquitectura",
        "diagram", "component diagram", "architecture diagram",
        "mermaid", "plantuml", "c4", "bpmn", "uml", "despliegue", "deployment"
    ]
    wants_diagram = any(t in uq.lower() for t in diagram_terms)

    # NEW: keywords para TÁCTICAS (ES/EN)
    tactics_terms = [
        "táctica", "tácticas", "tactic", "tactics",
        "estrategia", "estrategias", "strategy", "strategies",
        "cómo cumplir", "como cumplir", "how to satisfy",
        "how to meet", "how to achieve"
    ]
    wants_tactics = any(t in uq.lower() for t in tactics_terms)  # NEW

    sys_messages = [SystemMessage(content=makeSupervisorPrompt(state))]

    # baseline LLM (con fallback defensivo)
    try:
        resp = llm.with_structured_output(supervisorSchema).invoke(sys_messages)
        next_node = resp.get("nextNode", "investigator")
        local_q = resp.get("localQuestion", uq)
    except Exception:
        next_node, local_q = "investigator", uq

    intent_val = state.get("intent", "general")

        # 1) STYLE cuando el usuario lo pide explícitamente
    if wants_style or fu_intent == "style" or state.get("intent") == "style":
        next_node = "style"
        intent_val = "style"
        local_q = uq or (
            "Select the most appropriate architecture style for the current ASR."
            if state_lang == "en"
            else "Selecciona el estilo arquitectónico más adecuado para el ASR actual."
        )

    # 2) ASR después (no incluir tácticas aquí)
    elif any(x in uq.lower() for x in ["asr", "quality attribute scenario", "qas"]) or fu_intent == "make_asr":
        next_node = "asr"
        intent_val = "asr"
        local_q = f"Create a concrete QAS (ASR) for: {state['userQuestion']}"

    # 3) DIAGRAMA cuando lo piden
    elif wants_diagram or fu_intent in ("component_view", "deployment_view", "functional_view"):
        next_node = "diagram_agent"
        intent_val = "diagram"
        local_q = uq

    # 4) TÁCTICAS solo cuando el usuario las pide
    elif wants_tactics or fu_intent in ("explain_tactics", "tactics"):
        if not (state.get("current_asr") or (state.get("active_decisions") or {}).get("current_asr")):
            pq = state.get("pending_questions", []) + [
                "Necesito un ASR actual antes de proponer tacticas." if state_lang == "es" else "I need the current ASR before proposing tactics."
            ]
            return {
                **state,
                "nextNode": "unifier",
                "intent": "tactics",
                "language": state_lang,
                "response_language": state_lang,
                "pending_questions": pq,
            }
        next_node = "tactics"
        intent_val = "tactics"
        local_q = (
            "Propose architecture tactics to satisfy the previous ASR. "
            "Explain why each tactic helps and how it ties to the ASR response/measure."
        )

    # 4) Resto
    elif fu_intent in ("compare", "checklist"):
        next_node = "investigator"
        intent_val = "architecture"

    # evita unifier si no se visitó nada este turno
    if next_node == "unifier" and not (
        state.get("hasVisitedInvestigator") or state.get("hasVisitedCreator") or
        state.get("hasVisitedEvaluator") or state.get("hasVisitedASR") or
        state.get("hasVisitedDiagram")
    ):
        next_node = "investigator"; intent_val = "architecture"

    return {
        **state,
        "localQuestion": local_q,
        "nextNode": next_node,
        "intent": intent_val,
        "language": state_lang,
        "response_language": state_lang
    }
