
from typing import Literal
from langchain_core.messages import SystemMessage
from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, ClassifyOut
import os

llm = get_chat_model(temperature=0.0)

FOLLOWUP_PATTERNS = [
    ("explain_tactics", r"\b(tactics?|tácticas?).*(explain|describe|detalla|explica)|explica.*tácticas"),
    ("make_asr",        r"\b(asr|architecture significant requirement).*(make|create|example|ejemplo)|ejemplo.*asr"),
    ("component_view",  r"\b(component|diagrama de componentes|component diagram)"),
    ("deployment_view", r"\b(deployment|despliegue|deployment view)"),
    ("functional_view", r"\b(functional view|vista funcional)"),
    ("compare",         r"\b(compare|comparar).*?(latency|scalability|availability)"),
    ("checklist",       r"\b(checklist|lista de verificación|lista de verificacion)"),
]

def classifier_node(state: GraphState) -> GraphState:
    msg = state.get("userQuestion", "") or ""
    prompt = f"""
Classify the user's last message. Return JSON with:
- language: "en" or "es"
- intent: one of ["greeting","smalltalk","architecture","diagram","asr","tactics","style","other"]
- use_rag: true if this is a software-architecture question (ADD, tactics, latency, scalability,
  quality attributes, views, styles, diagrams, ASR), else false.

User message:
{msg}
"""
    out = llm.with_structured_output(ClassifyOut).invoke(prompt)

    low = msg.lower()
    intent = out["intent"]

    #disparadores de estilo arquitectónico
    style_triggers = [
        "style", "styles",
        "architecture style", "architectural style",
        "estilo", "estilos", "estilo arquitectónico", "estilos arquitectónicos"
    ]
    if any(k in low for k in style_triggers):
        intent = "style"


    tactics_triggers = [
        "tactic", "táctica", "tactica", "tácticas", "tactics", "tactcias",
        "strategy","estrategia",
        "cómo cumplir","como cumplir","how to meet","how to satisfy","how to achieve"
    ]
    if any(k in low for k in tactics_triggers):
        intent = "tactics"
    
    diagram_keywords = [
        "component diagram", "diagram", "diagrama", "diagrama de componentes",
        "diagrama de despliegue", "deployment diagram", "mermaid",
        "uml", "plantuml", "c4", "bpmn", "despliegue", "deployment"
    ]
    # Evita enrutar a diagrama por frases tipo "ese ASR" sin pedir diagrama explícito.
    if any(k in low for k in diagram_keywords) and intent not in ("asr", "style", "tactics"):
        intent = "diagram"


    # Prioriza el idioma ya detectado al inicio del turno (último mensaje del usuario)
    lang = state.get("language") or out["language"]

    return {
        **state,
        "language": lang,
        "intent": intent if intent in [
        "greeting",
        "smalltalk",
        "architecture",
        "diagram",
        "asr",
        "tactics",
        "style",
    ] else "general",

        "force_rag": bool(out["use_rag"]),
    }
