
from typing import Literal
from langchain_core.messages import SystemMessage
from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, ClassifyOut
import os
from src.graph.utils import _resolve_response_language

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
    forced_intent = ((state.get("turn_context") or {}).get("forced_intent") or "").strip().lower()
    if forced_intent in {"asr", "style", "tactics", "diagram", "general"}:
        msg_for_lang = state.get("userQuestion", "") or ""
        lang_forced = _resolve_response_language(state, msg_for_lang)
        return {
            **state,
            "language": lang_forced,
            "response_language": lang_forced,
            "intent": forced_intent,
        }

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
    
    diagram_triggers = [
        "component diagram", "diagram", "diagrama", "diagrama de componentes",
        "uml", "plantuml", "c4", "bpmn",
        "this asr", "este asr", "el asr", "ese asr", "anterior asr"
    ]
    # No pises estilo ni ASR cuando solo dicen "this ASR"
    if any(k in low for k in diagram_triggers) and intent not in ("asr", "style"):
        intent = "diagram"


    lang = _resolve_response_language(state, msg)

    return {
        **state,
        "language": lang,
        "response_language": lang,
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
