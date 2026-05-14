# -*- coding: utf-8 -*-

import re
from langchain_core.messages import SystemMessage
from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, supervisorSchema
from src.graph.nodes.classifier import FOLLOWUP_PATTERNS
from src.graph.utils import is_explicit_asr_request, is_asr_regenerate_request
from src.graph.consts import PHASE_INT, FUNNEL_INTENT_MIN_PHASE, PHASE_DISPLAY, PHASE_NEXT_TASK
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

def detect_lang(text: str) -> str | None:
    """Detect language from text. Returns None when there is no clear signal.

    Returning None (instead of a default) lets callers that already hold a
    prior language preserve it via `state.get("language") or detect_lang(uq) or "es"`.
    """
    t = (text or "").lower()
    es_hits = sum(w in t for w in [
        "qué","que","cómo","como","por qué","porque","cuál","cual",
        "hola","táctica","tactica","vista","despliegue","sistema",
        "para","con","una","uno","dame","quiero","necesito","genera",
        "muestra","explica","describe","define","crea","hazme","dime",
        "estilo","tácticas","diagrama","latencia","disponibilidad",
        "rendimiento","escalabilidad","seguridad","requerimiento",
        "arquitectura","servicio","microservicio","componente",
        "ahora","perfecto","vamos","este","ese","eso","mi","mis",
    ])
    en_hits = sum(w in t for w in [
        "what","how","why","which","hello","tactic","view","deployment",
        "component","please","give","show","explain","create","define",
        "style","diagram","latency","availability","performance",
        "scalability","security","requirement","architecture","service",
        "microservice","now","this","my",
    ])
    if es_hits > en_hits: return "es"
    if en_hits > es_hits: return "en"
    return None  # BUG-014: no signal — caller uses prior state language

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
    # BUG-013: also mark asr as done when routing_phase shows we've already passed it,
    # so completed_nodes stays consistent across turns even if hasVisitedASR was reset.
    _routing_phase = state.get("routing_phase") or "intake"
    if _routing_phase in ("asr", "style", "tactics", "tech", "done"):
        _append_unique(out, "asr")
    if bool(state.get("selected_asrs")) or bool(state.get("current_asr")) or bool(state.get("last_asr")):
        _append_unique(out, "asr")
    if "style_recommender" in turn_names:
        _append_unique(out, "style")
    if "tactics_advisor" in turn_names:
        _append_unique(out, "tactics")
    if state.get("hasVisitedTech") or "tech_advisor" in turn_names:
        _append_unique(out, "tech")
    if state.get("hasVisitedDiagram"):
        _append_unique(out, "diagram_agent")
    return out

import re as _re

_NEW_PROJECT_GREETING_RE = _re.compile(
    r"^\s*(?:hola\b|hi\b|hello\b|buenos\s+d[íi]as|buenas\s+tardes|hey\b)",
    _re.IGNORECASE,
)
_NEW_PROJECT_DESIGN_RE = _re.compile(
    r"\b(?:"
    r"quiero\s+dise[nñ]ar|quisiera\s+dise[nñ]ar|"
    r"necesito\s+(?:diseñar|crear|construir)\s+(?:la\s+)?(?:arquitectura|sistema)|"
    r"dise[nñ]ar\s+(?:la\s+)?arquitectura\s+de|"
    r"dise[nñ]ar\s+(?:el|un)\s+sistema|"
    r"I\s+(?:need|want)\s+to\s+(?:design|build|create)\s+(?:a|an|the)\s+(?:architecture|system)"
    r")\b",
    _re.IGNORECASE,
)

def _is_new_project_intro(uq: str) -> bool:
    """True when the message is a fresh project introduction (greeting + design intent, long enough)."""
    return (
        len(uq.split()) >= 20
        and bool(_NEW_PROJECT_GREETING_RE.search(uq))
        and bool(_NEW_PROJECT_DESIGN_RE.search(uq))
    )

def _build_block_message(current_phase: str, requested_phase: str, lang: str) -> str:
    cur_display = PHASE_DISPLAY.get(current_phase, {}).get(lang, current_phase)
    req_display = PHASE_DISPLAY.get(requested_phase, {}).get(lang, requested_phase)
    cur_task    = PHASE_NEXT_TASK.get(current_phase, {}).get(lang, "")

    if lang == "es":
        lines = [
            f"Estamos en la fase de **{cur_display}**.",
            f"Para llegar a **{req_display}** primero necesitamos completar la fase actual.",
        ]
        if cur_task:
            lines.append(f"La tarea pendiente ahora es: *{cur_task}*.")
        lines.append("¿Continuamos?")
    else:
        lines = [
            f"We are currently in the **{cur_display}** phase.",
            f"To reach **{req_display}** we need to complete the current phase first.",
        ]
        if cur_task:
            lines.append(f"The pending task right now is: *{cur_task}*.")
        lines.append("Shall we continue?")

    return "\n\n".join(lines)


def _infer_requested_nodes(uq: str, state: GraphState, forced: str | None) -> list[str]:
    low = (uq or "").lower()
    fu_intent = classify_followup(uq) or ""
    has_existing_asr = bool((state.get("current_asr") or state.get("last_asr") or "").strip())
    explicit_asr_request = is_explicit_asr_request(uq)

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
        "plantuml", "c4", "bpmn", "uml", "despliegue", "deployment", "graphviz", "dot"
    ]
    has_diagram_terms = any(t in low for t in diagram_terms)
    wants_diagram = has_diagram_terms or fu_intent in ("component_view", "deployment_view", "functional_view")
    # Solo respetar forced=diagram si el texto realmente menciona diagrama/despliegue.
    if forced == "diagram" and has_diagram_terms:
        wants_diagram = True

    tech_terms = [
        "tecnología", "tecnologias", "tecnologías", "technology", "tech stack",
        "framework", "library", "librería", "herramienta", "tool", "tools",
        "implementación", "implementacion", "implementation",
        "qué usar", "que usar", "what to use", "which library", "which framework",
        "propón tecnologías", "propón tecnologias", "propose technologies",
        "stack tecnológico", "stack tecnologico",
    ]
    wants_tech = any(t in low for t in tech_terms) or forced == "tech"

    wants_asr = (
        explicit_asr_request
        or fu_intent == "make_asr"
        or (forced == "asr" and not has_existing_asr)
        or forced == "asr_reject"
        or is_asr_regenerate_request(uq)
    )

    explicit_chain = wants_asr or wants_style or wants_tactics or wants_tech or wants_diagram
    if not explicit_chain:
        return []

    plan: list[str] = []

    if wants_asr:
        _append_unique(plan, "asr")

    if wants_style:
        if (not has_existing_asr) and ("asr" not in plan):
            _append_unique(plan, "asr")
        _append_unique(plan, "style")

    if wants_tactics:
        _phase_past_tactics = (state.get("current_phase") or "") in (
            "tech_proposals", "diagram", "done"
        )
        _tactics_already_done = bool(state.get("selected_tactics")) and _phase_past_tactics
        if not _tactics_already_done:
            if (not has_existing_asr) and ("asr" not in plan):
                _append_unique(plan, "asr")
            _append_unique(plan, "tactics")

    if wants_tech:
        if (not has_existing_asr) and ("asr" not in plan):
            _append_unique(plan, "asr")
        _append_unique(plan, "tech")

    if wants_diagram:
        _append_unique(plan, "diagram_agent")

    return plan

def makeSupervisorPrompt(state: GraphState) -> str:
    visited_nodes = []
    if state["hasVisitedInvestigator"]: visited_nodes.append("investigator")
    if state["hasVisitedEvaluator"]:    visited_nodes.append("evaluator")
    if state.get("hasVisitedASR", False): visited_nodes.append("asr")
    visited_nodes_str = ", ".join(visited_nodes) if visited_nodes else "none"
    doc_flag = "ON" if state.get("doc_only") else "OFF"

    proj_ctx = (state.get("project_context_text") or "").strip()
    project_block = f"\n{proj_ctx}\n" if proj_ctx else ""

    return f"""GLOSSARY: In this system "ASR" ALWAYS means Architecturally Significant Requirement (ADD 3.0). NEVER interpret "ASR" as Automatic Speech Recognition or any audio/voice technology.

You are a supervisor orchestrating: investigator, diagram_agent (diagrams via DOT/Graphviz), evaluator, and asr (Architecturally Significant Requirements advisor — ADD 3.0).
Choose the next worker and craft a specific sub-question.

Rules:
- DOC-ONLY mode is {doc_flag}.
- If DOC-ONLY is ON: DO NOT call or suggest any retrieval tool (no local_RAG). Answers MUST rely only on the PROJECT DOCUMENT context provided.
- If DOC-ONLY is OFF and user asks about ADD/architecture, prefer investigator (and it may call local_RAG).
- If user asks for a diagram, route to diagram_agent.
- If user asks for an ASR or a QAS (Architecturally Significant Requirement), route to asr.
- If two images are provided, evaluator may compare/analyze.
- Do not go directly to unifier unless at least one worker has produced output.
{project_block}
Visited so far: {visited_nodes_str}.
User question: {state["userQuestion"]}
Outputs: ['investigator','diagram_agent','evaluator','asr','unifier'].
"""

def supervisor_node(state: GraphState):
    uq = (state.get("userQuestion") or "")

    log.info(
        "supervisor: current_phase=%r intent=%r nextNode=%r",
        state.get("current_phase"), state.get("intent"), state.get("nextNode"),
    )

    if (state.get("current_phase") or "") in ("intro", "diagnosis") and (state.get("mode") or "professional") != "tutor":
        return {**state, "nextNode": "intake", "localQuestion": ""}

    # si ya hay un SVG listo en este turno, vamos directo al unifier
    d = state.get("diagram") or {}
    if d.get("ok") and d.get("svg_b64"):
        return {**state, "nextNode": "unifier", "intent": "diagram"}

    # BUG-025: New project detection — fires when a stale checkpoint has a
    # mid-session current_phase but the user is clearly starting a fresh project.
    # Without this, the M1 gate issues a "wrong phase" block instead of intake.
    _phase_now = (state.get("current_phase") or "intro")
    if _phase_now not in ("intro", "diagnosis") and _is_new_project_intro(uq):
        _np_lang = state.get("language") or detect_lang(uq) or "es"
        _np_lang = "es" if _np_lang == "es" else "en"
        return {
            **state,
            "current_phase": "intro",
            "new_project_flow": True,
            "routing_phase": "intake",
            "intake_fields": {},
            "intake_complete": False,
            "intake_current_field": 0,
            "current_asr": "",
            "last_asr": "",
            "selected_asrs": [],
            "asr_candidates": [],
            "style": "",
            "selected_style": "",
            "last_style": "",
            "style_candidates": [],
            "selected_tactics": [],
            "tactics_candidates": [],
            "tactics_struct": [],
            "tactics_list": [],
            "tech_candidates": [],
            "ledger_active": {},
            "completed_nodes": [],
            "nextNode": "intake",
            "localQuestion": "",
            "language": _np_lang,
        }

    # BUG-014: preserve prior language when detect_lang has no signal (returns None).
    state_lang = state.get("language") or detect_lang(uq) or "es"
    state_lang = "es" if state_lang == "es" else "en"

    # ─── Orientación para usuarios que regresan ──────────────────────────────
    # Fires when: user has passed intake (current_phase outside intro/diagnosis)
    # AND the message is a greeting or generic intent with no specific action.
    # Instead of falling through to investigator, summarize their progress and
    # tell them what to do next.
    _returning_intent = (state.get("intent") or "") in ("general", "greeting", "smalltalk")
    _has_phase_context = (state.get("current_phase") or "intro") not in ("intro", "diagnosis")

    if _returning_intent and _has_phase_context:
        _phase_now   = state.get("current_phase") or "intro"
        _task_hint   = PHASE_NEXT_TASK.get(_phase_now, {}).get(state_lang, "")
        _phase_label = PHASE_DISPLAY.get(_phase_now, {}).get(state_lang, _phase_now)
        _compact     = (state.get("ledger_dossier_compact") or "").strip()

        if state_lang == "es":
            _lines = [f"Bienvenido de vuelta. Estamos en la fase de **{_phase_label}**."]
            if _compact:
                _lines.append(_compact)
            if _task_hint:
                _lines.append(f"La siguiente tarea es: *{_task_hint}*. ¿Continuamos?")
        else:
            _lines = [f"Welcome back. We're in the **{_phase_label}** phase."]
            if _compact:
                _lines.append(_compact)
            if _task_hint:
                _lines.append(f"Next up: *{_task_hint}*. Shall we continue?")

        _completed = _augment_completed_nodes(state, list(state.get("completed_nodes") or []))
        return {
            **state,
            "endMessage": "\n\n".join(_lines),
            "nextNode": "unifier",
            "intent": "intake",
            "language": state_lang,
            "requested_nodes": [],
            "pending_nodes": [],
            "completed_nodes": _completed,
            "phase_redirect_hint": "",
        }
    # ────────────────────────────────────────────────────────────────────────

    # ─── M1: Gate de fase ADD 3.0 ───────────────────────────────────────────
    current_phase = (state.get("current_phase") or "intro")
    intent_raw = (state.get("intent") or "")
    min_phase_key = FUNNEL_INTENT_MIN_PHASE.get(intent_raw)

    if min_phase_key and PHASE_INT.get(current_phase, 0) < PHASE_INT[min_phase_key]:
        block_text = _build_block_message(current_phase, min_phase_key, state_lang)
        _sugs_es = ["Sí, continuemos", "Quiero cambiar el contexto del sistema"]
        _sugs_en = ["Yes, let's continue", "I want to change the system context"]
        # BUG-002 fix: preserve completed_nodes across phase-gate redirects.
        # Wiping it caused the supervisor to re-trigger already-done nodes
        # (e.g. ASR re-generation) on the turn immediately after the block.
        _completed_safe = _augment_completed_nodes(state, list(state.get("completed_nodes") or []))
        return {
            **state,
            "endMessage": block_text,
            "nextNode": "unifier",
            "intent": "intake",
            "language": state_lang,
            "suggestions": _sugs_es if state_lang == "es" else _sugs_en,
            "requested_nodes": [],
            "pending_nodes": [],
            "completed_nodes": _completed_safe,
            "phase_redirect_hint": "",
        }
    # ────────────────────────────────────────────────────────────────────────

    # Estado multi-intent del turno
    completed_nodes = _augment_completed_nodes(state, list(state.get("completed_nodes", []) or []))

    if intent_raw == "asr_confirm":
        return {
            **state,
            "nextNode": "asr_confirm",
            "intent": "asr_confirm",
            "language": state_lang,
            "requested_nodes": [],
            "pending_nodes": [],
            "completed_nodes": completed_nodes,
        }

    # BUG-054 / BUG-055: route the user's style selection directly to the
    # confirmation node — never to style_node (which would re-generate
    # candidates) or to the "Bienvenido de vuelta" fallback.
    if intent_raw == "style_confirm":
        return {
            **state,
            "nextNode": "style_confirm",
            "intent": "style_confirm",
            "language": state_lang,
            "requested_nodes": [],
            "pending_nodes": [],
            "completed_nodes": completed_nodes,
        }

    # BUG-012/007/013: route tactics confirmation directly to tactics_confirm_node.
    if intent_raw == "tactics_confirm":
        return {
            **state,
            "nextNode": "tactics_confirm",
            "intent": "tactics_confirm",
            "language": state_lang,
            "requested_nodes": [],
            "pending_nodes": [],
            "completed_nodes": completed_nodes,
        }

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
    # BUG-013: gate ASR with all available signals to prevent re-running after a failed turn.
    _routing_phase = state.get("routing_phase") or "intake"
    _has_existing_asr = (
        bool((state.get("current_asr") or state.get("last_asr") or "").strip())
        or bool(state.get("selected_asrs"))
        or _routing_phase in ("asr", "style", "tactics", "tech", "done")
    )
    _asr_already_done = ("asr" in completed_nodes) or _has_existing_asr
    explicit_regen = is_asr_regenerate_request(uq) or (state.get("intent") == "asr_reject")
    must_run_asr = ("asr" in requested_nodes) and (explicit_regen or not _asr_already_done)

    if must_run_asr:
        next_node = "asr"
        pending_nodes = [n for n in pending_nodes if n != "asr"]
    elif pending_nodes:
        if "style" in pending_nodes and "tactics" in pending_nodes:
            pending_nodes = [n for n in pending_nodes if n not in ("style", "tactics")]
            next_node = "style_tactics_parallel"
        else:
            next_node = pending_nodes.pop(0)
    elif requested_nodes:
        remaining = [n for n in requested_nodes if n not in completed_nodes]
        if "style" in remaining and "tactics" in remaining:
            next_node = "style_tactics_parallel"
            pending_nodes = [n for n in remaining if n not in ("style", "tactics")]
        elif remaining:
            next_node = remaining[0]
            pending_nodes = remaining[1:]
        else:
            next_node = "unifier"
    else:
        next_node = "investigator"

    if next_node in ("asr", "style", "tactics", "tech", "diagram_agent", "style_tactics_parallel"):
        if next_node == "diagram_agent":
            intent_val = "diagram"
        elif next_node == "style_tactics_parallel":
            intent_val = "tactics"
        else:
            intent_val = next_node
    elif next_node == "evaluator":
        intent_val = "architecture"
    else:
        intent_val = state.get("intent", "general")

    if next_node == "asr":
        local_q = (
            "GLOSSARY: ASR = Architecturally Significant Requirement (ADD 3.0)."
            " NEVER interpret ASR as Automatic Speech Recognition.\n\n"
            f"Create a concrete Architecturally Significant Requirement (ASR/QAS) for: {uq}"
        )
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
    elif next_node == "style_tactics_parallel":
        local_q = (
            "Selecciona el estilo arquitectónico y propón tácticas para el ASR actual."
            if state_lang == "es"
            else "Select the architecture style and propose tactics for the current ASR."
        )
    elif next_node == "tech":
        local_q = (
            "Propón tecnologías concretas para implementar las tácticas confirmadas."
            if state_lang == "es"
            else "Propose concrete technologies to implement the confirmed tactics."
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
            if next_node in ("asr", "style", "tactics", "tech", "diagram_agent"):
                intent_val = "diagram" if next_node == "diagram_agent" else next_node
            elif next_node == "evaluator":
                intent_val = "architecture"
        except Exception:
            pass

    # evita unifier si no se visitó nada este turno
    if next_node == "unifier" and not (
        state.get("hasVisitedInvestigator") or
        state.get("hasVisitedEvaluator") or state.get("hasVisitedASR") or
        state.get("hasVisitedDiagram") or state.get("hasVisitedTech") or
        completed_nodes
    ):
        next_node = "investigator"
        intent_val = "architecture"

    # M1: redirect hint para respuestas de smalltalk/general (unifier lo añade al final)
    free_intents = {"smalltalk", "general", "greeting", "architecture"}
    phase_redirect = ""
    if intent_raw in free_intents:
        task = PHASE_NEXT_TASK.get(current_phase, {}).get(state_lang, "")
        if task:
            if state_lang == "es":
                phase_redirect = f"> **Nota:** Cuando quieras, podemos continuar con: *{task}*."
            else:
                phase_redirect = f"> **Note:** Whenever you're ready, we can continue with: *{task}*."

    return {
        **state,
        "localQuestion": local_q,
        "nextNode": next_node,
        "intent": intent_val,
        "language": state_lang,
        "requested_nodes": requested_nodes,
        "pending_nodes": pending_nodes,
        "completed_nodes": completed_nodes,
        "phase_redirect_hint": phase_redirect,
    }
