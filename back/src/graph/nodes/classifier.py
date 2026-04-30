import os
import re as _re_mode
from functools import lru_cache
from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, ClassifyOut
from src.graph.index_resolver import resolve_quality_attribute
from src.graph.qa_registry import normalize_qa, supported_qas

llm = get_chat_model(temperature=0.0)


# ===== F2-T4: Heuristica de auto-routing de modo =====
# Triggers bilingues (ES/EN). Cada match suma 0.5 al score (cap 1.0).
TUTOR_TRIGGERS = (
    r"\bexpl[ií]came\b", r"\bense[ñn]ame\b", r"\bno\s+entiendo\b",
    r"\bqu[eé]\s+es\b", r"\bqu[eé]\s+significa\b",
    r"\bay[uú]dame\s+a\s+entender\b", r"\bduda\b", r"\bconcept[oa]\b",
    r"\bexplain\b", r"\bteach\s+me\b", r"\bwhat\s+is\b",
    r"\bdon'?t\s+understand\b", r"\bdoubt\b",
)
PROFESSIONAL_TRIGGERS = (
    r"\bimplementa(?:me|lo)?\b", r"\bdame\s+(?:el\s+)?c[oó]digo\b",
    r"\bmu[eé]strame\b", r"\bdiagrama\b", r"\bdise[ñn]a\b",
    r"\boptimiza\b", r"\bbenchmark\b", r"\barquitectura\s+de\b",
    r"\bimplement\b", r"\bgive\s+me\s+the\s+code\b", r"\bdesign\s+the\b",
    r"\bbuild\b", r"\boptimize\b", r"\bcompare\s+latency\b",
)

_MODE_SUGGESTION_THRESHOLD = float(os.getenv("MODE_SUGGESTION_THRESHOLD", "0.7"))


def _score_triggers(text: str, patterns: tuple) -> float:
    """Confianza en [0,1]: 1 match = 0.8, 2+ saturan a 1.0.

    Calibrado para que un unico verbo de intencion claro (ej. 'explicame',
    'dame el codigo') ya supere el umbral default 0.7, y multiples matches
    converjan rapido a 1.0.
    """
    matches = sum(1 for p in patterns if _re_mode.search(p, text or "", _re_mode.IGNORECASE))
    return 0.0 if matches == 0 else min(1.0, 0.8 + 0.2 * (matches - 1))


def suggest_mode(text: str, current_mode: str):
    """Devuelve `tutor` / `professional` si la confianza supera el umbral
    Y el modo sugerido difiere del actual. None si no aplica."""
    tutor_conf = _score_triggers(text, TUTOR_TRIGGERS)
    pro_conf = _score_triggers(text, PROFESSIONAL_TRIGGERS)
    if tutor_conf >= _MODE_SUGGESTION_THRESHOLD and tutor_conf > pro_conf:
        return "tutor" if current_mode != "tutor" else None
    if pro_conf >= _MODE_SUGGESTION_THRESHOLD and pro_conf > tutor_conf:
        return "professional" if current_mode != "professional" else None
    return None

FOLLOWUP_PATTERNS = [
    ("explain_tactics", r"\b(tactics?|tácticas?).*(explain|describe|detalla|explica)|explica.*tácticas"),
    ("make_asr",        r"\b(asr|architecture significant requirement).*(make|create|example|ejemplo)|ejemplo.*asr"),
    ("component_view",  r"\b(component|diagrama de componentes|component diagram)"),
    ("deployment_view", r"\b(deployment|despliegue|deployment view)"),
    ("functional_view", r"\b(functional view|vista funcional)"),
    ("compare",         r"\b(compare|comparar).*?(latency|scalability|availability)"),
    ("checklist",       r"\b(checklist|lista de verificación|lista de verificacion)"),
]

@lru_cache(maxsize=256)
def _classify_cached(msg: str, qa_opts_str: str) -> tuple:
    """Returns (language, intent, use_rag, quality_attribute). Cached by (msg, qa_opts_str)."""
    prompt = f"""
Classify the user's last message. Return JSON with:
- language: "en" or "es"
- intent: one of ["greeting","smalltalk","architecture","diagram","asr","tactics","style","other"]
- use_rag: true if this is a software-architecture question (ADD, tactics, latency, scalability,
  quality attributes, views, styles, diagrams, ASR), else false.
- quality_attribute: one of [{qa_opts_str}].
  Use "general" only if no clear quality attribute is requested.

User message:
{msg}
"""
    out = llm.with_structured_output(ClassifyOut).invoke(prompt)
    return (out["language"], out["intent"], bool(out["use_rag"]), out.get("quality_attribute", "general"))


@lru_cache(maxsize=128)
def _resolve_qa_cached(msg: str) -> str:
    """Cached wrapper around resolve_quality_attribute (deterministic, temperature=0.0)."""
    return resolve_quality_attribute(msg, llm)


def classifier_node(state: GraphState) -> GraphState:
    """Clasifica intención/idioma y fija QA operativo para el turno.

    Además de "resolved_index" (para RAG), este nodo propaga
    "quality_attribute" para que supervisor/router puedan decidir nodos
    específicos por QA (p. ej. style_latency vs style_scalability).
    """
    msg = state.get("userQuestion", "") or ""
    qa_ids = supported_qas()
    qa_opts = qa_ids + ["general"]
    qa_opts_str = ", ".join(f'"{q}"' for q in qa_opts)
    lang_raw, intent_raw, use_rag, qa_attr = _classify_cached(msg, qa_opts_str)

    low = msg.lower()
    intent = intent_raw

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
        "diagrama de despliegue", "deployment diagram",
        "uml", "plantuml", "c4", "bpmn", "despliegue", "deployment", "graphviz", "dot"
    ]
    # Evita enrutar a diagrama por frases tipo "ese ASR" sin pedir diagrama explícito.
    if any(k in low for k in diagram_keywords) and intent not in ("asr", "style", "tactics"):
        intent = "diagram"


    # Prioriza el idioma ya detectado al inicio del turno (último mensaje del usuario)
    lang = state.get("language") or lang_raw

    # QA primario clasificado junto al intent (misma invocación del classifier).
    qa_from_classifier = normalize_qa(qa_attr)

    # Resolución del índice QA para RAG.
    # Regla: usar QA del classifier primero; si no, resolver por fallback.
    resolved_index = "general"
    if use_rag:
        if qa_from_classifier != "general":
            resolved_index = qa_from_classifier
        else:
            resolved_index = _resolve_qa_cached(msg)

    # QA operativo del turno (prioridad):
    # 1) QA clasificado junto al intent,
    # 2) índice resuelto para RAG,
    # 3) valor previo del estado (continuidad).
    resolved_qa = normalize_qa(resolved_index)
    prev_qa = normalize_qa(state.get("quality_attribute", ""))

    if qa_from_classifier != "general":
        quality_attribute = qa_from_classifier
    elif resolved_qa != "general":
        quality_attribute = resolved_qa
    elif prev_qa != "general":
        quality_attribute = prev_qa
    else:
        quality_attribute = "general"

    # F2-T4: heuristica de auto-routing de modo. Si el usuario muestra
    # intencion de pedagogia/operatividad y el modo actual no coincide,
    # exponemos `mode_suggestion` para que el Frontend ofrezca el cambio.
    mode_suggestion = suggest_mode(msg, state.get("mode") or "professional")

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

        "force_rag": bool(use_rag),
        "resolved_index": resolved_index,
        "quality_attribute": quality_attribute,
        "mode_suggestion": mode_suggestion,
    }
