import re
from functools import lru_cache
from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, ClassifyOut
from src.graph.index_resolver import resolve_quality_attribute
from src.graph.qa_registry import detect_explicit_qa, normalize_qa, supported_qas

# Fast regex-based language detector — used during INTAKE to skip the LLM call
# while still updating language on every turn.
_ES_MARKERS = re.compile(
    r"[áéíóúüñ¿¡]"
    r"|\b(hola|gracias|cómo|como|qué|que|está|tienes|puedes|"
    r"necesito|quiero|tengo|los|las|con|para|sí|hay|bien|mal|"
    r"mi|tu|su|una|ninguna|ningún|"
    r"el|la|del|al|debe|deben|usuario|usuarios|sistema|"
    r"por|pero|también|cuando|donde|quien|"
    r"escalar|procesar|diseñar|implementar|manejar)\b",
    re.IGNORECASE,
)


def _detect_lang_fast(msg: str) -> str:
    return "es" if _ES_MARKERS.search(msg) else "en"

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
    # During INTAKE the supervisor routes to intake_node regardless of intent.
    # Skip the LLM call to preserve intent="intake" and avoid spurious QA overrides,
    # but still detect language so intake_node responds in the user's language.
    if (state.get("current_phase") or "") == "INTAKE":
        msg = state.get("userQuestion", "") or ""
        prior_lang = state.get("language") or "es"
        detected = _detect_lang_fast(msg)
        # Only switch to "en" when there is positive English evidence (message long
        # enough to carry signal). Short/ambiguous messages like "no", "ok", "genera"
        # must not flip the language the user established earlier in the session.
        lang = detected if (detected == "es" or len(msg.split()) > 2) else prior_lang
        return {**state, "language": lang}

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


    # Always use the LLM's fresh classification for the CURRENT message.
    # Stale state language is intentionally ignored so the language can switch
    # if the user changes their language between turns.
    lang = lang_raw or state.get("language") or "es"

    # QA primario clasificado junto al intent (misma invocación del classifier).
    qa_from_classifier = normalize_qa(qa_attr)
    explicit_qa_in_msg = detect_explicit_qa(msg)
    prev_qa = normalize_qa(state.get("quality_attribute", ""))
    has_existing_asr = bool(state.get("current_asr") or state.get("last_asr"))

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

    preserve_followup_qa = (
        intent in ("style", "tactics", "diagram")
        and has_existing_asr
        and explicit_qa_in_msg == "general"
        and prev_qa != "general"
    )

    if preserve_followup_qa:
        quality_attribute = prev_qa
        resolved_index = prev_qa
    elif qa_from_classifier != "general":
        quality_attribute = qa_from_classifier
    elif resolved_qa != "general":
        quality_attribute = resolved_qa
    elif prev_qa != "general":
        quality_attribute = prev_qa
    else:
        quality_attribute = "general"

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
    }
