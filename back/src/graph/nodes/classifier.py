from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, ClassifyOut
from src.graph.index_resolver import resolve_quality_attribute
from src.graph.qa_registry import normalize_qa, supported_qas

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
    """Clasifica intención/idioma y fija QA operativo para el turno.

    Además de "resolved_index" (para RAG), este nodo propaga
    "quality_attribute" para que supervisor/router puedan decidir nodos
    específicos por QA (p. ej. style_latency vs style_scalability).
    """
    msg = state.get("userQuestion", "") or ""
    qa_ids = supported_qas()
    qa_opts = qa_ids + ["general"]
    qa_opts_str = ", ".join(f'"{q}"' for q in qa_opts)
    prompt = f"""
Classify the user's last message. Return JSON with:
- language: "en" or "es"
- intent: one of ["greeting","smalltalk","architecture","diagram","asr","tactics","style","other"]
- use_rag: true if this is a software-architecture question (ADD, tactics, latency, scalability,
  quality attributes, views, styles, diagrams, ASR), else false.
- quality_attribute: one of [{qa_opts_str}].

Output MUST be JSON only.
User: {msg}
"""
    try:
        out = llm.with_structured_output(ClassifyOut).invoke(prompt)
        # 1. Normalizar QA
        qa = normalize_qa(out.quality_attribute)
        # 2. Resolver indice físico (p. ej. "latencia" -> "latency")
        idx = resolve_quality_attribute(qa)

        return {
            **state,
            "language": out.language,
            "intent": out.intent,
            "use_rag": out.use_rag,
            "quality_attribute": qa,
            "resolved_index": idx
        }
    except Exception:
        # Fallback si falla el LLM o la estructura
        return {
            **state,
            "language": "en",
            "intent": "architecture",
            "use_rag": True,
            "quality_attribute": "general",
            "resolved_index": "general"
        }
