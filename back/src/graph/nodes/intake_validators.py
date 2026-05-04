import logging
import re
from typing import Tuple, TypedDict

log = logging.getLogger(__name__)

_TECHNICAL_TERMS = re.compile(
    r"\b(módulo|servicio|api|componente|sistema|request|evento|endpoint|"
    r"base de datos|microservicio|caché|cola|latencia|throughput|"
    r"concurrencia|usuario|cliente|servidor)\b",
    re.IGNORECASE,
)

_SOURCE_CATEGORIES = re.compile(
    r"\b(usuario|user|sistema externo|external system|"
    r"evento interno|internal event|tiempo|time|timer|schedule)\b",
    re.IGNORECASE,
)

_METRIC_PATTERN = re.compile(
    r"\d+|[<>]=?|%\b|ms\b|rps\b|tps\b|sla\b|slo\b",
    re.IGNORECASE,
)

_NEGATION_PATTERN = re.compile(
    r"\b(ninguna|none|n/a|no hay|sin restricciones)\b",
    re.IGNORECASE,
)

_ALPHA_RE = re.compile(r'\b[a-záéíóúüñ]{3,}\b', re.IGNORECASE)


def _content_tokens(text: str) -> set:
    return {t.lower() for t in _ALPHA_RE.findall(text)}


def _is_copy_paste(index: int, value: str) -> bool:
    """True when >60 % of the question's content tokens appear in value.

    Uses the question as reference so that valid answers containing question
    vocabulary (e.g. 'sobrecarga', 'mantenimiento') are not falsely flagged.
    """
    value_tokens = _content_tokens(value)
    if len(value_tokens) < 4:
        return False
    for lang in ("es", "en"):
        q_tokens = _content_tokens(INTAKE_SCRIPT[index][f"question_{lang}"])
        if not q_tokens:
            continue
        common = value_tokens & q_tokens
        if len(common) / len(q_tokens) > 0.60:
            return True
    return False


INTAKE_SCRIPT = [
    {
        "field": "campo_0_requerimiento",
        "question_es": "¿Cuál es el requerimiento principal del sistema que deseas diseñar? Describe el objetivo principal, los componentes involucrados y las expectativas de calidad.",
        "question_en": "What is the main requirement of the system you want to design? Describe the main goal, the components involved, and quality expectations.",
        "rule": "len(tokens) >= 8 AND at least one technical term",
    },
    {
        "field": "campo_1_componentes",
        "question_es": "¿Cuáles son los componentes principales del sistema? Menciona servicios, módulos, APIs, bases de datos u otros elementos relevantes.",
        "question_en": "What are the main components of the system? Mention services, modules, APIs, databases, or other relevant elements.",
        "rule": "len(tokens) >= 8 AND at least one technical term",
    },
    {
        "field": "campo_2_fuente",
        "question_es": "¿Cuál es la fuente del estímulo? Por ejemplo: usuario, sistema externo, evento interno, tiempo/timer.",
        "question_en": "What is the source of the stimulus? For example: user, external system, internal event, time/timer.",
        "rule": "_SOURCE_CATEGORIES match",
    },
    {
        "field": "campo_3_estimulo",
        "question_es": "¿Cuál es el estímulo o trigger que activa el comportamiento del sistema? Describe el evento específico que dispara la respuesta.",
        "question_en": "What is the stimulus or trigger that activates system behavior? Describe the specific event that triggers the response.",
        "rule": "len(tokens) >= 8 AND at least one technical term",
    },
    {
        "field": "campo_4_ambientes",
        "question_es": "¿En qué ambientes o escenarios debe operar el sistema? Incluye métricas concretas: carga normal, sobrecarga, mantenimiento (ej: p95<200ms, 500rps).",
        "question_en": "In what environments or scenarios must the system operate? Include concrete metrics: normal load, overload, maintenance (e.g., p95<200ms, 500rps).",
        "rule": "_METRIC_PATTERN match",
    },
    {
        "field": "campo_5_prioridad_qa",
        "question_es": "¿Cuál es la prioridad de los atributos de calidad (QA)? Indica niveles o valores concretos: disponibilidad 99.9%, latencia <100ms, throughput 1000rps.",
        "question_en": "What is the priority of quality attributes (QA)? Provide concrete values: availability 99.9%, latency <100ms, throughput 1000rps.",
        "rule": "_METRIC_PATTERN match",
    },
    {
        "field": "campo_6_restricciones",
        "question_es": "¿Cuáles son las restricciones técnicas del proyecto? Por ejemplo: lenguaje de programación, infraestructura, presupuesto, regulaciones. Escribe 'ninguna' si no hay restricciones.",
        "question_en": "What are the technical constraints of the project? For example: programming language, infrastructure, budget, regulations. Write 'none' if there are no constraints.",
        "rule": "_NEGATION_PATTERN OR len(value.strip()) >= 10",
    },
    {
        "field": "campo_7_decisiones",
        "question_es": "¿Existen decisiones de diseño previas que debamos respetar? Por ejemplo: patrones arquitectónicos ya definidos, tecnologías elegidas, integraciones existentes. Escribe 'ninguna' si no hay.",
        "question_en": "Are there prior design decisions that must be respected? For example: already defined architectural patterns, chosen technologies, existing integrations. Write 'none' if there are none.",
        "rule": "_NEGATION_PATTERN OR len(value.strip()) >= 10",
    },
]


_COPY_PASTE_ERRORS: dict[str, str] = {
    "es": "Parece que pegaste la pregunta como respuesta. Por favor describe tu sistema con tu información específica.",
    "en": "It looks like you copied the question as your answer. Please describe your system with your specific information.",
}


def validate_field(index: int, value: str) -> Tuple[bool, str]:
    # Pre-check: reject copy-pasted question text before any other rule.
    if _is_copy_paste(index, value):
        return (False, "copy_paste")

    if index in (0, 1, 3):
        tokens = value.split()
        if len(tokens) < 8:
            return (
                False,
                "La respuesta es demasiado corta. Necesita al menos 8 palabras con vocabulario técnico concreto (servicio, módulo, API, componente, etc.).",
            )
        if not _TECHNICAL_TERMS.search(value):
            return (
                False,
                "No se detectó vocabulario técnico. Menciona al menos un término como: servicio, módulo, API, componente, sistema, endpoint, microservicio, etc.",
            )
        return (True, "")

    if index == 2:
        if not _SOURCE_CATEGORIES.search(value):
            return (
                False,
                "No se identificó la fuente del estímulo. Indica si proviene de: usuario, sistema externo, evento interno, tiempo/timer o schedule.",
            )
        return (True, "")

    if index in (4, 5):
        if not _METRIC_PATTERN.search(value):
            return (
                False,
                "No se encontró ninguna métrica concreta. Incluye números, comparaciones o unidades como: <200ms, 500rps, 99.9%, p95, SLA, SLO, TPS.",
            )
        return (True, "")

    if index in (6, 7):
        if _NEGATION_PATTERN.search(value) or len(value.strip()) >= 10:
            return (True, "")
        return (
            False,
            "La respuesta es demasiado corta. Describe las restricciones/decisiones previas, o escribe 'ninguna' / 'none' / 'n/a' si no aplica.",
        )

    return (True, "")


_REPROMPT_ERRORS: dict[int, dict[str, str]] = {
    0: {
        "es": "La respuesta es demasiado corta. Necesita al menos 8 palabras con vocabulario técnico concreto (servicio, módulo, API, componente, etc.).",
        "en": "Answer too short. Need at least 8 words with concrete technical vocabulary (service, module, API, component, etc.).",
    },
    1: {
        "es": "La respuesta es demasiado corta. Necesita al menos 8 palabras con vocabulario técnico concreto (servicio, módulo, API, componente, etc.).",
        "en": "Answer too short. Need at least 8 words with concrete technical vocabulary (service, module, API, component, etc.).",
    },
    2: {
        "es": "No se identificó la fuente del estímulo. Indica si proviene de: usuario, sistema externo, evento interno, tiempo/timer o schedule.",
        "en": "Stimulus source not identified. Indicate if it comes from: user, external system, internal event, time/timer or schedule.",
    },
    3: {
        "es": "La respuesta es demasiado corta. Necesita al menos 8 palabras con vocabulario técnico concreto (servicio, módulo, API, componente, etc.).",
        "en": "Answer too short. Need at least 8 words with concrete technical vocabulary (service, module, API, component, etc.).",
    },
    4: {
        "es": "No se encontró ninguna métrica concreta. Incluye números, comparaciones o unidades como: <200ms, 500rps, 99.9%, p95, SLA, SLO, TPS.",
        "en": "No concrete metric found. Include numbers, comparisons or units such as: <200ms, 500rps, 99.9%, p95, SLA, SLO, TPS.",
    },
    5: {
        "es": "No se encontró ninguna métrica concreta. Incluye números, comparaciones o unidades como: <200ms, 500rps, 99.9%, p95, SLA, SLO, TPS.",
        "en": "No concrete metric found. Include numbers, comparisons or units such as: <200ms, 500rps, 99.9%, p95, SLA, SLO, TPS.",
    },
    6: {
        "es": "La respuesta es demasiado corta. Describe las restricciones/decisiones previas, o escribe 'ninguna' / 'none' / 'n/a' si no aplica.",
        "en": "Answer too short. Describe the constraints/prior decisions, or write 'none' / 'n/a' if not applicable.",
    },
    7: {
        "es": "La respuesta es demasiado corta. Describe las restricciones/decisiones previas, o escribe 'ninguna' / 'none' / 'n/a' si no aplica.",
        "en": "Answer too short. Describe the constraints/prior decisions, or write 'none' / 'n/a' if not applicable.",
    },
}


def reprompt_message(index: int, lang: str, value: str = "") -> str:
    _lang = lang if lang in ("es", "en") else "es"
    if value and _is_copy_paste(index, value):
        error_msg = _COPY_PASTE_ERRORS[_lang]
    else:
        error_msg = _REPROMPT_ERRORS[index][_lang]
    key = "question_es" if _lang == "es" else "question_en"
    question = INTAKE_SCRIPT[index][key]
    return f"{error_msg}\n\n{question}"


# ── Semantic validation (LLM second layer) ───────────────────────────────────

class _SemanticOut(TypedDict):
    valid: bool
    reason: str


_FIELD_SEMANTIC_DESCRIPTIONS: dict[int, str] = {
    0: "main system requirement: objective, involved components, and quality expectations",
    1: "main system components: services, modules, APIs, databases, or other relevant elements",
    2: "stimulus source: who or what triggers the system (user, external system, internal event, timer/schedule)",
    3: "stimulus or trigger: the specific event that activates system behavior",
    4: "operating environments with concrete metrics: normal load, overload, maintenance (e.g., p95<200ms, 500rps)",
    5: "quality attribute priorities with concrete numeric values: availability %, latency ms, throughput rps",
    6: "technical constraints: programming language, infrastructure, budget, regulations — or explicit negation",
    7: "prior design decisions: patterns, technologies, existing integrations — or explicit negation",
}

# v1 — fixed prompt; bump version string before changing wording
_SEMANTIC_PROMPT_V1 = """\
You are a strict validator for an architecture intake interview.
Decide only whether the answer below contains real, specific information about the user's own system.
Do NOT evaluate grammar, writing style, or technical accuracy.

Field {index} — {field_description}

User's answer: "{value}"

Reject (valid=false) if ANY of the following apply:
- The answer is a copy or near-paraphrase of the question with no original content about the user's system.
- The answer is entirely generic and could apply to any system (no concrete specifics about THIS system).
- The answer only names the expected category without any detail (e.g., just "usuario" for a stimulus-source field).

Accept (valid=true) if:
- The answer contains at least one concrete detail specific to the user's actual system.

When rejecting, write reason as a single short sentence in {lang}. When accepting, set reason to empty string.\
"""


async def validate_field_semantic(index: int, value: str, lang: str) -> Tuple[bool, str]:
    """LLM-based semantic validation. Call only after validate_field() returns True.

    Fail-open: returns (True, "") if the LLM call fails so infra errors never block the user.
    """
    from src.graph.resources import llm  # lazy import — avoids circular deps at module load

    _lang = lang if lang in ("es", "en") else "es"
    field_desc = _FIELD_SEMANTIC_DESCRIPTIONS.get(index, f"field {index}")
    prompt = _SEMANTIC_PROMPT_V1.format(
        index=index,
        field_description=field_desc,
        value=value[:500],
        lang=_lang,
    )
    try:
        out = await llm.with_structured_output(_SemanticOut).ainvoke(prompt)
        valid = bool(out.get("valid", True))
        reason = str(out.get("reason") or "").strip()
        if not valid and not reason:
            reason = (
                "La respuesta no contiene información específica sobre tu sistema."
                if _lang == "es"
                else "The answer does not contain specific information about your system."
            )
        return (valid, reason)
    except Exception as exc:
        log.warning("validate_field_semantic: LLM call failed (fail-open): %s", exc)
        return (True, "")


if __name__ == "__main__":
    # Caso 1 — campo 4 válido (métricas concretas)
    assert validate_field(4, "normal: p95<200ms; sobrecarga: 500rps; mantenimiento: 2h los domingos") == (True, "")

    # Caso 2 — campo 4 inválido (sin números ni comparaciones)
    ok, msg = validate_field(4, "el sistema funciona bien en producción normalmente")
    assert ok is False
    assert "métrica" in msg.lower() or "número" in msg.lower() or "concreto" in msg.lower()

    # Caso 3 — campo 2 inválido (sin categoría válida)
    ok, msg = validate_field(2, "cuando hay mucha carga en el servidor")
    assert ok is False

    # Caso 4 — campo 6 con negación explícita
    assert validate_field(6, "ninguna") == (True, "")

    # Caso 5 — campo 0 con vocabulario técnico suficiente
    assert validate_field(0, "el microservicio de pagos debe procesar transacciones en menos de 300ms") == (True, "")

    # Caso 6 — campo 4: pregunta pegada como respuesta → rechazado
    ok, reason = validate_field(
        4,
        "¿En qué ambientes o escenarios debe operar el sistema? "
        "Incluye métricas concretas: carga normal, sobrecarga, mantenimiento (ej: p95<200ms, 500rps).",
    )
    assert ok is False, "campo 4 copy-paste debería ser rechazado"
    assert reason == "copy_paste", f"esperaba reason='copy_paste', got '{reason}'"

    # Caso 7 — campo 4 válido con vocabulario de la pregunta (no es copy-paste)
    assert validate_field(4, "carga normal: p95<200ms; sobrecarga: 500rps; mantenimiento: 2h domingos") == (True, ""), \
        "respuesta válida con términos de la pregunta fue rechazada incorrectamente"

    # Caso 8 — reprompt_message con copy-paste muestra mensaje específico (es)
    msg = reprompt_message(4, "es", "¿En qué ambientes o escenarios debe operar el sistema? Incluye métricas concretas: carga normal, sobrecarga, mantenimiento (ej: p95<200ms, 500rps).")
    assert "pegaste" in msg.lower(), f"esperaba mensaje de copy-paste en es, got: {msg[:100]}"

    # Caso 9 — reprompt_message sin copy-paste muestra mensaje genérico
    msg = reprompt_message(4, "es", "el sistema funciona bien en producción")
    assert "métrica" in msg.lower(), f"esperaba mensaje genérico de métrica, got: {msg[:100]}"

    print("Todos los casos pasaron ✓")

    # ── Semantic validation (integration — requires LLM) ─────────────────────
    import asyncio

    # Caso 10 — campo 4 con métricas reales → pasa semántica
    sem_ok, sem_reason = asyncio.run(
        validate_field_semantic(
            4, "normal: 300rps p95<200ms, sobrecarga: 1500rps, mantenimiento domingos 2am", "es"
        )
    )
    assert sem_ok is True, f"campo 4 con métricas reales debería pasar semántica, got reason: {sem_reason}"

    # Caso 11 — campo 2 con "usuario" solo pasa determinista pero falla semántica
    # (la categoría está presente pero no describe nada del sistema real)
    sem_ok, sem_reason = asyncio.run(validate_field_semantic(2, "usuario", "es"))
    assert sem_ok is False, f"'usuario' solo debería fallar semántica, got sem_ok={sem_ok}"
    assert sem_reason, "esperaba razón de rechazo en español"

    print("Semantic validation tests ✓")
