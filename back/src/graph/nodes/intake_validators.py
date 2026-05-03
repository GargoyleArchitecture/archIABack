import re
from typing import Tuple

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


def validate_field(index: int, value: str) -> Tuple[bool, str]:
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


def reprompt_message(index: int, lang: str) -> str:
    _, error_msg = validate_field(index, "")
    key = "question_es" if lang == "es" else "question_en"
    question = INTAKE_SCRIPT[index][key]
    return f"{error_msg}\n\n{question}"


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

    print("Todos los casos pasaron ✓")
