from __future__ import annotations
import logging
import re
from typing import Tuple
from pydantic import BaseModel

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


# ── Unified extraction + validation (LLM second layer) ───────────────────────

class FieldValidation(BaseModel):
    valid: bool
    reason: str  # vacío si válido; razón concreta en idioma del usuario si no


class MultiFieldExtractionResult(BaseModel):
    extracted_fields: dict[str, str | None]   # campo_X -> texto extraído o None
    validation: dict[str, FieldValidation]    # campo_X -> resultado de validación


_ADD3_CRITERIA: dict[int, str] = {
    0: "Must describe a concrete system requirement — not generic. Needs objective, quality expectation, or involved components. 'A system that handles requests' is NOT sufficient.",
    1: "Must name specific system components with at least one characteristic each. 'Frontend and backend' is NOT sufficient. Needs actual services, APIs, modules, or databases.",
    2: "Must explicitly identify the source category (user / external system / internal event / time/timer) AND contextualize it to the actual system. Just 'usuario' with no context is NOT sufficient.",
    3: "Must describe the specific triggering event AND mention which components from campo_1 it interacts with. Generic responses like 'when user sends a request' are NOT sufficient.",
    4: "Must cover ALL THREE operational conditions: normal, overload, AND maintenance. Each needs numeric metrics (rps, ms, %, intervals). Covering only normal operation is NOT sufficient.",
    5: "Must list quality attributes with concrete numeric values. 'High availability' without a percentage is NOT sufficient. Needs availability %, latency ms, throughput rps, or similar.",
    6: "Must list concrete technical constraints or explicit negation (ninguna/none/n/a). Vague mention of constraints is NOT sufficient.",
    7: "Must describe concrete prior architectural decisions or explicit negation. Generic statements like 'we follow best practices' are NOT sufficient.",
}

# v1 — fixed prompt; bump version string before changing wording
_UNIFIED_PROMPT_V1 = """\
# Architecture Intake Validator — ADD 3.0 / v1

## PROJECT CONTEXT
{project_context_text}

## PENDING FIELDS (you must attempt to extract these)
{pending_fields_spec}

## ADD 3.0 SUFFICIENCY CRITERIA
{criteria_spec}

## USER MESSAGE
"{user_message}"

## TASK
1. For each pending field, extract the portion of the user message that specifically answers it. Set to null if not addressed.
2. For each non-null extraction, validate it against the ADD 3.0 criterion AND the project context.
3. Write rejection reasons in {lang}. Reasons must name what is specifically missing (not "provide more detail").
4. Accept if the answer contains at least one concrete detail specific to the user's actual system AND meets the criterion.

Respond ONLY with valid JSON:
{{
  "extracted_fields": {{
    "campo_X": "extracted text or null"
  }},
  "validation": {{
    "campo_X": {{
      "valid": true,
      "reason": ""
    }}
  }}
}}
"""


async def extract_and_validate_fields(
    user_message: str,
    pending_indices: list[int],
    project_context_text: str,
    lang: str,
) -> MultiFieldExtractionResult | None:
    """Single LLM call: extrae respuestas para todos los campos pendientes + valida ADD 3.0.

    Retorna None en cualquier fallo (fail-open — el caller maneja el fallback).
    Nunca lanza excepción.
    """
    from src.graph.resources import llm  # lazy import — evita circular deps

    if not pending_indices:
        return MultiFieldExtractionResult(extracted_fields={}, validation={})

    _lang = lang if lang in ("es", "en") else "es"
    q_key = "question_es" if _lang == "es" else "question_en"

    pending_fields_spec = "\n".join(
        f"- {INTAKE_SCRIPT[i]['field']}: {INTAKE_SCRIPT[i][q_key]}"
        for i in pending_indices
    )
    criteria_spec = "\n".join(
        f"- {INTAKE_SCRIPT[i]['field']}: {_ADD3_CRITERIA[i]}"
        for i in pending_indices
    )
    context_snippet = (project_context_text or "").strip()[:800] or "(no project context provided)"

    prompt = _UNIFIED_PROMPT_V1.format(
        project_context_text=context_snippet,
        pending_fields_spec=pending_fields_spec,
        criteria_spec=criteria_spec,
        user_message=user_message[:600],
        lang=_lang,
    )

    try:
        result = await llm.with_structured_output(MultiFieldExtractionResult).ainvoke(prompt)
        return result
    except Exception as exc:
        log.warning("extract_and_validate_fields: LLM call failed (fail-open): %s", exc)
        return None


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

    # ── Unified extraction + validation (integration — requires LLM) ──────────
    import asyncio

    # Caso 10 — campo 4 con métricas de las tres condiciones → válido
    result = asyncio.run(extract_and_validate_fields(
        "normal: 300rps p95<200ms, sobrecarga: 1500rps, mantenimiento domingos 2am",
        [4], "", "es",
    ))
    assert result is not None
    fv = result.validation.get("campo_4_ambientes")
    assert fv is not None and fv.valid is True, f"expected valid, got: {fv}"

    # Caso 11 — "usuario" solo → semánticamente inválido
    result = asyncio.run(extract_and_validate_fields("usuario", [2], "", "es"))
    assert result is not None
    fv = result.validation.get("campo_2_fuente")
    assert fv is not None and fv.valid is False and fv.reason, \
        f"expected invalid with reason, got: {fv}"

    print("Semantic validation tests ✓")
