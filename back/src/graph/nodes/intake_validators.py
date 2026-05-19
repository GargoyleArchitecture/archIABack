from __future__ import annotations
import json
import logging
import re
from typing import Literal, Tuple
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Layer A: broad architectural vocabulary (protocols, infra, patterns, domain actors)
_TECHNICAL_TERMS = re.compile(
    r"\b("
    # original terms + plurals
    r"módulos?|servicios?|api|apis|componentes?|sistemas?|requests?|eventos?|"
    r"endpoints?|base de datos|bases de datos|microservicios?|cach[eé]s?|colas?|"
    r"latencia|throughput|concurrencia|usuarios?|clientes?|servidores?|"
    # protocols
    r"rest(?:ful)?|grpc|https?|websockets?|webrtc|mqtt|soap|graphql|"
    # infra / platforms
    r"kafka|redis|postgres(?:ql)?|mongo(?:db)?|mysql|sqlite|s3|cdn|pop|sfu|mcu|"
    r"gateways?|proxys?|proxies|broker|brokers?|balanceador|nginx|docker|kubernetes|k8s|"
    r"lambda|serverless|contenedores?|"
    # security / compliance
    r"oauth|jwt|tls|mtls|hipaa|gdpr|rbac|sso|autenticaci[oó]n|autorizaci[oó]n|"
    r"cifrado|encriptaci[oó]n|"
    # architecture patterns
    r"monolito|event.driven|pub.?sub|cqrs|saga|circuit.?breaker|"
    # domain entities that act as users/actors in telemedicine and typical projects
    r"pacientes?|m[eé]dicos?|m[eé]dica|doctor(?:as?|es)?|operador(?:as?|es)?|administrador(?:as?|es)?|"
    # generic system vocabulary
    r"funciones?|aplicaci[oó]n|aplicaciones|integraci[oó]n|integraciones|"
    r"notificaci[oó]n|notificaciones|videollamadas?|sesiones?|almacenamiento|"
    r"mensajer[ií]a|despliegue|r[eé]plica|r[eé]plicas|cpu|memoria|"
    # BUG-038: finance / integration vocabulary so fintech inputs like
    # "Gateway ISO 20022", "core bancario", "RTGS", "bus de eventos",
    # "payment gateway", "webhook" are recognised as technical terms.
    r"banco|bancos|banco origen|payment gateway|payment\s*gateway|pasarela de pagos|"
    r"iso\s?20022|swift|sepa|ach|rtgs|core\s+bancario|core\s+banking|core\s+ledger|"
    r"webhook|webhooks|bus de eventos|event bus|stream|streaming|"
    r"queue|queues|cola de mensajes|message queue|cola de eventos|"
    r"liquidaci[oó]n|settlement|conciliaci[oó]n|reconciliation|ledger|"
    r"tokenizaci[oó]n|tokenization|kyc|aml|antifraude|antifraud|fraud|fraude"
    r")\b",
    re.IGNORECASE,
)

# Layer B: bare uppercase acronyms not caught by Layer A word-boundary matching
_TECHNICAL_ACRONYM_RE = re.compile(
    r"\b(?:API|REST|HTTP|HTTPS|GRPC|RPC|SDK|UI|UX|DB|SQL|JSON|XML|YAML|UUID|"
    r"IP|TCP|UDP|TLS|SSL|JWT|SSO|CDN|SLA|SLO|SFU|MCU|CI|CD|"
    r"EHR|EMR|HIPAA|GDPR|RBAC|SAML|OIDC|CPU|RAM|VPN|DNS)\b",
)

_SOURCE_CATEGORIES = re.compile(
    r"\b("
    # user / actor synonyms
    r"usuario|user|pacientes?|m[eé]dicos?|m[eé]dica|doctor(?:as?|es)?|clientes?|"
    r"actores?|operador(?:as?|es)?|administrador(?:as?|es)?|admin|"
    # external system synonyms
    r"sistema externo|external system|servicio externo|api externa|"
    r"terceros?|integraci[oó]n externa|"
    # internal event
    r"evento interno|internal event|"
    # time / schedule synonyms
    r"tiempo|time|timer|schedule|cron|scheduler|tarea programada|job programado|timeout|"
    # BUG-038: finance / integration sources so fintech intake answers
    # (e.g. "banco origen", "API REST", "webhook", "cola Kafka", "payment gateway")
    # are accepted as stimulus sources.
    r"banco|banco origen|core\s+bancario|core\s+banking|"
    r"api\s+rest|rest\s+api|webhook|webhooks|"
    r"cola|queue|stream|streaming|"
    r"payment\s+gateway|pasarela\s+de\s+pagos|"
    r"iso\s?20022|swift|sepa|ach|rtgs|mtls|tls|broker"
    r")\b",
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
        if len(common) / len(q_tokens) > 0.80:
            return True
    return False


INTAKE_SCRIPT = [
    {
        "field": "campo_0_requerimiento",
        "question_es": "¿Cuál es el requerimiento principal del sistema que deseas diseñar? Describe el objetivo principal, el alcance funcional y las expectativas de calidad.",
        "question_en": "What is the main requirement of the system you want to design? Describe the main goal, the functional scope, and quality expectations.",
        "rule": "len(tokens) >= 8 AND at least one technical term",
    },
    {
        "field": "campo_1_alcance_funcional",
        "question_es": "¿Cuáles son las principales funcionalidades o áreas de responsabilidad que el sistema debe soportar? Si el sistema ya existe, describe qué hace actualmente en términos funcionales. Si estás construyendo desde cero, describe las capacidades que debe tener. No nombres componentes arquitectónicos — esos emergen del proceso de diseño ADD.",
        "question_en": "What are the main functionalities or areas of responsibility the system must support? If the system already exists, describe what it currently does in functional terms. If you are building from scratch, describe the capabilities it should have. Do not name architectural components — those emerge from the ADD design process.",
        "rule": "len(tokens) >= 8 AND at least one technical term",
    },
    {
        "field": "campo_2_fuente",
        "question_es": "¿Cuál es la fuente del estímulo? Por ejemplo: usuario, sistema externo, evento interno, tiempo/timer.",
        "question_en": "What is the source of the stimulus? For example: user, external system, internal event, time/timer.",
        "rule": "_SOURCE_CATEGORIES match",
        "optional": True,
    },
    {
        "field": "campo_3_estimulo",
        "question_es": "¿Cuál es el estímulo o trigger que activa el comportamiento del sistema? Describe el evento específico que dispara la respuesta.",
        "question_en": "What is the stimulus or trigger that activates system behavior? Describe the specific event that triggers the response.",
        "rule": "len(tokens) >= 5 AND at least one technical term",
        "optional": True,
    },
    {
        "field": "campo_4_ambientes",
        "question_es": "¿En qué ambientes o escenarios debe operar el sistema? Incluye métricas concretas para carga normal y al menos una condición de pico o sobrecarga (ej: normal p95<200ms a 100rps; pico p99<500ms a 800rps). Las ventanas de mantenimiento son opcionales.",
        "question_en": "In what environments or scenarios must the system operate? Include concrete metrics for normal load and at least one peak/overload condition (e.g., normal p95<200ms at 100rps; peak p99<500ms at 800rps). Maintenance windows are optional.",
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

    if index in (0, 1):
        tokens = value.split()
        if len(tokens) < 8:
            return (
                False,
                "La respuesta es demasiado corta. Necesita al menos 8 palabras con vocabulario técnico concreto (servicio, módulo, API, componente, etc.).",
            )
        if not (_TECHNICAL_TERMS.search(value) or _TECHNICAL_ACRONYM_RE.search(value)):
            return (
                False,
                "No se detectó vocabulario técnico. Menciona al menos un término arquitectónico como: servicio, API, componente, microservicio, REST, gRPC, WebRTC, kafka, JWT, gateway, etc.",
            )
        return (True, "")

    # BUG-030: campo_3 (stimulus) uses a lower word threshold than campo_0/1
    # because a concise technical description (5+ words) is sufficient.
    if index == 3:
        tokens = value.split()
        if len(tokens) < 5:
            return (
                False,
                "La respuesta es demasiado corta. Necesita al menos 5 palabras describiendo el evento técnico que dispara el comportamiento.",
            )
        if not (_TECHNICAL_TERMS.search(value) or _TECHNICAL_ACRONYM_RE.search(value)):
            return (
                False,
                "No se detectó vocabulario técnico. Menciona al menos un término arquitectónico como: servicio, API, componente, microservicio, REST, gRPC, WebRTC, kafka, JWT, gateway, etc.",
            )
        return (True, "")

    if index == 2:
        if not _SOURCE_CATEGORIES.search(value):
            return (
                False,
                "No se identificó la fuente del estímulo. Indica si proviene de: usuario/paciente/médico, sistema externo/API externa, evento interno, tiempo/timer/cron.",
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
        "es": "La respuesta es demasiado corta. Necesita al menos 8 palabras con vocabulario técnico concreto (servicio, API, REST, gRPC, WebRTC, microservicio, gateway, etc.).",
        "en": "Answer too short. Need at least 8 words with concrete technical vocabulary (service, API, REST, gRPC, WebRTC, microservice, gateway, etc.).",
    },
    1: {
        "es": "La respuesta es demasiado corta. Necesita al menos 8 palabras con vocabulario técnico concreto (servicio, API, REST, gRPC, WebRTC, microservicio, gateway, etc.).",
        "en": "Answer too short. Need at least 8 words with concrete technical vocabulary (service, API, REST, gRPC, WebRTC, microservice, gateway, etc.).",
    },
    2: {
        "es": "No se identificó la fuente del estímulo. Indica si proviene de: usuario/paciente/médico, sistema externo/API externa, evento interno, tiempo/timer/cron.",
        "en": "Stimulus source not identified. Indicate if it comes from: user/patient/doctor, external system/external API, internal event, time/timer/cron.",
    },
    3: {
        "es": "La respuesta es demasiado corta. Necesita al menos 5 palabras describiendo el evento técnico que dispara el comportamiento.",
        "en": "Answer too short. Need at least 5 words describing the technical event that triggers the behavior.",
    },
    4: {
        "es": "No se encontró ninguna métrica concreta. Tu respuesta debe cubrir al menos carga normal Y sobrecarga/pico, cada una con números/unidades: <200ms, 500rps, 99.9%, p95, SLA, SLO, TPS.",
        "en": "No concrete metric found. Your answer must cover at least normal load AND overload/peak, each with numbers/units: <200ms, 500rps, 99.9%, p95, SLA, SLO, TPS.",
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


class FieldAssessment(BaseModel):
    status: Literal["answered_valid", "answered_invalid", "not_addressed"]
    extracted_text: str | None = None
    reason: str = ""
    missing_details: list[str] = Field(default_factory=list)
    repair_prompt: str = ""


class MultiFieldAssessmentResult(BaseModel):
    fields: dict[str, FieldAssessment]


_ADD3_CRITERIA: dict[int, str] = {
    0: "Must describe a concrete system requirement — not generic. Needs objective, functional scope, or quality expectation. 'A system that handles requests' is NOT sufficient. Do NOT require the user to name architectural components (services, APIs, modules, databases) — those are ADD design outputs, not intake inputs.",
    1: "Must describe concrete functionalities or areas of responsibility the system must support. Whether building from scratch or describing an existing system, the answer must state what the system does or should do — not how it is structured. 'It handles requests' is NOT sufficient. Needs specific capabilities such as payment processing, user authentication, inventory management, or hotel search. Do NOT require the user to name architectural components (services, APIs, modules, databases) — those are ADD design outputs, not intake inputs.",
    2: "Must explicitly identify the source category (user / external system / internal event / time/timer) AND contextualize it to the actual system. Just 'usuario' with no context is NOT sufficient.",
    3: "Two-tier rule — diagnosis level only, not solution design. (A) Triggers WITH performance metrics (latency, TPM, concurrent users, timeouts): identify the system component that receives the event, indicate sync or async interaction, reference the endpoint or event name (approximate is acceptable), and include the associated metric (p95, TPM, timeout). (B) Triggers WITHOUT metrics (timers, webhooks, deployments, security events, internal events): sufficient to name the trigger and the component that processes it — no exact endpoint, retry policy, or cooldown required. The validator must NOT require in any case: autoscaling policies (threshold, cooldown, min/max replicas), HTTP response codes, retry or backoff policies, detailed failover mechanisms, or rollback behavior. These are solution details, not diagnosis details.",
    4: "Must cover at least two operational conditions with numeric metrics: normal load AND at least one stress condition (overload, peak, or burst). Maintenance windows or RTO/RPO are optional. Covering only normal operation is NOT sufficient.",
    5: "Must list quality attributes with concrete numeric values. 'High availability' without a percentage is NOT sufficient. Needs availability %, latency ms, throughput rps, or similar.",
    6: "Must list concrete technical constraints or explicit negation (ninguna/none/n/a). Vague mention of constraints is NOT sufficient.",
    7: "Must describe concrete prior architectural decisions or explicit negation. Generic statements like 'we follow best practices' are NOT sufficient.",
}

# v2 — adaptive repair guidance + explicit answer status
_UNIFIED_PROMPT_V2 = """\
# Architecture Intake Validator — ADD 3.0 / v2

## PROJECT CONTEXT
{project_context_text}

## PENDING FIELDS (you must attempt to extract these)
{pending_fields_spec}

## ADD 3.0 SUFFICIENCY CRITERIA
{criteria_spec}

## USER MESSAGE
"{user_message}"

## TASK
1. For each pending field, decide if the message: fully answers it, attempts it but is insufficient, or does not address it.
2. Use exactly one of these statuses per field: answered_valid, answered_invalid, not_addressed.
3. If answered_valid or answered_invalid, extract only the portion of the user message that answers that field.
4. Validate each attempted answer against the ADD 3.0 criterion AND the project context.
5. If answered_invalid, explain exactly what is missing in {lang}, list EVERY missing sub-aspect in `missing_details` (e.g. for `campo_4_ambientes` list each of "normal", "sobrecarga" that is absent — never just the first one), and write a short repair prompt telling the user exactly what to add or fix.
6. If answered_invalid, write a concise repair prompt telling the user what is missing or needs clarification — do NOT ask them to re-send what they already provided.
7. If not_addressed, set extracted_text to null and leave reason / repair_prompt empty.
8. Accept only if the answer contains concrete details specific to the user's actual system and meets the criterion.

Respond ONLY with valid JSON:
{{
  "fields": {{
    "campo_X": {{
      "status": "answered_valid",
      "extracted_text": "extracted text or null",
      "reason": "",
      "missing_details": [],
      "repair_prompt": ""
    }}
  }}
}}
"""


async def extract_and_validate_fields(
    user_message: str,
    pending_indices: list[int],
    project_context_text: str,
    lang: str,
) -> MultiFieldAssessmentResult | None:
    """Single LLM call: extrae respuestas para todos los campos pendientes + valida ADD 3.0.

    Retorna None en cualquier fallo (fail-open — el caller maneja el fallback).
    Nunca lanza excepción.
    """
    from src.graph.resources import llm  # lazy import — evita circular deps

    if not pending_indices:
        return MultiFieldAssessmentResult(fields={})

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

    prompt = _UNIFIED_PROMPT_V2.format(
        project_context_text=context_snippet,
        pending_fields_spec=pending_fields_spec,
        criteria_spec=criteria_spec,
        user_message=user_message[:5000],
        lang=_lang,
    )

    try:
        response = await llm.ainvoke(prompt)
        raw = getattr(response, "content", str(response)).strip()
        # Strip markdown code fences some models add around JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return MultiFieldAssessmentResult(**data)
    except Exception as exc:
        log.warning("extract_and_validate_fields: LLM call failed (fail-open): %s", exc)
        return None


def build_repair_prompt(index: int, lang: str, reason: str = "") -> str:
    _lang = lang if lang in ("es", "en") else "es"
    question = INTAKE_SCRIPT[index]["question_es" if _lang == "es" else "question_en"]

    if _lang == "es":
        templates = {
            0: "Reescribe tu respuesta indicando el objetivo principal del sistema, el alcance funcional y al menos una expectativa de calidad concreta. No nombres componentes arquitectónicos — esos emergen del proceso de diseño ADD, no en esta etapa.",
            1: "Reescribe tu respuesta describiendo las funcionalidades principales o áreas de responsabilidad del sistema. Si estás construyendo desde cero, menciona las capacidades que quieres que el sistema tenga. No nombres componentes arquitectónicos — eso se determina durante el proceso de diseño ADD, no en esta etapa.",
            2: "Reescribe tu respuesta indicando quién genera el estímulo y su contexto en tu sistema, por ejemplo usuario final, sistema externo, evento interno o timer.",
            3: "Reescribe tu respuesta describiendo el evento específico que dispara el comportamiento. Si tiene métricas (latencia, TPM, usuarios concurrentes, timeouts), indica el componente que lo procesa y si la llamada es síncrona o asíncrona. Si no tiene métricas (timer, webhook, despliegue, actor malicioso), basta con nombrar el evento de forma concreta.",
            4: "Reescribe tu respuesta cubriendo al menos carga normal Y sobrecarga/pico, e incluye métricas numéricas en cada caso (ms, rps, porcentajes).",
            5: "Reescribe tu respuesta listando los atributos de calidad prioritarios con valores concretos, por ejemplo disponibilidad 99.9%, latencia <100ms o throughput 1000rps.",
            6: "Reescribe tu respuesta indicando restricciones técnicas concretas, o escribe 'ninguna' si realmente no aplica.",
            7: "Reescribe tu respuesta indicando decisiones de diseño previas concretas que deban respetarse, o escribe 'ninguna' si no existe ninguna.",
        }
    else:
        templates = {
            0: "Rewrite your answer stating the system's main goal, the functional scope, and at least one concrete quality expectation. Do not name architectural components — those emerge from the ADD design process, not at this stage.",
            1: "Rewrite your answer describing the main functionalities or areas of responsibility of the system. If you are building from scratch, mention the capabilities you want the system to have. Do not name architectural components — those are determined during the ADD design process, not at this stage.",
            2: "Rewrite your answer stating who produces the stimulus and its context in your system, for example an end user, external system, internal event, or timer.",
            3: "Rewrite your answer describing the specific event that triggers the behavior. If it has associated metrics (latency, TPM, concurrent users, timeouts), indicate which component it interacts with and whether the call is sync or async. If it has no metrics (timer, webhook, deployment, malicious actor), naming the event specifically is sufficient.",
            4: "Rewrite your answer covering at least normal load AND overload/peak, with numeric metrics for each case (ms, rps, percentages).",
            5: "Rewrite your answer listing the priority quality attributes with concrete values, for example availability 99.9%, latency <100ms, or throughput 1000rps.",
            6: "Rewrite your answer stating concrete technical constraints, or write 'none' if there are truly none.",
            7: "Rewrite your answer stating concrete prior design decisions that must be respected, or write 'none' if there are none.",
        }

    prompt = templates[index]
    if reason:
        return f"{reason}\n\n{prompt}\n\n{question}"
    return f"{prompt}\n\n{question}"


async def extract_and_validate_fields_legacy(
    user_message: str,
    pending_indices: list[int],
    project_context_text: str,
    lang: str,
) -> MultiFieldExtractionResult | None:
    """Backward-compatible adapter for old callers/tests."""
    result = await extract_and_validate_fields(
        user_message=user_message,
        pending_indices=pending_indices,
        project_context_text=project_context_text,
        lang=lang,
    )
    if result is None:
        return None

    extracted_fields: dict[str, str | None] = {}
    validation: dict[str, FieldValidation] = {}
    for field_name, assessment in result.fields.items():
        extracted_fields[field_name] = assessment.extracted_text
        validation[field_name] = FieldValidation(
            valid=assessment.status == "answered_valid",
            reason=assessment.reason,
        )
    return MultiFieldExtractionResult(extracted_fields=extracted_fields, validation=validation)


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
