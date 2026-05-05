# -*- coding: utf-8 -*-
import logging
import re
from datetime import datetime, timezone

from src.graph.resources import llm
from src.graph.state import GraphState
from src.graph.nodes.intake_validators import (
    INTAKE_SCRIPT,
    validate_field,
    reprompt_message,
    extract_and_validate_fields,
    build_repair_prompt,
)
from src.ledger.store import (
    transition_phase,
    append_decision,
    load_ledger,
    save_ledger,
    LedgerValidationError,
    LedgerConcurrencyError,
)
from src.ledger.types import Phase, PhaseTransition

log = logging.getLogger(__name__)

_HAS_ASRS_RE = re.compile(
    r"\bsí\b|yes\b|tengo\b|ya defin[ií]\b|ya hay\b",
    re.IGNORECASE,
)
_NO_ASRS_RE = re.compile(
    r"\bno\b|ninguno\b|genera\b|prop[oó]n\b|propone\b|t[uú]\b|usted\b",
    re.IGNORECASE,
)

# Digression detection: a message is a digression only when the *intent* is to
# ask or request something off-topic from intake, NOT when architectural terms
# appear incidentally inside a descriptive answer.
# Rule: must have BOTH (a) a question or meta-imperative form AND (b) meta-terms.
_DIGRESSION_TERMS_RE = re.compile(
    r"\b(asr|diagrama|diagram|t[aá]cticas|tactics|estilo|style|"
    r"patr[oó]n|patron|pattern|add\b|atam|togaf|"
    r"eval[uú]a|evaluate|analiza|analyze|genera|generate|"
    r"dise[nñ]a|disena|design|arquitectura|architecture)\b",
    re.IGNORECASE,
)
# Question form: contains "?" or starts with interrogative word
_DIGRESSION_QUESTION_RE = re.compile(
    r"[?¿]"
    r"|^\s*[¿¡]"
    r"|^\s*(qué|que|cómo|como|cuál|cual|por qué|para qué|"
    r"what|how|which|why|explain|define|describe|tell me|"
    r"cuéntame|cuentame|explícame|explicame)\b",
    re.IGNORECASE | re.MULTILINE,
)
# Explicit meta-imperative at start of message (request for an off-topic action)
_DIGRESSION_IMPERATIVE_RE = re.compile(
    r"^\s*(muéstrame|muestrame|crea|genera|propón|propon|propone|"
    r"diseña|disena|evalúa|evalua|analiza|show me|create|generate|"
    r"propose|design|draw|build me|explain|describe)\b",
    re.IGNORECASE,
)


def _is_digression(uq: str) -> bool:
    """True only when the message is a question or explicit meta-request, not a descriptive answer."""
    if not _DIGRESSION_TERMS_RE.search(uq):
        return False
    return bool(_DIGRESSION_QUESTION_RE.search(uq)) or bool(_DIGRESSION_IMPERATIVE_RE.search(uq))

_ASR_QUESTION_ES = (
    "Ya tengo toda la información necesaria. "
    "¿Quieres que proponga los ASRs o ya tienes alguno definido?"
)
_ASR_QUESTION_EN = (
    "I have all the information needed. "
    "Would you like me to propose the ASRs, or do you already have some defined?"
)

_FIELD_LABELS: dict[int, dict[str, str]] = {
    0: {"es": "requerimiento",        "en": "requirement"},
    1: {"es": "componentes",          "en": "components"},
    2: {"es": "fuente del estímulo",  "en": "stimulus source"},
    3: {"es": "estímulo",             "en": "stimulus"},
    4: {"es": "ambientes",            "en": "environments"},
    5: {"es": "prioridad QA",         "en": "QA priority"},
    6: {"es": "restricciones",        "en": "constraints"},
    7: {"es": "decisiones previas",   "en": "prior decisions"},
}

_WELCOME_ES = (
    "¡Hola! Soy ArchIA. Antes de generar los Atributos de Calidad y Escenarios de Calidad (ASRs) "
    "necesito conocer bien tu proyecto. Te haré 8 preguntas cortas sobre el contexto del sistema.\n\n"
    "Empecemos con la primera:"
)
_WELCOME_EN = (
    "Hi! I'm ArchIA. Before generating the Architecture Significant Requirements (ASRs) "
    "I need to understand your project. I'll ask you 8 short questions about the system context.\n\n"
    "Let's start with the first one:"
)


def _welcome_message(lang: str) -> str:
    return _WELCOME_ES if lang == "es" else _WELCOME_EN


def _llm_digression(uq: str, lang: str) -> str:
    if lang == "es":
        prompt = f"Responde en 1-2 líneas esta pregunta de arquitectura: {uq[:200]}"
    else:
        prompt = f"Answer in 1-2 lines this architecture question: {uq[:200]}"
    response = llm.invoke(prompt)
    return str(response.content).strip()


def _build_feedback(
    saved: list[int],
    failed: list[dict],
    next_index: int,
    lang: str,
) -> str:
    parts: list[str] = []

    if saved:
        labels = [_FIELD_LABELS[i][lang] for i in saved]
        parts.append(
            f"Guardé: {', '.join(labels)}." if lang == "es"
            else f"Saved: {', '.join(labels)}."
        )

    if failed:
        failed_sorted = sorted(failed, key=lambda item: int(item["index"]))
        priority = failed_sorted[0]
        i = int(priority["index"])
        reason = str(priority.get("reason") or "").strip()
        repair_prompt = str(priority.get("repair_prompt") or "").strip()
        if reason:
            parts.append(f"→ {_FIELD_LABELS[i][lang]}: {reason}")
        parts.append(repair_prompt or build_repair_prompt(i, lang, reason))
    elif next_index < 8:
        q_key = "question_es" if lang == "es" else "question_en"
        parts.append(INTAKE_SCRIPT[next_index][q_key])

    return "\n\n".join(parts)


def _semantic_default_reason(lang: str) -> str:
    return (
        "La respuesta no contiene información específica sobre tu sistema."
        if lang == "es"
        else "The answer does not contain specific information about your system."
    )


def _failed_entry(index: int, lang: str, reason: str, repair_prompt: str = "") -> dict:
    clean_reason = (reason or "").strip() or _semantic_default_reason(lang)
    clean_repair = (repair_prompt or "").strip() or build_repair_prompt(index, lang, clean_reason)
    return {
        "index": index,
        "reason": clean_reason,
        "repair_prompt": clean_repair,
    }


async def _process_intake_turn(
    uq: str,
    intake_fields: dict,
    current_index: int,
    project_context_text: str,
    lang: str,
 ) -> tuple[dict, list[int], list[dict]]:
    """Extracción + validación multi-campo para un turno.

    Returns: (updated_intake_fields, saved_indices, failed_list[dict])
    Fail-open: si LLM falla, intenta determinista solo en current_index.
    """
    pending_indices = [
        i for i, s in enumerate(INTAKE_SCRIPT) if s["field"] not in intake_fields
    ]

    result = await extract_and_validate_fields(
        uq, pending_indices, project_context_text, lang
    )

    saved: list[int] = []
    failed: list[dict] = []

    if result is None:
        # Fail-open: determinista únicamente, campo activo únicamente
        det_ok, _ = validate_field(current_index, uq)
        if det_ok:
            intake_fields = dict(intake_fields)
            intake_fields[INTAKE_SCRIPT[current_index]["field"]] = uq
            saved.append(current_index)
        # Si det falla → saved=[], failed=[] → caller usa reprompt_message
        return intake_fields, saved, failed

    for i in pending_indices:
        field_name = INTAKE_SCRIPT[i]["field"]
        assessment = result.fields.get(field_name)
        if assessment is None or assessment.status == "not_addressed":
            continue

        extracted = (assessment.extracted_text or "").strip()
        if not extracted and assessment.status == "answered_invalid":
            failed.append(
                _failed_entry(i, lang, assessment.reason, assessment.repair_prompt)
            )
            continue
        if not extracted:
            continue

        # Primera línea: determinista
        det_ok, det_reason = validate_field(i, extracted)
        if not det_ok:
            failed.append(_failed_entry(i, lang, det_reason))
            continue

        # Segunda línea: semántica ADD 3.0 (ya computada en el LLM call)
        if assessment.status == "answered_invalid":
            failed.append(
                _failed_entry(
                    i,
                    lang,
                    assessment.reason or _semantic_default_reason(lang),
                    assessment.repair_prompt,
                )
            )
            continue

        intake_fields = dict(intake_fields)
        intake_fields[field_name] = extracted
        saved.append(i)

    return intake_fields, saved, failed


async def intake_node(state: GraphState) -> GraphState:
    uq = (state.get("userQuestion") or "").strip()
    lang = state.get("language") or "es"
    intake_fields = dict(state.get("intake_fields") or {})
    current_index = next(
        (i for i, s in enumerate(INTAKE_SCRIPT) if s["field"] not in intake_fields),
        8,
    )
    project_context_text = state.get("project_context_text") or ""
    intake_complete = state.get("intake_complete") or False

    asr_question = _ASR_QUESTION_ES if lang == "es" else _ASR_QUESTION_EN

    # Rama A: intake completo — procesar respuesta del arquitecto sobre ASRs
    if intake_complete:
        _user_id    = (state.get("user_id_for_prefs") or "").strip()
        _project_id = (state.get("project_id") or "").strip() or None
        _ts = datetime.now(timezone.utc).isoformat()

        if _HAS_ASRS_RE.search(uq):
            # A1 — el arquitecto ya tiene ASRs propios
            if _user_id:
                try:
                    append_decision(_user_id, _project_id, {
                        "id": "",
                        "kind": "constraint",
                        "phase": Phase.INTAKE.value,
                        "iteration": 0,
                        "qa": "",
                        "parents": [],
                        "payload": {"existing_asrs": [uq], "source": "architect_provided"},
                        "rationale": "",
                        "sources": [],
                        "status": "active",
                        "parent_status": "ok",
                        "superseded_by": None,
                        "rejection_reason": None,
                        "created_at": "",
                        "created_by_node": "intake_node",
                    })
                    transition_phase(_user_id, _project_id, PhaseTransition(
                        from_phase="INTAKE",
                        to_phase="ASR",
                        iteration=1,
                        triggered_by="user_request",
                        user_message=uq,
                        skipped_phases=[],
                        timestamp=_ts,
                    ))
                except (LedgerValidationError, LedgerConcurrencyError, Exception) as _exc:
                    log.warning("intake_node: ledger error (nonfatal): %s", _exc)

            _msg_es = (
                "Perfecto, he registrado tus ASRs. Los analizaré y refinaré si es necesario.\n"
                "Pasamos a la fase ASR."
            )
            _msg_en = (
                "Got it, I've registered your ASRs. I'll analyze and refine them as needed.\n"
                "Moving to the ASR phase."
            )
            return {
                **state,
                "intake_fields": intake_fields,
                "intake_current_field": 8,
                "intake_complete": True,
                "endMessage": _msg_es if lang == "es" else _msg_en,
                "nextNode": "unifier",
                "intent": "intake",
            }

        if _NO_ASRS_RE.search(uq):
            # A2 — ArchIA propone los ASRs.
            # Persists intake_v1 in the ledger, mirrors current_phase="ASR" in state,
            # and routes to asr_node in this same turn via the conditional intake edge.
            _updated_ledger = None
            if _user_id:
                try:
                    _ledger = load_ledger(_user_id, _project_id)
                    _ledger["project_context"]["intake_v1"] = intake_fields
                    save_ledger(_user_id, _ledger, _project_id)
                    transition_phase(_user_id, _project_id, PhaseTransition(
                        from_phase="INTAKE",
                        to_phase="ASR",
                        iteration=1,
                        triggered_by="user_request",
                        user_message=uq,
                        skipped_phases=[],
                        timestamp=_ts,
                    ))
                    _updated_ledger = _ledger
                except (LedgerValidationError, LedgerConcurrencyError, Exception) as _exc:
                    log.warning("intake_node: ledger error (nonfatal): %s", _exc)

            _a2: dict = {
                **state,
                "intake_fields": intake_fields,
                "intake_current_field": 8,
                "intake_complete": True,
                "current_phase": "ASR",  # mirror ledger transition so supervisor skips INTAKE gate
                "endMessage": "",         # asr_node will set the real response
                "nextNode": "asr",
                "intent": "asr",
            }
            if _updated_ledger is not None:
                # Provide asr_node with intake_v1 in state so intake context is injected
                # into the ASR prompt without waiting for the next context_loader reload.
                _a2["ledger"] = _updated_ledger
            return _a2

        # A3 — respuesta ambigua: repregunta con más claridad
        _clarify_es = (
            "No estoy seguro de entenderte. ¿Ya tienes ASRs definidos que quieras compartir, "
            "o prefieres que yo los proponga basándome en el contexto que me diste?"
        )
        _clarify_en = (
            "I'm not sure I understood. Do you already have ASRs defined that you'd like to share, "
            "or would you prefer that I propose them based on the context you provided?"
        )
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": 8,
            "intake_complete": True,
            "endMessage": _clarify_es if lang == "es" else _clarify_en,
            "nextNode": "unifier",
            "intent": "intake",
        }

    # Rama B: todos los campos validados en este turno
    if current_index >= 8:
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": 8,
            "intake_complete": True,
            "endMessage": asr_question,
            "nextNode": "unifier",
            "intent": "intake",
        }

    # Rama C: primer turno (sin campos previos) — validar o dar bienvenida
    if current_index == 0 and not intake_fields:
        if _is_digression(uq):
            brief = _llm_digression(uq, lang)
            re_question = INTAKE_SCRIPT[0][f"question_{lang}"]
            sep = "Para continuar necesito que respondas" if lang == "es" else "To continue I need you to answer"
            end_msg = f"{brief}\n\n---\n{sep}: {re_question}"
            return {
                **state,
                "intake_fields": intake_fields,
                "intake_current_field": 0,
                "intake_complete": False,
                "endMessage": end_msg,
                "nextNode": "unifier",
                "intent": "intake",
            }

        intake_fields, saved, failed = await _process_intake_turn(
            uq, intake_fields, 0, project_context_text, lang
        )
        new_index = next(
            (i for i, s in enumerate(INTAKE_SCRIPT) if s["field"] not in intake_fields), 8
        )
        target_index = min((int(item["index"]) for item in failed), default=new_index)

        if not saved and not failed:
            # Nada extraído (saludo, etc.) → bienvenida suave
            end_msg = f"{_welcome_message(lang)}\n\n{INTAKE_SCRIPT[0][f'question_{lang}']}"
        elif saved and new_index >= 8:
            # Todo respondido en el primer mensaje
            return {
                **state,
                "intake_fields": intake_fields,
                "intake_current_field": 8,
                "intake_complete": True,
                "endMessage": _build_feedback(saved, failed, 8, lang) + f"\n\n{asr_question}",
                "nextNode": "unifier",
                "intent": "intake",
            }
        else:
            end_msg = _build_feedback(saved, failed, target_index if saved or failed else 0, lang)

        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": target_index if saved or failed else 0,
            "intake_complete": False,
            "endMessage": end_msg,
            "nextNode": "unifier",
            "intent": "intake",
        }

    # Rama D: turno normal (campos en progreso)
    if _is_digression(uq):
        brief = _llm_digression(uq, lang)
        re_question = INTAKE_SCRIPT[current_index][f"question_{lang}"]
        sep = "Para continuar necesito que respondas" if lang == "es" else "To continue I need you to answer"
        end_msg = f"{brief}\n\n---\n{sep}: {re_question}"
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": current_index,
            "intake_complete": False,
            "endMessage": end_msg,
            "nextNode": "unifier",
            "intent": "intake",
        }

    intake_fields, saved, failed = await _process_intake_turn(
        uq, intake_fields, current_index, project_context_text, lang
    )
    new_index = next(
        (i for i, s in enumerate(INTAKE_SCRIPT) if s["field"] not in intake_fields), 8
    )
    target_index = min((int(item["index"]) for item in failed), default=new_index)

    if not saved and not failed:
        # Fail-open + determinista falló: reprompt campo activo
        end_msg = reprompt_message(current_index, lang, uq)
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": current_index,
            "intake_complete": False,
            "endMessage": end_msg,
            "nextNode": "unifier",
            "intent": "intake",
        }

    if new_index >= 8:
        summary = _build_feedback(saved, failed, 8, lang)
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": 8,
            "intake_complete": True,
            "endMessage": f"{summary}\n\n{asr_question}" if summary else asr_question,
            "nextNode": "unifier",
            "intent": "intake",
        }

    end_msg = _build_feedback(saved, failed, target_index, lang)
    return {
        **state,
        "intake_fields": intake_fields,
        "intake_current_field": target_index,
        "intake_complete": False,
        "endMessage": end_msg,
        "nextNode": "unifier",
        "intent": "intake",
    }


if __name__ == "__main__":
    import asyncio

    def _base(intake_fields, idx, uq, lang="es"):
        return {
            "userQuestion": uq,
            "language": lang,
            "intake_fields": intake_fields,
            "intake_current_field": idx,
            "intake_complete": False,
            "messages": [],
            "nextNode": "unifier",
            "intent": "general",
            "endMessage": "",
            "turn_messages": [],
        }

    async def _run_tests():
        # Test 1 — turno inicial con saludo → bienvenida + campo 0, no avanza
        r = await intake_node(_base({}, 0, "hola"))
        assert r["intake_current_field"] == 0, f"esperaba 0, got {r['intake_current_field']}"
        assert "requerimiento" in r["endMessage"].lower(), "esperaba 'requerimiento' en endMessage"

        # Test 2 — campo 0 válido → avanza a campo 1
        r = await intake_node(_base({}, 0, "el microservicio de pagos debe procesar 1000 transacciones/s con latencia <200ms"))
        assert r["intake_current_field"] == 1, f"esperaba 1, got {r['intake_current_field']}"
        assert "campo_0_requerimiento" in r["intake_fields"], "campo_0_requerimiento no guardado"

        # Test 3 — campo 4 inválido → reprompt sin avanzar
        fields_0_3 = {INTAKE_SCRIPT[i]["field"]: "valor de prueba" for i in range(4)}
        r = await intake_node(_base(fields_0_3, 4, "funciona bien en producción normalmente"))
        assert r["intake_current_field"] == 4, f"esperaba 4, got {r['intake_current_field']}"
        assert any(w in r["endMessage"].lower() for w in ["métrica", "número", "concreto", "rps", "ms"]), \
            f"esperaba mensaje de métrica, got: {r['endMessage'][:100]}"

        # Test 4 — digresión → re-pregunta campo actual (no avanza)
        r = await intake_node(_base({"campo_0_requerimiento": "descripcion"}, 1, "¿qué es el patrón CQRS?"))
        assert r["intake_current_field"] == 1, f"esperaba 1, got {r['intake_current_field']}"
        assert "Para continuar" in r["endMessage"] or "To continue" in r["endMessage"], \
            f"esperaba separador, got: {r['endMessage'][:100]}"

        # Test 5 — campo 7 válido → intake_complete=True, pregunta de ASRs
        fields_0_6 = {INTAKE_SCRIPT[i]["field"]: "valor suficientemente largo" for i in range(7)}
        r = await intake_node(_base(fields_0_6, 7, "ninguna restricción previa de diseño arquitectónico"))
        assert r["intake_complete"] is True, "esperaba intake_complete=True"
        assert r["intake_current_field"] == 8, f"esperaba 8, got {r['intake_current_field']}"
        assert "asr" in r["endMessage"].lower() or "ASR" in r["endMessage"], \
            f"esperaba pregunta de ASRs, got: {r['endMessage'][:100]}"

        print("Phase 2 tests ✓")

        # ── Smoke test — language="en" ─────────────────────────────────────────
        # ST-1: invalid first input → welcome + field-0 question in English
        r = await intake_node(_base({}, 0, "hello", "en"))
        assert r["intake_current_field"] == 0
        assert "requirement" in r["endMessage"].lower() or "main requirement" in r["endMessage"].lower(), \
            f"ST-1 expected English welcome, got: {r['endMessage'][:120]}"
        assert "requerimiento" not in r["endMessage"].lower(), \
            f"ST-1 Spanish text leaked into EN message: {r['endMessage'][:120]}"

        # ST-2: valid field-0 → field-1 question in English
        # Note: _TECHNICAL_TERMS includes 'api', 'request', 'endpoint', 'throughput' — use those in English inputs
        r = await intake_node(_base({}, 0, "the API gateway must process each request from external endpoints with throughput 500 rps under 200ms", "en"))
        assert r["intake_current_field"] == 1, \
            f"ST-2 expected advance to field 1, got {r['intake_current_field']}. msg: {r['endMessage'][:120]}"
        assert "component" in r["endMessage"].lower(), \
            f"ST-2 expected English field-1 question, got: {r['endMessage'][:120]}"

        # ST-3: invalid field-4 (no metric) → reprompt error + question in English
        fields_en = {INTAKE_SCRIPT[i]["field"]: "test value for field" for i in range(4)}
        r = await intake_node(_base(fields_en, 4, "the system works fine in production", "en"))
        assert r["intake_current_field"] == 4
        assert "metric" in r["endMessage"].lower(), \
            f"ST-3 expected English reprompt with 'metric', got: {r['endMessage'][:120]}"
        assert "métrica" not in r["endMessage"].lower(), \
            f"ST-3 Spanish text leaked into EN reprompt: {r['endMessage'][:120]}"

        # ST-4: field-7 valid → ASR question in English
        fields_en_06 = {INTAKE_SCRIPT[i]["field"]: "sufficient value for the field here" for i in range(7)}
        r = await intake_node(_base(fields_en_06, 7, "no prior architectural constraints", "en"))
        assert r["intake_complete"] is True
        assert "asr" in r["endMessage"].lower() or "ASR" in r["endMessage"], \
            f"ST-4 expected English ASR question, got: {r['endMessage'][:120]}"
        assert "ya tienes" not in r["endMessage"].lower(), \
            f"ST-4 Spanish text leaked into EN ASR question: {r['endMessage'][:120]}"

        # ST-5: intake_complete=True, "no" answer → A2 branch in English
        base_complete_en = {
            **_base({}, 0, "no", "en"),
            "intake_complete": True,
            "intake_current_field": 8,
        }
        r = await intake_node(base_complete_en)
        assert "propose" in r["endMessage"].lower() or "asr" in r["endMessage"].lower(), \
            f"ST-5 expected English A2 message, got: {r['endMessage'][:120]}"
        assert "propondré" not in r["endMessage"].lower() and "perfecto" not in r["endMessage"].lower(), \
            f"ST-5 Spanish text leaked into EN A2 message: {r['endMessage'][:120]}"

        # ST-6: intake_complete=True, "yes" answer → A1 branch in English
        base_complete_yes_en = {
            **_base({}, 0, "yes I already have ASRs", "en"),
            "intake_complete": True,
            "intake_current_field": 8,
        }
        r = await intake_node(base_complete_yes_en)
        assert "registered" in r["endMessage"].lower() or "asr" in r["endMessage"].lower(), \
            f"ST-6 expected English A1 message, got: {r['endMessage'][:120]}"
        assert "registrado" not in r["endMessage"].lower(), \
            f"ST-6 Spanish text leaked into EN A1 message: {r['endMessage'][:120]}"

        # ── Smoke test — language="es" still works ────────────────────────────
        # ST-7: invalid field-4 → reprompt in Spanish
        r = await intake_node(_base(fields_0_3, 4, "funciona bien en producción normalmente"))
        assert "métrica" in r["endMessage"].lower() or "número" in r["endMessage"].lower(), \
            f"ST-7 Spanish reprompt broken: {r['endMessage'][:120]}"

        # ST-8: field-7 valid → ASR question in Spanish
        r = await intake_node(_base(fields_0_6, 7, "ninguna restricción previa de diseño arquitectónico"))
        assert "ya tienes" in r["endMessage"].lower() or "asr" in r["endMessage"].lower(), \
            f"ST-8 Spanish ASR question broken: {r['endMessage'][:120]}"

        print("Language smoke tests ✓")

    asyncio.run(_run_tests())
