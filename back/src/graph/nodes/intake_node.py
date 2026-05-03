# -*- coding: utf-8 -*-
import re

from src.graph.resources import llm
from src.graph.state import GraphState
from src.graph.nodes.intake_validators import INTAKE_SCRIPT, validate_field, reprompt_message

_DIGRESSION_RE = re.compile(
    r"\b(asr|diagrama|diagram|t[aá]cticas|tactics|estilo|style|eval[uú]a|"
    r"evaluate|analiza|analyze|genera\b|generate|patr[oó]n|pattern|"
    r"arquitectura|architecture|dise[nñ]a|design)\b",
    re.IGNORECASE,
)

_ASR_QUESTION_ES = (
    "Ya tengo toda la información necesaria. "
    "¿Quieres que proponga los ASRs o ya tienes alguno definido?"
)
_ASR_QUESTION_EN = (
    "I have all the information needed. "
    "Would you like me to propose the ASRs, or do you already have some defined?"
)

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


def intake_node(state: GraphState) -> GraphState:
    uq = (state.get("userQuestion") or "").strip()
    lang = state.get("language") or "es"
    intake_fields = dict(state.get("intake_fields") or {})
    current_index = state.get("intake_current_field") or 0
    intake_complete = state.get("intake_complete") or False

    asr_question = _ASR_QUESTION_ES if lang == "es" else _ASR_QUESTION_EN

    # Rama A: intake completo, esperando respuesta sobre ASRs (Phase 4 extenderá esto)
    if intake_complete:
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": current_index,
            "intake_complete": True,
            "endMessage": asr_question,
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
        if _DIGRESSION_RE.search(uq):
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

        valid, _ = validate_field(0, uq)
        if valid:
            intake_fields[INTAKE_SCRIPT[0]["field"]] = uq
            end_msg = INTAKE_SCRIPT[1][f"question_{lang}"]
            return {
                **state,
                "intake_fields": intake_fields,
                "intake_current_field": 1,
                "intake_complete": False,
                "endMessage": end_msg,
                "nextNode": "unifier",
                "intent": "intake",
            }

        # Inválido en primer turno → bienvenida suave, sin mensaje de error
        end_msg = f"{_welcome_message(lang)}\n\n{INTAKE_SCRIPT[0][f'question_{lang}']}"
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": 0,
            "intake_complete": False,
            "endMessage": end_msg,
            "nextNode": "unifier",
            "intent": "intake",
        }

    # Rama D: turno normal (campos 0-7 con respuesta del usuario a validar)
    if _DIGRESSION_RE.search(uq):
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

    valid, _ = validate_field(current_index, uq)
    if not valid:
        end_msg = reprompt_message(current_index, lang)
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": current_index,
            "intake_complete": False,
            "endMessage": end_msg,
            "nextNode": "unifier",
            "intent": "intake",
        }

    # Válido: guardar y avanzar
    intake_fields[INTAKE_SCRIPT[current_index]["field"]] = uq
    new_index = current_index + 1

    if new_index >= 8:
        return {
            **state,
            "intake_fields": intake_fields,
            "intake_current_field": 8,
            "intake_complete": True,
            "endMessage": asr_question,
            "nextNode": "unifier",
            "intent": "intake",
        }

    end_msg = INTAKE_SCRIPT[new_index][f"question_{lang}"]
    return {
        **state,
        "intake_fields": intake_fields,
        "intake_current_field": new_index,
        "intake_complete": False,
        "endMessage": end_msg,
        "nextNode": "unifier",
        "intent": "intake",
    }


if __name__ == "__main__":
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

    # Test 1 — turno inicial con saludo → bienvenida + campo 0, no avanza
    r = intake_node(_base({}, 0, "hola"))
    assert r["intake_current_field"] == 0, f"esperaba 0, got {r['intake_current_field']}"
    assert "requerimiento" in r["endMessage"].lower(), "esperaba 'requerimiento' en endMessage"

    # Test 2 — campo 0 válido → avanza a campo 1
    r = intake_node(_base({}, 0, "el microservicio de pagos debe procesar 1000 transacciones/s con latencia <200ms"))
    assert r["intake_current_field"] == 1, f"esperaba 1, got {r['intake_current_field']}"
    assert "campo_0_requerimiento" in r["intake_fields"], "campo_0_requerimiento no guardado"

    # Test 3 — campo 4 inválido → reprompt sin avanzar
    fields_0_3 = {INTAKE_SCRIPT[i]["field"]: "valor de prueba" for i in range(4)}
    r = intake_node(_base(fields_0_3, 4, "funciona bien en producción normalmente"))
    assert r["intake_current_field"] == 4, f"esperaba 4, got {r['intake_current_field']}"
    assert any(w in r["endMessage"].lower() for w in ["métrica", "número", "concreto", "rps", "ms"]), \
        f"esperaba mensaje de métrica, got: {r['endMessage'][:100]}"

    # Test 4 — digresión → re-pregunta campo actual (no avanza)
    r = intake_node(_base({"campo_0_requerimiento": "descripcion"}, 1, "¿qué es el patrón CQRS?"))
    assert r["intake_current_field"] == 1, f"esperaba 1, got {r['intake_current_field']}"
    assert "Para continuar" in r["endMessage"] or "To continue" in r["endMessage"], \
        f"esperaba separador, got: {r['endMessage'][:100]}"

    # Test 5 — campo 7 válido → intake_complete=True, pregunta de ASRs
    fields_0_6 = {INTAKE_SCRIPT[i]["field"]: "valor suficientemente largo" for i in range(7)}
    r = intake_node(_base(fields_0_6, 7, "ninguna restricción previa de diseño arquitectónico"))
    assert r["intake_complete"] is True, "esperaba intake_complete=True"
    assert r["intake_current_field"] == 8, f"esperaba 8, got {r['intake_current_field']}"
    assert "asr" in r["endMessage"].lower() or "ASR" in r["endMessage"], \
        f"esperaba pregunta de ASRs, got: {r['endMessage'][:100]}"

    print("Phase 2 tests ✓")
