"""Adaptive intake flow tests."""

import asyncio
from unittest.mock import AsyncMock, patch

from src.graph.nodes.intake_node import intake_node
from src.graph.nodes.intake_validators import (
    FieldAssessment,
    INTAKE_SCRIPT,
    MultiFieldAssessmentResult,
    build_repair_prompt,
)


def _state(*, intake_fields=None, idx=0, uq="", lang="es"):
    return {
        "userQuestion": uq,
        "language": lang,
        "intake_fields": intake_fields or {},
        "intake_current_field": idx,
        "intake_complete": False,
        "project_context_text": "",
        "messages": [],
        "nextNode": "unifier",
        "intent": "general",
        "endMessage": "",
        "turn_messages": [],
    }


def _result(fields):
    return MultiFieldAssessmentResult(fields=fields)


def test_build_repair_prompt_for_metrics_is_targeted():
    msg = build_repair_prompt(4, "es", "Faltan métricas de sobrecarga.")
    assert "carga normal, sobrecarga y mantenimiento" in msg
    assert "ms" in msg or "rps" in msg


def test_intake_skips_questions_already_answered_in_same_message():
    llm_result = _result(
        {
            "campo_0_requerimiento": FieldAssessment(
                status="answered_valid",
                extracted_text="El sistema de pagos debe procesar transacciones con latencia menor a 200ms usando API y base de datos.",
            ),
            "campo_1_componentes": FieldAssessment(
                status="answered_valid",
                extracted_text="API gateway, servicio de pagos, base de datos PostgreSQL y cola Kafka para eventos de confirmación.",
            ),
            "campo_2_fuente": FieldAssessment(status="not_addressed"),
            "campo_3_estimulo": FieldAssessment(status="not_addressed"),
            "campo_4_ambientes": FieldAssessment(status="not_addressed"),
            "campo_5_prioridad_qa": FieldAssessment(status="not_addressed"),
            "campo_6_restricciones": FieldAssessment(status="not_addressed"),
            "campo_7_decisiones": FieldAssessment(status="not_addressed"),
        }
    )

    with patch("src.graph.nodes.intake_node.extract_and_validate_fields", new=AsyncMock(return_value=llm_result)):
        result = asyncio.run(
            intake_node(
                _state(
                    uq="Necesito un sistema de pagos con API gateway, servicio de pagos, PostgreSQL y Kafka; debe responder en menos de 200ms.",
                )
            )
        )

    assert result["intake_current_field"] == 2
    assert "campo_0_requerimiento" in result["intake_fields"]
    assert "campo_1_componentes" in result["intake_fields"]
    assert INTAKE_SCRIPT[2]["question_es"] in result["endMessage"]
    assert INTAKE_SCRIPT[1]["question_es"] not in result["endMessage"]


def test_intake_requests_rewrite_when_answer_is_insufficient():
    intake_fields = {INTAKE_SCRIPT[i]["field"]: f"valor suficientemente largo {i}" for i in range(4)}
    llm_result = _result(
        {
            "campo_4_ambientes": FieldAssessment(
                status="answered_invalid",
                extracted_text="normal: p95<200ms",
                reason="Solo cubre carga normal; faltan sobrecarga y mantenimiento con métricas.",
                repair_prompt="Reescribe tu respuesta cubriendo carga normal, sobrecarga y mantenimiento con métricas concretas en cada caso.",
            ),
            "campo_5_prioridad_qa": FieldAssessment(
                status="answered_valid",
                extracted_text="Disponibilidad 99.95%, latencia p95<200ms y throughput 1200rps.",
            ),
            "campo_6_restricciones": FieldAssessment(status="not_addressed"),
            "campo_7_decisiones": FieldAssessment(status="not_addressed"),
        }
    )

    with patch("src.graph.nodes.intake_node.extract_and_validate_fields", new=AsyncMock(return_value=llm_result)):
        result = asyncio.run(
            intake_node(
                _state(
                    intake_fields=intake_fields,
                    idx=4,
                    uq="Ambiente normal p95<200ms. Además la prioridad QA es disponibilidad 99.95% y throughput 1200rps.",
                )
            )
        )

    assert result["intake_current_field"] == 4
    assert "campo_5_prioridad_qa" in result["intake_fields"]
    assert "Solo cubre carga normal" in result["endMessage"]
    assert "sobrecarga y mantenimiento" in result["endMessage"]
    assert INTAKE_SCRIPT[6]["question_es"] not in result["endMessage"]


def test_intake_saves_future_answer_but_repairs_earlier_missing_field():
    intake_fields = {
        INTAKE_SCRIPT[0]["field"]: "Sistema de pagos con API, servicio y base de datos con objetivo de latencia.",
        INTAKE_SCRIPT[1]["field"]: "API gateway, servicio de pagos, PostgreSQL y Kafka para eventos.",
    }
    llm_result = _result(
        {
            "campo_2_fuente": FieldAssessment(
                status="answered_invalid",
                extracted_text="usuario",
                reason="Debes indicar qué tipo de usuario o actor externo genera el estímulo en tu sistema.",
                repair_prompt="Reescribe tu respuesta indicando si es un usuario final, un sistema externo, un evento interno o un timer, y menciona el contexto en tu sistema.",
            ),
            "campo_3_estimulo": FieldAssessment(
                status="answered_valid",
                extracted_text="Cuando el usuario confirma el pago en el checkout, el API gateway envía la solicitud al servicio de pagos.",
            ),
            "campo_4_ambientes": FieldAssessment(status="not_addressed"),
            "campo_5_prioridad_qa": FieldAssessment(status="not_addressed"),
            "campo_6_restricciones": FieldAssessment(status="not_addressed"),
            "campo_7_decisiones": FieldAssessment(status="not_addressed"),
        }
    )

    with patch("src.graph.nodes.intake_node.extract_and_validate_fields", new=AsyncMock(return_value=llm_result)):
        result = asyncio.run(
            intake_node(
                _state(
                    intake_fields=intake_fields,
                    idx=2,
                    uq="La fuente es un usuario. El estímulo ocurre cuando confirma el pago en checkout y el API gateway llama al servicio de pagos.",
                )
            )
        )

    assert result["intake_current_field"] == 2
    assert "campo_3_estimulo" in result["intake_fields"]
    assert "contexto en tu sistema" in result["endMessage"]
    assert INTAKE_SCRIPT[4]["question_es"] not in result["endMessage"]
