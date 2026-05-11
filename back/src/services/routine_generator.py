"""F5-T2: Orquestador del endpoint POST /generate-routine.

Separado del archivo `main.py` para que la lógica sea testeable sin
TestClient ni lifespan. El handler HTTP en `main.py` es una fachada delgada
que delega aquí.

Responsabilidades:
- Auth: verificar `X-Internal-Token` (mismo token que F4-T2).
- Trace: usar `X-Trace-Id` si viene; generar UUID si no.
- Carga del perfil del Store con curva de olvido aplicada (consistente con
  `boot_node` F3-T6 cuando el Store ya tiene datos; sin hidratación inversa
  desde Negocio porque ese path es del agente conversacional).
- Invocación del subgrafo `RoutineGenerator`.
- Logs estructurados con `trace_id`.
- Mapeo de fallos del grafo a HTTPException 500.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from fastapi import HTTPException, Request

from src.graph.resources import get_routine_graph, get_store
from src.graph.schemas.routine import RoutineOutput
from src.graph.services.decay import apply_decay_to_profile

log = logging.getLogger("routine_generator")

INTERNAL_TOKEN_HEADER = "X-Internal-Token"
TRACE_ID_HEADER = "X-Trace-Id"


# ============================================================================
# Auth
# ============================================================================

def verify_internal_token(request: Request) -> None:
    """F5-T2: rechaza la request con 401 si el header `X-Internal-Token` falta
    o no coincide con `INTERNAL_API_TOKEN` de env. Fail-closed: si la env var
    no está configurada, también rechaza (evita arrancar con auth deshabilitado
    por accidente)."""
    expected = (os.getenv("INTERNAL_API_TOKEN") or "").strip()
    received = (request.headers.get(INTERNAL_TOKEN_HEADER) or "").strip()
    if not expected or received != expected:
        raise HTTPException(status_code=401, detail="Invalid internal token")


# ============================================================================
# Trace
# ============================================================================

def make_trace_id(request: Request) -> str:
    """F5-T2: usa el `X-Trace-Id` provisto si viene; si no, genera un UUID4."""
    incoming = (request.headers.get(TRACE_ID_HEADER) or "").strip()
    if incoming:
        return incoming
    return str(uuid.uuid4())


# ============================================================================
# Profile loading
# ============================================================================

async def load_user_profile_for_routine(user_id: str) -> dict:
    """F5-T2: lee el perfil del usuario del Store y aplica la curva de olvido.

    No hace hidratación inversa desde Negocio (eso es responsabilidad del
    `boot_node` del agente conversacional, F3-T6). Si el Store no tiene
    perfil para este `user_id`, retornamos `{}` y el subgrafo
    (`select_weakness_node`) cae al fallback genérico.
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return {}

    try:
        store = get_store()
        ns = ("user", user_id, "profile")
        item = await store.aget(ns, key="profile")
    except Exception:
        log.exception(
            "load_user_profile_for_routine: store.aget failed for user_id=%s",
            user_id,
        )
        return {}

    if item is None:
        return {}

    return apply_decay_to_profile(item.value or {})


# ============================================================================
# Generation
# ============================================================================

async def generate_routine_for_user(
    user_id: str,
    target_weakness: Optional[str],
    *,
    trace_id: str,
) -> RoutineOutput:
    """F5-T2: invoca el subgrafo y devuelve el `RoutineOutput` final.

    Lanza HTTPException(500) si el subgrafo no produce un `final` (caso
    defensivo; el subgrafo está diseñado para siempre llenar `final` tras
    el validador).
    """
    user_id = (user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    user_profile = await load_user_profile_for_routine(user_id)

    log.info(
        "generate_routine started trace_id=%s user_id=%s target_weakness=%r profile_concepts=%d",
        trace_id,
        user_id,
        target_weakness,
        len((user_profile or {}).get("evaluated_concepts") or []),
    )

    graph = get_routine_graph()

    initial_state = {
        "user_id": user_id,
        "user_profile": user_profile,
        "target_weakness": target_weakness,
        "regen_count": 0,
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        log.exception(
            "generate_routine subgraph raised trace_id=%s user_id=%s: %s",
            trace_id,
            user_id,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Routine generation failed unexpectedly.",
        )

    final = (result or {}).get("final")
    if final is None or not isinstance(final, RoutineOutput):
        log.error(
            "generate_routine subgraph did not produce final trace_id=%s user_id=%s",
            trace_id,
            user_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Subgraph did not produce a final routine.",
        )

    log.info(
        "generate_routine completed trace_id=%s user_id=%s difficulty=%d concepts=%d regens=%d",
        trace_id,
        user_id,
        final.difficulty,
        len(final.expected_concepts),
        int(result.get("regen_count", 0) or 0),
    )

    # F11-T6: emite evento estructurado de routine_generated.
    try:
        from src.services.telemetry import emit as _emit_telemetry  # noqa: E402
        _emit_telemetry(
            "routine_generated",
            user_id=user_id,
            target_weakness=target_weakness or getattr(final, "target_weakness", None),
            difficulty=final.difficulty,
            concepts_count=len(final.expected_concepts),
            regen_count=int(result.get("regen_count", 0) or 0),
            trace_id=trace_id,
        )
    except Exception:
        # Telemetría no debe romper el endpoint.
        pass

    return final
