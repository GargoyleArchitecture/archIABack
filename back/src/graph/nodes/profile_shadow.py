"""Shadow Agent / Profile Evaluator (F3-T2).

Ejecuta una evaluacion del perfil del usuario fuera del path critico del
turno. Llamado por `fire_shadow_eval` desde el unifier al final del turno
si la cadencia se cumple. Escribe el resultado en el LangGraph Store bajo
el namespace `("user", user_id, "profile")`.

Cadencia y ventana son configurables via env:
- SHADOW_AGENT_EVERY_N_TURNS (default 4)
- SHADOW_AGENT_WINDOW_SIZE  (default 6)

La persistencia al Backend Negocio (PostgreSQL) llega con F3-T5 (cliente
HTTP). Esta capa solo escribe al Store in-process.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.resources import get_store, llm
from src.graph.schemas.profile import ConceptScore, UserProfileEvaluation
from src.services.profile_sync import sync_profile

log = logging.getLogger("shadow_agent")

SHADOW_EVERY_N = int(os.getenv("SHADOW_AGENT_EVERY_N_TURNS", "4"))
SHADOW_WINDOW = int(os.getenv("SHADOW_AGENT_WINDOW_SIZE", "6"))

# Tracking de tasks fire-and-forget para prevenir GC prematuro.
_pending_tasks: set = set()


def should_trigger(turn_count_since_eval: int) -> bool:
    """True si la cadencia se cumple y conviene disparar la evaluacion."""
    return (turn_count_since_eval or 0) >= SHADOW_EVERY_N


# ---- Evaluacion via LLM -------------------------------------------------

_EVAL_PROMPT = """\
You are a profiling agent. Analyze the recent exchange between USER and
ASSISTANT, then identify:
- strengths: architectural concepts the USER demonstrates mastery on.
- weaknesses: concepts the USER avoids, asks about, or gets wrong.
- evaluated_concepts: per-concept score in [0,1] with short evidence.
- confidence: overall confidence in [0,1] given the available signal.
- delta_from_previous: dict mapping concept name to mastery delta vs prior.
  Leave empty if you cannot infer it.

Be conservative: only include concepts with explicit signal in messages.
If signal is weak, return empty lists with confidence < 0.5.

RECENT EXCHANGE:
{window}
"""


def _format_messages_window(messages: list, window: int) -> str:
    """Devuelve texto pretty-printed con las ultimas `window` mensajes."""
    tail = list(messages or [])[-window:]
    out = []
    for m in tail:
        role = "USER" if isinstance(m, HumanMessage) else "ASSISTANT"
        content = getattr(m, "content", str(m))
        if not isinstance(content, str):
            content = str(content)
        content = content.replace("\n", " ").strip()[:600]
        out.append(f"[{role}] {content}")
    return "\n\n".join(out) if out else "(empty conversation)"


async def evaluate_messages(messages: list, llm_obj=None) -> UserProfileEvaluation:
    """Llama al LLM con structured_output(UserProfileEvaluation) sobre la ventana.

    Usa `method="function_calling"` para evitar el modo `json_schema` de
    OpenAI (langchain-openai >= 0.3) que exige `required` con todas las
    propiedades. Nuestro schema tiene defaults via `default_factory`
    (strengths/weaknesses/evaluated_concepts/delta_from_previous), por lo
    que el modo function_calling es la opcion compatible.
    """
    llm_obj = llm_obj if llm_obj is not None else llm
    structured = llm_obj.with_structured_output(
        UserProfileEvaluation, method="function_calling"
    )
    prompt = _EVAL_PROMPT.format(window=_format_messages_window(messages, SHADOW_WINDOW))
    return await structured.ainvoke(prompt)


# ---- Merge ponderado ----------------------------------------------------

def _union_names(prev_list, new_names):
    """Union case-insensitive preservando primer nombre canonico visto."""
    seen = {}
    for s in (prev_list or []):
        if isinstance(s, str) and s.strip():
            seen.setdefault(s.casefold(), s)
    for s in (new_names or []):
        if isinstance(s, str) and s.strip():
            seen.setdefault(s.casefold(), s)
    return list(seen.values())


def merge_profile(prev: dict, new: UserProfileEvaluation, alpha: float = 0.7) -> dict:
    """Mergea perfil previo con nueva evaluacion via promedio ponderado.

    - mastery_merged = alpha * new + (1 - alpha) * prev
    - strengths / weaknesses: union case-insensitive (prev gana en empate).
    - delta_from_previous: diff de mastery (positivo = mejora).
    - F3-T4: cada concepto evaluado en `new` recibe `last_seen_at = now_iso`
      (refuerzo). Conceptos solo-en-prev preservan su `last_seen_at` previo.
    """
    prev = prev or {}
    prev_concepts_idx = {
        (c.get("name") or "").casefold(): c
        for c in (prev.get("evaluated_concepts") or [])
        if isinstance(c, dict) and c.get("name")
    }

    now_iso = datetime.now(timezone.utc).isoformat()
    merged_concepts: List[dict] = []
    deltas: dict = {}
    seen_keys = set()

    for c in (new.evaluated_concepts or []):
        key = c.name.casefold()
        seen_keys.add(key)
        prev_c = prev_concepts_idx.get(key)
        if prev_c is not None:
            try:
                prev_m = float(prev_c.get("mastery") or 0.0)
            except (TypeError, ValueError):
                prev_m = 0.0
            new_m = alpha * float(c.mastery) + (1 - alpha) * prev_m
            deltas[c.name] = round(new_m - prev_m, 4)
        else:
            new_m = float(c.mastery)
            deltas[c.name] = round(new_m, 4)
        merged_concepts.append(
            {
                "name": c.name,
                "mastery": round(new_m, 4),
                "evidence": c.evidence or "",
                "last_seen_at": now_iso,
            }
        )

    # Conceptos en prev pero no en new se preservan (last_seen_at queda igual).
    for key, prev_c in prev_concepts_idx.items():
        if key not in seen_keys:
            merged_concepts.append(prev_c)

    strengths = _union_names(prev.get("strengths"), [c.name for c in (new.strengths or [])])
    weaknesses = _union_names(prev.get("weaknesses"), [c.name for c in (new.weaknesses or [])])

    return {
        "user_id": prev.get("user_id", ""),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "evaluated_concepts": merged_concepts,
        "confidence": float(new.confidence),
        "delta_from_previous": deltas,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---- Task entry point ---------------------------------------------------

async def shadow_eval_async(user_id: str, messages: list, llm_obj=None, store=None) -> Optional[dict]:
    """Punto de entrada del task fire-and-forget.

    - Si user_id vacio o messages < 2 -> no hace nada.
    - Si la evaluacion falla, NO escribe al Store (perfil previo intacto).
    - Logs explicitos: shadow_eval started/completed/failed.

    Devuelve el dict mergeado escrito al Store, o None si no se persistio.
    """
    user_id = (user_id or "").strip()
    if not user_id:
        log.debug("shadow_eval skipped: empty user_id")
        return None
    if len(messages or []) < 2:
        log.debug("shadow_eval skipped: messages window too small (%d)", len(messages or []))
        return None

    log.info("shadow_eval started for user_id=%s (msgs=%d)", user_id, len(messages))
    try:
        evaluation = await evaluate_messages(messages, llm_obj=llm_obj)
    except Exception as exc:
        log.exception("shadow_eval failed for user_id=%s: %s", user_id, exc)
        return None

    try:
        store_obj = store if store is not None else get_store()
        ns = ("user", user_id, "profile")
        existing = await store_obj.aget(ns, key="profile")
        prev = (existing.value if existing else {}) or {}
        if not isinstance(prev, dict):
            prev = {}
        prev.setdefault("user_id", user_id)
        merged = merge_profile(prev, evaluation)
        merged["user_id"] = user_id
        await store_obj.aput(ns, key="profile", value=merged)
    except Exception as exc:
        log.exception("shadow_eval persistence failed for user_id=%s: %s", user_id, exc)
        return None

    log.info(
        "shadow_eval completed for user_id=%s (concepts=%d, conf=%.2f)",
        user_id,
        len(merged.get("evaluated_concepts", [])),
        float(merged.get("confidence", 0.0)),
    )

    # F3-T5: sync HTTP a Backend Negocio. Best-effort: si falla, el perfil
    # ya esta seguro en el Store local. NUNCA lanza al caller.
    try:
        await sync_profile(user_id, merged)
    except Exception:
        log.exception("profile_sync raised unexpectedly for user_id=%s", user_id)

    return merged


def fire_shadow_eval(state: dict, final_text: str = "") -> Optional["asyncio.Task"]:
    """Lanza fire-and-forget el shadow eval si la cadencia se cumple.

    - Llama al loop async actual via `asyncio.get_running_loop()`.
    - Mantiene la task en `_pending_tasks` para evitar GC prematuro.
    - Devuelve la Task lanzada, o None si no se pudo disparar.
    """
    if not should_trigger(state.get("turn_count_since_eval", 0) or 0):
        return None
    user_id = (state.get("user_id") or "").strip()
    if not user_id:
        return None

    messages = list(state.get("messages") or [])
    if final_text:
        messages = messages + [AIMessage(content=final_text, name="unifier")]

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.warning("shadow_eval skipped: no running event loop")
        return None

    task = loop.create_task(shadow_eval_async(user_id, messages))
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
    return task
