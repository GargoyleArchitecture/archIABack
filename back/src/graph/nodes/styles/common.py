# -*- coding: utf-8 -*-

import json
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.graph.resources import llm, rag_trace_record
from src.graph.state import GraphState
from src.graph.qa_registry import normalize_qa
from src.graph.prompts.mode_prompts import apply_mode_prompt
from src.rag_agent import get_indexed_retriever
from src.graph.utils import _dedupe_snippets
from src.ledger import (
    append_decision,
    compute_active_view,
    load_ledger,
    render_dossier,
    render_dossier_compact,
    render_phase_prompt,
    LedgerValidationError,
    LedgerConcurrencyError,
)
from src.ledger.types import Phase

log = logging.getLogger("style_node")


@lru_cache(maxsize=64)
def _fetch_styles_rag(qa: str, resolved_index: str, k: int = 6) -> str:
    """Returns book_snippets string. Cached by (qa, resolved_index, k).
    Cache hit skips all ChromaDB queries."""
    queries = [
        f"{qa} architecture style",
        f"{qa} architectural style patterns",
        "architecture styles ADD 3.0",
        "Bass Clements Kazman architecture styles",
        "microservices layered event-driven architecture styles",
        "architecture style patterns and tradeoffs",
    ]
    _retriever = get_indexed_retriever(
        quality_attribute=normalize_qa(resolved_index or qa),
        content_type="estilos",
        k=k,
    )
    seen: set = set()
    gathered: list = []
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = {executor.submit(_retriever.invoke, q): q for q in queries}
        for future in as_completed(futures):
            try:
                for d in future.result():
                    key = (d.metadata.get("source_path"), d.metadata.get("page"))
                    if key in seen:
                        continue
                    seen.add(key)
                    gathered.append(d)
                    if len(gathered) >= 6:
                        break
            except Exception:
                pass
            if len(gathered) >= 6:
                break

    return _dedupe_snippets(gathered, max_items=5, max_chars=600)


# ---------------------------------------------------------------------------
# Pure helpers (Step 1 — P4)
# ---------------------------------------------------------------------------

def _build_dossier_asr_binding(ledger_active: dict, lang: str = "es") -> str:
    """Build a HARD-BINDING prompt block sourced directly from the active ASR in the ledger.
    Returns "" when no active ASR exists (first-turn sessions — zero-byte injection,
    behaviour bit-identical to today).
    """
    asr = (ledger_active or {}).get("asr")
    if not asr:
        return ""

    asr_id  = asr.get("id", "")
    qa      = asr.get("qa", "")
    payload = asr.get("payload") or {}
    rm      = payload.get("response_measure", "")
    domain  = payload.get("domain", "")
    summary = payload.get("summary", "")

    if lang == "en":
        return (
            f'\n{"=" * 60}\n'
            f'ACTIVE ASR BINDING — READ BEFORE SELECTING A STYLE:\n'
            f'  ASR ID:            {asr_id}\n'
            f'  Quality Attribute: {qa}\n'
            f'  Response Measure:  {rm}\n'
            f'  Domain:            {domain}\n'
            f'  Summary:           {summary[:200]}\n\n'
            f'REQUIREMENT: The chosen style MUST primarily improve "{qa}".\n'
            f'Your rationale MUST explicitly state how the chosen style helps\n'
            f'achieve: [{rm}].\n'
            f'Styles that do NOT directly improve "{qa}" MUST NOT be recommended.\n'
            f'{"=" * 60}\n'
        )
    return (
        f'\n{"=" * 60}\n'
        f'VINCULACIÓN CON EL ASR ACTIVO — LEER ANTES DE SELECCIONAR ESTILO:\n'
        f'  ID del ASR:          {asr_id}\n'
        f'  Atributo de Calidad: {qa}\n'
        f'  Medida de Respuesta: {rm}\n'
        f'  Dominio:             {domain}\n'
        f'  Resumen:             {summary[:200]}\n\n'
        f'REQUISITO: El estilo elegido DEBE mejorar principalmente "{qa}".\n'
        f'Tu justificación DEBE indicar explícitamente cómo el estilo ayuda\n'
        f'a lograr: [{rm}].\n'
        f'Estilos que NO mejoren "{qa}" directamente NO deben recomendarse.\n'
        f'{"=" * 60}\n'
    )


def _build_style_payload(
    data: dict,
    chosen_name: str,
    style1: dict,
    style2: dict,
    rationale: str,
) -> dict:
    candidates = []
    for s in (style1, style2):
        name = (s.get("name") or "").strip()
        if name:
            candidates.append({"name": name, "impact": (s.get("impact") or "").strip()})
    return {
        "name":       chosen_name,
        "candidates": candidates,
        "chosen":     chosen_name,
        "tradeoffs":  rationale,
    }


def _build_asr_parent_ref(ledger_active: dict) -> list:
    asr = (ledger_active or {}).get("asr")
    if not asr:
        return []
    return [{"id": asr["id"], "kind": "asr", "iteration": asr.get("iteration", 0)}]


# ---------------------------------------------------------------------------
# Ledger state refresh (Step 2 — P4)
# ---------------------------------------------------------------------------

def _refresh_ledger_state(
    state: dict,
    user_id: str,
    project_id,
    lang: str,
) -> None:
    try:
        fresh  = load_ledger(user_id, project_id, auto_migrate=False)
        active = compute_active_view(fresh)
        state["ledger"]                 = fresh
        state["ledger_active"]          = active
        state["design_dossier_md"]      = render_dossier(fresh, lang=lang)
        state["ledger_dossier_compact"] = render_dossier_compact(fresh, lang=lang)
        state["ledger_phase_prompt"]    = render_phase_prompt(fresh, lang=lang)
        state["current_phase"]          = fresh.get("current_phase", "INTAKE")
        state["ledger_pending_advance"] = fresh.get("pending_advance") or {}
        log.debug("style_node: ledger state refreshed phase=%s", state["current_phase"])
    except Exception as exc:
        log.warning("style_node: state refresh failed (nonfatal): %s", exc)


# ---------------------------------------------------------------------------
# QA resolution
# ---------------------------------------------------------------------------

def resolve_qa_for_style(state: GraphState, qa_override: str | None = None) -> str:
    """Resuelve el QA objetivo para el nodo de estilos.

    Prioridad:
    1) override explícito del wrapper,
    2) quality_attribute del estado,
    3) resolved_index del classifier,
    4) fallback general.
    """
    if qa_override:
        qa = normalize_qa(qa_override)
        if qa != "general":
            return qa

    qa_state = normalize_qa(state.get("quality_attribute", ""))
    if qa_state != "general":
        return qa_state

    qa_resolved = normalize_qa(state.get("resolved_index", ""))
    if qa_resolved != "general":
        return qa_resolved

    return "general"


# ---------------------------------------------------------------------------
# Main node implementation
# ---------------------------------------------------------------------------

def style_node_impl(state: GraphState, qa_override: str | None = None) -> GraphState:
    """Implementación común del nodo de estilos (ADD 3.0).

    Mantiene el contrato histórico del nodo:
    - genera candidatos + recomendación,
    - escribe style/selected_style/last_style,
    - publica turn_message con name='style_recommender'.
    """
    lang = state.get("language", "es")
    if lang == "en":
        directive = (
            "MANDATORY LANGUAGE: English.\n"
            "Your ENTIRE response MUST be in English. Do not mix languages."
        )
    else:
        directive = (
            "IDIOMA OBLIGATORIO: español.\n"
            "Tu respuesta COMPLETA debe estar en español. No mezcles idiomas."
        )
    style_hint = (state.get("user_style_hint") or "").strip()
    if style_hint:
        directive = f"{directive}\n{style_hint}"

    asr_text = (
        state.get("current_asr")
        or state.get("last_asr")
        or (state.get("userQuestion") or "")
    )
    qa = resolve_qa_for_style(state, qa_override=qa_override)
    ctx = (state.get("add_context") or "").strip()
    proj_ctx = (state.get("project_context_text") or "").strip()

    # Ground styles with the final QA resolved for the turn, not with a stale
    # classifier index that may drift on follow-up turns.
    book_snippets = _fetch_styles_rag(qa, qa, k=6)

    proj_ctx_block = ""
    if proj_ctx:
        proj_ctx_block = f"""
{"=" * 60}
PROJECT CONTEXT — MANDATORY CONSTRAINTS FOR STYLE SELECTION:
{proj_ctx}

IMPORTANT: The recommended style MUST be compatible with the listed tech stack.
Your rationale MUST explicitly explain how the chosen style fits (or requires adapting)
the specific technologies listed. Business rules must be respected in all trade-off analysis.
{"=" * 60}
"""

    # ── Dossier ASR binding (P4) ────────────────────────────────────────────
    dossier_binding_block = _build_dossier_asr_binding(
        state.get("ledger_active") or {}, lang
    )

    prompt = f"""{directive}
You are a software architect applying ADD 3.0.

Given the following Quality Attribute Scenario (ASR) and its business context,
propose 1-3 different architecture styles as reasonable candidates to solve this ASR,
and then recommend which of them is BETTER to satisfy this ASR,
explaining the recommendation in terms of its impact on the system and the quality attribute.
{proj_ctx_block}
{dossier_binding_block}
Quality attribute focus (e.g., availability, performance, latency, security, etc.):
{qa}

Additional session context:
{ctx or "(none)"}

ASR:
{asr_text}

GROUNDING (use this context from architecture documentation; prefer it over general knowledge):
{book_snippets or "(none)"}

You MUST respond with a VALID JSON object ONLY, with NO extra text, in the following form:

{{
  "style_1": {{
    "name": "Short name of style 1 (e.g., 'Layered', 'Microservices')",
    "impact": "Brief description of how this style impacts the ASR (pros, cons, trade-offs)."
  }},
  "style_2": {{
    "name": "Short name of style 2",
    "impact": "Brief description of how this style impacts the ASR (pros, cons, trade-offs)."
  }},
  "best_style": "style_1 or style_2 (choose ONE)",
  "rationale": "Explain why the chosen style is better for this ASR, based on its impact."
}}

Do NOT add comments or any text outside of this JSON object.
All string values in the JSON (name, impact, rationale) MUST be written in {"English" if lang == "en" else "español"}.
"""

    result = llm.invoke(apply_mode_prompt(state, prompt))
    raw = getattr(result, "content", str(result))

    try:
        data = json.loads(raw)
    except Exception:
        data = {}
        fallback_style = raw.splitlines()[0].strip() if raw.splitlines() else raw.strip()
        state["style"] = fallback_style
        state["selected_style"] = fallback_style
        state["last_style"] = fallback_style
        state["arch_stage"] = "STYLE"
        state["quality_attribute"] = qa
        state["endMessage"] = raw
        state["nextNode"] = "unifier"
        return state

    style1 = data.get("style_1", {}) or {}
    style2 = data.get("style_2", {}) or {}
    style1_name = style1.get("name", "").strip() or "Style 1"
    style2_name = style2.get("name", "").strip() or "Style 2"
    style1_impact = style1.get("impact", "").strip()
    style2_impact = style2.get("impact", "").strip()
    best_key = (data.get("best_style") or "").strip()
    rationale = data.get("rationale", "").strip()

    chosen_name = style2_name if best_key == "style_2" else style1_name

    # ── Scalar writes (unconditional) ────────────────────────────────────────
    state["style"] = chosen_name
    state["selected_style"] = chosen_name
    state["last_style"] = chosen_name
    state["arch_stage"] = "STYLE"
    state["quality_attribute"] = qa

    # ── Ledger write-back (P4) ───────────────────────────────────────────────
    _user_id    = (state.get("user_id_for_prefs") or "").strip()
    _project_id = (state.get("project_id") or "").strip() or None

    if _user_id:
        try:
            _payload = _build_style_payload(data, chosen_name, style1, style2, rationale)
            _parents = _build_asr_parent_ref(state.get("ledger_active") or {})
            _new_decision: dict = {
                "id":               "",
                "kind":             "style",
                "phase":            Phase.STYLE.value,
                "iteration":        0,
                "qa":               qa,
                "parents":          _parents,
                "payload":          _payload,
                "rationale":        rationale,
                "sources":          [],
                "status":           "active",
                "parent_status":    "ok",
                "superseded_by":    None,
                "rejection_reason": None,
                "created_at":       "",
                "created_by_node":  "style_node",
            }
            _saved = append_decision(_user_id, _project_id, _new_decision)
            log.info(
                "style_node: ledger ok id=%s qa=%s chosen=%s project=%s",
                _saved["id"], qa, chosen_name, _project_id,
            )
            _refresh_ledger_state(state, _user_id, _project_id, lang)

        except LedgerValidationError as _exc:
            log.warning("style_node: ledger validation error (nonfatal): %s", _exc)
        except LedgerConcurrencyError as _exc:
            log.warning("style_node: ledger concurrency error (nonfatal): %s", _exc)
        except Exception as _exc:
            log.warning("style_node: unexpected ledger error (nonfatal): %s", _exc)

    prev_mem = state.get("memory_text", "") or ""
    state["memory_text"] = (
        prev_mem
        + f"\n\n[STYLE_OPTIONS]\n1) {style1_name}\n2) {style2_name}\n"
        + f"[STYLE_CHOSEN]\n{chosen_name}\n"
    ).strip()

    if lang == "es":
        header = "## Estilos Arquitectónicos Candidatos"
        rec_label = "## Recomendación"
        because = "porque"
        impact_label = "Impacto"
        followups = [
            f"Explícame tácticas concretas para el ASR usando el estilo recomendado ({chosen_name}).",
            "Compárame más a fondo estos dos estilos para este ASR.",
        ]
    else:
        header = "## Candidate Architecture Styles"
        rec_label = "## Recommendation"
        because = "because"
        impact_label = "Impact"
        followups = [
            f"Explain concrete tactics for the ASR using the recommended style ({chosen_name}).",
            "Compare these two styles in more depth for this ASR.",
        ]

    content = (
        f"{header}\n\n"
        f"### 1. {style1_name}\n\n"
        f"- **{impact_label}:** {style1_impact}\n\n"
        f"### 2. {style2_name}\n\n"
        f"- **{impact_label}:** {style2_impact}\n\n"
        f"---\n\n"
        f"{rec_label}\n\n"
        f"**{chosen_name}** {because}:\n\n"
        f"{rationale}\n"
    )

    state["turn_messages"] = state.get("turn_messages", []) + [
        {"role": "assistant", "name": "style_recommender", "content": content}
    ]
    state["suggestions"] = followups
    state["endMessage"] = content
    state["nextNode"] = "unifier"

    return state
