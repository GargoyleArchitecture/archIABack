# -*- coding: utf-8 -*-

import json

from src.graph.resources import llm, rag_trace_record
from src.graph.state import GraphState
from src.graph.qa_registry import normalize_qa
from src.rag_agent import get_indexed_retriever
from src.graph.utils import _dedupe_snippets


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


def style_node_impl(state: GraphState, qa_override: str | None = None) -> GraphState:
    """Implementación común del nodo de estilos (ADD 3.0).

    Mantiene el contrato histórico del nodo:
    - genera candidatos + recomendación,
    - escribe style/selected_style/last_style,
    - publica turn_message con name='style_recommender'.
    """
    lang = state.get("language", "es")
    directive = "Answer in English." if lang == "en" else "Responde en español."
    style_hint = (state.get("user_style_hint") or "").strip()
    if style_hint:
        directive = f"{directive} {style_hint}"

    asr_text = (
        state.get("current_asr")
        or state.get("last_asr")
        or (state.get("userQuestion") or "")
    )
    qa = resolve_qa_for_style(state, qa_override=qa_override)
    ctx = (state.get("add_context") or "").strip()
    proj_ctx = (state.get("project_context_text") or "").strip()

    docs_list = []
    try:
        queries = [
            f"{qa} architecture style",
            f"{qa} architectural style patterns",
            "architecture styles ADD 3.0",
            "Bass Clements Kazman architecture styles",
            "microservices layered event-driven architecture styles",
            "architecture style patterns and tradeoffs",
        ]
        _retriever = get_indexed_retriever(
            quality_attribute=normalize_qa(state.get("resolved_index") or qa),
            content_type="estilos",
            k=6,
        )
        seen = set()
        gathered = []
        for q in queries:
            try:
                for d in _retriever.invoke(q):
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
        docs_list = gathered
        rag_trace_record(query=" | ".join(queries), docs=docs_list)
    except Exception:
        docs_list = []

    book_snippets = _dedupe_snippets(docs_list, max_items=5, max_chars=600)

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

    prompt = f"""{directive}
You are a software architect applying ADD 3.0.

Given the following Quality Attribute Scenario (ASR) and its business context,
propose 1-3 different architecture styles as reasonable candidates to solve this ASR,
and then recommend which of them is BETTER to satisfy this ASR,
explaining the recommendation in terms of its impact on the system and the quality attribute.
{proj_ctx_block}
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
"""

    result = llm.invoke(prompt)
    raw = getattr(result, "content", str(result))

    try:
        data = json.loads(raw)
    except Exception:
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

    state["style"] = chosen_name
    state["selected_style"] = chosen_name
    state["last_style"] = chosen_name
    state["arch_stage"] = "STYLE"
    state["quality_attribute"] = qa

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
        followups = [
            f"Explícame tácticas concretas para el ASR usando el estilo recomendado ({chosen_name}).",
            "Compárame más a fondo estos dos estilos para este ASR.",
        ]
    else:
        header = "## Candidate Architecture Styles"
        rec_label = "## Recommendation"
        because = "because"
        followups = [
            f"Explain concrete tactics for the ASR using the recommended style ({chosen_name}).",
            "Compare these two styles in more depth for this ASR.",
        ]

    content = (
        f"{header}\n\n"
        f"### 1. {style1_name}\n\n"
        f"- **Impact:** {style1_impact}\n\n"
        f"### 2. {style2_name}\n\n"
        f"- **Impact:** {style2_impact}\n\n"
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
