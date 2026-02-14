
import json
from src.graph.state import GraphState
from src.graph.resources import llm
from src.graph.utils import _resolve_response_language, _sync_contract_from_legacy, _clip_text

def style_node(state: GraphState) -> GraphState:
    """
    Architecture style node (ADD 3.0):

    - Proposes EXACTLY 2 candidate styles.
    - Evaluates the impact of each one on the ASR.
    - Recommends one of them.
    - Stores only the recommended style as the active style in the pipeline.
    """
    _sync_contract_from_legacy(state)
    lang = _resolve_response_language(state)
    directive = "Answer in English." if lang == "en" else "Responde en español."

    # 1) Recover ASR, quality attribute, and business context
    asr_text = (
        state.get("current_asr")
        or state.get("last_asr")
        or (state.get("userQuestion") or "")
    )
    qa = state.get("quality_attribute", "")
    ctx = (state.get("add_context") or "").strip()

    # 2) Prompt: ask for JSON with 2 styles + recommendation (PROMPT 100% IN ENGLISH)
    prompt = f"""{directive}
You are a software architect applying ADD 3.0.

Given the following Quality Attribute Scenario (ASR) and its business context,
propose exactly TWO different architecture styles as reasonable candidates,
and then recommend which of the two is BETTER to satisfy this ASR,
explaining the recommendation in terms of its impact on the quality attribute.

Quality attribute focus (e.g., availability, performance, latency, security, etc.):
{qa}

Business / context:
{ctx or "(none)"}

ASR:
{asr_text}

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

    # 3) Parse JSON (fallback if it fails)
    try:
        data = json.loads(raw)
    except Exception:
        # If no valid JSON: at least store one line as style
        fallback_style = raw.splitlines()[0].strip()
        state["style"] = fallback_style
        state["selected_style"] = fallback_style
        state["last_style"] = fallback_style
        state["arch_stage"] = "STYLE"
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

    if best_key == "style_2":
        chosen_name = style2_name
    else:
        # Default to style_1 if not clear
        best_key = "style_1"
        chosen_name = style1_name

    # 4) Store ONLY the recommended style in the ADD 3.0 state
    state["style"] = chosen_name
    state["selected_style"] = chosen_name
    state["last_style"] = chosen_name
    state["arch_stage"] = "STYLE"
    active = dict(state.get("active_decisions") or {})
    active["arch_stage"] = "STYLE"
    active["style"] = chosen_name
    log = list(active.get("decision_log") or [])
    log.append({"type": "style", "value": _clip_text(chosen_name, 30), "source_turn": len(state.get("messages", []))})
    active["decision_log"] = log[-20:]
    state["active_decisions"] = active
    state["response_language"] = lang

    # Update rich memory (long-term text)
    prev_mem = state.get("memory_text", "") or ""
    state["memory_text"] = (
        prev_mem
        + f"\n\n[STYLE_OPTIONS]\n1) {style1_name}\n2) {style2_name}\n"
        + f"[STYLE_CHOSEN]\n{chosen_name}\n"
    ).strip()

    # 5) Build user-facing message (in ES or EN)
    if lang == "es":
        header = "He identificado dos estilos arquitectónicos candidatos para tu ASR:"
        rec_label = "Recomendación"
        because = "porque"
        followups = [
            f"Explícame tácticas concretas para el ASR usando el estilo recomendado ({chosen_name}).",
            "Compárame más a fondo estos dos estilos para este ASR.",
        ]
    else:
        header = "I have identified two candidate architecture styles for your ASR:"
        rec_label = "Recommendation"
        because = "because"
        followups = [
            f"Explain concrete tactics for the ASR using the recommended style ({chosen_name}).",
            "Compare these two styles in more depth for this ASR.",
            ]

    content = (
        f"{header}\n\n"
        f"1) {style1_name}\n"
        f"   - Impact: {style1_impact}\n\n"
        f"2) {style2_name}\n"
        f"   - Impact: {style2_impact}\n\n"
        f"{rec_label}: **{chosen_name}** {because}:\n"
        f"{rationale}\n"
    )

    state["turn_messages"] = state.get("turn_messages", []) + [
        {"role": "assistant", "name": "style_recommender", "content": content}
    ]
    state["suggestions"] = followups
    state["endMessage"] = content
    state["nextNode"] = "unifier"

    return state
