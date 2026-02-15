
import re
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.graph.state import GraphState
from src.graph.resources import llm, retriever, _HAS_VERTEX
from src.graph.utils import _push_turn
from src.graph.nodes.supervisor import _looks_like_eval
from src.graph.nodes.tools import theory_tool, viability_tool, needs_tool, analyze_tool

def _pick_asr_to_evaluate(state: GraphState) -> str:
    if state.get("last_asr"):
        return state["last_asr"]
    uq = (state.get("userQuestion") or "")
    m = re.search(r"(evaluate|evalúa|evaluar|check|review)\s+(this|este)\s+asr\s*:?\s*(.+)$", uq, re.I | re.S)
    if m:
        return m.group(3).strip()
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and (getattr(m, "name", "") or "") == "asr_recommender" and m.content:
            return m.content
    return ""

def _book_snippets_for_eval(retriever, concern_hint: str = "") -> str:
    q = "quality attribute scenario parts stimulus source environment artifact response response measure"
    if concern_hint:
        q = concern_hint + " " + q
    try:
        docs = list(retriever.invoke(q))
    except Exception:
        docs = []
    # 4 fragmentos de 300 chars c/u
    seen, out = set(), []
    for d in docs:
        t = (d.page_content or "").strip().replace("\n", " ")
        t = t[:300]
        if t and t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= 4: break
    return "\n\n".join(out)

def getEvaluatorPrompt(image_path1: str, image_path2: str) -> str:
    i1 = f"\nthis is the first image path: {image_path1}" if image_path1 else ""
    i2 = f"\nthis is the second image path: {image_path2}" if image_path2 else ""
    return f"""You are an expert in software-architecture evaluation.
Use:
- Theory Tool (correctness)
- Viability Tool (feasibility)
- Needs Tool (requirements alignment)
- Analyze Tool (compare two diagrams){i1}{i2}
Keep answers short and decisive."""

def evaluator_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    uq = (state.get("userQuestion") or "")
    concern_hint = "latency" if re.search(r"latenc", uq, re.I) else ("scalability" if re.search(r"scalab", uq, re.I) else "")
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()

    # --- MODO 1: evaluación de ASR ---
    if _looks_like_eval(uq):
        asr_text = _pick_asr_to_evaluate(state)
        if not asr_text:
            short = "No encuentro un ASR para evaluar. Pega el texto del ASR o pide que genere uno primero." if lang=="es" \
                    else "I couldn't find an ASR to evaluate. Paste the ASR text or ask me to create one first."
            _push_turn(state, role="assistant", name="evaluator", content=short)
            return {**state, "messages": state["messages"] + [AIMessage(content=short, name="evaluator")], "hasVisitedEvaluator": True}

        if doc_only and ctx_doc:
            book_snips = f"[DOC] {ctx_doc[:1500]}"
        else:
            book_snips = _book_snippets_for_eval(retriever, concern_hint)

        directive = "Responde en español." if lang=="es" else "Answer in English."
        eval_prompt = f"""{directive}
You are evaluating a Quality Attribute Scenario (Architecture Significant Requirement).

BOOK_SNIPPETS (ground your critique in these ideas; keep it short):
{book_snips}

ASR_TO_EVALUATE:
{asr_text}

Write a compact evaluation with EXACTLY these sections (plain text, no Markdown):

Verdict:
  One line: Good / Weak / Invalid, with a short reason.

Gaps:
  3–6 bullets pointing missing or vague parts against the canonical QAS fields (Source, Stimulus, Environment, Artifact, Response, Response Measure).

Quality:
  3–5 bullets about measurability, precision of Response Measure (p95/p99, thresholds), clarity of stimulus, realism.

Risks & Tactics:
  3–5 bullets on plausible risks and which tactics mitigate them (use tactic names verbatim).

Rewrite (improved ASR):
  Provide a tightened ASR using the same QAS structure (Summary, Context, Scenario with the 6 fields, Response Measure). Keep it realistic and measurable.

References:
  List 2–5 short items only if grounded by BOOK_SNIPPETS; otherwise write "None".
"""
        eval_prompt = ("DOC-ONLY mode: ON. Reason exclusively from the PROJECT DOCUMENT.\n\n" + eval_prompt) if doc_only else eval_prompt
        result = llm.invoke(eval_prompt)
        content = getattr(result, "content", str(result)).strip()

        _push_turn(state, role="system", name="evaluator_system", content=eval_prompt)
        _push_turn(state, role="assistant", name="evaluator", content=content)

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=content, name="evaluator")],
            "hasVisitedEvaluator": True
        }

    # --- MODO 2 (fallback): tools variados ---
    tools = [theory_tool, viability_tool, needs_tool]
    if _HAS_VERTEX:
        tools.append(analyze_tool)
    evaluator_agent = create_react_agent(llm, tools=tools)

    eval_prompt = getEvaluatorPrompt(state.get("imagePath1",""), state.get("imagePath2",""))
    ctx_add = (state.get("add_context") or "").strip()[:1500]
    if doc_only and ctx_doc:
        eval_prompt = f"DOC-ONLY: use exclusively this PROJECT DOCUMENT.\n{ctx_doc[:1500]}\n\n" + eval_prompt
    elif ctx_add:
        eval_prompt = f"PROJECT CONTEXT:\n{ctx_add}\n\n" + eval_prompt
    _push_turn(state, role="system", name="evaluator_system", content=eval_prompt)

    messages_with_system = [SystemMessage(content=eval_prompt)] + state["messages"]
    result = evaluator_agent.invoke({
        "messages": messages_with_system,
        "userQuestion": state.get("userQuestion",""),
        "localQuestion": state.get("localQuestion",""),
        "imagePath1": state.get("imagePath1",""),
        "imagePath2": state.get("imagePath2","")
    })

    for msg in result["messages"]:
        _push_turn(state, role="assistant", name="evaluator", content=str(getattr(msg, "content", msg)))

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="evaluator") for msg in result["messages"]],
        "hasVisitedEvaluator": True
    }
