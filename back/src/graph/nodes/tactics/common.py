import re
import os
import json
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.resources import llm, log, rag_trace_record
from src.rag_agent import get_indexed_retriever
from src.utils.json_helpers import (
    extract_json_array,
    strip_first_json_fence,
    normalize_tactics_json,
    build_json_from_markdown,
)
from src.graph.utils import (
    _dedupe_snippets,
    _clip_text,
    _push_turn,
    _json_only_repair_pass,
)
from src.graph.consts import TACTICS_JSON_EXAMPLE
from src.graph.qa_registry import normalize_qa


def guess_quality_attribute(text: str) -> str:
    """Heurística legacy para QA cuando no hay señal explícita."""
    low = (text or "").lower()
    if "latenc" in low or "response time" in low:
        return "latencia"
    if "scalab" in low or "throughput" in low:
        return "escalabilidad"
    if "availab" in low or "uptime" in low:
        return "availability"
    if "secur" in low:
        return "security"
    if "modifiab" in low or "change" in low:
        return "modifiability"
    if "reliab" in low or "fault" in low:
        return "reliability"
    return "performance"


def resolve_qa_for_tactics(state: GraphState, asr_text: str, qa_override: str | None = None) -> str:
    """Resuelve QA final para tácticas con prioridad explícita."""
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

    qa_from_asr = normalize_qa(asr_text)
    if qa_from_asr != "general":
        return qa_from_asr

    return guess_quality_attribute(asr_text)


def tactics_node_impl(state: GraphState, qa_override: str | None = None) -> GraphState:
    """Implementación común del nodo de tácticas (ADD 3.0)."""
    lang = state.get("language", "es")
    directive = "Answer in English." if lang == "en" else "Responde en español."
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()
    ctx_add = (state.get("add_context") or "").strip()
    ctx = (ctx_doc if (doc_only and ctx_doc) else ctx_add)[:2000]

    asr_text = (
        state.get("current_asr")
        or state.get("asr_text")
        or state.get("last_asr")
        or ""
    )
    if not asr_text:
        uq = state.get("userQuestion", "") or ""
        m = re.search(r"(?:^|\n)\s*ASR\s*:?\s*(.+)$", uq, flags=re.I | re.S)
        asr_text = (m.group(1).strip() if m else "")

    qa = resolve_qa_for_tactics(state, asr_text=asr_text, qa_override=qa_override)
    style_text = state.get("style") or state.get("selected_style") or state.get("last_style") or ""

    docs_list = []
    if doc_only and ctx_doc:
        book_snippets = f"[DOC] {ctx_doc[:2000]}"
    else:
        try:
            queries = [
                f"{qa} architectural tactics",
                f"{qa} tactics performance scalability latency availability security modifiability",
                "Bass Clements Kazman performance and scalability tactics",
                "quality attribute tactics list",
            ]
            _retriever = get_indexed_retriever(
                quality_attribute=normalize_qa(state.get("resolved_index") or qa),
                content_type="tacticas",
                k=6,
            )
            seen = set()
            gathered = []
            for q in queries:
                for d in _retriever.invoke(q):
                    key = (d.metadata.get("source_path"), d.metadata.get("page"))
                    if key in seen:
                        continue
                    seen.add(key)
                    gathered.append(d)
                    if len(gathered) >= 6:
                        break
                if len(gathered) >= 6:
                    break
            docs_list = gathered
            rag_trace_record(query=" | ".join(queries), docs=docs_list)
        except Exception:
            docs_list = []
        book_snippets = _dedupe_snippets(docs_list, max_items=5, max_chars=600)

    prompt = f"""{directive}
You are an expert software architect applying Attribute-Driven Design 3.0 (ADD 3.0).

We ALREADY HAVE an ASR / Quality Attribute Scenario. That ASR is an ADD 3.0 architectural driver.
Your job now is to continue the ADD 3.0 process by selecting architectural tactics.

PROJECT CONTEXT (if any)
{ctx or "None"}

ASR (driver to satisfy):
{asr_text or "(none provided)"}

Primary quality attribute (guessed):
{qa}
Selected architecture style (if any):
{style_text or "(none)"}


GROUNDING (use ONLY this context; if DOC-ONLY, this is the exclusive source):
{book_snippets or "(none)"}

If DOC-ONLY is ON, do not rely on knowledge beyond the PROJECT DOCUMENT even if you “know” typical tactics. If the document does not support a tactic, state “not supported by the document”.

You MUST output THREE sections, in EXACT order:

(0) Which is the ASR and it´s style (if any):
- 3–5 concise lines.
- Explicitly link back to the ASR's Source, Stimulus, Artifact, Environment and Response Measure. Also its architectonic style.

(1) TACTICS (TOP-3 with highest success probability):
Select EXACTLY THREE architectural tactics that maximally satisfy this ASR GIVEN the selected style.
For EACH tactic include: Name, Rationale, Consequences / Trade-offs, When to use, Why it ranks in TOP-3, Sucess probability.

(2) JSON:
Return ONE code fence starting with ```json and ending with ``` that contains ONLY a JSON array with EXACTLY 3 objects.
- Use dot as decimal separator (e.g., 0.82), never commas.
- Do not use percent signs, just 0..1 floats for success_probability.
- Do not add any prose or markdown outside the JSON fence.

Example shape (values are illustrative — adjust to your tactics):
{TACTICS_JSON_EXAMPLE}
"""
    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", str(resp)).strip()

    log.debug("tactics raw (first 400): %s", raw[:400].replace("\n", " "))
    log.debug("has ```json fence? %s", bool(re.search(r"```json", raw, re.I)))

    struct = extract_json_array(raw) or []
    if not (isinstance(struct, list) and struct):
        struct = _json_only_repair_pass(llm, asr_text=asr_text, qa=qa, style_text=style_text, md_preview=raw) or []
    if not (isinstance(struct, list) and struct):
        struct = build_json_from_markdown(raw, top_n=3)
    struct = normalize_tactics_json(struct, top_n=3)

    md_only = strip_first_json_fence(raw)
    if os.getenv("SHOW_TACTICS_JSON", "0") == "1":
        md_only = f"{md_only}\n\n```json\n{json.dumps(struct, ensure_ascii=False, indent=2)}\n```"
    else:
        md_only = re.sub(r"\n?\(?2\)?\s*JSON\s*:?\s*$", "", md_only, flags=re.I | re.M).rstrip()
    if (not md_only) and isinstance(struct, list) and struct:
        md_only = "\n".join(f"- {it.get('name','')}: {it.get('rationale','')}" for it in struct if isinstance(it, dict))

    src_lines = []
    for d in (docs_list or []):
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page = md.get("page_label") or md.get("page")
        path = md.get("source_path") or md.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str} — {path}")
    if src_lines:
        src_lines = list(dict.fromkeys([_clip_text(s, 60) for s in src_lines]))[:6]
    src_block = "SOURCES:\n" + ("\n".join(src_lines) if src_lines else "- (no local sources)")

    _push_turn(state, role="system", name="tactics_system", content=prompt)
    _push_turn(state, role="assistant", name="tactics_advisor", content=md_only)
    _push_turn(state, role="assistant", name="tactics_sources", content=src_block)

    msgs = [AIMessage(content=md_only, name="tactics_advisor"), AIMessage(content=src_block, name="tactics_sources")]

    state["tactics_md"] = md_only
    state["tactics_struct"] = struct if isinstance(struct, list) else []
    state["tactics_list"] = [(it.get("name") or "").strip() for it in (struct or []) if isinstance(it, dict) and it.get("name")]
    state["arch_stage"] = "TACTICS"
    state["quality_attribute"] = qa
    if asr_text:
        state["current_asr"] = asr_text

    state["endMessage"] = md_only
    state["intent"] = "tactics"
    state["nextNode"] = "unifier"
    prev_msgs = state.get("messages", [])
    return {**state, "messages": prev_msgs + msgs}
