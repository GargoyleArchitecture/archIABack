# -*- coding: utf-8 -*-
import re

from langchain_core.messages import AIMessage
from src.graph.resources import llm, rag_trace_record
from src.graph.state import GraphState
from src.graph.consts import MARKDOWN_FORMAT_DIRECTIVE
from src.graph.utils import (
    _clip_text,
    _dedupe_snippets,
    _sanitize_response,
    _strip_tactics_sections,
)
from src.rag_agent import get_indexed_retriever
from src.graph.qa_registry import normalize_qa, qa_to_focus_label


def asr_node(state: GraphState) -> GraphState:
    """Genera ASR y deja QA coherente para nodos siguientes (style/tactics)."""
    lang = state.get("language", "es")
    uq = state.get("userQuestion", "") or ""
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()

    # Heurística del atributo
    concern = (
        "scalability"
        if re.search(r"scalab", uq, re.I)
        else "latency"
        if re.search(r"latenc", uq, re.I)
        else "performance"
    )

    # QA operativo para el pipeline: prioriza classifier (resolved_index) cuando exista.
    qa_from_classifier = normalize_qa(state.get("resolved_index", ""))
    qa_from_text = normalize_qa(concern)
    qa_pipeline = qa_from_classifier if qa_from_classifier != "general" else qa_from_text
    qa_focus = qa_to_focus_label(qa_pipeline, default=concern)

    # Dominio típico si el usuario no lo da
    low = uq.lower()
    if any(k in low for k in ["e-comm", "commerce", "shop", "checkout"]):
        domain = "e-commerce flash sale"
    elif "api" in low:
        domain = "public REST API with burst traffic"
    elif any(k in low for k in ["stream", "kafka"]):
        domain = "event streaming pipeline"
    else:
        domain = "e-commerce flash sale"

    # === RAG (saltable) ===
    docs_list = []
    if state.get("force_rag", False) and not doc_only:
        try:
            query = f"{qa_focus} quality attribute scenario latency measure stimulus environment artifact response response measure"
            _retriever = get_indexed_retriever(
                quality_attribute=(state.get("resolved_index") or qa_pipeline),
                content_type="asr",
                k=6,
            )
            docs_raw = list(_retriever.invoke(query))
            docs_list = docs_raw[:6]
        except Exception:
            docs_list = []

    book_snippets = _dedupe_snippets(docs_list, max_items=6, max_chars=800)

    directive = "Answer in English." if lang == "en" else "Responde en español."
    ctx = (
        ctx_doc if (doc_only and ctx_doc) else (state.get("add_context") or "")
    ).strip()[:2000]

    prompt = f"""{directive}
You are an expert software architect following Attribute-Driven Design 3.0 (ADD 3.0).

Your job is to create 1-5 concrete Architecture Significant Requirement(s) (ASR)
that will be used as an architectural driver.

Each ASR MUST:
- Follow the classic QAS structure: Source, Stimulus, Environment, Artifact, Response, Response Measure.
- Be measurable, with a clear SINGLE Response Measure (SLO/SLA, e.g. p95 < X ms under Y load, error rate, availability, etc.).
- Be realistic for production systems in the given domain.
- Follow a single quality attribute focus (e.g. latency, scalability, availability) inferred from the user question.

Relevant domain or workload (you must stay coherent with this):
{domain}

Quality attribute focus inferred from the user message:
{qa_focus}

User input to ground this ASR:
{uq}

PROJECT CONTEXT (if any):
{ctx or "None"}

OPTIONAL BOOK CONTEXT (only if not in DOC-ONLY mode):
{book_snippets or "None"}

OUTPUT FORMAT (MANDATORY):

Use this Markdown structure:

## ASR

**ASR complete:** <one single sentence that concisely states Source, Stimulus, Environment, Artifact, Response and Response Measure in natural language>

### Scenario

- **Source:** <who initiates the stimulus>
- **Stimulus:** <what happens / event that triggers the behavior>
- **Environment:** <when / in which operating conditions this happens>
- **Artifact:** <what part of the system is stimulated>
- **Response:** <what the system must do>
- **Response Measure:** <how success is measured with clear numeric thresholds>

Rules:
- The line that starts with "**ASR complete:**" MUST be a single sentence.
- Then the section "### Scenario" with each of the six fields as bold-labeled list items.
- Do NOT add any other sections (no 'Architectural Driver Summary', no 'Summary', no 'Context' headings).
- Do NOT talk about tactics, styles or next steps here.
- Keep the numbers realistic and measurable (p95 / p99, RPS, error rate, availability, etc.).
- Answer entirely in the requested language.
{MARKDOWN_FORMAT_DIRECTIVE}
"""

    result = llm.invoke(prompt)
    content_raw = getattr(result, "content", str(result))
    content = _sanitize_response(content_raw)
    content = _strip_tactics_sections(content)

    # === Fuentes (si hubo RAG) ===
    src_lines = []
    for d in docs_list or []:
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page = md.get("page_label") or md.get("page")
        path = md.get("source_path") or md.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str} — {path}")
    if src_lines:
        src_lines = [_clip_text(s, 60) for s in src_lines]
        src_lines = list(dict.fromkeys(src_lines))[:4]

    src_block = "SOURCES:\n" + (
        "\n".join(src_lines) if src_lines else "- (no local sources)"
    )

    # Traza + memoria de turno
    state["turn_messages"] = state.get("turn_messages", []) + [
        {"role": "system", "name": "asr_system", "content": prompt},
        {"role": "assistant", "name": "asr_recommender", "content": content},
        {"role": "assistant", "name": "asr_sources", "content": src_block},
    ]
    state["messages"] = state["messages"] + [
        AIMessage(content=content, name="asr_recommender"),
        AIMessage(content=src_block, name="asr_sources"),
    ]

    # Memoria viva del chat
    state["last_asr"] = content
    refs_list = [
        ln.lstrip("- ").strip()
        for ln in src_block.splitlines()
        if ln.strip() and not ln.lower().startswith("sources")
    ]
    state["asr_sources_list"] = refs_list
    prev_mem = state.get("memory_text", "") or ""
    state["memory_text"] = (prev_mem + f"\n\n[LAST_ASR]\n{content}\n").strip()

    # Metadatos
    state["quality_attribute"] = qa_pipeline
    state["arch_stage"] = "ASR"
    state["current_asr"] = content

    # Señales de fin de turno
    state["endMessage"] = content
    state["hasVisitedASR"] = True
    state["force_rag"] = False
    state["nextNode"] = "unifier"

    return state
