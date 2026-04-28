# -*- coding: utf-8 -*-
import logging
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

log = logging.getLogger("asr_node")


# ---------------------------------------------------------------------------
# Compile-time regexes (Step 1 — P3)
# ---------------------------------------------------------------------------

_HISTORY_HEADING_RE = re.compile(
    r"^##\s+(?:History\s+\(superseded\s*/\s*rejected\)|"
    r"Historial\s+\(reemplazadas\s*/\s*rechazadas\))\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_ASR_SUMMARY_RE = re.compile(r"\*\*ASR\s+complete\s*:\*\*\s*(.+)", re.IGNORECASE)

_ASR_FIELD_RE: dict[str, re.Pattern] = {
    "source":           re.compile(r"-\s*\*\*Source\s*:\*\*\s*(.+)",             re.IGNORECASE),
    "stimulus":         re.compile(r"-\s*\*\*Stimulus\s*:\*\*\s*(.+)",           re.IGNORECASE),
    "environment":      re.compile(r"-\s*\*\*Environment\s*:\*\*\s*(.+)",        re.IGNORECASE),
    "artifact":         re.compile(r"-\s*\*\*Artifact\s*:\*\*\s*(.+)",           re.IGNORECASE),
    "response":         re.compile(r"-\s*\*\*Response\s*:\*\*\s*(.+)",           re.IGNORECASE),
    "response_measure": re.compile(r"-\s*\*\*Response\s+Measure\s*:\*\*\s*(.+)", re.IGNORECASE),
}

_NONE_MARKER_RE = re.compile(r"_\((?:ninguna aún|none yet)\)_", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pure helpers (Step 1 — P3, not yet called by asr_node)
# ---------------------------------------------------------------------------

def _extract_dossier_history(dossier_md: str) -> str:
    """Return the history section from design_dossier_md, or '' if absent/empty."""
    if not dossier_md:
        return ""
    m = _HISTORY_HEADING_RE.search(dossier_md)
    if not m:
        return ""
    section = dossier_md[m.start():].strip()
    content_lines = [
        ln for ln in section.splitlines()
        if ln.strip()
        and not ln.strip().startswith("##")
        and not _NONE_MARKER_RE.fullmatch(ln.strip())
    ]
    return section if content_lines else ""


def _build_asr_payload(content: str, domain: str) -> dict:
    """Parse structured ASR markdown into a ledger payload dict."""
    m = _ASR_SUMMARY_RE.search(content)
    summary = m.group(1).strip() if m else _clip_text(content.strip(), 300)

    payload: dict = {
        "summary":          summary,
        "source":           "",
        "stimulus":         "",
        "environment":      "",
        "artifact":         "",
        "response":         "",
        "response_measure": "",
        "domain":           domain or "",
    }
    for field_key, pattern in _ASR_FIELD_RE.items():
        fm = pattern.search(content)
        if fm:
            payload[field_key] = fm.group(1).strip()
    return payload


def _build_sources_from_docs(docs_list: list) -> list[dict]:
    """Convert RAG Document objects to ledger source dicts (title, page, path)."""
    seen: set = set()
    result: list = []
    for d in docs_list or []:
        md = d.metadata or {}
        title = (md.get("source_title") or md.get("title") or "doc").strip()
        page  = md.get("page_label") or md.get("page")
        path  = (md.get("source_path") or md.get("source") or "").strip()
        key   = (title, str(page), path)
        if key in seen:
            continue
        seen.add(key)
        entry: dict = {"title": title, "path": path}
        if page is not None:
            entry["page"] = page
        result.append(entry)
        if len(result) >= 4:
            break
    return result


def _refresh_ledger_state(
    state: dict,
    user_id: str,
    project_id: str | None,
    lang: str,
) -> None:
    """Refresh ledger-derived state fields in-place after a successful append_decision."""
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
        log.debug("asr_node: ledger state refreshed phase=%s", state["current_phase"])
    except Exception as exc:
        log.warning("asr_node: state refresh failed (nonfatal): %s", exc)


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
    style_hint = (state.get("user_style_hint") or "").strip()
    if style_hint:
        directive = f"{directive} {style_hint}"

    ctx = (
        ctx_doc if (doc_only and ctx_doc) else (state.get("add_context") or "")
    ).strip()[:2000]
    proj_ctx = (state.get("project_context_text") or "").strip()

    # ── Dossier history injection (P3) ─────────────────────────────────────
    history_block = _extract_dossier_history(
        (state.get("design_dossier_md") or "").strip()
    )
    history_clipped = _clip_text(history_block, 1500) if history_block else ""
    prior_asr_section = (
        f'\n{"=" * 60}\n'
        f'PRIOR ASR HISTORY — DO NOT REPEAT THESE DESIGNS:\n'
        f'{history_clipped}\n\n'
        f'IMPORTANT: Your new ASR MUST be meaningfully different '
        f'in at least its Response Measure or Stimulus.\n'
        f'{"=" * 60}\n'
    ) if history_clipped else ""

    prompt = f"""{directive}
You are an expert software architect following Attribute-Driven Design 3.0 (ADD 3.0).

Your job is to create 1-5 concrete Architecture Significant Requirement(s) (ASR)
that will be used as an architectural driver.

Each ASR MUST:
- Follow the classic QAS structure: Source, Stimulus, Environment, Artifact, Response, Response Measure.
- Be measurable, with a clear SINGLE Response Measure (SLO/SLA, e.g. p95 < X ms under Y load, error rate, availability, etc.).
- Be realistic for production systems in the given domain.
- Follow a single quality attribute focus (e.g. latency, scalability, availability) inferred from the user question.

{"=" * 60}
PROJECT CONTEXT — YOU MUST RESPECT THESE CONSTRAINTS:
{proj_ctx if proj_ctx else "(none — no project configured)"}

IMPORTANT: If a tech stack is listed above, the ASR's Artifact and Response MUST reference
those specific technologies. If business rules are listed, the ASR scenario MUST be coherent
with them. Do NOT use generic placeholders like "the system" when a real stack is provided.
{"=" * 60}
{prior_asr_section}
Relevant domain or workload (you must stay coherent with this):
{domain}

Quality attribute focus inferred from the user message:
{qa_focus}

User input to ground this ASR:
{uq}

Additional session context (if any):
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

    # ── Ledger write-back (P3) ────────────────────────────────────────────
    _user_id    = (state.get("user_id_for_prefs") or "").strip()
    _project_id = (state.get("project_id") or "").strip() or None

    if _user_id:
        try:
            _new_decision: dict = {
                "id":               "",
                "kind":             "asr",
                "phase":            Phase.ASR.value,
                "iteration":        0,
                "qa":               qa_pipeline,
                "parents":          [],
                "payload":          _build_asr_payload(content, domain),
                "rationale":        "",
                "sources":          _build_sources_from_docs(docs_list),
                "status":           "active",
                "parent_status":    "ok",
                "superseded_by":    None,
                "rejection_reason": None,
                "created_at":       "",
                "created_by_node":  "asr_node",
            }
            _saved = append_decision(_user_id, _project_id, _new_decision)
            log.info(
                "asr_node: ledger ok id=%s qa=%s project=%s",
                _saved["id"], qa_pipeline, _project_id,
            )
            _refresh_ledger_state(state, _user_id, _project_id, lang)
        except LedgerValidationError as _exc:
            log.warning("asr_node: ledger validation error (nonfatal): %s", _exc)
        except LedgerConcurrencyError as _exc:
            log.warning("asr_node: ledger concurrency error (nonfatal): %s", _exc)
        except Exception as _exc:
            log.warning("asr_node: unexpected ledger error (nonfatal): %s", _exc)

    # Señales de fin de turno
    state["endMessage"] = content
    state["hasVisitedASR"] = True
    state["force_rag"] = False
    state["nextNode"] = "unifier"

    return state
