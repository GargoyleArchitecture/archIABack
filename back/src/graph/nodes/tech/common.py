import re
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.resources import llm, log, rag_trace_record
from src.rag_agent import get_indexed_retriever
from src.utils.json_helpers import extract_json_array, strip_first_json_fence
from src.graph.utils import _dedupe_snippets, _clip_text, _push_turn
from src.graph.consts import TECH_JSON_EXAMPLE, MARKDOWN_FORMAT_DIRECTIVE
from src.graph.prompts.mode_prompts import apply_mode_prompt
from src.graph.qa_registry import normalize_qa
from src.graph.nodes.tactics.common import resolve_qa_for_tactics, _refresh_ledger_state
from src.ledger import (
    append_decision,
    compute_active_view,
    get_all_active_asrs,
    LedgerValidationError,
    LedgerConcurrencyError,
)
from src.ledger.types import Phase

_tech_log = logging.getLogger("tech_node")

_UNVERIFIED_WARNING_ES = "⚠ No respaldado por la base de conocimiento — verificar independientemente."
_UNVERIFIED_WARNING_EN = "⚠ Not backed by the knowledge base — verify independently."


@lru_cache(maxsize=64)
def _fetch_tech_rag(qa: str, resolved_index: str, k: int = 6) -> tuple:
    """Returns (book_snippets: str, src_meta: tuple). Cached by (qa, resolved_index, k)."""
    queries = [
        f"{qa} technology implementation framework",
        f"{qa} tools libraries infrastructure",
        f"technology stack for {qa} architectural tactics",
        "Bass Clements Kazman technology choices",
    ]
    _retriever = get_indexed_retriever(
        quality_attribute=normalize_qa(resolved_index or qa),
        content_type="tecnologias",
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

    book_snippets = _dedupe_snippets(gathered, max_items=5, max_chars=600)

    src_meta = []
    for d in gathered:
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page = md.get("page_label") or md.get("page")
        path = md.get("source_path") or md.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_meta.append((title, page_str, path))

    return book_snippets, tuple(src_meta)


def _normalize_tech_json(items: list) -> list:
    """Valida campos requeridos y normaliza la lista de propuestas tecnológicas."""
    out = []
    for i, it in enumerate(items or [], 1):
        if not isinstance(it, dict):
            continue
        name = (it.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "id": (it.get("id") or f"TECH-{i}").strip(),
            "name": name,
            "tactic": (it.get("tactic") or "").strip(),
            "asr_id": (it.get("asr_id") or "").strip(),
            "rationale": (it.get("rationale") or "").strip(),
            "rag_backed": bool(it.get("rag_backed", False)),
        })
    return out


def _build_tech_parent_refs(ledger_active: dict) -> list:
    active = ledger_active or {}
    refs = []
    for kind in ("asr", "style", "tactic"):
        entry = active.get(kind)
        if entry:
            refs.append({"id": entry["id"], "kind": kind, "iteration": entry.get("iteration", 0)})
    return refs


def _build_selected_context(state: GraphState) -> str:
    """Assembles a rich context block from the full ADD 3.0 decision chain."""
    lines = []

    selected_asrs = state.get("selected_asrs") or []
    asr_candidates = state.get("asr_candidates") or []
    if selected_asrs and asr_candidates:
        lines.append("### Selected ASRs")
        for asr_id in selected_asrs:
            for asr in asr_candidates:
                if isinstance(asr, dict) and asr.get("id") == asr_id:
                    lines.append(f"- **{asr_id}**: {asr.get('scenario') or asr.get('description') or asr_id}")
    elif state.get("current_asr") or state.get("last_asr"):
        lines.append("### ASR")
        lines.append((state.get("current_asr") or state.get("last_asr") or "").strip())

    selected_style = state.get("selected_style") or state.get("style") or state.get("last_style") or ""
    if selected_style:
        lines.append("\n### Selected Style")
        lines.append(selected_style.strip())

    selected_tactics = state.get("selected_tactics") or []
    tactics_candidates = state.get("tactics_candidates") or []
    if selected_tactics:
        lines.append("\n### Confirmed Tactics")
        for tac_id in selected_tactics:
            matched = next(
                (t for t in tactics_candidates if isinstance(t, dict) and t.get("id") == tac_id),
                None,
            )
            if matched:
                lines.append(f"- **{matched.get('name', tac_id)}** (id: {tac_id}): {matched.get('rationale', '')[:120]}")
            else:
                lines.append(f"- {tac_id}")

    return "\n".join(lines)


def tech_node_impl(
    state: GraphState,
    qa_override: str | None = None,
) -> GraphState:
    """Implementación común del nodo de tecnologías (ADD 3.0 — Point 5)."""
    lang = state.get("language", "es")

    # ── Gate: requiere tácticas confirmadas ─────────────────────────────────
    selected_tactics = state.get("selected_tactics") or []
    if not selected_tactics:
        msg = (
            "Primero necesito que confirmes las tácticas antes de proponer tecnologías."
            if lang == "es"
            else "I need you to confirm tactics before proposing technologies."
        )
        # hasVisitedTech=True so _augment_completed_nodes marks 'tech' complete
        # and supervisor routes to unifier instead of looping back here.
        return {
            **state,
            "endMessage": msg,
            "nextNode": "unifier",
            "intent": "tech",
            "hasVisitedTech": True,
        }

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
        or ""
    )
    qa = resolve_qa_for_tactics(state, asr_text=asr_text, qa_override=qa_override)
    proj_ctx = (state.get("project_context_text") or "").strip()

    # ── RAG retrieval ────────────────────────────────────────────────────────
    book_snippets, src_meta = _fetch_tech_rag(qa, qa, k=6)
    rag_trace_record(
        query=" | ".join([
            f"{qa} technology implementation framework",
            f"{qa} tools libraries infrastructure",
            f"technology stack for {qa} architectural tactics",
            "Bass Clements Kazman technology choices",
        ])
    )

    # ── Context blocks ───────────────────────────────────────────────────────
    decision_chain = _build_selected_context(state)

    proj_ctx_block = ""
    if proj_ctx:
        proj_ctx_block = (
            f'\n{"=" * 60}\n'
            f"PROJECT CONTEXT — MANDATORY CONSTRAINTS FOR TECHNOLOGY SELECTION:\n"
            f"{proj_ctx}\n\n"
            f"IMPORTANT: All proposed technologies MUST be compatible with the listed tech stack and business rules.\n"
            f'{"=" * 60}\n'
        )

    # ── Active ledger binding ────────────────────────────────────────────────
    ledger_active = state.get("ledger_active") or {}
    active_asr = ledger_active.get("asr") or {}
    response_measure = (active_asr.get("payload") or {}).get("response_measure", "")
    binding_block = ""
    if response_measure:
        binding_block = (
            f'\n{"=" * 60}\n'
            f"ACTIVE ASR BINDING:\n"
            f"  Response Measure: {response_measure}\n"
            f"  All technologies MUST demonstrably address this response measure.\n"
            f'{"=" * 60}\n'
        )

    # ── Multi-ASR priority constraint (P7) ─────────────────────────────────
    _ledger = state.get("ledger") or {}
    _all_asrs = get_all_active_asrs(_ledger) if _ledger.get("decisions") else []
    _priority_block = ""
    if _all_asrs:
        _top_qa = _all_asrs[0].get("qa", "")
        if _top_qa:
            if lang == "en":
                _priority_block = (
                    f'\n{"=" * 60}\n'
                    f'PRIORITY CONSTRAINT (P7):\n'
                    f'  Highest-priority QA: {_top_qa}\n'
                    f'  If a technology negatively impacts "{_top_qa}", add a\n'
                    f'  "priority_conflict" field (one sentence) to its JSON entry.\n'
                    f'{"=" * 60}\n'
                )
            else:
                _priority_block = (
                    f'\n{"=" * 60}\n'
                    f'RESTRICCIÓN DE PRIORIDAD (P7):\n'
                    f'  QA de mayor prioridad: {_top_qa}\n'
                    f'  Si una tecnología impacta negativamente "{_top_qa}", agrega un\n'
                    f'  campo "priority_conflict" (una oración) en su entrada JSON.\n'
                    f'{"=" * 60}\n'
                )

    prompt = f"""{directive}
You are an architecture technology advisor following ADD 3.0.

Your job is to propose CONCRETE technologies that implement the CONFIRMED architectural tactics below.
Each technology must directly trace back to one tactic and one ASR.
{proj_ctx_block}
{binding_block}
{_priority_block}
## ADD 3.0 Decision Chain (confirmed by user)
{decision_chain or "(no prior decisions in state)"}

## Knowledge Base Grounding
Use the following excerpts when proposing technologies.
If a technology appears in the excerpts, set "rag_backed": true.
If a technology comes only from your general knowledge, set "rag_backed": false.
Do NOT omit a good technology just because it lacks RAG backing — flag it honestly instead.

{book_snippets or "(no RAG snippets available)"}

## Output Instructions

Produce TWO sections in EXACT order:

### 1. Technology Proposals (Markdown)
For each proposed technology write a ### heading with its name.
Include: **Tactic it implements**, **ASR it addresses**, **Rationale**, **Trade-offs**, **RAG-backed** (yes/no).
{MARKDOWN_FORMAT_DIRECTIVE}

### 2. JSON
Return ONE code fence starting with ```json containing a JSON array with EXACTLY 3 objects.
Required fields per object:
- "id": string like "TECH-1", "TECH-2", "TECH-3"
- "name": technology name
- "tactic": the tactic name it implements
- "asr_id": the ASR id it addresses (e.g. "ASR-1")
- "rationale": 1-2 sentence justification
- "rag_backed": boolean — true if the technology appears in the RAG snippets, false if from general knowledge only

Example shape (values are illustrative):
{TECH_JSON_EXAMPLE}

{"RECORDATORIO FINAL: toda tu respuesta debe estar en español." if lang == "es" else "FINAL REMINDER: your entire response must be in English."}
"""

    resp = llm.invoke(apply_mode_prompt(state, prompt))
    raw = getattr(resp, "content", str(resp)).strip()

    log.debug("tech_node raw (first 400): %s", raw[:400].replace("\n", " "))

    struct = extract_json_array(raw) or []
    struct = _normalize_tech_json(struct)

    # ── Apply unverified warning to non-RAG items ────────────────────────────
    warning = _UNVERIFIED_WARNING_ES if lang == "es" else _UNVERIFIED_WARNING_EN
    for item in struct:
        if not item.get("rag_backed"):
            item["rationale"] = f"{item['rationale']} {warning}".strip()

    md_only = strip_first_json_fence(raw)
    md_only = re.sub(r"\n?###\s+2\.\s*JSON\s*:?\s*$", "", md_only, flags=re.I | re.M).rstrip()
    if not md_only and struct:
        md_only = "\n".join(
            f"- **{it['name']}** ({it['tactic']}): {it['rationale'][:100]}"
            for it in struct
        )

    src_lines = [
        _clip_text(f"- {title}{page_str} — {path}", 60)
        for title, page_str, path in src_meta
    ]
    src_lines = list(dict.fromkeys(src_lines))[:6]
    src_block = "SOURCES:\n" + ("\n".join(src_lines) if src_lines else "- (no local sources)")

    _push_turn(state, role="system", name="tech_system", content=prompt)
    _push_turn(state, role="assistant", name="tech_advisor", content=md_only)
    _push_turn(state, role="assistant", name="tech_sources", content=src_block)

    msgs = [
        AIMessage(content=md_only, name="tech_advisor"),
        AIMessage(content=src_block, name="tech_sources"),
    ]

    # ── Scalar writes ────────────────────────────────────────────────────────
    state["tech_candidates"] = struct
    state["quality_attribute"] = qa
    if asr_text:
        state["current_asr"] = asr_text

    # ── Ledger write-back ────────────────────────────────────────────────────
    _user_id = (state.get("user_id_for_prefs") or "").strip()
    _project_id = (state.get("project_id") or "").strip() or None

    if _user_id:
        try:
            rag_count = sum(1 for it in struct if it.get("rag_backed"))
            _parents = _build_tech_parent_refs(state.get("ledger_active") or {})
            _new_decision: dict = {
                "id": "",
                "kind": "tech",
                "phase": Phase.TECH_PROPOSALS.value,
                "iteration": 0,
                "qa": qa,
                "parents": _parents,
                "payload": {
                    "items": struct,
                    "rag_coverage": f"{rag_count}/{len(struct)}" if struct else "0/0",
                },
                "rationale": "",
                "sources": [],
                "status": "active",
                "parent_status": "ok",
                "superseded_by": None,
                "rejection_reason": None,
                "created_at": "",
                "created_by_node": "tech_node",
            }
            _saved = append_decision(_user_id, _project_id, _new_decision)
            _tech_log.info(
                "tech_node: ledger ok id=%s qa=%s items=%d project=%s",
                _saved["id"], qa, len(struct), _project_id,
            )
            _refresh_ledger_state(state, _user_id, _project_id, lang)

        except LedgerValidationError as _exc:
            _tech_log.warning("tech_node: ledger validation error (nonfatal): %s", _exc)
        except LedgerConcurrencyError as _exc:
            _tech_log.warning("tech_node: ledger concurrency error (nonfatal): %s", _exc)
        except Exception as _exc:
            _tech_log.warning("tech_node: unexpected ledger error (nonfatal): %s", _exc)

    state["endMessage"] = md_only
    state["intent"] = "tech"
    state["nextNode"] = "unifier"
    state["hasVisitedTech"] = True
    prev_msgs = state.get("messages", [])
    return {**state, "messages": prev_msgs + msgs}
