import re
import logging
from datetime import datetime, timezone
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
    transition_phase,
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


def _parse_tech_from_markdown(md: str) -> list[dict]:
    """Bug C fallback: parse tech proposals from ### heading blocks when JSON extraction fails."""
    items = []
    sections = re.split(r"\n###\s+", "\n" + (md or ""))
    for i, section in enumerate(sections[1:], 1):
        lines = section.strip().split("\n")
        if not lines:
            continue
        name = lines[0].strip()
        if not name or re.match(r"^\d+\.", name):  # skip numbered headings like "2. JSON"
            continue
        body = "\n".join(lines[1:])
        tactic = ""
        asr_id = ""
        rationale = ""
        rag_backed = False
        m = re.search(r"\*{1,2}[Tt]actic[^*\n]*\*{0,2}[:\s]+(.+)", body)
        if m:
            tactic = m.group(1).strip()
        m = re.search(r"\*{1,2}ASR[^*\n]*\*{0,2}[:\s]+(.+)", body)
        if m:
            asr_id = m.group(1).strip()
        m = re.search(r"\*{1,2}[Rr]ationale[^*\n]*\*{0,2}[:\s]+(.+)", body)
        if m:
            rationale = m.group(1).strip()
        m = re.search(r"\*{1,2}RAG.backed[^*\n]*\*{0,2}[:\s]+(yes|no|true|false)", body, re.I)
        if m:
            rag_backed = m.group(1).lower() in ("yes", "true")
        items.append({
            "id": f"TECH-{i}",
            "name": name,
            "tactic": tactic,
            "asr_id": asr_id,
            "rationale": rationale,
            "rag_backed": rag_backed,
        })
    return items


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

Your job is to propose CONCRETE technologies that implement ONLY the CONFIRMED tactics listed below.
Each technology must directly trace back to one confirmed tactic and one ASR.
HARD CONSTRAINT: Do NOT propose technologies for any tactic not listed under "Confirmed Tactics" above.
{proj_ctx_block}
{binding_block}
{_priority_block}
## ADD 3.0 Decision Chain (confirmed by user)
{decision_chain or "(no prior decisions in state)"}

## Knowledge Base Grounding
Use the following excerpts when proposing technologies.
MANDATORY: Every JSON object MUST include "rag_backed": true or false — this field is REQUIRED.
  - Set "rag_backed": true ONLY if the technology name explicitly appears in the excerpts below.
  - Set "rag_backed": false if the technology comes from your general knowledge, even if it is a good fit.
  - Never omit this field. Omitting it is an error.

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
- "tactic": the tactic name it implements (MUST be one of the confirmed tactics above)
- "asr_id": the ASR id it addresses — use the human-readable ID shown to the user (e.g. "A1", "A2")
- "rationale": 1-2 sentence justification
- "rag_backed": boolean — REQUIRED. true if the technology name appears in the RAG excerpts, false otherwise

Example shape (values are illustrative):
{TECH_JSON_EXAMPLE}

{"RECORDATORIO FINAL: toda tu respuesta debe estar en español." if lang == "es" else "FINAL REMINDER: your entire response must be in English."}
"""

    resp = llm.invoke(apply_mode_prompt(state, prompt))
    raw = getattr(resp, "content", str(resp)).strip()

    log.debug("tech_node raw (first 400): %s", raw[:400].replace("\n", " "))

    # Compute markdown view first so the fallback parser can use it.
    md_only = strip_first_json_fence(raw)
    md_only = re.sub(r"\n?###\s+2\.\s*JSON\s*:?\s*$", "", md_only, flags=re.I | re.M).rstrip()
    # Strip any remaining trailing code fences (bare JSON arrays that the LLM
    # appends after the markdown section).
    md_only = re.sub(r"\n*```(?:json|JSON)?\s*\[[\s\S]*?\]\s*```\s*$", "", md_only).rstrip()

    struct = extract_json_array(raw) or []
    struct = _normalize_tech_json(struct)

    # Bug C: if JSON extraction failed but the LLM did produce proposals, parse them
    # from the markdown ### heading blocks so the ledger never stores items:[].
    if not struct and raw.strip():
        _tech_log.warning(
            "tech_node: extract_json_array returned empty from %d-char response; "
            "running markdown fallback parser",
            len(raw),
        )
        struct = _normalize_tech_json(_parse_tech_from_markdown(md_only))

    # ── Apply unverified warning to non-RAG items ────────────────────────────
    warning = _UNVERIFIED_WARNING_ES if lang == "es" else _UNVERIFIED_WARNING_EN
    for item in struct:
        if not item.get("rag_backed"):
            item["rationale"] = f"{item['rationale']} {warning}".strip()

    if not md_only and struct:
        md_only = "\n".join(
            f"- **{it['name']}** ({it['tactic']}): {it['rationale'][:100]}"
            for it in struct
        )

    # Never expose server filesystem paths in references.
    src_lines = [
        _clip_text(f"- {title}{page_str}", 60)
        for title, page_str, _path in src_meta
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

    # ── Issue 2-bis: QA queue cycling ────────────────────────────────────────
    # After completing the tech proposals for one QA, check if more QAs are
    # queued. If so, transition the ledger back to style_table for the next QA
    # and emit a "now designing QA X" hint in the message.
    _qa_queue = list(state.get("selected_qa_queue") or [])
    _current_qa = state.get("quality_attribute") or qa or ""
    _remaining_qas: list[str] = []
    if _current_qa and _current_qa in _qa_queue:
        _idx = _qa_queue.index(_current_qa)
        _remaining_qas = _qa_queue[_idx + 1:]
    elif len(_qa_queue) > 1:
        _remaining_qas = _qa_queue[1:]

    if _remaining_qas and _user_id:
        _next_qa = _remaining_qas[0]
        state["selected_qa_queue"] = _remaining_qas
        try:
            _ledger_for_cycle = state.get("ledger") or {}
            _cycle_trans = {
                "from_phase":    Phase.TECH_PROPOSALS.value,
                "to_phase":      Phase.STYLE_TABLE.value,
                "iteration":     int(_ledger_for_cycle.get("current_iteration", 0)) + 1,
                "triggered_by":  "qa_queue_cycle",
                "user_message":  "",
                "skipped_phases": [],
                "timestamp":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            transition_phase(_user_id, _project_id, _cycle_trans)
            _refresh_ledger_state(state, _user_id, _project_id, lang)
        except Exception as _exc:
            _tech_log.warning("tech_node: QA queue phase cycle failed (nonfatal): %s", _exc)

        state["quality_attribute"] = _next_qa
        state["routing_phase"] = "style"
        if lang == "es":
            _hint = (
                f"\n\n---\n\n**Ciclo de diseño para _{_current_qa}_ completado.**\n"
                f"Ahora diseñamos el siguiente ASR: **{_next_qa}**. "
                f"Propón los estilos arquitectónicos para este atributo de calidad."
            )
        else:
            _hint = (
                f"\n\n---\n\n**Design cycle for _{_current_qa}_ complete.**\n"
                f"Now designing the next ASR: **{_next_qa}**. "
                f"Propose the architecture styles for this quality attribute."
            )
        md_only += _hint
    else:
        state["selected_qa_queue"] = []

    state["endMessage"] = md_only
    state["intent"] = "tech"
    state["nextNode"] = "unifier"
    state["hasVisitedTech"] = True

    # Persist completed_nodes and routing_phase across turns.
    _done = list(state.get("completed_nodes") or [])
    for _n in ("asr", "tech"):
        if _n not in _done:
            _done.append(_n)
    state["completed_nodes"] = _done
    if not _remaining_qas:
        state["routing_phase"] = "tech"

    prev_msgs = state.get("messages", [])
    return {**state, "messages": prev_msgs + msgs}
