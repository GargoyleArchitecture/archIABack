import re
import os
import json
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.graph.consts import TACTICS_JSON_EXAMPLE, MARKDOWN_FORMAT_DIRECTIVE
from src.graph.prompts.mode_prompts import apply_mode_prompt
from src.graph.qa_registry import normalize_qa
from datetime import datetime, timezone
from src.ledger import (
    append_decision,
    compute_active_view,
    get_all_active_asrs,
    load_ledger,
    render_dossier,
    render_dossier_compact,
    render_phase_prompt,
    transition_phase,
    LedgerValidationError,
    LedgerConcurrencyError,
)
from src.ledger.types import Phase, PhaseTransition

_tac_log = logging.getLogger("tactics_node")


def _active_view_with_primary_asr(state: GraphState) -> dict:
    # BUG-056: order matters here. The user types ASRs in priority order
    # (selected_asrs[0] = highest); classifier→asr_confirm preserves that as
    # ledger append order. But state["ledger_active"]["asr"] comes from
    # compute_active_view(), which collapses every active ASR to the
    # LAST-appended one (ledger/store.py:330-336 — ASRs intentionally do
    # NOT supersede each other per store.py:72-82). Reading .asr from the
    # raw view silently swaps in the wrong driver on multi-ASR selection.
    view = dict(state.get("ledger_active") or {})
    ledger = state.get("ledger") or {}
    if ledger.get("decisions"):
        asrs = get_all_active_asrs(ledger)
        if asrs:
            view["asr"] = asrs[0]
    return view


@lru_cache(maxsize=64)
def _fetch_tactics_rag(qa: str, resolved_index: str, k: int = 6, queries_override: tuple | None = None) -> tuple:
    """Returns (book_snippets: str, src_meta: tuple of (title, page_str, path)).
    Cached by (qa, resolved_index, k, queries_override). Cache hit skips all ChromaDB queries."""
    queries = list(queries_override) if queries_override else [
        f"{qa} architectural tactics",
        f"{qa} tactics performance scalability latency availability security modifiability",
        "Bass Clements Kazman performance and scalability tactics",
        "quality attribute tactics list",
    ]
    _retriever = get_indexed_retriever(
        quality_attribute=normalize_qa(resolved_index or qa),
        content_type="tacticas",
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

def _allowed_tactic_names_from_lines(lines: list) -> list:
    """Extrae el nombre canónico de líneas tipo 'Nombre — descripción'."""
    out: list = []
    for raw in lines or []:
        line = (raw or "").strip()
        if not line or line.startswith("#"):
            continue
        if " — " in line:
            out.append(line.split(" — ", 1)[0].strip())
        elif " – " in line:
            out.append(line.split(" – ", 1)[0].strip())
        else:
            out.append(line)
    return out


def _canonicalize_tactic_name(name: str, allowed: list) -> str:
    """Fuerza el nombre a uno del catálogo permitido (mejor esfuerzo)."""
    n = (name or "").strip()
    if not allowed:
        return n
    if not n:
        return allowed[0]
    nf = n.casefold()
    for a in allowed:
        if a.casefold() == nf:
            return a
    for a in allowed:
        ac = a.casefold()
        if ac in nf or nf in ac:
            return a
    return allowed[0]

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


# ---------------------------------------------------------------------------
# Pure helpers (Step 4 — P4)
# ---------------------------------------------------------------------------

def _build_dossier_design_binding(ledger_active: dict, lang: str = "es") -> str:
    """Build a HARD-BINDING prompt block sourced from the active ASR + active style.
    Returns "" when either is missing (first-turn sessions, pre-style sessions).
    Both required: tactics without a confirmed ASR and style are structurally incomplete.
    """
    active = ledger_active or {}
    asr    = active.get("asr")
    style  = active.get("style")
    if not asr or not style:
        return ""

    asr_id        = asr.get("id", "")
    qa            = asr.get("qa", "")
    asr_payload   = asr.get("payload") or {}
    rm            = asr_payload.get("response_measure", "")
    # Bug A fix: prefer the human-friendly ID (A1/A2/…) over the ULID when
    # building the prompt so the LLM cites "A2" in `traces_to_asr` instead of
    # emitting the ULID, which leaks into the tactics table column.
    human_id      = (asr_payload.get("candidate_id") or "").upper().strip()
    asr_ref       = human_id or asr_id
    style_id      = style.get("id", "")
    style_payload = style.get("payload") or {}
    style_chosen  = style_payload.get("chosen", "")
    style_trades  = style_payload.get("tradeoffs", "")[:200]

    if lang == "en":
        return (
            f'\n{"=" * 60}\n'
            f'ACTIVE DESIGN DECISIONS — BINDING CONSTRAINTS FOR TACTICS:\n'
            f'  ASR ID:            {asr_ref}\n'
            f'  Quality Attribute: {qa}\n'
            f'  Response Measure:  {rm}\n\n'
            f'  Active Style:      {style_chosen}  (id: {style_id})\n'
            f'  Style Tradeoffs:   {style_trades}\n\n'
            f'REQUIREMENTS:\n'
            f'1. Each tactic\'s "traces_to_asr" field MUST be the ASR id "{asr_ref}" '
            f'(NOT the ULID, NOT the response measure verbatim).\n'
            f'2. Tactics MUST realize style "{style_chosen}" — do NOT contradict its tradeoffs.\n'
            f'3. Tactics that conflict with "{style_chosen}" MUST be excluded with explanation.\n'
            f'{"=" * 60}\n'
        )
    return (
        f'\n{"=" * 60}\n'
        f'DECISIONES DE DISEÑO ACTIVAS — RESTRICCIONES VINCULANTES PARA TÁCTICAS:\n'
        f'  ID del ASR:          {asr_ref}\n'
        f'  Atributo de Calidad: {qa}\n'
        f'  Medida de Respuesta: {rm}\n\n'
        f'  Estilo Activo:       {style_chosen}  (id: {style_id})\n'
        f'  Compromisos:         {style_trades}\n\n'
        f'REQUISITOS:\n'
        f'1. El campo "traces_to_asr" de cada táctica DEBE ser el id del ASR "{asr_ref}" '
        f'(NO el ULID, NO la medida de respuesta verbatim).\n'
        f'2. Las tácticas DEBEN realizar el estilo "{style_chosen}" — no contradigan sus compromisos.\n'
        f'3. Las tácticas que conflictúen con "{style_chosen}" DEBEN excluirse con explicación.\n'
        f'{"=" * 60}\n'
    )


def _build_tactic_payload(items: list) -> dict:
    return {"items": items}


def _build_parent_refs(ledger_active: dict) -> list:
    active = ledger_active or {}
    refs   = []
    asr    = active.get("asr")
    style  = active.get("style")
    if asr:
        refs.append({"id": asr["id"], "kind": "asr",   "iteration": asr.get("iteration", 0)})
    if style:
        refs.append({"id": style["id"], "kind": "style", "iteration": style.get("iteration", 0)})
    return refs


_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _validate_tactic_traces(
    items: list,
    response_measure: str,
    human_asr_id: str = "",
) -> list:
    """Post-processing guard:
    - If LLM emitted an empty traces_to_asr, fill a sensible default so the
      ledger payload is structurally complete.
    - Bug A fix: if the LLM emitted the bare ULID (26-char Crockford base32),
      replace it with the human-friendly ASR id (e.g. "A2") so the rendered
      "ASR al que aplica" column is readable.
    Mutates and returns the list.
    """
    fallback = (
        human_asr_id
        or (f"Satisfies Response Measure: {response_measure}" if response_measure else "")
    )
    human = (human_asr_id or "").strip()
    for item in items:
        if not isinstance(item, dict):
            continue
        val = (item.get("traces_to_asr") or "").strip()
        if not val:
            item["traces_to_asr"] = fallback
            continue
        if human and _ULID_RE.match(val):
            # Bare ULID — swap for human id.
            item["traces_to_asr"] = human
    return items


def _build_multi_asr_tactics_constraint(all_asrs: list[dict], lang: str) -> str:
    """Prompt block: flag tactics that conflict with the highest-priority ASR."""
    if not all_asrs:
        return ""
    top_asr = all_asrs[0]
    top_qa = top_asr.get("qa", "")
    top_rm = (top_asr.get("payload") or {}).get("response_measure", "")
    if not top_qa:
        return ""
    if lang == "en":
        return (
            f'\n{"=" * 60}\n'
            f'HIGHEST-PRIORITY ASR CONSTRAINT:\n'
            f'  QA: {top_qa}\n'
            f'  Response Measure: {top_rm}\n\n'
            f'RULE: If a tactic conflicts with "{top_qa}", it MAY still appear '
            f'in the list, but you MUST add a "conflict_note" field (one sentence) '
            f'explaining the tradeoff. The architect decides — do not hide options.\n'
            f'{"=" * 60}\n'
        )
    return (
        f'\n{"=" * 60}\n'
        f'RESTRICCIÓN DEL ASR DE MAYOR PRIORIDAD:\n'
        f'  QA: {top_qa}\n'
        f'  Medida de Respuesta: {top_rm}\n\n'
        f'REGLA: Si una táctica conflictúa con "{top_qa}", PUEDE seguir en la lista, '
        f'pero DEBES agregar un campo "conflict_note" (una oración) explicando el '
        f'tradeoff. El arquitecto decide — no ocultes opciones.\n'
        f'{"=" * 60}\n'
    )


def _sanitize_md_cell(text: str, max_chars: int = 120) -> str:
    """Sanitize a string for safe use inside a Markdown table cell."""
    text = (text or "").replace("\n", " ").replace("\r", " ")
    text = text.replace("|", "\\|").replace("`", "'")
    text = text.strip()
    if len(text) > max_chars:
        truncated = text[:max_chars].rsplit(" ", 1)[0]
        text = (truncated or text[:max_chars]) + "…"
    return text


def _render_tactics_fallback_table(struct: list, lang: str) -> str:
    """Render the tactics struct as the spec table when the LLM markdown is unusable."""
    if lang == "es":
        header = "| ID | Táctica | ASR al que aplica | Efecto esperado | Riesgo si se omite |"
        sep    = "|----|---------|-------------------|-----------------|---------------------|"
        prompt_q = "Escribe el ID (T1, T2, T3) de la(s) táctica(s) a profundizar."
    else:
        header = "| ID | Tactic | ASR addressed | Expected effect | Risk if omitted |"
        sep    = "|----|--------|---------------|-----------------|-----------------|"
        prompt_q = "Type the ID (T1, T2, T3) of the tactic(s) you want to expand."
    rows = []
    for i, it in enumerate(struct[:3], 1):
        if not isinstance(it, dict):
            continue
        name      = _sanitize_md_cell(it.get("name", ""), 50)
        asr_ref   = _sanitize_md_cell(it.get("traces_to_asr", ""), 60)
        rationale = _sanitize_md_cell(it.get("rationale", ""), 80)
        risk      = _sanitize_md_cell(it.get("consequences", "") or it.get("tradeoff", ""), 80)
        rows.append(f"| T{i} | {name} | {asr_ref} | {rationale} | {risk} |")
    if not rows:
        return ""
    return "\n".join([header, sep, *rows, "", prompt_q])


def _render_conflict_flags(struct: list, all_asrs: list[dict], lang: str) -> str:
    """If any tactic has a conflict_note, build a visible warning block."""
    if not all_asrs:
        return ""
    top_qa = all_asrs[0].get("qa", "")
    notes = []
    for item in struct:
        if not isinstance(item, dict):
            continue
        cn = (item.get("conflict_note") or "").strip()
        if cn:
            notes.append(f"- **{item.get('name', '?')}**: {cn}")
    if not notes:
        return ""
    if lang == "es":
        header = f"\n\n---\n⚠️ **Conflictos con ASR prioritario ({top_qa}):**\n"
    else:
        header = f"\n\n---\n⚠️ **Conflicts with highest-priority ASR ({top_qa}):**\n"
    return header + "\n".join(notes)


# ---------------------------------------------------------------------------
# Ledger state refresh (Step 5 — P4)
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
        state["current_phase"]          = fresh.get("current_phase") or "intro"
        state["ledger_pending_advance"] = fresh.get("pending_advance") or {}
        _tac_log.debug("tactics_node: ledger state refreshed phase=%s", state["current_phase"])
    except Exception as exc:
        _tac_log.warning("tactics_node: state refresh failed (nonfatal): %s", exc)


# ---------------------------------------------------------------------------
# Main node implementation
# ---------------------------------------------------------------------------

def tactics_node_impl(
    state: GraphState,
    qa_override: str | None = None,
    preferred_tactics: list | None = None,
    preferred_group_label: str | None = None,
    restrict_to_preferred_tactics: bool = False,
    rag_queries_override: list | None = None,
) -> GraphState:
    """Implementación común del nodo de tácticas (ADD 3.0)."""
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
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()
    ctx_add = (state.get("add_context") or "").strip()
    ctx = (ctx_doc if (doc_only and ctx_doc) else ctx_add)[:2000]
    proj_ctx = (state.get("project_context_text") or "").strip()

    asr_text = (
        state.get("current_asr")
        or state.get("asr_text")
        or state.get("last_asr")
        or ""
    )
    # BUG-050: when state["current_asr"] is empty (e.g. context_loader didn't
    # repopulate after an in-process turn), read the active ASR from the ledger
    # so the user does NOT have to paste the ASR context manually.
    if not asr_text:
        _led_asr = (_active_view_with_primary_asr(state).get("asr") or {}).get("payload") or {}
        asr_text = (
            _led_asr.get("summary")
            or _led_asr.get("scenario")
            or _led_asr.get("response")
            or ""
        ).strip()
    if not asr_text:
        uq = state.get("userQuestion", "") or ""
        m = re.search(r"(?:^|\n)\s*ASR\s*:?\s*(.+)$", uq, flags=re.I | re.S)
        asr_text = (m.group(1).strip() if m else "")

    qa = resolve_qa_for_tactics(state, asr_text=asr_text, qa_override=qa_override)
    style_text = state.get("style") or state.get("selected_style") or state.get("last_style") or ""
    # BUG-050: mirror the ASR fallback for style — pull from ledger_active.style
    # before falling back to the raw user message.
    if not style_text:
        _led_style = ((state.get("ledger_active") or {}).get("style") or {}).get("payload") or {}
        style_text = (
            _led_style.get("chosen")
            or _led_style.get("name")
            or ""
        ).strip()

    src_meta: tuple = ()
    if doc_only and ctx_doc:
        book_snippets = f"[DOC] {ctx_doc[:2000]}"
    else:
        _rag_queries = tuple(rag_queries_override) if rag_queries_override else None
        book_snippets, src_meta = _fetch_tactics_rag(
            qa,
            qa,
            k=6,
            queries_override=_rag_queries,
        )
        rag_trace_record(
            query=" | ".join([
                f"{qa} architectural tactics",
                f"{qa} tactics performance scalability latency availability security modifiability",
                "Bass Clements Kazman performance and scalability tactics",
                "quality attribute tactics list",
            ])
        )

    preferred_block = ""
    allowed_names: list = []
    if preferred_tactics:
        group_label = (preferred_group_label or "Preferred tactics").strip()
        _tac_items: list[str] = []
        _has_groups = False
        for _t in preferred_tactics:
            _t_s = str(_t).strip()
            if not _t_s:
                continue
            if _t_s.startswith("#"):
                _tac_items.append(f"\n{_t_s[1:].strip()}:")
                _has_groups = True
            else:
                _tac_items.append(f"  - {_t_s}")
        items = "\n".join(_tac_items)
        allowed_names = _allowed_tactic_names_from_lines(preferred_tactics)
        if restrict_to_preferred_tactics and allowed_names:
            allowed_csv = ", ".join(f'"{n}"' for n in allowed_names)
            _multi_group_guidance = ""
            if _has_groups:
                _multi_group_guidance = (
                    "\nSELECTION GUIDANCE:\n"
                    "An ASR has one primary stimulus → response chain. Identify which ONE group best matches this ASR:\n"
                    "  * DETECT — The ASR's response is about knowing WHEN or WHETHER a failure is occurring.\n"
                    "  * RECOVER — The ASR's response is about RESTORING service or state after a failure.\n"
                    "  * PREVENT — The ASR's response is about ELIMINATING or REDUCING the probability of failure.\n"
                    "Select ALL THREE tactics from ONLY that group. Do not mix groups.\n"
                )
            preferred_block = (
                f"\n\nALLOWED TACTICS ONLY ({group_label}):\n"
                f"{items}\n\n"
                "HARD CONSTRAINTS:\n"
                "- You MUST select EXACTLY THREE tactics for the TOP-3.\n"
                "- EVERY tactic name in sections (1) and (2) MUST be one of the allowed canonical names "
                f"listed above (before the em dash), exactly from this set: [{allowed_csv}].\n"
                "- Do NOT introduce any other tactic names outside the allowed list above.\n"
                "- If documentation grounding conflicts, still obey the allowed list; you may note doc limitations in prose.\n"
                f"{_multi_group_guidance}"
            )
        else:
            preferred_block = (
                f"\n\nPRIORITY TACTIC GROUP ({group_label}):\n"
                f"{items}\n"
                "Prioritize these tactics in your TOP-3 when they fit the ASR and selected style."
            )

    restriction_clause = ""
    if restrict_to_preferred_tactics and allowed_names:
        restriction_clause = (
            "\nFor section (1) and the JSON in section (2): tactic names MUST come ONLY from the ALLOWED TACTICS list above.\n"
        )

    proj_ctx_block = ""
    if proj_ctx:
        proj_ctx_block = f"""
{"=" * 60}
PROJECT CONTEXT — MANDATORY CONSTRAINTS FOR TACTIC SELECTION:
{proj_ctx}

IMPORTANT: All proposed tactics MUST be compatible with the listed tech stack and business rules.
Mention specific technologies from the stack when describing how each tactic would be implemented.
{"=" * 60}
"""

    # ── Dossier design binding (P4) ─────────────────────────────────────────
    # BUG-056: pass the primary-ASR-corrected view so the dossier binds tactics
    # to selected_asrs[0], not to whichever ASR was last-appended to the ledger.
    _primary_view = _active_view_with_primary_asr(state)
    dossier_binding_block = _build_dossier_design_binding(_primary_view, lang)
    # Extract response_measure for traces validation fallback
    _active_asr = _primary_view.get("asr")
    _response_measure = ((_active_asr or {}).get("payload") or {}).get("response_measure", "")

    # ── Multi-ASR consistency constraint (P7) ──────────────────────────────
    _ledger = state.get("ledger") or {}
    _all_asrs = get_all_active_asrs(_ledger) if _ledger.get("decisions") else []
    multi_asr_constraint = _build_multi_asr_tactics_constraint(_all_asrs, lang)

    # BUG-048: produce a tactics CANDIDATE TABLE (T1/T2/T3), not multi-section
    # prose with code blocks. Internal JSON payload is still required for the
    # ledger but goes inside a fenced block that we strip BEFORE the user sees
    # the message (BUG-049).
    _active_asr_payload = _active_asr or {}
    _asr_id_for_tactics = (_active_asr_payload.get("payload") or {}).get("candidate_id") or "A1"

    if lang == "es":
        _col_header = "| ID | Táctica | ASR al que aplica | Efecto esperado | Riesgo si se omite |"
        _table_sep  = "|----|---------|-------------------|-----------------|---------------------|"
        _row_hint   = f"| T1 | <nombre> | {_asr_id_for_tactics} | <una oración> | <una oración> |"
        _select_q   = "Escribe el ID (T1, T2, T3) de la(s) táctica(s) a profundizar."
        _final_rmd  = "RECORDATORIO FINAL: responde completamente en español."
    else:
        _col_header = "| ID | Tactic | ASR addressed | Expected effect | Risk if omitted |"
        _table_sep  = "|----|--------|---------------|-----------------|-----------------|"
        _row_hint   = f"| T1 | <name> | {_asr_id_for_tactics} | <one sentence> | <one sentence> |"
        _select_q   = "Type the ID (T1, T2, T3) of the tactic(s) you want to expand."
        _final_rmd  = "FINAL REMINDER: answer entirely in English."

    prompt = f"""{directive}
You are an expert software architect applying Attribute-Driven Design 3.0 (ADD 3.0).

We ALREADY HAVE an ASR (Quality Attribute Scenario) and a selected architecture style.
Your job now is to propose the TOP-3 tactics that realise the ASR under that style.
{proj_ctx_block}
{dossier_binding_block}
{multi_asr_constraint}
Additional session context (if any):
{ctx or "None"}

ASR (driver to satisfy):
{asr_text or "(none provided)"}

Primary quality attribute (guessed):
{qa}
Selected architecture style (if any):
{style_text or "(none)"}
{preferred_block}


GROUNDING (use ONLY this context; if DOC-ONLY, this is the exclusive source):
{book_snippets or "(none)"}

If DOC-ONLY is ON, do not rely on knowledge beyond the PROJECT DOCUMENT even if you "know" typical tactics. If the document does not support a tactic, state "not supported by the document".
{restriction_clause}
OUTPUT FORMAT (MANDATORY) — output the Markdown table below FIRST, then the
selection prompt, then a ```json fence with the internal payload.

{_col_header}
{_table_sep}
{_row_hint}

Hard rules for the table:
- EXACTLY 3 rows (T1, T2, T3). No more, no less.
- Each cell is ONE short sentence. No bullet lists, no sub-headings, no code fences inside cells.
- "ASR addressed" MUST be the ASR ID(s) (e.g. {_asr_id_for_tactics}), never free-text.
- NEVER include YAML, Go, Python, JSON, k6, checklists or any prose outside the table itself.
- Tactic names MUST be canonical (e.g. "Circuit Breaker", "Ping/Echo", "Load Shedding", "Bulkhead").

After the table, on a new line, write EXACTLY this selection prompt:
{_select_q}

ABSOLUTE STOP RULE: After the line above you MUST output ONLY the ```json fence
and NOTHING ELSE. Do NOT add any "Solución concreta", checklists, implementation
details, YAML, code snippets, deployment configs, or trade-off paragraphs in this
response. Implementation details belong to a LATER step (post-confirmation), not here.

THEN — and only then — append one ```json fenced block containing a JSON array of
EXACTLY 3 objects (T1, T2, T3) for internal ledger use:
- Use dot as decimal separator (0.82, never 0,82).
- success_probability is a float in [0, 1].
- Each object MUST include "name", "rationale", "traces_to_asr" (one sentence citing the ASR's Response Measure), "consequences", and "success_probability".
- If the ALLOWED/PRIORITY list is restrictive, each object's "name" MUST match one allowed canonical name.
- The JSON fence is internal; the user never sees it. Do NOT add any extra prose around it.

Example JSON shape (values are illustrative — adjust to your tactics):
{TACTICS_JSON_EXAMPLE}

{MARKDOWN_FORMAT_DIRECTIVE}

{_final_rmd}
"""
    resp = llm.invoke(apply_mode_prompt(state, prompt))
    raw = getattr(resp, "content", str(resp)).strip()

    log.debug("tactics raw (first 400): %s", raw[:400].replace("\n", " "))
    log.debug("has ```json fence? %s", bool(re.search(r"```json", raw, re.I)))

    struct = extract_json_array(raw) or []
    if not (isinstance(struct, list) and struct):
        struct = _json_only_repair_pass(llm, asr_text=asr_text, qa=qa, style_text=style_text, md_preview=raw) or []
    if not (isinstance(struct, list) and struct):
        struct = build_json_from_markdown(raw, top_n=3)
    struct = normalize_tactics_json(struct, top_n=3)

    if restrict_to_preferred_tactics and allowed_names and isinstance(struct, list):
        taken: set = set()

        def _pick_unused_fallback() -> str:
            for cand in allowed_names:
                if cand.casefold() not in taken:
                    return cand
            return allowed_names[0]

        for it in struct:
            if not isinstance(it, dict):
                continue
            canon = _canonicalize_tactic_name(str(it.get("name", "")), allowed_names)
            if canon.casefold() in taken:
                canon = _pick_unused_fallback()
            it["name"] = canon
            taken.add(canon.casefold())

        while isinstance(struct, list) and len(struct) < 3:
            cand = _pick_unused_fallback()
            struct.append(
                {
                    "name": cand,
                    "rationale": "",
                    "categories": ["fault-detection"],
                    "success_probability": 0.5,
                    "rank": len(struct) + 1,
                }
            )
            taken.add(cand.casefold())

        struct = normalize_tactics_json(struct, top_n=3)

    # BUG-049: never expose the raw JSON payload to the user. The JSON is
    # internal ledger payload; debugging relies on logs, not chat output.
    md_only = strip_first_json_fence(raw)
    md_only = re.sub(r"\n?\(?2\)?\s*JSON\s*:?\s*$", "", md_only, flags=re.I | re.M).rstrip()
    # Bug A fix: if the LLM emitted the ASR's ULID in the "ASR al que aplica"
    # column instead of the friendly id (A1/A2/…), swap it back. The ULID is
    # internal; users should see "A2", not "01KRQE2BCJ34FPYVT302BYZY3N".
    # We swap in three places:
    #   (1) struct items' `traces_to_asr` field — for the fallback renderer and
    #       any downstream consumers that read struct directly.
    #   (2) the markdown the LLM produced — for the user-visible chat bubble.
    #   (3) the ledger write later in the function (handled via the
    #       `human_asr_id` arg to _validate_tactic_traces).
    # BUG-056: swap ULID→human-id using the primary ASR (selected_asrs[0]),
    # not whatever last-appended ASR ledger_active.asr happens to point at.
    _active_asr_for_swap = _active_view_with_primary_asr(state).get("asr") or {}
    _ulid_for_swap = (_active_asr_for_swap.get("id") or "").strip()
    _human_for_swap = ((_active_asr_for_swap.get("payload") or {}).get("candidate_id") or "").upper().strip()
    if _ulid_for_swap and _human_for_swap and _ULID_RE.match(_ulid_for_swap):
        md_only = md_only.replace(_ulid_for_swap, _human_for_swap)
        if isinstance(struct, list):
            for _it in struct:
                if isinstance(_it, dict):
                    _val = (_it.get("traces_to_asr") or "").strip()
                    if _val == _ulid_for_swap or _ULID_RE.match(_val):
                        _it["traces_to_asr"] = _human_for_swap
    # BUG-S3-002: truncate anything the LLM appended after the selection prompt.
    _select_marker = _select_q.strip()
    if _select_marker and _select_marker in md_only:
        _idx = md_only.index(_select_marker)
        md_only = md_only[: _idx + len(_select_marker)].rstrip()
    if (not md_only) and isinstance(struct, list) and struct:
        # BUG-048 fallback: render as a one-row table per item with the same
        # column schema as the spec, not a bullet list.
        md_only = _render_tactics_fallback_table(struct, lang)

    # ── Post-LLM conflict flags (P7) ──────────────────────────────────────
    _conflict_block = _render_conflict_flags(struct, _all_asrs, lang)
    if _conflict_block:
        md_only += _conflict_block

    # BUG-016: never expose server filesystem paths in references.
    src_lines = [
        _clip_text(f"- {title}{page_str}", 60)
        for title, page_str, _path in src_meta
    ]
    src_lines = list(dict.fromkeys(src_lines))[:6]
    src_block = "SOURCES:\n" + ("\n".join(src_lines) if src_lines else "- (no local sources)")

    _push_turn(state, role="system", name="tactics_system", content=prompt)
    _push_turn(state, role="assistant", name="tactics_advisor", content=md_only)
    _push_turn(state, role="assistant", name="tactics_sources", content=src_block)

    msgs = [AIMessage(content=md_only, name="tactics_advisor"), AIMessage(content=src_block, name="tactics_sources")]

    # ── Scalar writes (unconditional) ────────────────────────────────────────
    state["tactics_md"] = md_only
    _struct_list = struct if isinstance(struct, list) else []
    state["tactics_struct"] = _struct_list
    # BUG-010/012: populate tactics_candidates so the classifier's tactics_confirm
    # block can resolve T1/T2/T3 IDs without needing tactics_struct separately.
    state["tactics_candidates"] = _struct_list
    state["tactics_list"] = [(it.get("name") or "").strip() for it in (_struct_list or []) if isinstance(it, dict) and it.get("name")]
    state["quality_attribute"] = qa
    if asr_text:
        state["current_asr"] = asr_text

    # ── Ledger write-back (P4) ───────────────────────────────────────────────
    _user_id    = (state.get("user_id_for_prefs") or "").strip()
    _project_id = (state.get("project_id") or "").strip() or None

    if _user_id:
        try:
            _items   = _validate_tactic_traces(
                list(state.get("tactics_struct") or []),
                _response_measure,
                human_asr_id=_asr_id_for_tactics,
            )
            # BUG-056: parent ref must point at the PRIMARY active ASR.
            _parents = _build_parent_refs(_active_view_with_primary_asr(state))
            _qa      = state.get("quality_attribute") or qa
            _new_decision: dict = {
                "id":               "",
                "kind":             "tactic",
                "phase":            Phase.TACTICS_TABLE.value,
                "iteration":        0,
                "qa":               _qa,
                "parents":          _parents,
                "payload":          _build_tactic_payload(_items),
                "rationale":        "",
                "sources":          [],
                "status":           "active",
                "parent_status":    "ok",
                "superseded_by":    None,
                "rejection_reason": None,
                "created_at":       "",
                "created_by_node":  "tactics_node",
            }
            _saved = append_decision(_user_id, _project_id, _new_decision)
            _tac_log.info(
                "tactics_node: ledger ok id=%s qa=%s items=%d project=%s",
                _saved["id"], _qa, len(_items), _project_id,
            )
            _refresh_ledger_state(state, _user_id, _project_id, lang)

        except LedgerValidationError as _exc:
            _tac_log.warning("tactics_node: ledger validation error (nonfatal): %s", _exc)
        except LedgerConcurrencyError as _exc:
            _tac_log.warning("tactics_node: ledger concurrency error (nonfatal): %s", _exc)
        except Exception as _exc:
            _tac_log.warning("tactics_node: unexpected ledger error (nonfatal): %s", _exc)

    state["endMessage"] = md_only
    state["intent"] = "tactics"
    state["nextNode"] = "unifier"

    # BUG-013: persist completed_nodes and routing_phase across turns.
    _done = list(state.get("completed_nodes") or [])
    for _n in ("asr", "style", "tactics"):
        if _n not in _done:
            _done.append(_n)
    state["completed_nodes"] = _done
    state["routing_phase"] = "tactics"

    prev_msgs = state.get("messages", [])
    return {**state, "messages": prev_msgs + msgs}
