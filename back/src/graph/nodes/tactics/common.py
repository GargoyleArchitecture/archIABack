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
from src.graph.qa_registry import normalize_qa
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

_tac_log = logging.getLogger("tactics_node")


@lru_cache(maxsize=64)
def _fetch_tactics_rag(qa: str, resolved_index: str, k: int = 6) -> tuple:
    """Returns (book_snippets: str, src_meta: tuple of (title, page_str, path)).
    Cached by (qa, resolved_index, k). Cache hit skips all ChromaDB queries."""
    queries = [
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
        if not line:
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
    style_id      = style.get("id", "")
    style_payload = style.get("payload") or {}
    style_chosen  = style_payload.get("chosen", "")
    style_trades  = style_payload.get("tradeoffs", "")[:200]

    if lang == "en":
        return (
            f'\n{"=" * 60}\n'
            f'ACTIVE DESIGN DECISIONS — BINDING CONSTRAINTS FOR TACTICS:\n'
            f'  ASR ID:            {asr_id}\n'
            f'  Quality Attribute: {qa}\n'
            f'  Response Measure:  {rm}\n\n'
            f'  Active Style:      {style_chosen}  (id: {style_id})\n'
            f'  Style Tradeoffs:   {style_trades}\n\n'
            f'REQUIREMENTS:\n'
            f'1. Each tactic\'s "traces_to_asr" field MUST cite: "{rm}"\n'
            f'2. Tactics MUST realize style "{style_chosen}" — do NOT contradict its tradeoffs.\n'
            f'3. Tactics that conflict with "{style_chosen}" MUST be excluded with explanation.\n'
            f'{"=" * 60}\n'
        )
    return (
        f'\n{"=" * 60}\n'
        f'DECISIONES DE DISEÑO ACTIVAS — RESTRICCIONES VINCULANTES PARA TÁCTICAS:\n'
        f'  ID del ASR:          {asr_id}\n'
        f'  Atributo de Calidad: {qa}\n'
        f'  Medida de Respuesta: {rm}\n\n'
        f'  Estilo Activo:       {style_chosen}  (id: {style_id})\n'
        f'  Compromisos:         {style_trades}\n\n'
        f'REQUISITOS:\n'
        f'1. El campo "traces_to_asr" de cada táctica DEBE citar: "{rm}"\n'
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


def _validate_tactic_traces(items: list, response_measure: str) -> list:
    """Post-processing guard: if LLM emitted an empty traces_to_asr, fill a
    sensible default so the ledger payload is structurally complete.
    Mutates and returns the list.
    """
    default = f"Satisfies Response Measure: {response_measure}" if response_measure else ""
    for item in items:
        if isinstance(item, dict) and not (item.get("traces_to_asr") or "").strip():
            item["traces_to_asr"] = default
    return items


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
        state["current_phase"]          = fresh.get("current_phase", "INTAKE")
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
) -> GraphState:
    """Implementación común del nodo de tácticas (ADD 3.0)."""
    lang = state.get("language", "es")
    directive = "Answer in English." if lang == "en" else "Responde en español."
    style_hint = (state.get("user_style_hint") or "").strip()
    if style_hint:
        directive = f"{directive} {style_hint}"
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
    if not asr_text:
        uq = state.get("userQuestion", "") or ""
        m = re.search(r"(?:^|\n)\s*ASR\s*:?\s*(.+)$", uq, flags=re.I | re.S)
        asr_text = (m.group(1).strip() if m else "")

    qa = resolve_qa_for_tactics(state, asr_text=asr_text, qa_override=qa_override)
    style_text = state.get("style") or state.get("selected_style") or state.get("last_style") or ""

    src_meta: tuple = ()
    if doc_only and ctx_doc:
        book_snippets = f"[DOC] {ctx_doc[:2000]}"
    else:
        book_snippets, src_meta = _fetch_tactics_rag(
            qa,
            qa,
            k=6,
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
        items = "\n".join(f"- {t}" for t in preferred_tactics if str(t).strip())
        allowed_names = _allowed_tactic_names_from_lines(preferred_tactics)
        if restrict_to_preferred_tactics and allowed_names:
            allowed_csv = ", ".join(f'"{n}"' for n in allowed_names)
            preferred_block = (
                f"\n\nALLOWED TACTICS ONLY ({group_label}):\n"
                f"{items}\n\n"
                "HARD CONSTRAINTS:\n"
                "- You MUST select EXACTLY THREE tactics for the TOP-3.\n"
                "- EVERY tactic name in sections (1) and (2) MUST be one of the allowed canonical names "
                f"listed above (before the em dash), exactly from this set: [{allowed_csv}].\n"
                "- Do NOT introduce any other tactic names (no recovery/repair/redundancy tactics unless they appear in the allowed list).\n"
                "- If documentation grounding conflicts, still obey the allowed list; you may note doc limitations in prose.\n"
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
    dossier_binding_block = _build_dossier_design_binding(
        state.get("ledger_active") or {}, lang
    )
    # Extract response_measure for traces validation fallback
    _active_asr = (state.get("ledger_active") or {}).get("asr")
    _response_measure = ((_active_asr or {}).get("payload") or {}).get("response_measure", "")

    prompt = f"""{directive}
You are an expert software architect applying Attribute-Driven Design 3.0 (ADD 3.0).

We ALREADY HAVE an ASR / Quality Attribute Scenario. That ASR is an ADD 3.0 architectural driver.
Your job now is to continue the ADD 3.0 process by selecting architectural tactics.
{proj_ctx_block}
{dossier_binding_block}
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
You MUST output THREE sections, in EXACT order.
Use Markdown formatting for sections (0) and (1). Section (2) is JSON only.
{MARKDOWN_FORMAT_DIRECTIVE}

## ASR & Style Context
3-5 concise lines. Explicitly link back to the ASR's **Source**, **Stimulus**, **Artifact**, **Environment** and **Response Measure**. Also its architectonic style.

## Tactics (TOP-3)
Select EXACTLY THREE architectural tactics that maximally satisfy this ASR GIVEN the selected style.
For EACH tactic use a ### heading with the tactic name and include: **Rationale**, **Consequences / Trade-offs**, **When to use**, **Why it ranks in TOP-3**, **Success probability**.

(2) JSON:
Return ONE code fence starting with ```json and ending with ``` that contains ONLY a JSON array with EXACTLY 3 objects.
- Use dot as decimal separator (e.g., 0.82), never commas.
- Do not use percent signs, just 0..1 floats for success_probability.
- Each object MUST include a "traces_to_asr" field: one sentence citing the ASR's Response Measure that this tactic helps satisfy.
- If the ALLOWED/PRIORITY list is RESTRICTIVE (i.e., tactics must be chosen only from it), then each JSON object's "name" MUST exactly match one allowed canonical tactic name from that list. Otherwise, you SHOULD PREFER names from the PRIORITY list but MAY use other reasonable canonical tactics when appropriate.
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

    md_only = strip_first_json_fence(raw)
    if os.getenv("SHOW_TACTICS_JSON", "0") == "1":
        md_only = f"{md_only}\n\n```json\n{json.dumps(struct, ensure_ascii=False, indent=2)}\n```"
    else:
        md_only = re.sub(r"\n?\(?2\)?\s*JSON\s*:?\s*$", "", md_only, flags=re.I | re.M).rstrip()
    if (not md_only) and isinstance(struct, list) and struct:
        md_only = "\n".join(f"- {it.get('name','')}: {it.get('rationale','')}" for it in struct if isinstance(it, dict))

    src_lines = [
        _clip_text(f"- {title}{page_str} — {path}", 60)
        for title, page_str, path in src_meta
    ]
    src_lines = list(dict.fromkeys(src_lines))[:6]
    src_block = "SOURCES:\n" + ("\n".join(src_lines) if src_lines else "- (no local sources)")

    _push_turn(state, role="system", name="tactics_system", content=prompt)
    _push_turn(state, role="assistant", name="tactics_advisor", content=md_only)
    _push_turn(state, role="assistant", name="tactics_sources", content=src_block)

    msgs = [AIMessage(content=md_only, name="tactics_advisor"), AIMessage(content=src_block, name="tactics_sources")]

    # ── Scalar writes (unconditional) ────────────────────────────────────────
    state["tactics_md"] = md_only
    state["tactics_struct"] = struct if isinstance(struct, list) else []
    state["tactics_list"] = [(it.get("name") or "").strip() for it in (struct or []) if isinstance(it, dict) and it.get("name")]
    state["arch_stage"] = "TACTICS"
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
            )
            _parents = _build_parent_refs(state.get("ledger_active") or {})
            _qa      = state.get("quality_attribute") or qa
            _new_decision: dict = {
                "id":               "",
                "kind":             "tactic",
                "phase":            Phase.TACTICS.value,
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
    prev_msgs = state.get("messages", [])
    return {**state, "messages": prev_msgs + msgs}
