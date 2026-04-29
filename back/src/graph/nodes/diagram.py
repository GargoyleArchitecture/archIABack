import base64
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.consts import (
    DOT_SYSTEM,
    DOT_SYSTEM_EXPAND,
    DOT_SYSTEM_OVERVIEW,
    EXPAND_TARGET_DETAILED,
    EXPAND_TARGET_MEDIUM,
)
from src.graph.resources import llm, log
from src.graph.state import GraphState
from src.services.diagram_ir import (
    DiagramLevel,
    build_diagram_model,
    normalize_text,
    parse_diagram_level,
    parse_dot_to_model,
    to_detail_level,
)
from src.services.diagram_render import render_dot, render_dot_drawio, render_svg_b64
from src.ledger import (
    append_decision, compute_active_view, load_ledger,
    render_dossier, render_dossier_compact, render_phase_prompt,
    LedgerValidationError, LedgerConcurrencyError,
)
from src.ledger.types import Phase

try:
    from graphviz import Source
except Exception:
    Source = None


def _sanitize_dot(raw: Any) -> str:
    """Extract a valid DOT graph from LLM output while preserving UTF-8 text."""
    txt = normalize_text(raw).strip()
    if not txt:
        return ""

    m = re.search(r"```dot\s*(.*?)```", txt, flags=re.I | re.S)
    if not m:
        m = re.search(r"```(?:\w+)?\s*(.*?)```", txt, flags=re.I | re.S)
    if m:
        txt = normalize_text(m.group(1)).strip()

    m = re.search(r"(?:strict\s+)?digraph\s+[A-Za-z_][A-Za-z0-9_]*\s*\{[\s\S]*\}", txt, flags=re.I)
    if m:
        txt = m.group(0).strip()
    else:
        m = re.search(r"\{[\s\S]*\}", txt, flags=re.S)
        if m:
            body = m.group(0).strip().strip("{}").strip()
            txt = "digraph G {\n" + body + "\n}"
        else:
            lines = [normalize_text(ln).rstrip() for ln in txt.splitlines() if ln.strip()]
            body = "\n".join(lines)
            txt = "digraph G {\n" + body + "\n}"

    return normalize_text(txt).strip()


def _llm_nl_to_dot(natural_prompt: str, *, level: DiagramLevel, expand: bool = False, lang: str = "es") -> str:
    """Generate DOT code via LLM. Uses expansion prompt when building on a prior level."""
    if expand:
        target_desc = EXPAND_TARGET_MEDIUM if level == DiagramLevel.MEDIUM else EXPAND_TARGET_DETAILED
        system_prompt = DOT_SYSTEM_EXPAND.format(target_description=target_desc)
    else:
        system_prompt = DOT_SYSTEM_OVERVIEW if level == DiagramLevel.OVERVIEW else DOT_SYSTEM

    lang_note = (
        "Node and edge labels in the DOT graph MUST be written in Spanish (español)."
        if lang == "es"
        else "Node and edge labels in the DOT graph MUST be written in English."
    )
    system_prompt = f"{system_prompt}\n\n{lang_note}"
    msgs = [SystemMessage(content=normalize_text(system_prompt)), HumanMessage(content=normalize_text(natural_prompt))]
    resp = llm.invoke(msgs)
    raw = getattr(resp, "content", str(resp)) or ""
    return _sanitize_dot(raw)


def _render_dot_svg_b64(dot_code: str) -> str:
    """Render DOT to base64 SVG. Tries renderer, falls back to python-graphviz."""
    engine = os.getenv("GRAPHVIZ_ENGINE", "dot").strip() or "dot"
    try:
        return render_svg_b64(dot_code, engine=engine)
    except Exception:
        if Source is None:
            raise RuntimeError("python-graphviz is not installed")
        src = Source(normalize_text(dot_code), format="svg", engine=engine)
        svg_bytes = normalize_text(src.pipe(format="svg")).encode("utf-8")
        return base64.b64encode(svg_bytes).decode("ascii")


def _infer_level_from_user_request(user_q: str) -> DiagramLevel:
    """Infer diagram level from natural language request."""
    low_q = normalize_text(user_q).lower()

    explicit = re.search(r"\b(?:level|nivel)\s*([1-3])\b", low_q)
    if explicit:
        return DiagramLevel(int(explicit.group(1)))

    detailed_keywords = [
        "detailed",
        "detallado",
        "full",
        "completo",
        "complete",
        "all components",
        "todos los componentes",
    ]
    if any(kw in low_q for kw in detailed_keywords):
        return DiagramLevel.DETAILED

    medium_keywords = [
        "more detail",
        "más detalle",
        "mas detalle",
        "expand",
        "expandir",
        "ampliar",
    ]
    if any(kw in low_q for kw in medium_keywords):
        return DiagramLevel.MEDIUM

    return DiagramLevel.OVERVIEW


def _resolve_diagram_level(state: GraphState, user_q: str) -> DiagramLevel:
    """Resolve level from explicit state override or user prompt."""
    level_override = state.get("diagram_level")
    if level_override is None:
        level_override = state.get("diagram_detail_level")

    if level_override is not None:
        try:
            return parse_diagram_level(level_override)
        except ValueError:
            log.warning("diagram_orchestrator_node: invalid diagram level override %r", level_override)

    return _infer_level_from_user_request(user_q)


def _build_diagram_parent_refs(state: GraphState) -> list[dict]:
    """Extract style + tactic refs from ledger_active for diagram parents."""
    active = state.get("ledger_active") or {}
    refs = []
    style = active.get("style")
    if style and style.get("id"):
        refs.append({"id": style["id"], "kind": "style", "iteration": style.get("iteration", 0)})
    tactic = active.get("tactic")
    if tactic and tactic.get("id"):
        refs.append({"id": tactic["id"], "kind": "tactic", "iteration": tactic.get("iteration", 0)})
    return refs


def _build_diagram_payload(diagram_obj: dict) -> dict:
    """Map diagram_orchestrator output to ledger payload contract."""
    return {
        "level": diagram_obj.get("level"),
        "dot": diagram_obj.get("dot", ""),
        "dot_drawio": diagram_obj.get("dot_drawio", ""),
        "svg_b64": diagram_obj.get("svg_b64", ""),
        "focus": diagram_obj.get("detail_level", ""),
        "mapping": diagram_obj.get("overview_mapping"),
    }


def _refresh_ledger_state(state: GraphState, user_id: str, project_id: str | None, lang: str) -> None:
    """Reload ledger into state after a successful append_decision."""
    ledger = load_ledger(user_id, project_id)
    active = compute_active_view(ledger)
    state["ledger"] = ledger
    state["ledger_active"] = active
    state["design_dossier_md"] = render_dossier(ledger, lang=lang)
    state["current_phase"] = ledger.get("current_phase", "INTAKE")
    state["ledger_dossier_compact"] = render_dossier_compact(ledger, lang=lang)
    state["ledger_phase_prompt"] = render_phase_prompt(ledger, lang=lang)
    state["ledger_pending_advance"] = ledger.get("pending_advance") or {}


def diagram_orchestrator_node(state: GraphState) -> GraphState:
    _lang = (state.get("language") or "es")
    user_q = normalize_text(state.get("localQuestion") or state.get("userQuestion") or "").strip()
    requested_level = _resolve_diagram_level(state, user_q)

    # --- Expansion mode: reuse previous level's diagram as base for consistency ---
    diagram_history = dict(state.get("diagram_history") or {})
    expand_mode = False
    expand_from_dot = ""
    prev_level_num = int(requested_level) - 1
    if prev_level_num >= 1:
        expand_from_dot = str(
            diagram_history.get(prev_level_num)
            or diagram_history.get(str(prev_level_num))
            or ""
        )
        if expand_from_dot.strip():
            expand_mode = True

    asr_text = normalize_text(state.get("current_asr") or state.get("last_asr") or "").strip()
    style_text = normalize_text(
        state.get("style")
        or state.get("selected_style")
        or state.get("last_style")
        or ""
    ).strip()

    tactics_names: list[str] = []
    tactics_struct = state.get("tactics_struct") or []
    if isinstance(tactics_struct, list):
        for it in tactics_struct:
            if isinstance(it, dict) and it.get("name"):
                tactics_names.append(normalize_text(it["name"]))

    if not tactics_names:
        tactics_md = normalize_text(state.get("tactics_md") or "").strip()
        if tactics_md:
            for line in tactics_md.splitlines():
                line = re.sub(r"^\s*[-*]\s*", "", normalize_text(line)).strip()
                if line:
                    tactics_names.append(line)

    tactics_names = tactics_names[:8]
    tactics_block = "\n".join(f"- {t}" for t in tactics_names) if tactics_names else "- (none selected yet)"

    add_context = normalize_text(state.get("add_context") or "").strip()
    doc_context = normalize_text(state.get("doc_context") or "").strip()
    memory_text = normalize_text(state.get("memory_text") or "").strip()

    sections: list[str] = []
    if add_context:
        sections.append(f"Business / project context:\n{add_context}")
    if doc_context:
        sections.append(f"Project documents context (RAG):\n{doc_context}")
    if memory_text:
        sections.append(f"Conversation memory (ASR/style/tactics decisions):\n{memory_text}")

    sections.append(
        "Quality Attribute Scenario (ASR):\n"
        f"{asr_text or '(not explicitly defined; infer from context and request)'}"
    )
    sections.append(
        "Chosen architecture style:\n"
        f"{style_text or '(not explicitly chosen; infer a reasonable style for the ASR)'}"
    )
    sections.append("Selected tactics:\n" + tactics_block)
    sections.append(
        "User diagram request:\n"
        + (user_q or "Generate a deployment/component diagram aligned with the ASR and tactics.")
    )

    # If expanding from a previous level, prepend its DOT as context
    if expand_mode and expand_from_dot:
        expansion_header = (
            f"PREVIOUS DIAGRAM (Level {prev_level_num}) — your base to expand:\n\n"
            f"{expand_from_dot}\n\n"
            f"Expand ALL the above components to Level {int(requested_level)} detail. "
            f"Keep every original component and add finer-grained sub-components."
        )
        sections.insert(0, expansion_header)

    full_prompt = "\n\n---\n\n".join(sections)

    dot_code = ""
    try:
        dot_code = _llm_nl_to_dot(full_prompt, level=requested_level, expand=expand_mode, lang=_lang)
    except Exception as exc:
        log.warning("diagram_orchestrator_node: DOT generation failed: %s", exc)

    diagram_obj: dict[str, Any] = {}
    if dot_code:
        try:
            detailed_model = parse_dot_to_model(dot_code)
            ir_model, level_mapping = build_diagram_model(
                detailed_model,
                requested_level,
                overview_max_nodes=15,
                medium_max_nodes=30,
            )

            final_dot = render_dot(ir_model)
            svg_b64 = _render_dot_svg_b64(final_dot)
            dot_drawio_code = render_dot_drawio(ir_model)
            detail_level = to_detail_level(requested_level).value

            diagram_obj = {
                "ok": True,
                "format": "svg",
                "engine": os.getenv("GRAPHVIZ_ENGINE", "dot"),
                "level": int(requested_level),
                "detail_level": detail_level,
                "svg_b64": svg_b64,
                "dot": final_dot,
                "dot_raw": dot_code,
                "dot_drawio": dot_drawio_code,
                "node_count": len(ir_model.nodes),
                "edge_count": len(ir_model.edges),
            }
            if requested_level != DiagramLevel.DETAILED:
                diagram_obj["overview_mapping"] = level_mapping

            # Update diagram history for cross-level expansion
            diagram_history[int(requested_level)] = final_dot
            # Invalidate higher levels — they were based on a previous version
            for lvl in range(int(requested_level) + 1, 4):
                diagram_history.pop(lvl, None)

        except Exception as exc:
            log.warning("diagram_orchestrator_node: Graphviz render failed: %s", exc)
            diagram_obj = {
                "ok": False,
                "format": "svg",
                "engine": os.getenv("GRAPHVIZ_ENGINE", "dot"),
                "level": int(requested_level),
                "detail_level": to_detail_level(requested_level).value,
                "dot": dot_code,
                "error": str(exc),
            }

    state["diagram"] = diagram_obj
    state["diagram_history"] = diagram_history
    state["hasVisitedDiagram"] = True
    state["intent"] = "diagram"

    # --- Ledger write-back (nonfatal) ---
    _user_id = state.get("user_id_for_prefs") or ""
    _project_id = state.get("project_id")
    # _lang already set at the top of this function
    if _user_id and diagram_obj.get("ok"):
        _qa = (
            (state.get("ledger_active") or {}).get("asr", {}) or {}
        ).get("qa", state.get("quality_attribute", ""))
        _parent_refs = _build_diagram_parent_refs(state)
        try:
            append_decision(_user_id, _project_id, {
                "id": "",
                "kind": "diagram",
                "phase": state.get("current_phase") or Phase.DIAGRAM,
                "iteration": 0,
                "qa": _qa,
                "parents": _parent_refs,
                "payload": _build_diagram_payload(diagram_obj),
                "rationale": f"Level {diagram_obj.get('level')} diagram generated for {_qa or 'current design'}",
                "sources": [],
                "status": "active",
                "parent_status": "ok",
                "superseded_by": None,
                "rejection_reason": None,
                "created_at": "",
                "created_by_node": "diagram_orchestrator_node",
            })
            _refresh_ledger_state(state, _user_id, _project_id, _lang)
            log.info("diagram_orchestrator_node: ledger write ok — kind=diagram qa=%s", _qa)
        except LedgerValidationError as exc:
            log.warning("diagram_orchestrator_node: ledger validation error: %s", exc)
        except LedgerConcurrencyError as exc:
            log.warning("diagram_orchestrator_node: ledger concurrency error: %s", exc)
        except Exception as exc:
            log.warning("diagram_orchestrator_node: ledger write failed: %s", exc)

    return state
