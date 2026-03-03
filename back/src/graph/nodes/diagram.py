import base64
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.consts import DOT_SYSTEM, DOT_SYSTEM_OVERVIEW
from src.graph.resources import llm, log
from src.graph.state import GraphState
from src.services.diagram_ir import (
    DetailLevel,
    DiagramModel,
    build_overview,
    build_expanded_view,
    parse_dot_to_model,
)
from src.services.diagram_render import (
    render_dot,
    render_dot_drawio,
    render_drawio,
    render_svg,
    render_svg_b64,
)

try:
    from graphviz import Source
except Exception:
    Source = None


def _sanitize_dot(raw: str) -> str:
    if not raw:
        return ""

    txt = raw.replace("\r\n", "\n").strip()

    m = re.search(r"```dot\s*(.*?)```", txt, flags=re.I | re.S)
    if not m:
        m = re.search(r"```(?:\w+)?\s*(.*?)```", txt, flags=re.I | re.S)
    if m:
        txt = m.group(1).strip()

    m = re.search(r"(?:strict\s+)?digraph\s+[A-Za-z_][A-Za-z0-9_]*\s*\{[\s\S]*\}", txt, flags=re.I)
    if m:
        txt = m.group(0).strip()
    else:
        m = re.search(r"\{[\s\S]*\}", txt, flags=re.S)
        if m:
            body = m.group(0).strip().strip("{}").strip()
            txt = "digraph G {\n" + body + "\n}"
        else:
            lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
            body = "\n".join(lines)
            txt = "digraph G {\n" + body + "\n}"

    return txt.encode("ascii", errors="ignore").decode("ascii").strip()


def _llm_nl_to_dot(natural_prompt: str, *, detail_level: str = "detailed") -> str:
    """Generate DOT code via LLM. Uses overview or detailed system prompt."""
    system_prompt = DOT_SYSTEM_OVERVIEW if detail_level == "overview" else DOT_SYSTEM
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=natural_prompt)]
    resp = llm.invoke(msgs)
    raw = getattr(resp, "content", str(resp)) or ""
    return _sanitize_dot(raw)


def _render_dot_svg_b64(dot_code: str) -> str:
    """Render DOT to base64 SVG. Tries new renderer, falls back to legacy."""
    engine = os.getenv("GRAPHVIZ_ENGINE", "dot").strip() or "dot"
    try:
        return render_svg_b64(dot_code, engine=engine)
    except Exception:
        # Fallback to legacy path
        if Source is None:
            raise RuntimeError("python-graphviz is not installed")
        src = Source(dot_code, format="svg", engine=engine)
        svg_bytes = src.pipe(format="svg")
        return base64.b64encode(svg_bytes).decode("ascii")


def diagram_orchestrator_node(state: GraphState) -> GraphState:
    user_q = (state.get("localQuestion") or state.get("userQuestion") or "").strip()

    # --- Determine detail level ---
    # Check if the user request hints at a detail level
    detail_level = "overview"  # default
    low_q = user_q.lower()
    detail_keywords = [
        "detailed", "detallado", "full", "completo", "complete",
        "all components", "todos los componentes",
        "more detail", "más detalle", "mas detalle",
        "expand", "expandir", "ampliar",
    ]
    if any(kw in low_q for kw in detail_keywords):
        detail_level = "detailed"

    # Also check explicit state override
    if state.get("diagram_detail_level"):
        detail_level = state["diagram_detail_level"]

    asr_text = (state.get("current_asr") or state.get("last_asr") or "").strip()

    style_text = (
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
                tactics_names.append(str(it["name"]))

    if not tactics_names:
        tactics_md = (state.get("tactics_md") or "").strip()
        if tactics_md:
            for line in tactics_md.splitlines():
                line = re.sub(r"^\s*[-*]\s*", "", line).strip()
                if line:
                    tactics_names.append(line)

    tactics_names = tactics_names[:8]
    tactics_block = "\n".join(f"- {t}" for t in tactics_names) if tactics_names else "- (none selected yet)"

    add_context = (state.get("add_context") or "").strip()
    doc_context = (state.get("doc_context") or "").strip()
    memory_text = (state.get("memory_text") or "").strip()

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

    full_prompt = "\n\n---\n\n".join(sections)

    dot_code = ""
    try:
        dot_code = _llm_nl_to_dot(full_prompt, detail_level=detail_level)
    except Exception as e:
        log.warning("diagram_orchestrator_node: DOT generation failed: %s", e)

    diagram_obj = {}
    if dot_code:
        try:
            # Parse DOT into IR
            ir_model = parse_dot_to_model(dot_code)
            ir_model.detail_level = DetailLevel(detail_level)

            # If overview was requested AND the LLM produced too many nodes,
            # collapse programmatically
            overview_mapping = None
            if detail_level == "overview" and len(ir_model.nodes) > 20:
                ir_model, overview_mapping = build_overview(ir_model, max_nodes=15)

            # Re-render DOT from IR for consistency
            final_dot = render_dot(ir_model)

            # Render SVG
            svg_b64 = _render_dot_svg_b64(final_dot)

            # Prepare export variants
            dot_drawio_code = render_dot_drawio(ir_model)

            diagram_obj = {
                "ok": True,
                "format": "svg",
                "engine": os.getenv("GRAPHVIZ_ENGINE", "dot"),
                "svg_b64": svg_b64,
                "dot": final_dot,
                "dot_raw": dot_code,        # original LLM output DOT
                "dot_drawio": dot_drawio_code,
                "detail_level": detail_level,
                "node_count": len(ir_model.nodes),
                "edge_count": len(ir_model.edges),
            }

            # Store overview mapping for potential expansion requests
            if overview_mapping:
                diagram_obj["overview_mapping"] = overview_mapping

        except Exception as e:
            log.warning("diagram_orchestrator_node: Graphviz render failed: %s", e)
            diagram_obj = {
                "ok": False,
                "format": "svg",
                "engine": os.getenv("GRAPHVIZ_ENGINE", "dot"),
                "dot": dot_code,
                "detail_level": detail_level,
                "error": str(e),
            }

    state["diagram"] = diagram_obj
    state["hasVisitedDiagram"] = True
    state["intent"] = "diagram"

    return state
