# -*- coding: utf-8 -*-
"""
diagram_export – Export the LangGraph workflow as DOT / SVG.

This module is COMPLETELY SEPARATE from diagram_orchestrator_node
(which generates architecture diagrams *for the user*). This module
exports the agent's own internal workflow graph for development
documentation and editing in draw.io.

Usage (CLI):
    python -m src.services.diagram_export --format dot --output workflow.dot
    python -m src.services.diagram_export --format svg --output workflow.svg

Usage (programmatic):
    from src.services.diagram_export import export_workflow
    dot_str = export_workflow(fmt="dot")
"""
from __future__ import annotations

import enum
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from src.services.diagram_ir import normalize_text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class NodeType(enum.Enum):
    """Semantic classification of a workflow node."""
    START = "start"
    END = "end"
    AGENT = "agent"
    TOOL = "tool"
    ROUTER = "router"
    PROCESSOR = "processor"


@dataclass(frozen=True)
class NodeModel:
    """A single node in the workflow graph."""
    id: str
    label: str
    node_type: NodeType


@dataclass(frozen=True)
class EdgeModel:
    """A directed edge between two nodes."""
    source: str
    target: str
    label: Optional[str] = None
    is_conditional: bool = False


@dataclass
class GraphModel:
    """Pure representation of a workflow graph (independent of Graphviz)."""
    title: str = "ArchIA Workflow"
    nodes: List[NodeModel] = field(default_factory=list)
    edges: List[EdgeModel] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

# Maps the node *name* (as registered in workflow.py) to a semantic type.
# Unknown nodes fall back to PROCESSOR.
_NODE_TYPE_MAP: dict[str, NodeType] = {
    "__start__": NodeType.START,
    "__end__": NodeType.END,
    "boot": NodeType.PROCESSOR,
    "classifier": NodeType.ROUTER,
    "supervisor": NodeType.AGENT,
    "investigator": NodeType.AGENT,
    "diagram_agent": NodeType.TOOL,
    "evaluator": NodeType.AGENT,
    "unifier": NodeType.PROCESSOR,
    "asr": NodeType.AGENT,
    "style": NodeType.AGENT,
    "tactics": NodeType.AGENT,
}


def classify_node(name: str) -> NodeType:
    """Return the semantic type for a workflow node name."""
    if name in _NODE_TYPE_MAP:
        return _NODE_TYPE_MAP[name]
    # Dynamic QA-specific nodes (e.g. style_latency, tactics_scalability)
    if name.startswith("style"):
        return NodeType.AGENT
    if name.startswith("tactics"):
        return NodeType.AGENT
    return NodeType.PROCESSOR


def humanize_label(name: str) -> str:
    """Convert an internal node name to a human-readable label.

    Examples:
        supervisor      -> Supervisor
        diagram_agent   -> Diagram Agent
        __start__       -> START
        __end__          -> END
        style_latency   -> Style Latency
    """
    if name == "__start__":
        return "START"
    if name == "__end__":
        return "END"
    cleaned = name.replace("_node", "")
    return cleaned.replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Extraction from a compiled LangGraph
# ---------------------------------------------------------------------------

def extract_from_langgraph(compiled_graph: object) -> GraphModel:
    """Build a *GraphModel* from a compiled LangGraph graph object.

    Parameters
    ----------
    compiled_graph:
        The value of ``graph`` exported from ``src.graph``.
        Must support ``.get_graph()`` which returns an object with
        ``.nodes`` (dict) and ``.edges`` (list of tuples/objects).
    """
    drawable = compiled_graph.get_graph()

    model = GraphModel()

    # --- Nodes ---
    seen_ids: set[str] = set()
    for node_id in drawable.nodes:
        ntype = classify_node(str(node_id))
        model.nodes.append(
            NodeModel(id=str(node_id), label=humanize_label(str(node_id)), node_type=ntype)
        )
        seen_ids.add(str(node_id))

    # --- Edges ---
    for edge in drawable.edges:
        # LangGraph edge objects expose .source / .target (and optionally .data or .conditional)
        if hasattr(edge, "source") and hasattr(edge, "target"):
            src = str(edge.source)
            tgt = str(edge.target)
            is_cond = getattr(edge, "conditional", False)
            lbl = getattr(edge, "data", None)
            if isinstance(lbl, dict):
                lbl = lbl.get("label")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, tgt = str(edge[0]), str(edge[1])
            is_cond = False
            lbl = None
        else:
            continue

        # Ensure both endpoints exist (they should, but be defensive)
        for ep in (src, tgt):
            if ep not in seen_ids:
                model.nodes.append(
                    NodeModel(id=ep, label=humanize_label(ep), node_type=classify_node(ep))
                )
                seen_ids.add(ep)

        model.edges.append(EdgeModel(source=src, target=tgt, label=str(lbl) if lbl else None, is_conditional=is_cond))

    # Stable ordering for deterministic output
    model.nodes.sort(key=lambda n: n.id)
    model.edges.sort(key=lambda e: (e.source, e.target))

    return model


# ---------------------------------------------------------------------------
# DOT rendering
# ---------------------------------------------------------------------------

# Visual styles per NodeType – (shape, fillcolor, fontcolor, extra attrs)
_DOT_STYLES: dict[NodeType, dict[str, str]] = {
    NodeType.START: {
        "shape": "circle",
        "fillcolor": "#48BB78",
        "fontcolor": "#FFFFFF",
        "width": "0.4",
        "fixedsize": "true",
        "label": "",
    },
    NodeType.END: {
        "shape": "doublecircle",
        "fillcolor": "#F56565",
        "fontcolor": "#FFFFFF",
        "width": "0.4",
        "fixedsize": "true",
        "label": "",
    },
    NodeType.AGENT: {
        "shape": "box",
        "style": "rounded,filled",
        "fillcolor": "#4299E1",
        "fontcolor": "#FFFFFF",
    },
    NodeType.TOOL: {
        "shape": "component",
        "style": "filled",
        "fillcolor": "#ED8936",
        "fontcolor": "#FFFFFF",
    },
    NodeType.ROUTER: {
        "shape": "diamond",
        "style": "filled",
        "fillcolor": "#9F7AEA",
        "fontcolor": "#FFFFFF",
    },
    NodeType.PROCESSOR: {
        "shape": "box",
        "style": "filled",
        "fillcolor": "#A0AEC0",
        "fontcolor": "#1A202C",
    },
}


def _escape_dot_id(raw: str) -> str:
    """Escape a string for use as a DOT identifier."""
    text = normalize_text(raw).strip().replace("\n", "_").replace("\t", "_")
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _escape_dot_label(raw: str) -> str:
    """Escape a string for use inside a DOT label attribute."""
    return normalize_text(raw).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_dot(graph: GraphModel) -> str:
    """Convert a *GraphModel* to a Graphviz DOT string."""
    lines: list[str] = []
    lines.append("digraph G {")
    lines.append('    graph [rankdir=LR, splines=ortho, nodesep=0.6, ranksep=0.9, '
                 'fontsize=12, labelloc=t, bgcolor="transparent"];')
    lines.append('    node [fontname="Helvetica", fontsize=10];')
    lines.append('    edge [color="#718096", arrowsize=0.7, penwidth=1.1];')
    lines.append("")

    # Nodes
    for node in graph.nodes:
        attrs = dict(_DOT_STYLES.get(node.node_type, _DOT_STYLES[NodeType.PROCESSOR]))
        # Use humanized label unless the style overrides it (START/END use empty label)
        if "label" not in attrs:
            attrs["label"] = _escape_dot_label(node.label)
        attr_str = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
        lines.append(f"    {_escape_dot_id(node.id)} [{attr_str}];")

    lines.append("")

    # Edges
    for edge in graph.edges:
        parts: list[str] = []
        if edge.label:
            parts.append(f'label="{_escape_dot_label(edge.label)}"')
        if edge.is_conditional:
            parts.append('style="dashed"')
            parts.append('color="#E53E3E"')
        attr_str = f" [{', '.join(parts)}]" if parts else ""
        lines.append(f"    {_escape_dot_id(edge.source)} -> {_escape_dot_id(edge.target)}{attr_str};")

    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def render_svg(dot_string: str) -> bytes:
    """Render a DOT string to SVG bytes using the system `dot` command.

    Raises *RuntimeError* if the ``dot`` binary is not available or fails.
    """
    try:
        result = subprocess.run(
            ["dot", "-Tsvg"],
            input=normalize_text(dot_string),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Graphviz 'dot' binary not found. "
            "Install Graphviz (https://graphviz.org/download/) or use format=dot."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Graphviz rendering timed out (>30s).")

    if result.returncode != 0:
        stderr = normalize_text(result.stderr)
        raise RuntimeError(f"Graphviz render failed (exit {result.returncode}): {stderr}")

    return normalize_text(result.stdout).encode("utf-8")


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def export_workflow(*, fmt: str = "dot") -> str | bytes:
    """High-level helper: extract the current workflow graph and export it.

    Parameters
    ----------
    fmt : ``"dot"`` or ``"svg"``

    Returns
    -------
    str (DOT) or bytes (SVG)
    """
    # Lazy import to avoid pulling LangGraph at module-import time,
    # which is important for testing with mocks.
    from src.graph import graph as compiled_graph  # noqa: E402

    model = extract_from_langgraph(compiled_graph)
    dot = render_dot(model)

    if fmt == "dot":
        return dot
    if fmt == "svg":
        return render_svg(dot)
    raise ValueError(f"Unsupported format: {fmt!r}. Use 'dot' or 'svg'.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli_main(argv: Sequence[str] | None = None) -> None:
    """Minimal CLI: ``python -m src.services.diagram_export [options]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export the ArchIA LangGraph workflow diagram."
    )
    parser.add_argument(
        "--format", choices=["dot", "svg"], default="dot",
        help="Output format (default: dot)."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path. Defaults to stdout for DOT, or workflow.svg for SVG."
    )
    args = parser.parse_args(argv)

    result = export_workflow(fmt=args.format)

    if args.output:
        mode = "wb" if isinstance(result, bytes) else "w"
        open_kwargs = {"encoding": "utf-8"} if mode == "w" else {}
        with open(args.output, mode, **open_kwargs) as fh:
            fh.write(result)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        if isinstance(result, bytes):
            out_path = "workflow.svg"
            with open(out_path, "wb") as fh:
                fh.write(result)
            print(f"Wrote {out_path}", file=sys.stderr)
        else:
            sys.stdout.write(result)


if __name__ == "__main__":
    _cli_main()
