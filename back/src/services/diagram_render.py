# -*- coding: utf-8 -*-
"""
diagram_render – Render a DiagramModel to DOT, SVG, draw.io XML, or flattened DOT.

This module consumes the renderer-agnostic ``DiagramModel`` from
``diagram_ir`` and produces output in the requested format.

Supported formats:
    - ``dot``       : Standard Graphviz DOT (with clusters, compound edges, etc.)
    - ``svg``       : SVG via Graphviz ``dot`` binary
    - ``drawio``    : Native draw.io / mxGraph XML (positions from Graphviz JSON)
    - ``dot_drawio``: Flattened DOT safe for draw.io import (no clusters/ports/HTML)
"""
from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from src.services.diagram_ir import (
    DiagramEdge,
    DiagramModel,
    DiagramNode,
    EdgeKind,
    NodeKind,
    normalize_text,
)

log = logging.getLogger("diagram_render")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DOT_GRAPH_ATTRS = (
    'rankdir=LR, splines=ortho, nodesep=0.5, ranksep=0.8, '
    'fontsize=12, labelloc=t, bgcolor="transparent"'
)

_DOT_NODE_DEFAULTS = (
    'shape=box, style="rounded,filled", fillcolor="#2D3748", '
    'color="#4A5568", fontname="Helvetica", fontsize=10, fontcolor="#FFFFFF"'
)

_DOT_EDGE_DEFAULTS = (
    'color="#A0AEC0", arrowsize=0.7, penwidth=1.1, fontcolor="#FFFFFF"'
)

# Per-kind overrides for node shape/color
_KIND_STYLES: Dict[NodeKind, Dict[str, str]] = {
    NodeKind.DATABASE:     {"shape": "cylinder",  "fillcolor": "#2B6CB0"},
    NodeKind.QUEUE:        {"shape": "cds",       "fillcolor": "#D69E2E"},
    NodeKind.CACHE:        {"shape": "hexagon",   "fillcolor": "#E53E3E"},
    NodeKind.GATEWAY:      {"shape": "trapezium", "fillcolor": "#38A169"},
    NodeKind.LOADBALANCER: {"shape": "invtrapezium", "fillcolor": "#319795"},
    NodeKind.CDN:          {"shape": "tab",       "fillcolor": "#805AD5"},
    NodeKind.CLIENT:       {"shape": "box",       "fillcolor": "#4A5568"},
    NodeKind.EXTERNAL:     {"shape": "box",       "fillcolor": "#718096", "style": "dashed,filled"},
    NodeKind.CLUSTER:      {"shape": "box3d",     "fillcolor": "#4A5568"},
    NodeKind.SERVICE:      {"shape": "box",       "fillcolor": "#2D3748"},
    NodeKind.GENERIC:      {},
}

_EDGE_STYLES: Dict[EdgeKind, Dict[str, str]] = {
    EdgeKind.ASYNC:   {"style": "dashed"},
    EdgeKind.DATA:    {"style": "bold", "color": "#63B3ED"},
    EdgeKind.DEPENDS: {"style": "dotted", "color": "#FC8181"},
    EdgeKind.SYNC:    {},
    EdgeKind.GENERIC: {},
}


# ---------------------------------------------------------------------------
# ID / label escaping
# ---------------------------------------------------------------------------

def _escape_dot_id(raw: str) -> str:
    """Escape for use as a DOT identifier."""
    text = normalize_text(raw).strip().replace("\n", "_").replace("\t", "_")
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _escape_dot_label(raw: str) -> str:
    """Escape for use inside a DOT label= attribute."""
    text = normalize_text(raw)
    return (
        text
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("<", "\\<")
        .replace(">", "\\>")
    )


def _escape_xml(text: str) -> str:
    """Minimal XML escaping for attribute values."""
    return (normalize_text(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;'))


# ---------------------------------------------------------------------------
# DOT rendering (full fidelity – clusters, compound edges, etc.)
# ---------------------------------------------------------------------------

def render_dot(model: DiagramModel) -> str:
    """Render a ``DiagramModel`` as a Graphviz DOT string.

    This is the *full-fidelity* DOT that may include clusters and other
    Graphviz-specific features.
    """
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append(f"    graph [{_DOT_GRAPH_ATTRS}];")
    lines.append(f"    node [{_DOT_NODE_DEFAULTS}];")
    lines.append(f"    edge [{_DOT_EDGE_DEFAULTS}];")

    if model.title:
        lines.append(f'    label="{_escape_dot_label(model.title)}";')

    lines.append("")

    # Build set of grouped node IDs
    grouped_ids = {n.id for n in model.nodes if n.group_id}

    # --- Clusters ---
    for group in model.groups:
        members = model.nodes_in_group(group.id)
        if not members:
            continue
        lines.append(f"    subgraph {_escape_dot_id(group.id)} {{")
        lines.append(f'        label="{_escape_dot_label(group.label)}";')
        lines.append('        style="rounded,filled"; fillcolor="#1A202C"; color="#718096";')
        lines.append('        fontcolor="#CBD5E0"; fontsize=11;')
        for node in sorted(members, key=lambda n: n.id):
            lines.append(f"        {_render_node_line(node)}")
        lines.append("    }")
        lines.append("")

    # --- Ungrouped nodes ---
    for node in model.nodes:
        if node.id not in grouped_ids:
            lines.append(f"    {_render_node_line(node)}")

    lines.append("")

    # --- Edges ---
    for edge in model.edges:
        lines.append(f"    {_render_edge_line(edge)}")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _render_node_line(node: DiagramNode) -> str:
    """Produce a single DOT node declaration line."""
    attrs: Dict[str, str] = {}
    kind_style = _KIND_STYLES.get(node.kind, {})
    attrs.update(kind_style)
    attrs["label"] = _escape_dot_label(node.label)

    attr_str = ", ".join(f'{k}="{v}"' for k, v in sorted(attrs.items()))
    return f"{_escape_dot_id(node.id)} [{attr_str}];"


def _render_edge_line(edge: DiagramEdge) -> str:
    """Produce a single DOT edge declaration line."""
    attrs: Dict[str, str] = {}
    kind_style = _EDGE_STYLES.get(edge.kind, {})
    attrs.update(kind_style)
    if edge.label:
        attrs["label"] = _escape_dot_label(edge.label)

    attr_str = ""
    if attrs:
        attr_str = " [" + ", ".join(f'{k}="{v}"' for k, v in sorted(attrs.items())) + "]"
    return f"{_escape_dot_id(edge.source_id)} -> {_escape_dot_id(edge.target_id)}{attr_str};"


# ---------------------------------------------------------------------------
# SVG rendering via Graphviz
# ---------------------------------------------------------------------------

def render_svg(dot_string: str, engine: str = "dot") -> bytes:
    """Render DOT to SVG bytes using the system Graphviz binary.

    Falls back to python-graphviz ``Source.pipe()`` if available.
    """
    # Try subprocess first (more reliable)
    try:
        result = subprocess.run(
            [engine, "-Tsvg"],
            input=normalize_text(dot_string),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode == 0:
            return normalize_text(result.stdout).encode("utf-8")
        stderr = normalize_text(result.stderr)
        log.warning("Graphviz render failed (exit %d): %s", result.returncode, stderr)
    except FileNotFoundError:
        log.info("System %s binary not found, trying python-graphviz", engine)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Graphviz rendering timed out (>30s)")

    # Fallback: python-graphviz
    try:
        from graphviz import Source
        src = Source(normalize_text(dot_string), format="svg", engine=engine)
        return normalize_text(src.pipe(format="svg")).encode("utf-8")
    except ImportError:
        raise RuntimeError(
            f"Neither system '{engine}' binary nor python-graphviz is available. "
            "Install Graphviz: https://graphviz.org/download/"
        )


def render_svg_b64(dot_string: str, engine: str = "dot") -> str:
    """Render DOT to a base64-encoded SVG string."""
    svg_bytes = render_svg(dot_string, engine=engine)
    return base64.b64encode(svg_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# Flattened DOT for draw.io import  (dot_drawio)
# ---------------------------------------------------------------------------

def render_dot_drawio(model: DiagramModel) -> str:
    """Render a ``DiagramModel`` as a flat, draw.io-compatible DOT string.

    Guarantees:
    - NO ``subgraph cluster_*`` blocks
    - NO compound edges (lhead / ltail)
    - NO port references (node:port)
    - NO HTML-like labels (< ... >)
    - Simple ASCII node IDs
    - Explicit node declarations before edges
    - All connectivity preserved

    Import into draw.io:
        Extras → Edit Diagram → paste this DOT → click "Close"
        or: File → Import from → Text → paste DOT
    """
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append('    graph [rankdir=LR, splines=true, nodesep=0.5, ranksep=0.8, fontsize=12];')
    lines.append('    node [shape=box, style="rounded,filled", fillcolor="#2D3748", '
                 'fontname="Helvetica", fontsize=10, fontcolor="#FFFFFF"];')
    lines.append('    edge [color="#A0AEC0", arrowsize=0.7, penwidth=1.1];')
    lines.append("")

    # --- All nodes explicitly declared (no clusters) ---
    for node in model.nodes:
        label = _escape_dot_label(node.label)
        # Indicate original cluster membership in label if node was grouped
        if node.group_id:
            group = model.group_by_id(node.group_id)
            if group:
                label = f"{_escape_dot_label(group.label)}\\n{label}"

        attrs: Dict[str, str] = {"label": label}
        kind_style = _KIND_STYLES.get(node.kind, {})
        # Only safe shape attributes (skip anything draw.io won't understand)
        for k in ("shape", "fillcolor", "style"):
            if k in kind_style:
                attrs[k] = kind_style[k]

        attr_str = ", ".join(f'{k}="{v}"' for k, v in sorted(attrs.items()))
        lines.append(f"    {_escape_dot_id(node.id)} [{attr_str}];")

    lines.append("")

    # --- Edges (simple, no ports, no compound) ---
    for edge in model.edges:
        parts: List[str] = []
        if edge.label:
            parts.append(f'label="{_escape_dot_label(edge.label)}"')
        kind_style = _EDGE_STYLES.get(edge.kind, {})
        for k, v in sorted(kind_style.items()):
            parts.append(f'{k}="{v}"')

        attr_str = ""
        if parts:
            attr_str = " [" + ", ".join(parts) + "]"
        lines.append(f"    {_escape_dot_id(edge.source_id)} -> {_escape_dot_id(edge.target_id)}{attr_str};")

    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Native draw.io / mxGraph XML export  –  HIGH FIDELITY
# ---------------------------------------------------------------------------

_DPI = 72.0           # Graphviz: 72 points per inch
_GV_PAD = 8.0         # small padding around the canvas

# Node styles keyed by NodeKind
_DRAWIO_NODE_STYLES: Dict[NodeKind, str] = {
    NodeKind.SERVICE:      "rounded=1;whiteSpace=wrap;html=1;fillColor=#2D3748;fontColor=#FFFFFF;strokeColor=#4A5568;",
    NodeKind.DATABASE:     "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;fillColor=#2B6CB0;fontColor=#FFFFFF;strokeColor=#4A5568;",
    NodeKind.QUEUE:        "rounded=1;whiteSpace=wrap;html=1;fillColor=#D69E2E;fontColor=#FFFFFF;strokeColor=#B7791F;",
    NodeKind.CACHE:        "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=20;fillColor=#E53E3E;fontColor=#FFFFFF;strokeColor=#C53030;",
    NodeKind.GATEWAY:      "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=20;fillColor=#38A169;fontColor=#FFFFFF;strokeColor=#276749;",
    NodeKind.LOADBALANCER: "rounded=1;whiteSpace=wrap;html=1;fillColor=#319795;fontColor=#FFFFFF;strokeColor=#285E61;",
    NodeKind.CDN:          "rounded=1;whiteSpace=wrap;html=1;fillColor=#805AD5;fontColor=#FFFFFF;strokeColor=#6B46C1;",
    NodeKind.CLIENT:       "rounded=1;whiteSpace=wrap;html=1;fillColor=#4A5568;fontColor=#FFFFFF;strokeColor=#2D3748;",
    NodeKind.EXTERNAL:     "rounded=1;whiteSpace=wrap;html=1;dashed=1;fillColor=#718096;fontColor=#FFFFFF;strokeColor=#4A5568;",
    NodeKind.CLUSTER:      "rounded=1;whiteSpace=wrap;html=1;fillColor=#4A5568;fontColor=#FFFFFF;strokeColor=#2D3748;shadow=1;",
    NodeKind.GENERIC:      "rounded=1;whiteSpace=wrap;html=1;fillColor=#2D3748;fontColor=#FFFFFF;strokeColor=#4A5568;",
}

# Container (cluster) style
_DRAWIO_CONTAINER_STYLE = (
    "rounded=1;whiteSpace=wrap;html=1;container=1;collapsible=0;"
    "fillColor=#1A202C;fontColor=#CBD5E0;strokeColor=#718096;"
    "verticalAlign=top;fontStyle=1;fontSize=12;spacingTop=5;"
)

# Edge base + per-kind modifiers
_DRAWIO_EDGE_BASE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
    "jettySize=auto;html=1;strokeColor=#A0AEC0;fontColor=#FFFFFF;"
)

_DRAWIO_EDGE_MODS: Dict[EdgeKind, str] = {
    EdgeKind.ASYNC:   "dashed=1;",
    EdgeKind.DATA:    "strokeColor=#63B3ED;strokeWidth=2;",
    EdgeKind.DEPENDS: "dashed=1;strokeColor=#FC8181;",
    EdgeKind.SYNC:    "",
    EdgeKind.GENERIC: "",
}


# ---------------------------------------------------------------------------
#  Graphviz JSON helpers
# ---------------------------------------------------------------------------

def _graphviz_json_layout(dot_string: str, engine: str = "dot") -> Optional[Dict[str, Any]]:
    """Run Graphviz with ``-Tjson`` to obtain computed positions.

    Returns the parsed JSON dict, or ``None`` if Graphviz is unavailable.
    """
    try:
        result = subprocess.run(
            [engine, "-Tjson"],
            input=normalize_text(dot_string),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("Graphviz JSON layout failed: %s", normalize_text(result.stderr))
            return None
        return json.loads(normalize_text(result.stdout))
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        log.info("Graphviz JSON layout unavailable: %s", exc)
        return None


def _gv_point(pt_str: str) -> Tuple[float, float]:
    """Parse a Graphviz point string ``"x,y"`` into ``(x, y)`` floats."""
    parts = pt_str.replace(",", " ").split()
    return float(parts[0]), float(parts[1])


def _parse_gv_bb(bb_str: str) -> Tuple[float, float, float, float]:
    """Parse a Graphviz bounding-box ``"llx,lly,urx,ury"``."""
    parts = bb_str.replace(",", " ").split()
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


# ---------------------------------------------------------------------------
#  Position-extraction back-ends
# ---------------------------------------------------------------------------

def _extract_gv_positions(
    layout: Dict[str, Any],
) -> Tuple[
    Dict[str, Tuple[float, float, float, float]],      # node_id → (x, y, w, h)
    Dict[str, Tuple[float, float, float, float]],      # cluster_id → (x, y, w, h)
    float,                                               # canvas_width
    float,                                               # canvas_height
]:
    """Convert Graphviz JSON positions to draw.io coordinates.

    * Flips Y-axis  (Graphviz bottom-up → draw.io top-down).
    * Converts node centre + inches → top-left + pixels.
    * Extracts cluster bounding boxes.
    """
    bb_str = layout.get("bb", "0,0,800,600")
    gx1, gy1, gx2, gy2 = _parse_gv_bb(bb_str)
    max_y = gy2
    canvas_w = gx2 - gx1 + 2 * _GV_PAD
    canvas_h = gy2 - gy1 + 2 * _GV_PAD

    node_geom: Dict[str, Tuple[float, float, float, float]] = {}
    cluster_geom: Dict[str, Tuple[float, float, float, float]] = {}

    for obj in layout.get("objects", []):
        name = obj.get("name", "")

        if "pos" in obj:
            # ---- regular node ----
            cx, cy = _gv_point(obj["pos"])
            w = float(obj.get("width", "1.5")) * _DPI
            h = float(obj.get("height", "0.5")) * _DPI
            x = cx - w / 2 + _GV_PAD - gx1
            y = (max_y - cy) - h / 2 + _GV_PAD
            node_geom[name] = (x, y, w, h)

        elif "bb" in obj and name.startswith("cluster"):
            # ---- cluster / subgraph ----
            bx1, by1, bx2, by2 = _parse_gv_bb(obj["bb"])
            x = bx1 - gx1 + _GV_PAD
            y = (max_y - by2) + _GV_PAD       # flip
            w = bx2 - bx1
            h = by2 - by1
            cluster_geom[name] = (x, y, w, h)

    return node_geom, cluster_geom, canvas_w, canvas_h


def _grid_positions(
    model: DiagramModel,
) -> Tuple[
    Dict[str, Tuple[float, float, float, float]],
    Dict[str, Tuple[float, float, float, float]],
    float,
    float,
]:
    """Fallback grid layout when Graphviz JSON is unavailable.

    Creates containers for groups and places nodes inside them.
    All coordinates are *absolute* draw.io pixels.
    """
    NODE_W, NODE_H = 140.0, 50.0
    COL_STEP, ROW_STEP = 180.0, 80.0
    PAD = 20.0
    LABEL_H = 30.0   # vertical space for cluster label

    node_geom: Dict[str, Tuple[float, float, float, float]] = {}
    cluster_geom: Dict[str, Tuple[float, float, float, float]] = {}

    x_offset = PAD

    # --- grouped nodes (inside cluster containers) ---
    for group in model.groups:
        members = model.nodes_in_group(group.id)
        if not members:
            continue
        members_sorted = sorted(members, key=lambda n: n.id)
        cols = max(1, min(len(members_sorted), 3))
        rows = (len(members_sorted) + cols - 1) // cols
        cluster_w = PAD * 2 + cols * COL_STEP
        cluster_h = PAD + LABEL_H + rows * ROW_STEP + PAD
        cluster_x = x_offset
        cluster_y = PAD
        cluster_geom[group.id] = (cluster_x, cluster_y, cluster_w, cluster_h)

        for i, node in enumerate(members_sorted):
            r, c = divmod(i, cols)
            nx = cluster_x + PAD + c * COL_STEP + (COL_STEP - NODE_W) / 2
            ny = cluster_y + PAD + LABEL_H + r * ROW_STEP + (ROW_STEP - NODE_H) / 2
            node_geom[node.id] = (nx, ny, NODE_W, NODE_H)

        x_offset += cluster_w + PAD

    # --- ungrouped nodes ---
    ungrouped = sorted(
        [n for n in model.nodes if not n.group_id],
        key=lambda n: n.id,
    )
    for i, node in enumerate(ungrouped):
        nx = x_offset + (COL_STEP - NODE_W) / 2
        ny = PAD + i * ROW_STEP + (ROW_STEP - NODE_H) / 2
        node_geom[node.id] = (nx, ny, NODE_W, NODE_H)

    # canvas dimensions
    heights: List[float] = []
    for gid in cluster_geom:
        _, cy, _, ch = cluster_geom[gid]
        heights.append(cy + ch)
    if ungrouped:
        heights.append(PAD + len(ungrouped) * ROW_STEP)
    canvas_w = max(800.0, x_offset + (COL_STEP + PAD if ungrouped else 0))
    canvas_h = max(600.0, (max(heights) if heights else PAD) + PAD)

    return node_geom, cluster_geom, canvas_w, canvas_h


# ---------------------------------------------------------------------------
#  Main draw.io renderer
# ---------------------------------------------------------------------------

def render_drawio(model: DiagramModel, engine: str = "dot") -> bytes:
    """Render a ``DiagramModel`` as a native draw.io / mxGraph XML file.

    Strategy
    --------
    1. Render the IR to full-fidelity DOT (with clusters).
    2. Run ``dot -Tjson`` to extract Graphviz-computed layout.
    3. Build mxGraph XML with:
       - **Container cells** for clusters (from Graphviz bounding-boxes).
       - **Node cells** at Graphviz-computed positions **inside** their
         parent containers (coordinates are relative to container).
       - **Edge cells** with orthogonal auto-routing between the
         correctly-positioned nodes.
    4. Y-axis is flipped (Graphviz bottom-up → draw.io top-down).
    5. Falls back to a grid layout if Graphviz is unavailable.

    The resulting ``.drawio`` file preserves the same visual structure
    as the SVG so architects can open and edit it immediately.
    """
    # Step 1 — DOT
    dot_string = render_dot(model)

    # Step 2 — Graphviz JSON layout (or fallback)
    layout = _graphviz_json_layout(dot_string, engine=engine)
    if layout:
        node_geom, cluster_geom, canvas_w, canvas_h = _extract_gv_positions(layout)
    else:
        node_geom, cluster_geom, canvas_w, canvas_h = _grid_positions(model)

    # Step 3 — Build mxGraph XML
    mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device")
    diagram_el = ET.SubElement(mxfile, "diagram", name="Architecture", id="arch_1")
    mx_model = ET.SubElement(
        diagram_el, "mxGraphModel",
        dx="0", dy="0", grid="1", gridSize="10",
        guides="1", tooltips="1", connect="1",
        arrows="1", fold="1", page="1",
        pageScale="1",
        pageWidth=str(max(1169, int(canvas_w) + 100)),
        pageHeight=str(max(827, int(canvas_h) + 100)),
    )
    root = ET.SubElement(mx_model, "root")

    # Default parent cells (required by mxGraph)
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    cell_counter = 2
    node_cell_ids: Dict[str, str] = {}
    cluster_cell_ids: Dict[str, str] = {}

    # ---- cluster container cells ----
    for group in model.groups:
        if group.id not in cluster_geom:
            continue
        cell_id = str(cell_counter)
        cell_counter += 1
        cluster_cell_ids[group.id] = cell_id

        gx, gy, gw, gh = cluster_geom[group.id]

        cell = ET.SubElement(
            root, "mxCell",
            id=cell_id,
            value=_escape_xml(group.label),
            style=_DRAWIO_CONTAINER_STYLE,
            vertex="1",
            parent="1",
        )
        ET.SubElement(
            cell, "mxGeometry",
            x=str(round(gx)), y=str(round(gy)),
            width=str(round(gw)), height=str(round(gh)),
            **{"as": "geometry"},
        )

    # ---- node cells ----
    for node in model.nodes:
        cell_id = str(cell_counter)
        cell_counter += 1
        node_cell_ids[node.id] = cell_id

        abs_x, abs_y, w, h = node_geom.get(node.id, (40.0, 40.0, 140.0, 50.0))

        # Determine parent: container or root layer
        parent_id = "1"
        if node.group_id and node.group_id in cluster_cell_ids:
            parent_id = cluster_cell_ids[node.group_id]
            # Convert absolute → relative to container top-left
            cx, cy, _, _ = cluster_geom[node.group_id]
            abs_x -= cx
            abs_y -= cy

        style = _DRAWIO_NODE_STYLES.get(node.kind, _DRAWIO_NODE_STYLES[NodeKind.GENERIC])

        cell = ET.SubElement(
            root, "mxCell",
            id=cell_id,
            value=_escape_xml(node.label),
            style=style,
            vertex="1",
            parent=parent_id,
        )
        ET.SubElement(
            cell, "mxGeometry",
            x=str(round(abs_x)), y=str(round(abs_y)),
            width=str(round(w)), height=str(round(h)),
            **{"as": "geometry"},
        )

    # ---- edge cells ----
    for edge in model.edges:
        cell_id = str(cell_counter)
        cell_counter += 1

        src_cell = node_cell_ids.get(edge.source_id, "1")
        tgt_cell = node_cell_ids.get(edge.target_id, "1")

        edge_style = _DRAWIO_EDGE_BASE + _DRAWIO_EDGE_MODS.get(edge.kind, "")
        value = _escape_xml(edge.label) if edge.label else ""

        cell = ET.SubElement(
            root, "mxCell",
            id=cell_id,
            value=value,
            style=edge_style,
            edge="1",
            source=src_cell,
            target=tgt_cell,
            parent="1",
        )
        ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})

    # Serialize
    xml_str = ET.tostring(mxfile, encoding="unicode", xml_declaration=True)
    return xml_str.encode("utf-8")
