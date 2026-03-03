# Diagram Pipeline – Developer Guide

## Architecture Overview

The diagram pipeline uses a **renderer-agnostic Intermediate Representation (IR)** to separate concerns:

```
User Request → LLM (DOT generation) → DOT Parser → DiagramModel (IR)
                                                        │
                    ┌───────────────────────────────────┤
                    ▼               ▼           ▼       ▼
                render_dot()   render_svg()  render_     render_
                (full DOT)    (Graphviz)    dot_drawio() drawio()
                                            (flat DOT)   (mxGraph XML)
```

### Key Modules

| Module | Path | Purpose |
|--------|------|---------|
| `diagram_ir` | `back/src/services/diagram_ir.py` | IR data model, DOT parser, overview builder, expansion |
| `diagram_render` | `back/src/services/diagram_render.py` | All renderers: DOT, SVG, dot_drawio, draw.io XML |
| `diagram.py` | `back/src/graph/nodes/diagram.py` | LangGraph node: LLM prompt → DOT → IR → export |
| `consts.py` | `back/src/graph/consts.py` | System prompts: `DOT_SYSTEM` (detailed), `DOT_SYSTEM_OVERVIEW` |

### Data Model (IR)

```python
DiagramModel:
    nodes: List[DiagramNode]       # {id, label, kind, group_id}
    edges: List[DiagramEdge]       # {source_id, target_id, label, kind}
    groups: List[DiagramGroup]     # {id, label, parent_id}
    detail_level: DetailLevel      # "overview" | "detailed"
    title: str
    metadata: dict

NodeKind:  service | database | queue | cache | gateway | loadbalancer |
           cdn | client | external | cluster | generic

EdgeKind:  sync | async | data | depends | generic
```

---

## Progressive Disclosure

### Default: Overview (5–15 nodes)

By default, diagram generation uses `detail_level="overview"`. The LLM receives `DOT_SYSTEM_OVERVIEW` which instructs it to produce 5–15 high-level nodes. If the LLM exceeds 20 nodes, `build_overview()` programmatically collapses them:

1. Groups (clusters) → one overview node per group
2. Ungrouped nodes remain as-is
3. Internal edges within a collapsed group are removed
4. Parallel edges between overview nodes are aggregated

### Detailed

When the user says "more detail", "detailed", "full diagram", etc., the node uses `detail_level="detailed"` with the original `DOT_SYSTEM` prompt (unchanged from before).

### Focused Expansion

Via `GET /diagram/export?focus=<overview_node_id>`, a single overview cluster can be expanded to show its internal detailed nodes plus context edges.

---

## API Endpoints

### `GET /diagram/export` – User architecture diagrams

Export the last generated architecture diagram in various formats.

| Param | Values | Default | Description |
|-------|--------|---------|-------------|
| `session_id` | string | (required) | Session that produced the diagram |
| `format` | `svg`, `dot`, `dot_drawio`, `drawio` | `svg` | Output format |
| `detail_level` | `overview`, `detailed` | `overview` | Abstraction level |
| `focus` | string | (none) | Overview node ID to expand |

#### curl Examples

```bash
# SVG overview (default)
curl "http://localhost:8000/diagram/export?session_id=abc123&format=svg" -o arch_overview.svg

# Full-detail DOT
curl "http://localhost:8000/diagram/export?session_id=abc123&format=dot&detail_level=detailed" -o arch_detailed.dot

# draw.io-compatible flat DOT
curl "http://localhost:8000/diagram/export?session_id=abc123&format=dot_drawio" -o arch_drawio.dot

# Native .drawio file
curl "http://localhost:8000/diagram/export?session_id=abc123&format=drawio" -o architecture.drawio

# Expand a specific overview node
curl "http://localhost:8000/diagram/export?session_id=abc123&format=svg&detail_level=detailed&focus=grp_backend" -o backend_expanded.svg
```

### `GET /diagrams` – Internal workflow graph (unchanged)

Exports the LangGraph agent's own workflow for documentation.

```bash
curl "http://localhost:8000/diagrams?format=dot" -o workflow.dot
curl "http://localhost:8000/diagrams?format=svg" -o workflow.svg
```

### `POST /message` – Chat endpoint (unchanged API, enhanced diagram output)

The diagram object in the response now includes additional fields:

```json
{
  "diagram": {
    "ok": true,
    "format": "svg",
    "engine": "dot",
    "svg_b64": "PHN2Zy...",
    "dot": "digraph G { ... }",
    "dot_raw": "digraph G { ... }",
    "dot_drawio": "digraph G { ... }",
    "detail_level": "overview",
    "node_count": 8,
    "edge_count": 10,
    "overview_mapping": {"grp_backend": ["auth_svc", "order_svc", ...]}
  }
}
```

---

## How to Import into draw.io

### Option A: Native .drawio file (recommended)

1. Download: `GET /diagram/export?session_id=...&format=drawio`
2. Open in draw.io (desktop or web)
3. All nodes and edges are preserved with positions
4. Edit freely

### Option B: Flat DOT import

1. Download: `GET /diagram/export?session_id=...&format=dot_drawio`
2. Open draw.io
3. Go to **Extras → Edit Diagram** (or press Ctrl+Shift+X)
4. Paste the DOT content
5. Click **Close**
6. draw.io will auto-layout the diagram

**Why not import the regular DOT?**
draw.io's DOT import is limited. It doesn't support:
- `subgraph cluster_*` blocks
- Compound edges (`lhead`/`ltail`)
- Port references (`node:port`)
- HTML-like labels (`<TABLE>...</TABLE>`)
- `splines=ortho`

The `dot_drawio` format avoids all of these, ensuring **100% connectivity preservation**.

---

## Troubleshooting

### "No diagram found for this session"
**Cause:** No diagram has been generated yet for this `session_id`.
**Fix:** Send a diagram request via `POST /message` first (e.g., "Generate a deployment diagram").

### "Graphviz 'dot' binary not found"
**Cause:** System Graphviz is not installed.
**Fix:** Install Graphviz: https://graphviz.org/download/
- Windows: `choco install graphviz` or download MSI
- macOS: `brew install graphviz`
- Linux: `apt install graphviz`

The SVG and drawio exports require the system `dot` binary. The `dot` and `dot_drawio` text exports work without it.

### "draw.io shows different layout than SVG"
**Expected.** draw.io uses its own layout engine. The node shapes and connectivity are preserved; only positions differ. Use `format=drawio` for the closest match (positions computed by Graphviz are embedded in the XML).

### "Edges missing after draw.io import"
**Cause:** You're importing the full-fidelity DOT (with clusters/compound edges).
**Fix:** Use `format=dot_drawio` instead. This flat DOT is specifically designed for draw.io compatibility.

### Overview has too many nodes
**Fix:** The system targets 5–15 nodes. If the LLM produces more, `build_overview()` collapses them. You can adjust `max_nodes` in the code. The overview prompt (`DOT_SYSTEM_OVERVIEW`) can also be tuned.

### Tests
```bash
# Run all diagram pipeline tests (51 tests)
poetry run pytest back/tests/test_diagram_pipeline.py -v

# Run existing workflow export tests (20 tests)
poetry run pytest back/tests/test_diagram_export.py -v

# Run all tests
poetry run pytest back/tests/ -v
```
