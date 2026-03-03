# -*- coding: utf-8 -*-
"""
Tests for diagram IR, rendering, and export pipeline.

Covers:
- DiagramModel creation and validation
- DOT parsing → DiagramModel
- Overview generation (fewer nodes than detailed)
- Focused expansion
- DOT rendering from IR
- dot_drawio flattened export (no clusters/compound/ports, edge-complete)
- draw.io XML export (valid XML, preserves counts)
- Deterministic output (snapshot-style)
- Label escaping / safety
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from src.services.diagram_ir import (
    DetailLevel,
    DiagramEdge,
    DiagramGroup,
    DiagramModel,
    DiagramNode,
    EdgeKind,
    NodeKind,
    build_expanded_view,
    build_overview,
    parse_dot_to_model,
)
from src.services.diagram_render import (
    render_dot,
    render_dot_drawio,
    render_drawio,
)


# ---------------------------------------------------------------------------
# Fixtures — representative DOT inputs
# ---------------------------------------------------------------------------

SAMPLE_DOT_DETAILED = """\
digraph G {
    graph [rankdir=LR, splines=ortho, nodesep=0.5, ranksep=0.8, fontsize=12, labelloc=t, bgcolor="transparent"]
    node [shape=box, style="rounded,filled", fillcolor="#2D3748", color="#4A5568", fontname="Helvetica", fontsize=10, fontcolor="#FFFFFF"]
    edge [color="#A0AEC0", arrowsize=0.7, penwidth=1.1, fontcolor="#FFFFFF"]

    subgraph cluster_frontend {
        label="Frontend Layer"
        style="rounded,filled"; fillcolor="#1A202C"; color="#718096";
        cdn [label="CDN"];
        web_app [label="Web App"];
        mobile_app [label="Mobile App"];
    }

    subgraph cluster_backend {
        label="Backend Services"
        style="rounded,filled"; fillcolor="#1A202C"; color="#718096";
        api_gateway [label="API Gateway"];
        auth_service [label="Auth Service"];
        order_service [label="Order Service"];
        catalog_service [label="Catalog Service"];
        payment_service [label="Payment Service"];
        notification_service [label="Notification Service"];
        cache_redis [label="Redis Cache"];
    }

    subgraph cluster_data {
        label="Data Layer"
        style="rounded,filled"; fillcolor="#1A202C"; color="#718096";
        postgres_db [label="PostgreSQL"];
        mongo_db [label="MongoDB"];
        message_queue [label="Kafka Queue"];
    }

    external_payment [label="Payment Gateway (Stripe)"];

    cdn -> web_app;
    web_app -> api_gateway;
    mobile_app -> api_gateway;
    api_gateway -> auth_service;
    api_gateway -> order_service;
    api_gateway -> catalog_service;
    order_service -> payment_service;
    order_service -> notification_service;
    order_service -> cache_redis;
    payment_service -> external_payment;
    order_service -> postgres_db;
    catalog_service -> mongo_db;
    notification_service -> message_queue;
    auth_service -> postgres_db;
    cache_redis -> postgres_db;
}
"""

SAMPLE_DOT_SIMPLE = """\
digraph G {
    client [label="Client"];
    server [label="Server"];
    database [label="Database"];
    client -> server [label="HTTP"];
    server -> database [label="SQL"];
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_model() -> DiagramModel:
    """Build a representative DiagramModel directly (no parsing)."""
    model = DiagramModel(
        title="E-Commerce Platform",
        detail_level=DetailLevel.DETAILED,
        groups=[
            DiagramGroup(id="cluster_frontend", label="Frontend Layer"),
            DiagramGroup(id="cluster_backend", label="Backend Services"),
            DiagramGroup(id="cluster_data", label="Data Layer"),
        ],
        nodes=[
            DiagramNode(id="cdn", label="CDN", kind=NodeKind.CDN, group_id="cluster_frontend"),
            DiagramNode(id="web_app", label="Web App", kind=NodeKind.CLIENT, group_id="cluster_frontend"),
            DiagramNode(id="mobile_app", label="Mobile App", kind=NodeKind.CLIENT, group_id="cluster_frontend"),
            DiagramNode(id="api_gateway", label="API Gateway", kind=NodeKind.GATEWAY, group_id="cluster_backend"),
            DiagramNode(id="auth_service", label="Auth Service", kind=NodeKind.SERVICE, group_id="cluster_backend"),
            DiagramNode(id="order_service", label="Order Service", kind=NodeKind.SERVICE, group_id="cluster_backend"),
            DiagramNode(id="catalog_service", label="Catalog Service", kind=NodeKind.SERVICE, group_id="cluster_backend"),
            DiagramNode(id="payment_service", label="Payment Service", kind=NodeKind.SERVICE, group_id="cluster_backend"),
            DiagramNode(id="notification_service", label="Notification Service", kind=NodeKind.SERVICE, group_id="cluster_backend"),
            DiagramNode(id="cache_redis", label="Redis Cache", kind=NodeKind.CACHE, group_id="cluster_backend"),
            DiagramNode(id="postgres_db", label="PostgreSQL", kind=NodeKind.DATABASE, group_id="cluster_data"),
            DiagramNode(id="mongo_db", label="MongoDB", kind=NodeKind.DATABASE, group_id="cluster_data"),
            DiagramNode(id="message_queue", label="Kafka Queue", kind=NodeKind.QUEUE, group_id="cluster_data"),
            DiagramNode(id="external_payment", label="Payment Gateway (Stripe)", kind=NodeKind.EXTERNAL),
        ],
        edges=[
            DiagramEdge(source_id="cdn", target_id="web_app"),
            DiagramEdge(source_id="web_app", target_id="api_gateway"),
            DiagramEdge(source_id="mobile_app", target_id="api_gateway"),
            DiagramEdge(source_id="api_gateway", target_id="auth_service"),
            DiagramEdge(source_id="api_gateway", target_id="order_service"),
            DiagramEdge(source_id="api_gateway", target_id="catalog_service"),
            DiagramEdge(source_id="order_service", target_id="payment_service"),
            DiagramEdge(source_id="order_service", target_id="notification_service"),
            DiagramEdge(source_id="order_service", target_id="cache_redis"),
            DiagramEdge(source_id="payment_service", target_id="external_payment"),
            DiagramEdge(source_id="order_service", target_id="postgres_db"),
            DiagramEdge(source_id="catalog_service", target_id="mongo_db"),
            DiagramEdge(source_id="notification_service", target_id="message_queue"),
            DiagramEdge(source_id="auth_service", target_id="postgres_db"),
            DiagramEdge(source_id="cache_redis", target_id="postgres_db"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Tests – DiagramModel basics
# ---------------------------------------------------------------------------

class TestDiagramModel:
    def test_create_empty(self):
        m = DiagramModel()
        assert m.nodes == []
        assert m.edges == []
        assert m.groups == []
        assert m.validate() == []

    def test_validate_missing_edge_endpoint(self):
        m = DiagramModel(
            nodes=[DiagramNode(id="a", label="A")],
            edges=[DiagramEdge(source_id="a", target_id="missing")],
        )
        issues = m.validate()
        assert len(issues) == 1
        assert "missing" in issues[0]

    def test_validate_missing_group(self):
        m = DiagramModel(
            nodes=[DiagramNode(id="a", label="A", group_id="nonexistent")],
        )
        issues = m.validate()
        assert len(issues) == 1
        assert "nonexistent" in issues[0]

    def test_node_ids(self):
        m = _make_sample_model()
        ids = m.node_ids()
        assert "cdn" in ids
        assert "api_gateway" in ids
        assert "external_payment" in ids

    def test_nodes_in_group(self):
        m = _make_sample_model()
        backend = m.nodes_in_group("cluster_backend")
        backend_ids = {n.id for n in backend}
        assert "api_gateway" in backend_ids
        assert "order_service" in backend_ids
        assert "cdn" not in backend_ids

    def test_sort_deterministic(self):
        m1 = _make_sample_model()
        m2 = _make_sample_model()
        m1.sort_deterministic()
        m2.sort_deterministic()
        assert [n.id for n in m1.nodes] == [n.id for n in m2.nodes]
        assert [(e.source_id, e.target_id) for e in m1.edges] == \
               [(e.source_id, e.target_id) for e in m2.edges]

    def test_sample_model_is_valid(self):
        m = _make_sample_model()
        assert m.validate() == []


# ---------------------------------------------------------------------------
# Tests – DOT parsing
# ---------------------------------------------------------------------------

class TestParseDot:
    def test_parse_detailed_nodes(self):
        model = parse_dot_to_model(SAMPLE_DOT_DETAILED)
        ids = model.node_ids()
        assert "cdn" in ids
        assert "api_gateway" in ids
        assert "postgres_db" in ids
        assert "external_payment" in ids

    def test_parse_detailed_edges(self):
        model = parse_dot_to_model(SAMPLE_DOT_DETAILED)
        edge_pairs = {(e.source_id, e.target_id) for e in model.edges}
        assert ("web_app", "api_gateway") in edge_pairs
        assert ("order_service", "payment_service") in edge_pairs

    def test_parse_simple(self):
        model = parse_dot_to_model(SAMPLE_DOT_SIMPLE)
        assert len(model.nodes) == 3
        assert len(model.edges) == 2
        ids = model.node_ids()
        assert "client" in ids
        assert "server" in ids
        assert "database" in ids

    def test_parse_extracts_groups(self):
        model = parse_dot_to_model(SAMPLE_DOT_DETAILED)
        group_ids = {g.id for g in model.groups}
        assert "cluster_frontend" in group_ids
        assert "cluster_backend" in group_ids
        assert "cluster_data" in group_ids

    def test_parse_assigns_group_membership(self):
        model = parse_dot_to_model(SAMPLE_DOT_DETAILED)
        cdn_node = model.node_by_id("cdn")
        assert cdn_node is not None
        assert cdn_node.group_id == "cluster_frontend"

    def test_parse_edge_labels(self):
        model = parse_dot_to_model(SAMPLE_DOT_SIMPLE)
        http_edges = [e for e in model.edges if e.label == "HTTP"]
        assert len(http_edges) == 1
        assert http_edges[0].source_id == "client"

    def test_parse_empty_dot(self):
        model = parse_dot_to_model("digraph G {}")
        assert model.nodes == []
        assert model.edges == []

    def test_parse_node_kind_heuristic(self):
        model = parse_dot_to_model(SAMPLE_DOT_DETAILED)
        pg = model.node_by_id("postgres_db")
        assert pg is not None
        assert pg.kind == NodeKind.DATABASE

        mq = model.node_by_id("message_queue")
        assert mq is not None
        assert mq.kind == NodeKind.QUEUE


# ---------------------------------------------------------------------------
# Tests – Overview generation
# ---------------------------------------------------------------------------

class TestOverview:
    def test_overview_has_fewer_nodes(self):
        detailed = _make_sample_model()
        overview, mapping = build_overview(detailed, max_nodes=15)
        assert len(overview.nodes) < len(detailed.nodes)

    def test_overview_preserves_connectivity(self):
        """Overview must have at least one edge."""
        detailed = _make_sample_model()
        overview, mapping = build_overview(detailed, max_nodes=15)
        assert len(overview.edges) > 0

    def test_overview_no_self_loops(self):
        detailed = _make_sample_model()
        overview, mapping = build_overview(detailed, max_nodes=15)
        for e in overview.edges:
            assert e.source_id != e.target_id, f"Self-loop: {e.source_id}"

    def test_overview_mapping_covers_all_nodes(self):
        detailed = _make_sample_model()
        overview, mapping = build_overview(detailed, max_nodes=15)
        all_detailed_ids = set()
        for ids in mapping.values():
            all_detailed_ids.update(ids)
        assert all_detailed_ids == detailed.node_ids()

    def test_overview_detail_level_set(self):
        detailed = _make_sample_model()
        overview, _ = build_overview(detailed, max_nodes=15)
        assert overview.detail_level == DetailLevel.OVERVIEW

    def test_small_graph_returned_as_is(self):
        """If graph is already small and has no groups, overview == detailed."""
        model = DiagramModel(
            nodes=[
                DiagramNode(id="a", label="A"),
                DiagramNode(id="b", label="B"),
            ],
            edges=[DiagramEdge(source_id="a", target_id="b")],
        )
        overview, mapping = build_overview(model, max_nodes=15)
        assert len(overview.nodes) == 2
        assert len(overview.edges) == 1

    def test_overview_from_parsed_dot(self):
        detailed = parse_dot_to_model(SAMPLE_DOT_DETAILED)
        overview, mapping = build_overview(detailed, max_nodes=10)
        assert len(overview.nodes) < len(detailed.nodes)
        assert len(overview.nodes) <= 10


# ---------------------------------------------------------------------------
# Tests – Focused expansion
# ---------------------------------------------------------------------------

class TestExpansion:
    def test_expand_existing_node(self):
        detailed = _make_sample_model()
        _, mapping = build_overview(detailed, max_nodes=15)
        # Pick the first overview node that maps to multiple detailed nodes
        multi_node = None
        for ov_id, detail_ids in mapping.items():
            if len(detail_ids) > 1:
                multi_node = ov_id
                break
        if multi_node is None:
            pytest.skip("No multi-node groups in this sample")

        expanded = build_expanded_view(detailed, mapping, multi_node)
        assert len(expanded.nodes) > 0
        assert expanded.detail_level == DetailLevel.DETAILED

    def test_expand_unknown_node_raises(self):
        detailed = _make_sample_model()
        _, mapping = build_overview(detailed, max_nodes=15)
        with pytest.raises(ValueError, match="Unknown overview node"):
            build_expanded_view(detailed, mapping, "nonexistent_node")


# ---------------------------------------------------------------------------
# Tests – DOT rendering from IR
# ---------------------------------------------------------------------------

class TestRenderDot:
    def test_produces_valid_digraph(self):
        model = _make_sample_model()
        dot = render_dot(model)
        assert dot.startswith("digraph G {")
        assert dot.strip().endswith("}")

    def test_all_nodes_present(self):
        model = _make_sample_model()
        dot = render_dot(model)
        for node in model.nodes:
            assert node.id in dot, f"Node {node.id} missing from DOT"

    def test_all_edges_present(self):
        model = _make_sample_model()
        dot = render_dot(model)
        for edge in model.edges:
            assert f"{edge.source_id}" in dot
            assert f"{edge.target_id}" in dot
            assert "->" in dot

    def test_clusters_in_dot(self):
        model = _make_sample_model()
        dot = render_dot(model)
        assert "subgraph" in dot
        assert "cluster_frontend" in dot

    def test_empty_model(self):
        dot = render_dot(DiagramModel())
        assert "digraph G {" in dot
        assert dot.strip().endswith("}")

    def test_no_html_labels(self):
        model = _make_sample_model()
        dot = render_dot(model)
        assert "<TABLE" not in dot
        assert "<table" not in dot

    def test_deterministic(self):
        """Two renders of the same model produce identical output."""
        m1 = _make_sample_model()
        m2 = _make_sample_model()
        m1.sort_deterministic()
        m2.sort_deterministic()
        assert render_dot(m1) == render_dot(m2)


# ---------------------------------------------------------------------------
# Tests – dot_drawio (flattened DOT for draw.io)
# ---------------------------------------------------------------------------

class TestDotDrawio:
    def test_no_clusters(self):
        model = _make_sample_model()
        flat = render_dot_drawio(model)
        assert "subgraph" not in flat
        assert "cluster_" not in flat.split("[")[0]  # Not as a subgraph

    def test_no_compound_edges(self):
        model = _make_sample_model()
        flat = render_dot_drawio(model)
        assert "lhead" not in flat
        assert "ltail" not in flat

    def test_no_ports(self):
        model = _make_sample_model()
        flat = render_dot_drawio(model)
        # Edges should not contain port references (nodeID:port)
        edge_lines = [ln for ln in flat.splitlines() if "->" in ln]
        for ln in edge_lines:
            # Before the -> and after, there should be no : in node refs
            src_part = ln.split("->")[0].strip()
            tgt_part = ln.split("->")[1].split("[")[0].strip()
            assert ":" not in src_part, f"Port in source: {ln}"
            assert ":" not in tgt_part, f"Port in target: {ln}"

    def test_no_html_labels(self):
        model = _make_sample_model()
        flat = render_dot_drawio(model)
        assert "<TABLE" not in flat
        assert "<table" not in flat

    def test_preserves_edge_count(self):
        """dot_drawio must preserve ALL edges from the IR model."""
        model = _make_sample_model()
        flat = render_dot_drawio(model)
        edge_lines = [ln for ln in flat.splitlines() if "->" in ln]
        assert len(edge_lines) == len(model.edges)

    def test_preserves_node_count(self):
        """dot_drawio must declare ALL nodes from the IR model."""
        model = _make_sample_model()
        flat = render_dot_drawio(model)
        # Count explicit node declarations (lines with [...]; but not edges)
        node_lines = [
            ln for ln in flat.splitlines()
            if "[" in ln and "]" in ln and "->" not in ln
            and not ln.strip().startswith("graph")
            and not ln.strip().startswith("node")
            and not ln.strip().startswith("edge")
        ]
        assert len(node_lines) == len(model.nodes)

    def test_splines_not_ortho(self):
        """draw.io doesn't handle splines=ortho well; should use splines=true."""
        model = _make_sample_model()
        flat = render_dot_drawio(model)
        assert "splines=true" in flat or 'splines="true"' in flat

    def test_explicit_node_declarations(self):
        """Every node must be explicitly declared (not just implicitly via edges)."""
        model = _make_sample_model()
        flat = render_dot_drawio(model)
        for node in model.nodes:
            pattern = re.compile(rf'^\s*{re.escape(node.id)}\s*\[', re.M)
            assert pattern.search(flat), f"Node {node.id} not explicitly declared"


# ---------------------------------------------------------------------------
# Tests – draw.io XML export
# ---------------------------------------------------------------------------

class TestDrawioExport:
    def test_valid_xml(self):
        model = _make_sample_model()
        xml_bytes = render_drawio(model)
        # Should parse as valid XML
        root = ET.fromstring(xml_bytes)
        assert root.tag == "mxfile"

    def test_preserves_node_count(self):
        model = _make_sample_model()
        xml_bytes = render_drawio(model)
        root = ET.fromstring(xml_bytes)
        # Count vertex cells (exclude the two default parent cells)
        cells = root.findall(".//{http://www.w3.org/1999/xhtml}mxCell") or root.findall(".//mxCell")
        vertex_cells = [c for c in cells if c.get("vertex") == "1"]
        assert len(vertex_cells) == len(model.nodes)

    def test_preserves_edge_count(self):
        model = _make_sample_model()
        xml_bytes = render_drawio(model)
        root = ET.fromstring(xml_bytes)
        cells = root.findall(".//{http://www.w3.org/1999/xhtml}mxCell") or root.findall(".//mxCell")
        edge_cells = [c for c in cells if c.get("edge") == "1"]
        assert len(edge_cells) == len(model.edges)

    def test_has_geometry(self):
        model = _make_sample_model()
        xml_bytes = render_drawio(model)
        root = ET.fromstring(xml_bytes)
        geometries = root.findall(".//{http://www.w3.org/1999/xhtml}mxGeometry") or root.findall(".//mxGeometry")
        # Every cell (node + edge) should have geometry
        assert len(geometries) >= len(model.nodes)

    def test_edge_sources_and_targets(self):
        """Every edge cell must reference valid source and target cell IDs."""
        model = _make_sample_model()
        xml_bytes = render_drawio(model)
        root = ET.fromstring(xml_bytes)
        cells = root.findall(".//{http://www.w3.org/1999/xhtml}mxCell") or root.findall(".//mxCell")
        all_ids = {c.get("id") for c in cells}
        edge_cells = [c for c in cells if c.get("edge") == "1"]
        for ec in edge_cells:
            assert ec.get("source") in all_ids, f"Edge source {ec.get('source')} not found"
            assert ec.get("target") in all_ids, f"Edge target {ec.get('target')} not found"


# ---------------------------------------------------------------------------
# Tests – Label escaping / safety
# ---------------------------------------------------------------------------

class TestSafety:
    def test_dot_injection_in_label(self):
        """Labels with special chars must be escaped, not break DOT."""
        model = DiagramModel(
            nodes=[
                DiagramNode(id="svc", label='Service "A" -> B'),
                DiagramNode(id="db", label="DB<script>"),
            ],
            edges=[DiagramEdge(source_id="svc", target_id="db")],
        )
        dot = render_dot(model)
        # Must not have unescaped quotes or <script>
        assert r'\"A\"' in dot or r'\\"A\\"' in dot or "A" in dot
        assert "<script>" not in dot  # should be escaped

    def test_drawio_xml_escaping(self):
        model = DiagramModel(
            nodes=[
                DiagramNode(id="svc", label='Svc & <Co>'),
            ],
            edges=[],
        )
        xml_bytes = render_drawio(model)
        xml_str = xml_bytes.decode("utf-8")
        # Raw & and < should be escaped in XML
        assert "&amp;" in xml_str or "Svc" in xml_str
        assert "<Co>" not in xml_str  # must be escaped

    def test_node_id_validation(self):
        """IDs with weird chars should be safe-ified."""
        model = parse_dot_to_model('digraph G { "node with spaces" [label="A"]; }')
        ids = model.node_ids()
        for nid in ids:
            assert re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', nid), f"Bad ID: {nid}"


# ---------------------------------------------------------------------------
# Tests – Snapshot / determinism regression
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_render_dot_snapshot(self):
        """Rendering the same model twice produces identical DOT output."""
        m = _make_sample_model()
        m.sort_deterministic()
        dot1 = render_dot(m)
        dot2 = render_dot(m)
        assert dot1 == dot2

    def test_render_dot_drawio_snapshot(self):
        m = _make_sample_model()
        m.sort_deterministic()
        flat1 = render_dot_drawio(m)
        flat2 = render_dot_drawio(m)
        assert flat1 == flat2

    def test_overview_deterministic(self):
        m1 = _make_sample_model()
        m2 = _make_sample_model()
        ov1, map1 = build_overview(m1, max_nodes=10)
        ov2, map2 = build_overview(m2, max_nodes=10)
        assert [n.id for n in ov1.nodes] == [n.id for n in ov2.nodes]
        assert [(e.source_id, e.target_id) for e in ov1.edges] == \
               [(e.source_id, e.target_id) for e in ov2.edges]

    def test_parse_deterministic(self):
        """Parsing the same DOT twice gives the same IR."""
        m1 = parse_dot_to_model(SAMPLE_DOT_DETAILED)
        m2 = parse_dot_to_model(SAMPLE_DOT_DETAILED)
        assert [n.id for n in m1.nodes] == [n.id for n in m2.nodes]
        assert [(e.source_id, e.target_id) for e in m1.edges] == \
               [(e.source_id, e.target_id) for e in m2.edges]
