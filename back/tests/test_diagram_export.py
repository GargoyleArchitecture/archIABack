# -*- coding: utf-8 -*-
"""
Unit tests for src.services.diagram_export.

All tests use mocks – no full server, LangGraph runtime, or Graphviz binary required.
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Optional

from src.services.diagram_export import (
    NodeType,
    NodeModel,
    EdgeModel,
    GraphModel,
    classify_node,
    humanize_label,
    extract_from_langgraph,
    render_dot,
)


# ---------------------------------------------------------------------------
# Helpers – mock LangGraph drawable graph
# ---------------------------------------------------------------------------

@dataclass
class _MockEdge:
    source: str
    target: str
    conditional: bool = False
    data: Optional[str] = None


class _MockDrawable:
    """Minimal stand-in for what LangGraph's ``compiled_graph.get_graph()`` returns."""

    def __init__(self, nodes: dict, edges: list[_MockEdge]):
        self.nodes = nodes
        self.edges = edges


class _MockCompiledGraph:
    def __init__(self, drawable: _MockDrawable):
        self._drawable = drawable

    def get_graph(self):
        return self._drawable


def _make_simple_graph() -> _MockCompiledGraph:
    """Simulate the ArchIA workflow (simplified)."""
    nodes = {
        "__start__": {},
        "__end__": {},
        "boot": {},
        "classifier": {},
        "supervisor": {},
        "investigator": {},
        "diagram_agent": {},
        "evaluator": {},
        "unifier": {},
        "asr": {},
        "style": {},
        "tactics": {},
    }
    edges = [
        _MockEdge("__start__", "boot"),
        _MockEdge("boot", "classifier"),
        _MockEdge("classifier", "supervisor"),
        _MockEdge("supervisor", "investigator", conditional=True),
        _MockEdge("supervisor", "diagram_agent", conditional=True),
        _MockEdge("supervisor", "evaluator", conditional=True),
        _MockEdge("supervisor", "asr", conditional=True),
        _MockEdge("supervisor", "style", conditional=True),
        _MockEdge("supervisor", "tactics", conditional=True),
        _MockEdge("supervisor", "unifier", conditional=True),
        _MockEdge("investigator", "supervisor"),
        _MockEdge("diagram_agent", "supervisor"),
        _MockEdge("evaluator", "supervisor"),
        _MockEdge("asr", "supervisor"),
        _MockEdge("style", "supervisor"),
        _MockEdge("tactics", "supervisor"),
        _MockEdge("unifier", "__end__"),
    ]
    return _MockCompiledGraph(_MockDrawable(nodes, edges))


# ---------------------------------------------------------------------------
# Tests – humanize_label
# ---------------------------------------------------------------------------

class TestHumanizeLabel:
    def test_start(self):
        assert humanize_label("__start__") == "START"

    def test_end(self):
        assert humanize_label("__end__") == "END"

    def test_simple(self):
        assert humanize_label("supervisor") == "Supervisor"

    def test_underscore(self):
        assert humanize_label("diagram_agent") == "Diagram Agent"

    def test_node_suffix_removed(self):
        assert humanize_label("classifier_node") == "Classifier"

    def test_compound(self):
        assert humanize_label("style_latency") == "Style Latency"


# ---------------------------------------------------------------------------
# Tests – classify_node
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def test_known_nodes(self):
        assert classify_node("__start__") == NodeType.START
        assert classify_node("__end__") == NodeType.END
        assert classify_node("supervisor") == NodeType.AGENT
        assert classify_node("classifier") == NodeType.ROUTER
        assert classify_node("diagram_agent") == NodeType.TOOL
        assert classify_node("unifier") == NodeType.PROCESSOR
        assert classify_node("boot") == NodeType.PROCESSOR

    def test_dynamic_style_node(self):
        assert classify_node("style_latency") == NodeType.AGENT

    def test_dynamic_tactics_node(self):
        assert classify_node("tactics_scalability") == NodeType.AGENT

    def test_unknown_defaults_to_processor(self):
        assert classify_node("some_random_node") == NodeType.PROCESSOR


# ---------------------------------------------------------------------------
# Tests – extract_from_langgraph
# ---------------------------------------------------------------------------

class TestExtraction:
    @pytest.fixture
    def model(self) -> GraphModel:
        return extract_from_langgraph(_make_simple_graph())

    def test_extracts_all_nodes(self, model: GraphModel):
        ids = {n.id for n in model.nodes}
        expected = {
            "__start__", "__end__", "boot", "classifier", "supervisor",
            "investigator", "diagram_agent", "evaluator", "unifier",
            "asr", "style", "tactics",
        }
        assert ids == expected

    def test_extracts_all_edges(self, model: GraphModel):
        # 17 edges in the mock
        assert len(model.edges) == 17

    def test_node_types_correct(self, model: GraphModel):
        by_id = {n.id: n for n in model.nodes}
        assert by_id["__start__"].node_type == NodeType.START
        assert by_id["__end__"].node_type == NodeType.END
        assert by_id["supervisor"].node_type == NodeType.AGENT
        assert by_id["classifier"].node_type == NodeType.ROUTER
        assert by_id["diagram_agent"].node_type == NodeType.TOOL
        assert by_id["unifier"].node_type == NodeType.PROCESSOR

    def test_conditional_edges_marked(self, model: GraphModel):
        cond = [e for e in model.edges if e.is_conditional]
        # supervisor -> 7 conditional edges in the mock
        assert len(cond) == 7

    def test_deterministic_ordering(self, model: GraphModel):
        """Two calls produce identical output (sorted nodes and edges)."""
        model2 = extract_from_langgraph(_make_simple_graph())
        assert [n.id for n in model.nodes] == [n.id for n in model2.nodes]
        assert [(e.source, e.target) for e in model.edges] == [(e.source, e.target) for e in model2.edges]


# ---------------------------------------------------------------------------
# Tests – render_dot
# ---------------------------------------------------------------------------

class TestRenderDot:
    def test_produces_valid_digraph(self):
        model = extract_from_langgraph(_make_simple_graph())
        dot = render_dot(model)
        assert dot.startswith("digraph G {")
        assert dot.strip().endswith("}")

    def test_handles_empty_graph(self):
        dot = render_dot(GraphModel())
        assert "digraph G {" in dot
        assert dot.strip().endswith("}")

    def test_conditional_edges_dashed(self):
        model = extract_from_langgraph(_make_simple_graph())
        dot = render_dot(model)
        assert 'style="dashed"' in dot

    def test_node_ids_present(self):
        model = extract_from_langgraph(_make_simple_graph())
        dot = render_dot(model)
        for node in model.nodes:
            # Node ID must appear somewhere (possibly quoted)
            assert node.id in dot or f'"{node.id}"' in dot

    def test_no_html_labels(self):
        model = extract_from_langgraph(_make_simple_graph())
        dot = render_dot(model)
        assert "<TABLE" not in dot
        assert "<table" not in dot
