# -*- coding: utf-8 -*-
"""
diagram_ir – Intermediate Representation for architecture diagrams.

This module defines a renderer-agnostic data model for diagrams.
All exporters (DOT, SVG, draw.io, dot_drawio) consume this IR
instead of raw DOT strings.

The IR also powers *progressive disclosure*: a ``DiagramModel`` at
detail_level="detailed" can be collapsed into an "overview" model
via ``build_overview()``.
"""
from __future__ import annotations

import enum
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger("diagram_ir")


_SAFE_CONTROL_CHARS = {"\n", "\t"}


def normalize_text(value: Any) -> str:
    """Normalize arbitrary text payloads to deterministic UTF-8-safe ``str``.

    Rules:
    - ``None`` -> ``""``
    - ``bytes`` -> UTF-8 strict decode; fallback to ``errors="replace"`` with warning
    - everything else -> ``str(value)``
    - normalize to NFC
    - replace control characters except ``\\n`` and ``\\t`` with a space
    """
    if value is None:
        return ""

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = value.decode("utf-8", errors="replace")
            log.warning("normalize_text: invalid UTF-8 bytes decoded with replacement fallback")
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    cleaned_chars: List[str] = []
    for ch in text:
        if unicodedata.category(ch).startswith("C") and ch not in _SAFE_CONTROL_CHARS:
            cleaned_chars.append(" ")
            continue
        cleaned_chars.append(ch)
    return "".join(cleaned_chars)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DetailLevel(str, enum.Enum):
    """Requested abstraction level for diagram rendering."""
    OVERVIEW = "overview"
    MEDIUM = "medium"
    DETAILED = "detailed"


class DiagramLevel(enum.IntEnum):
    """Numeric level for progressive disclosure."""
    OVERVIEW = 1
    MEDIUM = 2
    DETAILED = 3


class NodeKind(str, enum.Enum):
    """Semantic kind of a diagram node."""
    SERVICE = "service"
    DATABASE = "database"
    QUEUE = "queue"
    CACHE = "cache"
    GATEWAY = "gateway"
    LOADBALANCER = "loadbalancer"
    CDN = "cdn"
    CLIENT = "client"
    EXTERNAL = "external"
    CLUSTER = "cluster"       # represents a collapsed group in overview
    GENERIC = "generic"


class EdgeKind(str, enum.Enum):
    """Semantic kind of a diagram edge."""
    SYNC = "sync"
    ASYNC = "async"
    DATA = "data"
    DEPENDS = "depends"
    GENERIC = "generic"


_DETAIL_LEVEL_TO_DIAGRAM_LEVEL: Dict[DetailLevel, DiagramLevel] = {
    DetailLevel.OVERVIEW: DiagramLevel.OVERVIEW,
    DetailLevel.MEDIUM: DiagramLevel.MEDIUM,
    DetailLevel.DETAILED: DiagramLevel.DETAILED,
}
_DIAGRAM_LEVEL_TO_DETAIL_LEVEL: Dict[DiagramLevel, DetailLevel] = {
    DiagramLevel.OVERVIEW: DetailLevel.OVERVIEW,
    DiagramLevel.MEDIUM: DetailLevel.MEDIUM,
    DiagramLevel.DETAILED: DetailLevel.DETAILED,
}


def parse_diagram_level(raw_level: Any, *, default: DiagramLevel = DiagramLevel.OVERVIEW) -> DiagramLevel:
    """Parse a level from int/string/enum with clear validation errors."""
    if raw_level is None or raw_level == "":
        return default
    if isinstance(raw_level, DiagramLevel):
        return raw_level
    if isinstance(raw_level, DetailLevel):
        return _DETAIL_LEVEL_TO_DIAGRAM_LEVEL[raw_level]
    if isinstance(raw_level, bool):
        raise ValueError(
            "Invalid diagram level 'bool'. Supported levels: 1 (overview), 2 (medium), 3 (detailed)."
        )

    if isinstance(raw_level, int):
        try:
            return DiagramLevel(raw_level)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported diagram level '{raw_level}'. Supported levels: 1 (overview), 2 (medium), 3 (detailed)."
            ) from exc

    low = normalize_text(raw_level).strip().lower()
    aliases = {
        "overview": DiagramLevel.OVERVIEW,
        "high": DiagramLevel.OVERVIEW,
        "high_level": DiagramLevel.OVERVIEW,
        "high-level": DiagramLevel.OVERVIEW,
        "summary": DiagramLevel.OVERVIEW,
        "medium": DiagramLevel.MEDIUM,
        "intermediate": DiagramLevel.MEDIUM,
        "expanded": DiagramLevel.MEDIUM,
        "more_detail": DiagramLevel.MEDIUM,
        "more-detail": DiagramLevel.MEDIUM,
        "detailed": DiagramLevel.DETAILED,
        "detail": DiagramLevel.DETAILED,
        "full": DiagramLevel.DETAILED,
    }
    if low in aliases:
        return aliases[low]
    level_match = re.search(r"\b(?:level|nivel)\s*([1-3])\b", low)
    if level_match:
        return DiagramLevel(int(level_match.group(1)))
    if low.isdigit():
        try:
            return DiagramLevel(int(low))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported diagram level '{raw_level}'. Supported levels: 1 (overview), 2 (medium), 3 (detailed)."
            ) from exc

    raise ValueError(
        f"Unsupported diagram level '{raw_level}'. Supported levels: 1 (overview), 2 (medium), 3 (detailed)."
    )


def to_detail_level(level: DiagramLevel | int | DetailLevel | str) -> DetailLevel:
    """Map numeric/string level to legacy string detail level."""
    parsed = parse_diagram_level(level)
    return _DIAGRAM_LEVEL_TO_DETAIL_LEVEL[parsed]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class DiagramNode:
    """A single node in the architecture diagram."""
    id: str
    label: str
    kind: NodeKind = NodeKind.GENERIC
    group_id: Optional[str] = None      # cluster / subsystem this node belongs to
    metadata: Dict[str, str] = field(default_factory=dict, compare=False)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True, order=True)
class DiagramEdge:
    """A directed edge between two nodes."""
    source_id: str
    target_id: str
    label: Optional[str] = None
    kind: EdgeKind = EdgeKind.GENERIC
    metadata: Dict[str, str] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, order=True)
class DiagramGroup:
    """An optional grouping / cluster of nodes (e.g., a layer or subsystem)."""
    id: str
    label: str
    parent_id: Optional[str] = None     # nested clusters
    metadata: Dict[str, str] = field(default_factory=dict, compare=False)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class DiagramModel:
    """Pure, renderer-agnostic representation of an architecture diagram."""
    nodes: List[DiagramNode] = field(default_factory=list)
    edges: List[DiagramEdge] = field(default_factory=list)
    groups: List[DiagramGroup] = field(default_factory=list)
    title: str = ""
    detail_level: DetailLevel = DetailLevel.DETAILED
    metadata: Dict[str, str] = field(default_factory=dict)

    # ---- helpers --------------------------------------------------------

    def node_ids(self) -> Set[str]:
        return {n.id for n in self.nodes}

    def node_by_id(self, nid: str) -> Optional[DiagramNode]:
        for n in self.nodes:
            if n.id == nid:
                return n
        return None

    def group_by_id(self, gid: str) -> Optional[DiagramGroup]:
        for g in self.groups:
            if g.id == gid:
                return g
        return None

    def nodes_in_group(self, gid: str) -> List[DiagramNode]:
        return [n for n in self.nodes if n.group_id == gid]

    def validate(self) -> List[str]:
        """Return a list of validation warnings (empty = valid)."""
        issues: List[str] = []
        nids = self.node_ids()
        for e in self.edges:
            if e.source_id not in nids:
                issues.append(f"Edge source '{e.source_id}' not in nodes")
            if e.target_id not in nids:
                issues.append(f"Edge target '{e.target_id}' not in nodes")
        gids = {g.id for g in self.groups}
        for n in self.nodes:
            if n.group_id and n.group_id not in gids:
                issues.append(f"Node '{n.id}' references missing group '{n.group_id}'")
        return issues

    def sort_deterministic(self) -> None:
        """Sort nodes, edges, and groups for deterministic output."""
        self.nodes.sort(key=lambda n: n.id)
        self.edges.sort(key=lambda e: (e.source_id, e.target_id, e.label or ""))
        self.groups.sort(key=lambda g: g.id)


# ---------------------------------------------------------------------------
# DOT Parser  –  parse raw DOT string -> DiagramModel
# ---------------------------------------------------------------------------


def _safe_id(raw: str) -> str:
    """Normalise a string into a safe DOT / IR node identifier."""
    cleaned = normalize_text(raw).strip().strip('"').strip()
    # Replace non-alphanumeric with underscore
    safe = re.sub(r'[^A-Za-z0-9_]', '_', cleaned)
    # Ensure starts with letter/underscore
    if safe and safe[0].isdigit():
        safe = '_' + safe
    return safe or '_unknown'


def _unquote(s: str) -> str:
    """Remove surrounding quotes from a DOT string."""
    s = normalize_text(s).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return normalize_text(s[1:-1].replace('\\"', '"').replace('\\\\', '\\'))
    return s


def _extract_attr(attr_str: str, key: str) -> Optional[str]:
    """Extract a single attribute value from a DOT attribute string."""
    attr_str = normalize_text(attr_str)
    # Match key="value" or key=value
    m = re.search(
        rf'\b{re.escape(key)}\s*=\s*"([^"]*)"',
        attr_str,
        re.I,
    )
    if m:
        return normalize_text(m.group(1))
    m = re.search(
        rf'\b{re.escape(key)}\s*=\s*([^\s,;\]]+)',
        attr_str,
        re.I,
    )
    if m:
        return normalize_text(m.group(1))
    return None


def _guess_node_kind(node_id: str, label: str, attrs: str) -> NodeKind:
    """Heuristic: map node id/label/shape to a NodeKind."""
    combined = normalize_text(node_id + " " + label + " " + attrs).lower()
    if any(k in combined for k in ("database", "db", "postgres", "mysql", "mongo", "redis_db", "datastore", "storage")):
        return NodeKind.DATABASE
    if any(k in combined for k in ("queue", "kafka", "rabbitmq", "sqs", "pubsub", "mq", "broker")):
        return NodeKind.QUEUE
    if any(k in combined for k in ("cache", "redis", "memcache", "varnish")):
        return NodeKind.CACHE
    if any(k in combined for k in ("gateway", "api_gateway", "apigw", "api_gw", "ingress")):
        return NodeKind.GATEWAY
    if any(k in combined for k in ("load_balancer", "lb", "loadbalancer", "balancer", "alb", "nlb", "elb", "haproxy", "nginx_lb")):
        return NodeKind.LOADBALANCER
    if any(k in combined for k in ("cdn", "cloudfront", "akamai", "fastly")):
        return NodeKind.CDN
    if any(k in combined for k in ("client", "browser", "user", "mobile", "frontend", "web_app", "spa")):
        return NodeKind.CLIENT
    if any(k in combined for k in ("external", "third_party", "3rd_party", "vendor", "saas")):
        return NodeKind.EXTERNAL
    shape = _extract_attr(attrs, "shape") or ""
    if "cylinder" in shape or "doublecircle" in shape.lower():
        return NodeKind.DATABASE
    return NodeKind.GENERIC


def parse_dot_to_model(dot_source: Any) -> DiagramModel:
    """Parse a DOT source string into a DiagramModel.

    This is a pragmatic regex-based parser adequate for LLM-generated DOT.
    It handles:
    - Explicit node declarations with attributes
    - Edge declarations (->)
    - Subgraph clusters (subgraph cluster_*)
    - Implicit node references in edges

    It does NOT attempt to be a full DOT parser.
    """
    normalized_dot = normalize_text(dot_source)
    model = DiagramModel(detail_level=DetailLevel.DETAILED)
    seen_nodes: Dict[str, DiagramNode] = {}

    # --- Phase 1: Extract clusters / subgraphs ---
    cluster_re = re.compile(
        r'subgraph\s+(cluster_[A-Za-z0-9_]+)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        re.S | re.I,
    )
    for cm in cluster_re.finditer(normalized_dot):
        gid = cm.group(1)
        body = cm.group(2)
        glabel = _extract_attr(body, "label") or gid.replace("cluster_", "").replace("_", " ").title()
        model.groups.append(DiagramGroup(id=gid, label=normalize_text(glabel)))

    # Build group membership from cluster bodies
    group_node_map: Dict[str, str] = {}  # node_id -> group_id
    for cm in cluster_re.finditer(normalized_dot):
        gid = cm.group(1)
        body = cm.group(2)
        # Find node IDs mentioned in declarations or edges inside this cluster
        for nid_match in re.finditer(r'(?:^|\s|;)([A-Za-z_][A-Za-z0-9_]*)\s*[\[;{\n]', body):
            nid = nid_match.group(1).strip()
            if nid not in ('subgraph', 'graph', 'node', 'edge', 'digraph', 'label', 'style', 'color', 'fillcolor', 'fontname', 'fontsize', 'fontcolor', 'rankdir', 'splines', 'nodesep', 'ranksep', 'bgcolor', 'labelloc', 'shape', 'rank'):
                group_node_map[nid] = gid

    # --- Phase 2: Extract explicit node declarations ---
    # Match: nodeID [attr1=val1, ...];
    node_decl_re = re.compile(
        r'(?:^|;)\s*("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]*)\]\s*;?',
        re.M,
    )
    skip_keywords = {'graph', 'node', 'edge', 'digraph', 'subgraph', 'cluster'}

    for nm in node_decl_re.finditer(normalized_dot):
        raw_id = _unquote(nm.group(1))
        if raw_id.lower() in skip_keywords:
            continue
        attrs = nm.group(2)
        nid = _safe_id(raw_id)
        label = normalize_text(_extract_attr(attrs, "label") or raw_id.replace("_", " "))
        kind = _guess_node_kind(nid, label, attrs)
        gid = group_node_map.get(nid) or group_node_map.get(raw_id)
        node = DiagramNode(id=nid, label=label, kind=kind, group_id=gid)
        seen_nodes[nid] = node
        # Also map raw_id for edge resolution
        if raw_id != nid:
            seen_nodes[raw_id] = node

    # --- Phase 3: Extract edges ---
    edge_re = re.compile(
        r'("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*)'
        r'(?::([A-Za-z0-9_]+))?\s*'   # optional port
        r'->\s*'
        r'("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*)'
        r'(?::([A-Za-z0-9_]+))?\s*'   # optional port
        r'(?:\[([^\]]*)\])?\s*;?',
        re.M,
    )

    for em in edge_re.finditer(normalized_dot):
        src_raw = _unquote(em.group(1))
        tgt_raw = _unquote(em.group(3))
        attrs = em.group(5) or ""

        src_id = _safe_id(src_raw)
        tgt_id = _safe_id(tgt_raw)

        # Create implicit nodes if not seen
        for rid, sid in [(src_raw, src_id), (tgt_raw, tgt_id)]:
            if sid not in seen_nodes and rid not in seen_nodes:
                gid = group_node_map.get(sid) or group_node_map.get(rid)
                node = DiagramNode(
                    id=sid,
                    label=normalize_text(rid.replace("_", " ")),
                    kind=_guess_node_kind(sid, rid, ""),
                    group_id=gid,
                )
                seen_nodes[sid] = node

        raw_elabel = _extract_attr(attrs, "label")
        elabel = normalize_text(raw_elabel) if raw_elabel is not None else None
        ekind = EdgeKind.GENERIC
        if elabel:
            ll = elabel.lower()
            if any(k in ll for k in ("async", "event", "queue", "pub")):
                ekind = EdgeKind.ASYNC
            elif any(k in ll for k in ("data", "replicate", "sync_data", "backup")):
                ekind = EdgeKind.DATA

        model.edges.append(DiagramEdge(
            source_id=src_id,
            target_id=tgt_id,
            label=elabel,
            kind=ekind,
        ))

    # Deduplicate nodes (use the version with most info)
    final_nodes: Dict[str, DiagramNode] = {}
    for node in seen_nodes.values():
        if node.id not in final_nodes:
            final_nodes[node.id] = node
        else:
            existing = final_nodes[node.id]
            # Prefer the one with a group or non-GENERIC kind
            if (node.group_id and not existing.group_id) or \
               (node.kind != NodeKind.GENERIC and existing.kind == NodeKind.GENERIC):
                final_nodes[node.id] = node

    model.nodes = list(final_nodes.values())
    model.sort_deterministic()
    return model


# ---------------------------------------------------------------------------
# Overview builder  –  collapse detailed model into overview
# ---------------------------------------------------------------------------

def build_overview(
    detailed: DiagramModel,
    *,
    max_nodes: int = 15,
) -> Tuple[DiagramModel, Dict[str, List[str]]]:
    """Collapse a detailed DiagramModel into an overview.

    Strategy:
    1. If the detailed model has groups, collapse each group into a single
       overview node.  Ungrouped nodes remain as-is.
    2. If there are no groups, or the result still exceeds *max_nodes*,
       use a simple heuristic: keep "boundary" nodes (clients, gateways,
       external, LBs) and merge interior services by first-letter or kind.
    3. Aggregate parallel edges between the same pair of overview nodes.

    Returns
    -------
    (overview_model, mapping)
        mapping: overview_node_id -> [detailed_node_id, ...]
    """
    if len(detailed.nodes) <= max_nodes and not detailed.groups:
        # Already small enough; return a copy with level set to OVERVIEW
        overview = DiagramModel(
            nodes=list(detailed.nodes),
            edges=list(detailed.edges),
            groups=[],
            title=normalize_text(detailed.title),
            detail_level=DetailLevel.OVERVIEW,
            metadata=dict(detailed.metadata),
        )
        mapping = {n.id: [n.id] for n in detailed.nodes}
        return overview, mapping

    # --- Phase 1: Determine mapping detailed_node → overview_node ---
    mapping: Dict[str, List[str]] = {}      # overview_id -> [detailed_ids]
    detail_to_overview: Dict[str, str] = {} # detailed_id -> overview_id

    if detailed.groups:
        # Collapse each group into one node
        for group in detailed.groups:
            members = detailed.nodes_in_group(group.id)
            if not members:
                continue
            ov_id = "grp_" + _safe_id(group.id.replace("cluster_", ""))
            mapping[ov_id] = [m.id for m in members]
            for m in members:
                detail_to_overview[m.id] = ov_id

        # Ungrouped nodes stay as-is
        for node in detailed.nodes:
            if node.id not in detail_to_overview:
                detail_to_overview[node.id] = node.id
                mapping[node.id] = [node.id]
    else:
        # No groups — keep boundary nodes, merge interior by kind
        boundary_kinds = {NodeKind.CLIENT, NodeKind.GATEWAY, NodeKind.LOADBALANCER,
                          NodeKind.CDN, NodeKind.EXTERNAL}
        interior_by_kind: Dict[NodeKind, List[DiagramNode]] = {}

        for node in detailed.nodes:
            if node.kind in boundary_kinds or len(detailed.nodes) <= max_nodes:
                detail_to_overview[node.id] = node.id
                mapping[node.id] = [node.id]
            else:
                interior_by_kind.setdefault(node.kind, []).append(node)

        for kind, members in interior_by_kind.items():
            if len(members) <= 2:
                # Few enough to keep individually
                for m in members:
                    detail_to_overview[m.id] = m.id
                    mapping[m.id] = [m.id]
            else:
                # Merge them
                ov_id = f"group_{kind.value}"
                mapping[ov_id] = [m.id for m in members]
                for m in members:
                    detail_to_overview[m.id] = ov_id

    # --- Phase 2: Build overview nodes ---
    overview_nodes: Dict[str, DiagramNode] = {}
    for ov_id, detail_ids in mapping.items():
        if len(detail_ids) == 1:
            # Keep original node
            orig = detailed.node_by_id(detail_ids[0])
            if orig:
                overview_nodes[ov_id] = DiagramNode(
                    id=ov_id,
                    label=normalize_text(orig.label),
                    kind=orig.kind,
                    group_id=None,
                )
        else:
            # Find the group label or synthesize one
            # Check if these nodes share a group
            group_ids = {detailed.node_by_id(did).group_id
                         for did in detail_ids
                         if detailed.node_by_id(did) and detailed.node_by_id(did).group_id}
            if group_ids:
                gid = next(iter(group_ids))
                grp = detailed.group_by_id(gid)
                label = grp.label if grp else ov_id.replace("_", " ").title()
            else:
                label = ov_id.replace("_", " ").replace("group ", "").title()
                label = f"{label} ({len(detail_ids)})"

            overview_nodes[ov_id] = DiagramNode(
                id=ov_id,
                label=normalize_text(label),
                kind=NodeKind.CLUSTER,
                group_id=None,
            )

    # --- Phase 3: Build overview edges (aggregate parallel) ---
    edge_agg: Dict[Tuple[str, str], List[Optional[str]]] = {}
    for edge in detailed.edges:
        ov_src = detail_to_overview.get(edge.source_id, edge.source_id)
        ov_tgt = detail_to_overview.get(edge.target_id, edge.target_id)
        if ov_src == ov_tgt:
            continue  # skip internal edges within a collapsed group
        key = (ov_src, ov_tgt)
        edge_agg.setdefault(key, []).append(edge.label)

    overview_edges: List[DiagramEdge] = []
    for (src, tgt), labels in sorted(edge_agg.items()):
        # Aggregate labels
        unique_labels = sorted(set(lb for lb in labels if lb))
        if len(unique_labels) == 0:
            agg_label = None
        elif len(unique_labels) == 1:
            agg_label = unique_labels[0]
        elif len(unique_labels) <= 3:
            agg_label = normalize_text(" | ".join(unique_labels))
        else:
            agg_label = normalize_text(f"{unique_labels[0]} (+{len(unique_labels) - 1} more)")
        overview_edges.append(DiagramEdge(
            source_id=src,
            target_id=tgt,
            label=agg_label,
            kind=EdgeKind.GENERIC,
        ))

    overview = DiagramModel(
        nodes=list(overview_nodes.values()),
        edges=overview_edges,
        groups=[],
        title=normalize_text(detailed.title),
        detail_level=DetailLevel.OVERVIEW,
        metadata=dict(detailed.metadata),
    )
    overview.sort_deterministic()

    return overview, mapping


def build_medium(
    detailed: DiagramModel,
    *,
    max_nodes: int = 30,
) -> Tuple[DiagramModel, Dict[str, List[str]]]:
    """Collapse a detailed model into an intermediate (level-2) view."""
    if len(detailed.nodes) <= max_nodes:
        medium = DiagramModel(
            nodes=list(detailed.nodes),
            edges=list(detailed.edges),
            groups=list(detailed.groups),
            title=detailed.title,
            detail_level=DetailLevel.MEDIUM,
            metadata=dict(detailed.metadata),
        )
        medium.sort_deterministic()
        return medium, {n.id: [n.id] for n in detailed.nodes}

    boundary_kinds = {
        NodeKind.CLIENT,
        NodeKind.GATEWAY,
        NodeKind.LOADBALANCER,
        NodeKind.CDN,
        NodeKind.EXTERNAL,
        NodeKind.DATABASE,
    }

    mapping: Dict[str, List[str]] = {}
    detail_to_medium: Dict[str, str] = {}
    medium_label: Dict[str, str] = {}
    medium_kind: Dict[str, NodeKind] = {}

    for node in sorted(detailed.nodes, key=lambda n: n.id):
        if node.kind in boundary_kinds:
            mid = node.id
            label = node.label
            kind = node.kind
        else:
            grp = node.group_id or "global"
            mid = f"med_{_safe_id(grp)}_{node.kind.value}"
            grp_label = ""
            if node.group_id:
                group = detailed.group_by_id(node.group_id)
                grp_label = (group.label if group else node.group_id).strip()
            if grp_label:
                label = f"{grp_label} {node.kind.value.title()}"
            elif node.kind != NodeKind.GENERIC:
                label = f"{node.kind.value.title()} Services"
            else:
                label = "Internal Services"
            kind = NodeKind.CLUSTER

        mapping.setdefault(mid, []).append(node.id)
        detail_to_medium[node.id] = mid
        medium_label.setdefault(mid, normalize_text(label))
        medium_kind.setdefault(mid, kind)

    if len(mapping) > max_nodes:
        fallback, fallback_mapping = build_overview(detailed, max_nodes=max_nodes)
        fallback.detail_level = DetailLevel.MEDIUM
        fallback.metadata["collapsed_via"] = "overview_fallback"
        return fallback, fallback_mapping

    medium_nodes: Dict[str, DiagramNode] = {}
    for mid, member_ids in sorted(mapping.items()):
        if len(member_ids) == 1 and medium_kind[mid] != NodeKind.CLUSTER:
            original = detailed.node_by_id(member_ids[0])
            if original is not None:
                medium_nodes[mid] = DiagramNode(
                    id=mid,
                    label=normalize_text(original.label),
                    kind=original.kind,
                    group_id=None,
                )
                continue

        medium_nodes[mid] = DiagramNode(
            id=mid,
            label=medium_label[mid],
            kind=medium_kind[mid],
            group_id=None,
        )

    edge_agg: Dict[Tuple[str, str], List[str]] = {}
    for edge in detailed.edges:
        src = detail_to_medium.get(edge.source_id, edge.source_id)
        tgt = detail_to_medium.get(edge.target_id, edge.target_id)
        if src == tgt:
            continue
        edge_agg.setdefault((src, tgt), []).append(normalize_text(edge.label))

    medium_edges: List[DiagramEdge] = []
    for (src, tgt), labels in sorted(edge_agg.items()):
        cleaned = sorted({lb for lb in labels if lb})
        if not cleaned:
            agg_label: Optional[str] = None
        elif len(cleaned) == 1:
            agg_label = cleaned[0]
        elif len(cleaned) <= 3:
            agg_label = " | ".join(cleaned)
        else:
            agg_label = f"{cleaned[0]} (+{len(cleaned) - 1} more)"
        medium_edges.append(DiagramEdge(source_id=src, target_id=tgt, label=agg_label))

    medium = DiagramModel(
        nodes=list(medium_nodes.values()),
        edges=medium_edges,
        groups=[],
        title=normalize_text(detailed.title),
        detail_level=DetailLevel.MEDIUM,
        metadata=dict(detailed.metadata),
    )
    medium.sort_deterministic()
    return medium, mapping


def build_diagram_model(
    detailed: DiagramModel,
    level: DiagramLevel | int | str | DetailLevel = DiagramLevel.OVERVIEW,
    *,
    overview_max_nodes: int = 15,
    medium_max_nodes: int = 30,
) -> Tuple[DiagramModel, Dict[str, List[str]]]:
    """Build a level-specific model while keeping rendering logic independent."""
    parsed_level = parse_diagram_level(level)

    if parsed_level == DiagramLevel.OVERVIEW:
        model, mapping = build_overview(detailed, max_nodes=overview_max_nodes)
    elif parsed_level == DiagramLevel.MEDIUM:
        model, mapping = build_medium(detailed, max_nodes=medium_max_nodes)
    else:
        model = DiagramModel(
            nodes=list(detailed.nodes),
            edges=list(detailed.edges),
            groups=list(detailed.groups),
            title=normalize_text(detailed.title),
            detail_level=DetailLevel.DETAILED,
            metadata=dict(detailed.metadata),
        )
        model.sort_deterministic()
        mapping = {n.id: [n.id] for n in model.nodes}

    model.metadata["diagram_level"] = str(int(parsed_level))
    model.metadata["detail_level"] = model.detail_level.value
    return model, mapping


# ---------------------------------------------------------------------------
# Focused expansion  –  expand a single overview node
# ---------------------------------------------------------------------------

def build_expanded_view(
    detailed: DiagramModel,
    overview_mapping: Dict[str, List[str]],
    focus_node_id: str,
) -> DiagramModel:
    """Build a focused expansion of a single overview cluster.

    Shows all detailed nodes that belong to *focus_node_id*,
    plus any external nodes they connect to (shown as stubs with
    their overview-level label).

    Returns a DiagramModel at DETAILED level containing only the
    focused subgraph.
    """
    detail_ids = set(overview_mapping.get(focus_node_id, []))
    if not detail_ids:
        raise ValueError(f"Unknown overview node: {focus_node_id!r}")

    # Reverse mapping: detail_id -> overview_id
    detail_to_overview: Dict[str, str] = {}
    for ov_id, dids in overview_mapping.items():
        for did in dids:
            detail_to_overview[did] = ov_id

    # Collect all edges touching our focus set
    internal_edges: List[DiagramEdge] = []
    context_node_ids: Set[str] = set()

    for edge in detailed.edges:
        src_in = edge.source_id in detail_ids
        tgt_in = edge.target_id in detail_ids
        if src_in or tgt_in:
            internal_edges.append(edge)
            if not src_in:
                context_node_ids.add(edge.source_id)
            if not tgt_in:
                context_node_ids.add(edge.target_id)

    # Build nodes
    focus_nodes: List[DiagramNode] = []
    for node in detailed.nodes:
        if node.id in detail_ids:
            focus_nodes.append(node)
        elif node.id in context_node_ids:
            # External context node — show with a distinct label
            ov_id = detail_to_overview.get(node.id, node.id)
            focus_nodes.append(DiagramNode(
                id=node.id,
                label=normalize_text(f"[ext] {node.label}"),
                kind=NodeKind.EXTERNAL,
                group_id=None,
            ))

    expanded = DiagramModel(
        nodes=focus_nodes,
        edges=internal_edges,
        groups=[],
        title=normalize_text(f"Expanded: {focus_node_id}"),
        detail_level=DetailLevel.DETAILED,
        metadata={"focus_node": focus_node_id},
    )
    expanded.sort_deterministic()
    return expanded
