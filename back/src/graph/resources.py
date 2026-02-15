
import os
import requests
import logging
from pathlib import Path
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from dotenv import load_dotenv, find_dotenv
# Automatically find and load .env regardless of where the script is started
load_dotenv(find_dotenv())

from src.services.llm_factory import get_chat_model
from src.rag_agent import get_retriever

# (Opcional) GCP Vision for image compare – protegido con try/except
try:
    from vertexai.generative_models import GenerativeModel
    from vertexai.preview.generative_models import Image
    _HAS_VERTEX = True
except Exception:
    _HAS_VERTEX = False
    GenerativeModel = None
    Image = None

# LangGraph builder + checkpointer
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.graph.state import GraphState

# Setup Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("graph")

# ========== Resources ==========

llm = get_chat_model(temperature=0.0)

# ===== RAG trace (per-request via session id) =====
_rag_trace_store: dict[str, dict] = {}
_rag_trace_session: str | None = None

def rag_trace_set_session(session_id: str | None) -> None:
    global _rag_trace_session
    _rag_trace_session = session_id

def rag_trace_reset(session_id: str | None) -> None:
    if session_id:
        _rag_trace_store[session_id] = {
            "attempted": False,
            "hit_count": 0,
            "queries": [],
            "sources": [],
        }

def rag_trace_get(session_id: str | None) -> dict:
    if not session_id:
        return {"attempted": False, "hit_count": 0, "queries": [], "sources": []}
    t = _rag_trace_store.get(session_id) or {}
    return {
        "attempted": bool(t.get("attempted")),
        "hit_count": int(t.get("hit_count") or 0),
        "queries": list(t.get("queries") or []),
        "sources": list(t.get("sources") or []),
    }

def rag_trace_record(*, query: str | None = None, docs: list | None = None, session_id: str | None = None) -> None:
    sid = session_id or _rag_trace_session
    if not sid:
        return
    t = _rag_trace_store.get(sid) or {"attempted": False, "hit_count": 0, "queries": [], "sources": []}
    t["attempted"] = True
    if query:
        t["queries"].append(query)
    if docs:
        t["hit_count"] = int(t.get("hit_count") or 0) + len(docs)
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            title = md.get("source_title") or md.get("title") or "doc"
            page = md.get("page_label") or md.get("page")
            path = md.get("source_path") or md.get("source") or ""
            page_str = f" (p.{page})" if page is not None else ""
            t["sources"].append(f"{title}{page_str} — {path}")
    _rag_trace_store[sid] = t

# Lazy retriever: initialized on first access to avoid import-time OpenAI errors
_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever

# Property-like access — modules that import `retriever` will get this proxy
class _LazyRetriever:
    """Proxy that delays retriever creation until first method call."""
    def __getattr__(self, name):
        return getattr(_get_retriever(), name)
    def invoke(self, *a, **kw):
        # Always trace any retriever invocation (global RAG signal).
        query = a[0] if a else kw.get("query")
        docs = _get_retriever().invoke(*a, **kw)
        try:
            rag_trace_record(query=query if isinstance(query, str) else None, docs=list(docs or []))
        except Exception:
            pass
        return docs

retriever = _LazyRetriever()

# State-graph builder & checkpointer
sqlite_saver = MemorySaver()
builder = StateGraph(GraphState)

# Sesión HTTP con retries y timeouts
def _make_http() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        connect=3,
        read=3,
        status=3,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET","POST"]),
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "ArchIA/diagram-orchestrator"})
    return s

_HTTP = _make_http()
