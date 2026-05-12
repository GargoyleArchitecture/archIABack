
import os
import threading
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
from langgraph.store.memory import InMemoryStore
from src.graph.state import GraphState

# Setup Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("graph")

# ========== Resources ==========

llm = get_chat_model(temperature=0.0)

# ========== RAG Trace ==========

_rag_trace_store: dict[str, dict] = {}
_rag_trace_session: str | None = None
_rag_trace_lock = threading.Lock()


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
    with _rag_trace_lock:
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
        return _get_retriever().invoke(*a, **kw)

retriever = _LazyRetriever()

# State-graph builder. Checkpointer (AsyncSqliteSaver) y store (InMemoryStore)
# se inyectan via build_graph() desde el lifespan de FastAPI (ver src/main.py).
builder = StateGraph(GraphState)

# Holders de la instancia compilada y el store. Se fijan al inicio del lifespan
# y se consumen via get_graph()/get_store() desde los call-sites.
_graph_holder: dict = {"instance": None}
_store_holder: dict = {"instance": None}


def set_graph(graph_instance) -> None:
    """Fija la instancia del grafo compilado. Llamar desde el lifespan."""
    _graph_holder["instance"] = graph_instance


def get_graph():
    """Devuelve la instancia del grafo compilado. Lanza si lifespan no corrió."""
    g = _graph_holder["instance"]
    if g is None:
        raise RuntimeError(
            "Graph not initialized. The FastAPI lifespan must run first."
        )
    return g


def set_store(store_instance) -> None:
    """Fija la instancia del Store cross-thread. Llamar desde el lifespan."""
    _store_holder["instance"] = store_instance


def get_store():
    """Devuelve el Store cross-thread. Lanza si lifespan no corrio."""
    s = _store_holder["instance"]
    if s is None:
        raise RuntimeError(
            "Store not initialized. The FastAPI lifespan must run first."
        )
    return s


def make_inmemory_store() -> InMemoryStore:
    """Factory del Store cross-thread.

    InMemoryStore es ephemeral por proceso. Cuando LangGraph publique un
    backend SQLite oficial para Store (o se decida adoptar
    `langgraph-checkpoint-postgres` tambien para Store), reemplazar aqui.
    """
    return InMemoryStore()

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
