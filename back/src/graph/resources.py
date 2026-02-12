
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
