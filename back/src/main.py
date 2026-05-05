# -*- coding: utf-8 -*-
# src/main.py

import logging
from typing import Optional
from pathlib import Path

import json
import os, re, sqlite3, base64

log = logging.getLogger("main")

# ===================== UTF-8 Encoding Fix =======================
def fix_utf8_encoding(text: str) -> str:
    """
    Fix double-encoded UTF-8 text.
    When UTF-8 bytes are misinterpreted as Latin-1 and re-encoded,
    characters like "ó" become "Ã³" and "á" become "Ã¡".
    This function detects and reverses that corruption.
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    try:
        # Check if text has the pattern of double-encoded UTF-8
        if re.search(r'[\u00C0-\u00DF][\u0080-\u00BF]', text):
            # Convert to bytes as if it were Latin-1, then decode as UTF-8
            fixed = text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
            return fixed if fixed else text
    except Exception:
        pass
    
    return text

def fix_utf8_recursive(obj):
    """Recursively fix UTF-8 encoding in all strings within an object."""
    if isinstance(obj, str):
        return fix_utf8_encoding(obj)
    elif isinstance(obj, list):
        return [fix_utf8_recursive(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: fix_utf8_recursive(v) for k, v in obj.items()}
    return obj

def _sse(data: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


from dotenv import load_dotenv
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

from fastapi import UploadFile, File, Form, HTTPException, Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware


from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import (
    build_graph,
    get_graph,
    set_graph,
    set_store,
    make_inmemory_store,
)
from src.rag_agent import create_or_load_vectorstore
from src.memory import (
    init as memory_init,
    get as memory_get,
    set_kv as memory_set,
    load_arch_flow,
    save_arch_flow,
    project_key,
)
from src.graph.utils import is_explicit_asr_request
from src.services.doc_ingest import extract_pdf_text
from src.ledger.store import load_ledger as _load_ledger, compute_active_view as _compute_active_view
memory_init()

# ===================== Deteccion simple de idioma (ES/EN) ==========================
def detect_lang(q: str) -> str:
    ql = (q or "").lower()
    if re.search(r"[\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00bf\u00a1]", ql):
        return "es"
    if re.search(r"\b(what|how|why|when|which|where|who|the|and|or|if|is|are|can|do|does|should|would)\b", ql): return "en"
    ascii_ratio = sum(1 for c in q if ord(c) < 128) / max(1, len(q))
    return "en" if ascii_ratio > 0.97 else "es"

# ===================== Lifespan ==========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = Path(__file__).resolve().parent.parent / "state_db" / "graph_checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        store = make_inmemory_store()
        compiled = build_graph(saver, store=store)
        set_graph(compiled)
        set_store(store)
        try:
            create_or_load_vectorstore()
            print("[startup] RAG listo")
        except Exception as e:
            print(f"[startup] RAG init omitido: {e}")
        print(f"[startup] Checkpointer listo: {db_path}")
        yield
        print("[shutdown] Cerrando app...")

# Una sola instancia de FastAPI
app = FastAPI(title="ArquIA API", lifespan=lifespan)

# ===================== UTF-8 Middleware =======================
class UTF8Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Asegurar que la respuesta especifica UTF-8
        if "content-type" in response.headers:
            content_type = response.headers["content-type"]
            if "application/json" in content_type and "charset" not in content_type:
                response.headers["content-type"] = "application/json; charset=utf-8"
        return response

app.add_middleware(UTF8Middleware)

# ===================== Paths ==========================
BACK_DIR = Path(__file__).resolve().parent.parent  # .../back/
IMAGES_DIR = BACK_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = BACK_DIR / "docs_uploads"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_DIR = BACK_DIR / "feedback_db"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DB_PATH = FEEDBACK_DIR / "feedback.db"

# ===================== DB Feedback ======================
def init_feedback_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(FEEDBACK_DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS message_feedback (
            session_id   TEXT NOT NULL,
            message_id   INTEGER NOT NULL,
            thumbs_up    INTEGER DEFAULT 0,
            thumbs_down  INTEGER DEFAULT 0,
            PRIMARY KEY (session_id, message_id)
        )
        """
    )
    conn.commit()
    return conn

feedback_conn = init_feedback_db()

def get_next_message_id(session_id: str) -> int:
    cur = feedback_conn.cursor()
    cur.execute("SELECT MAX(message_id) FROM message_feedback WHERE session_id = ?", (session_id,))
    row = cur.fetchone()
    return (row[0] or 0) + 1

def upsert_feedback(session_id: str, message_id: int, up: int = 0, down: int = 0):
    feedback_conn.execute(
        "INSERT OR REPLACE INTO message_feedback (session_id, message_id, thumbs_up, thumbs_down) VALUES (?, ?, ?, ?)",
        (session_id, message_id, up, down),
    )
    feedback_conn.commit()

def update_feedback(session_id: str, message_id: int, up: int, down: int):
    feedback_conn.execute(
        "UPDATE message_feedback SET thumbs_up = ?, thumbs_down = ? WHERE session_id = ? AND message_id = ?",
        (up, down, session_id, message_id),
    )
    feedback_conn.commit()

# ===================== CORS ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== Helpers de tema ===================
def _normalize_topic(xx: str) -> str:
    x = (xx or "").lower()
    if "latenc" in x or "latencia" in x: return "latency"
    if "scalab" in x or "escalabilidad" in x: return "scalability"
    if "availab" in x or "disponibilidad" in x: return "availability"
    if "perform" in x or "rendim" in x: return "performance"
    return ""

def _extract_topic_from_text(q: str) -> str:
    return _normalize_topic(q)

def _needs_topic_hint(q: str) -> bool:
    low = (q or "").lower()
    mentions_tactics = bool(re.search(r"\btactic|\btactica|\btacticas|\bt\u00e1ctica|\bt\u00e1cticas", low))
    has_topic = bool(_extract_topic_from_text(low))
    return mentions_tactics and not has_topic

# ===================== ASR helpers =======================
ASR_HEAD_RE = re.compile(
    r"\b(ASR|Architecture[-\s]?Significant[-\s]?Requirement|Requisit[oa]\s+Significativ[oa]\s+de\s+Arquitectura)\b[:\uFF1A]?",
    re.I,
)

def _looks_like_make_asr(msg: str) -> bool:
    return is_explicit_asr_request(msg)

def _extract_asr_from_message(msg: str) -> str:
    if not msg: return ""
    m = re.search(r"(?:review|evaluate|revisar|evalua\w*)\s+(?:this|este|esta)?\s*asr\s*[:\uFF1A]\s*(.+)$", msg, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"(?:review|evaluate|revisar|evalua\w*)\s+(?:this|este|esta)?\s*asr[\s,)\-:]*\s*(.+)$", msg, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"\basr\s*[:\uFF1A]\s*(.+)$", msg, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"(?:review|evaluate|revisar|evalua\w*)\s+(?:this|este|esta)?\s*asr\s*\((.+)\)", msg, re.I | re.S)
    if m: return m.group(1).strip()
    return ""

def _extract_asr_from_result_text(text: str) -> str:
    if not text: return ""
    m = re.search(r"```asr\s*([\s\S]*?)```", text, re.I)
    if m: return m.group(1).strip()
    if re.search(r"\b(Summary|Context|Design\s+tactics|Trade[-\s]?offs|Acceptance\s+criteria|Validation\s+plan)\b", text, re.I):
        return text.strip()
    m = ASR_HEAD_RE.search(text)
    if m:
        start = m.start()
        asr = text[start:]
        asr = re.split(r"\n\s*#{1,6}\s|\n\s*(?:Rationale|Razonamiento|Conclusiones)\s*[:\uFF1A]", asr, maxsplit=1)[0]
        return asr.strip()
    m = re.search(r"(?:^|\n)\s*[-*]\s*ASR\s*[:\uFF1A]\s*(.+)", text, re.I)
    if m: return m.group(1).strip()
    return ""

def _wants_diagram_of_that_asr(msg: str) -> bool:
    if not msg: return False
    low = msg.lower()
    wants_diagram = any(k in low for k in ["diagram", "diagrama"])
    mentions_component = any(k in low for k in ["component diagram", "diagram component", "componentes", "de componentes"])
    mentions_that_asr = any(k in low for k in ["that asr", "ese asr", "esa asr", "dicho asr", "ese requisito"])
    return (wants_diagram and mentions_that_asr) or (mentions_component and mentions_that_asr)

def _wants_style(txt: str) -> bool:
    low = (txt or "").lower()
    keys = [
        # EN
        "architecture style",
        "architectural style",
        "architecture styles",
        "architectural styles",
        "style for this asr",
        "styles for this asr",
        "style for the asr",
        "what style", "which style",
        # ES
        "estilo de arquitectura",
        "estilos de arquitectura",
        "estilo arquitectonico", "estilo arquitect\u00f3nico",
        "estilos arquitectonicos", "estilos arquitect\u00f3nicos",
        "estilos para este asr",
        "que estilo", "qu\u00e9 estilo",
    ]
    # Tambien capturamos frases donde se combinan "style" y "asr".
    return (
        any(k in low for k in keys)
        or ("style" in low and "asr" in low)
        or ("estilos" in low and "arquitect" in low)
        or ("estilo" in low and "arquitect" in low)
    )


def _wants_tactics(txt: str) -> bool:
    low = (txt or "").lower()
    keys = [
        "tactica", "tacticas", "t\u00e1ctica", "t\u00e1cticas",
        "tactic", "tactics",
        "estrategia", "estrategias", "strategy", "strategies",
        "como cumplir", "c\u00f3mo cumplir",
        "how to satisfy", "how to meet", "how to achieve"
    ]
    return any(k in low for k in keys)

def _wants_deployment(txt: str) -> bool:
    low = (txt or "").lower()
    keys = [
        "despliegue", "deployment", "deployment diagram",
        "diagrama de despliegue",
        "plantuml", "graphviz", "dot"
    ]
    return any(k in low for k in keys)


# ===================== Health ===========================
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== /diagrams (workflow export) ==============
from fastapi import Query, Response

@app.get("/diagrams")
def diagrams(format: str = Query("dot", regex="^(dot|svg)$")):
    """Export the LangGraph workflow graph for documentation / draw.io import.

    Query params:
        format: ``dot`` (default) or ``svg``

    This endpoint is SEPARATE from the diagram_agent node, which generates
    architecture diagrams *for the user* via LLM + Graphviz.
    """
    from src.services.diagram_export import export_workflow

    try:
        result = export_workflow(fmt=format)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if format == "svg":
        return Response(
            content=result,
            media_type="image/svg+xml; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="workflow.svg"'},
        )
    # DOT
    return Response(
        content=result,
        media_type="text/vnd.graphviz; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="workflow.dot"'},
    )


# ===================== /diagram/export (user architecture diagrams) ==============
@app.get("/diagram/export")
async def diagram_export(
    session_id: str = Query(..., description="Session that produced the diagram"),
    format: str = Query("svg", regex="^(svg|dot|dot_drawio|drawio)$"),
    detail_level: str = Query("overview", regex="^(overview|detailed)$"),
    level: Optional[str] = Query(None, description="Diagram level: 1=overview, 2=medium, 3=detailed"),
    focus: Optional[str] = Query(None, description="Overview node ID to expand (optional)"),
    project_id: Optional[str] = Query(None, description="Project scope for ledger lookup"),
):
    """Export the last generated architecture diagram in the requested format.

    Query params:
        session_id   : Session that produced the diagram (required).
        format       : ``svg`` | ``dot`` | ``dot_drawio`` | ``drawio`` (default: svg).
        detail_level : ``overview`` | ``detailed`` (legacy; default: overview).
        level        : ``1`` | ``2`` | ``3`` (optional, preferred).
        focus        : An overview node ID to expand (optional; requires detail_level=detailed).

    The ``dot_drawio`` format produces a flattened DOT file safe for draw.io import
    (no clusters, no compound edges, no ports, no HTML labels).

    The ``drawio`` format produces a native .drawio (mxGraph XML) file.
    """
    from src.services.diagram_ir import (
        DiagramLevel,
        build_diagram_model,
        build_expanded_view,
        parse_diagram_level,
        parse_dot_to_model,
        to_detail_level,
    )
    from src.services.diagram_render import (
        render_dot as render_dot_from_ir,
        render_dot_drawio,
        render_drawio,
        render_svg as render_svg_from_dot,
        render_svg_async as render_svg_async_from_dot,
    )

    # P5: read from ledger; fallback to arch_flow for pre-migration sessions
    user_id = session_id
    project_id = (project_id or "").strip() or None

    ledger = _load_ledger(user_id, project_id)
    active = _compute_active_view(ledger)
    diagram_decision = active.get("diagram")

    if diagram_decision:
        payload = diagram_decision.get("payload") or {}
        dot_code = payload.get("dot") or payload.get("dot_raw") or ""
        _detail_level_hint = payload.get("focus") or "overview"
        _overview_mapping = payload.get("mapping") or {}
    else:
        arch_flow = load_arch_flow(user_id, project_id)
        last_diagram = arch_flow.get("last_diagram") or {}
        dot_code = last_diagram.get("dot") or last_diagram.get("dot_raw") or ""
        _detail_level_hint = last_diagram.get("detail_level") or "overview"
        _overview_mapping = last_diagram.get("overview_mapping") or {}

    if not dot_code:
        raise HTTPException(
            status_code=404,
            detail="No diagram found for this session. Generate one first via POST /message.",
        )

    try:
        requested_level = parse_diagram_level(level if level is not None else detail_level)
        if focus and requested_level != DiagramLevel.DETAILED:
            raise HTTPException(
                status_code=400,
                detail="Parameter 'focus' requires level=3 (or detail_level=detailed).",
            )

        detailed_model = parse_dot_to_model(dot_code)
        ir_model, level_mapping = build_diagram_model(
            detailed_model,
            requested_level,
            overview_max_nodes=15,
            medium_max_nodes=30,
        )
        detail_level_name = to_detail_level(requested_level).value

        if focus:
            mapping_for_focus = _overview_mapping or level_mapping
            ir_model = build_expanded_view(
                detailed_model,
                mapping_for_focus,
                focus,
            )
            detail_level_name = "detailed"

        engine = os.getenv("GRAPHVIZ_ENGINE", "dot").strip() or "dot"

        if format == "svg":
            dot_str = render_dot_from_ir(ir_model)
            svg_bytes = await render_svg_async_from_dot(dot_str, engine=engine)
            return Response(
                content=svg_bytes,
                media_type="image/svg+xml; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="architecture_{detail_level_name}.svg"'},
            )

        elif format == "dot":
            dot_str = render_dot_from_ir(ir_model)
            return Response(
                content=dot_str,
                media_type="text/vnd.graphviz; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="architecture_{detail_level_name}.dot"'},
            )

        elif format == "dot_drawio":
            flat_dot = render_dot_drawio(ir_model)
            return Response(
                content=flat_dot,
                media_type="text/vnd.graphviz; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="architecture_{detail_level_name}_drawio.dot"'},
            )

        elif format == "drawio":
            drawio_bytes = render_drawio(ir_model, engine=engine)
            return Response(
                content=drawio_bytes,
                media_type="application/xml; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="architecture_{detail_level_name}.drawio"'},
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        log.exception("diagram_export: unexpected error")
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}")

def _stream_post_process(
    result: dict,
    user_id: str,
    message_id: int,
    session_id: str,
    user_intent: str,
    made_asr: bool,
    message: str,
    arch_flow: dict,
    asr_key: str = "current_asr",
    project_id: Optional[str] = None,
) -> None:
    """Post-processing after the graph finishes: persist feedback, memory, arch_flow."""
    upsert_feedback(session_id=session_id, message_id=message_id, up=0, down=0)

    low = message.lower()
    if "latencia" in low:
        memory_set(user_id, "topic", "latencia")
    elif "escalabilidad" in low:
        memory_set(user_id, "topic", "escalabilidad")
    if "asr" in low:
        memory_set(user_id, "asr_notes", message)

    end_msg = result.get("endMessage", "") or ""
    asr_from_result = _extract_asr_from_result_text(end_msg)
    if asr_from_result:
        memory_set(user_id, asr_key, asr_from_result)
    elif made_asr and len(end_msg) > 80:
        memory_set(user_id, asr_key, end_msg.strip())

    if result.get("hasVisitedASR"):
        arch_flow["current_asr"] = memory_get(user_id, asr_key, "")
        arch_flow["quality_attribute"] = result.get(
            "asr_quality_attribute",
            arch_flow.get("quality_attribute", "")
        )
        arch_flow["add_context"] = result.get(
            "asr_context",
            arch_flow.get("add_context", "")
        )
        arch_flow["stage"] = "ASR"

    style_text = (
        result.get("style")
        or result.get("selected_style")
        or result.get("last_style")
    )
    if style_text and result.get("arch_stage") == "STYLE":
        arch_flow["style"] = style_text
        arch_flow["stage"] = "STYLE"

    tactics_json = result.get("tactics_struct") or None
    tactics_md   = result.get("tactics_md") or ""
    if user_intent == "tactics" and (tactics_json or tactics_md):
        arch_flow["tactics"] = tactics_json or []
        arch_flow["stage"] = "TACTICS"

    diagram_obj = result.get("diagram") or {}
    if diagram_obj.get("ok") and diagram_obj.get("dot"):
        # P5 guard: skip arch_flow write when ledger already captured the diagram
        _diagram_in_ledger = bool(
            (result.get("ledger_active") or {}).get("diagram")
        )
        if not _diagram_in_ledger:
            arch_flow["last_diagram"] = {
                "dot": diagram_obj.get("dot", ""),
                "dot_raw": diagram_obj.get("dot_raw", ""),
                "dot_drawio": diagram_obj.get("dot_drawio", ""),
                "detail_level": diagram_obj.get("detail_level", "overview"),
                "level": diagram_obj.get("level", 1),
                "overview_mapping": diagram_obj.get("overview_mapping"),
            }

    result_diagram_history = result.get("diagram_history") or {}
    if result_diagram_history:
        arch_flow["diagram_levels"] = {str(k): v for k, v in result_diagram_history.items()}

    if _wants_deployment(message):
        arch_flow["stage"] = "DEPLOYMENT"

    # Persist dual context (survives across turns once loaded)
    if result.get("project_context_text"):
        arch_flow["project_context_text"] = result["project_context_text"]
    if result.get("user_style_hint"):
        arch_flow["user_style_hint"] = result["user_style_hint"]

    save_arch_flow(user_id, arch_flow, project_id)


# ===================== /message =========================
@app.post("/message")
async def message(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    mode: str = Form(None),
    user_id: str = Form(None),
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
    project_id: Optional[str] = Form(None),
):
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")

    # F2-T2: validacion de mode (default seguro: professional).
    if mode is None or mode == "":
        mode = "professional"
    if mode not in ("tutor", "professional"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode!r}. Use 'tutor' or 'professional'.",
        )

    # F2-T2: user_id estable. Prioridad: Form > X-User-Id > session_id.
    user_id = (user_id or request.headers.get("X-User-Id") or session_id).strip()
    authorization_header = (request.headers.get("Authorization") or "").strip()
    authorization_parts = authorization_header.split()
    api_token = (
        authorization_parts[1]
        if len(authorization_parts) == 2 and authorization_parts[0].lower() == "bearer"
        else ""
    )
    project_id = (project_id or "").strip() or None          # normalizar: "" → None
    _raw_project_id = project_id
    try:
        arch_flow = load_arch_flow(user_id, project_id)
    except ValueError:
        log.warning(
            "Invalid project_id=%r received in /message; falling back to default arch_flow",
            _raw_project_id,
            exc_info=True,
        )
        project_id = None
        arch_flow = load_arch_flow(user_id, project_id)
    asr_key = project_key("current_asr", project_id)         # clave escopada por proyecto

    # ID incremental para feedback por mensaje
    message_id = get_next_message_id(session_id)

    # Usar un thread_id POR SESION, no por mensaje
    thread_id = session_id

    # --- Adjuntos (imagen o PDF) ---
    def _is_pdf(up):
        return bool(up and up.filename and (
            (up.content_type or "").lower().startswith("application/pdf")
            or up.filename.lower().endswith(".pdf")
        ))

    async def _save(up, dst_dir):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", up.filename or "file")
        p = dst_dir / f"{session_id}__{safe}"
        with open(p, "wb") as f:
            f.write(await up.read())
        return p

    image_path1, image_path2 = "", ""
    doc_context, doc_only = "", False

    # image1
    if image1 and image1.filename:
        if _is_pdf(image1):
            p = await _save(image1, DOCS_DIR)
            doc_context = extract_pdf_text(str(p), max_chars=8000) or ""
            doc_only = bool(doc_context.strip())
        else:
            p = await _save(image1, IMAGES_DIR)
            image_path1 = str(p)

    # image2
    if image2 and image2.filename:
        if _is_pdf(image2):
            p = await _save(image2, DOCS_DIR)
            extra = extract_pdf_text(str(p), max_chars=8000) or ""
            doc_context = (doc_context + "\n\n" + extra).strip() if extra else doc_context
            doc_only = bool(doc_context.strip())
        else:
            p = await _save(image2, IMAGES_DIR)
            image_path2 = str(p)

    # --- Turno actual como HumanMessage(s) ---
    turn_messages = [HumanMessage(content=message)]
    if image_path1:
        turn_messages.append(HumanMessage(content=f"[image_path_1] {image_path1}"))
    if image_path2:
        turn_messages.append(HumanMessage(content=f"[image_path_2] {image_path2}"))
    if doc_only and doc_context:
        # visible en el turno para trazabilidad
        turn_messages.append(HumanMessage(content=f"[DOCUMENT_EXCERPT]\n{doc_context[:4000]}"))

    # --- Memoria previa (MEJORADA) ---
    last_topic = memory_get(user_id, "topic", "")

    # âžœ FIX: antes se usaba uploaded_pdf_snippets (no existe). Usamos doc_context.
    pdf_context_turn = doc_context  # FIX

    if pdf_context_turn:
        # Persistimos en arch_flow.add_context (append no destructivo)
        af = dict(arch_flow)
        prev_ctx = (af.get("add_context") or "").strip()
        af["add_context"] = (prev_ctx + "\n\n" + pdf_context_turn).strip() if prev_ctx else pdf_context_turn
        save_arch_flow(user_id, af, project_id)
        arch_flow = af  # usarlo ya mismo

    stored_current_asr = (
        memory_get(user_id, asr_key, "")
        or arch_flow.get("current_asr", "")
    ).strip()

    memory_text = (
        f"Stage: {arch_flow.get('stage','')}\n"
        f"Quality Attribute: {arch_flow.get('quality_attribute','')}\n"
        f"Business / Context: {arch_flow.get('add_context','')}\n"
        f"Current ASR:\n{stored_current_asr}\n\n"
        f"Architecture style: {arch_flow.get('style','')}\n"
        f"Tactics so far: {arch_flow.get('tactics', [])}\n"
        f"User last topic: {last_topic}"
    ).strip() or "N/A"

    # --- ASR pegado por el usuario (si lo hay) ---
    asr_in_msg = _extract_asr_from_message(message)
    if asr_in_msg:
        memory_set(user_id, asr_key, asr_in_msg)
    made_asr = _looks_like_make_asr(message)

    # --- Config del grafo ---
    config = {"configurable": {"thread_id": thread_id, "api_token": api_token}, "recursion_limit": 20}
    user_lang = detect_lang(message)

    # --- Heuristicas locales ---
    explicit_asr_request = is_explicit_asr_request(message)
    topic_hint = _extract_topic_from_text(message) or _extract_topic_from_text(last_topic)
    msg_low = message.lower()
    force_rag = (
        explicit_asr_request or
        _needs_topic_hint(message) or
        bool(re.search(
            r"\b(add|qas|tactic|tactica|t\u00e1ctica|latenc|scalab|throughput|rendim|availability|disponib|diagrama|diagram)\b",
            msg_low
        ))
    )

    if doc_only:
        force_rag = False  # DOC-ONLY desactiva RAG

    has_existing_asr = bool(stored_current_asr)

    user_intent = "general"
    if explicit_asr_request:
        user_intent = "asr"
    elif not has_existing_asr:
        # Si aún no hay ASR, cualquier cosa va a ASR primero
        user_intent = "asr"
    elif _wants_style(message):
        # Ya hay ASR y el usuario está pidiendo estilos
        user_intent = "style"
    elif _wants_tactics(message):
        user_intent = "tactics"
    elif _wants_deployment(message):
        user_intent = "diagram"


    # --- Limpieza parcial del estado (sin borrar historial persistente del grafo) ---
    try:
        get_graph().update_state(config, {"values": {
            "endMessage": "",

            "diagram": {},  # FIX: dict vacío, no None
            "hasVisitedDiagram": False,
            "turn_messages": [],
            "requested_nodes": [],
            "pending_nodes": [],
            "completed_nodes": [],
            "current_asr": stored_current_asr,
        }})
    except Exception:
        pass

    # --- Invocación del grafo (streaming SSE) ---
    input_state = {
        "messages": turn_messages,
        "userQuestion": message,
        "localQuestion": "",
        # F2-T1 / F2-T2: campos de modo y perfilado.
        "mode": mode,
        "mode_suggestion": None,  # lo rellena classifier en F2-T4
        "user_id": user_id,
        "user_profile": {},  # se hidratara en F3-T3 (boot_node)
        "hasVisitedInvestigator": False,
        "hasVisitedEvaluator": False,
        "hasVisitedASR": False,
        "nextNode": "supervisor",
        "requested_nodes": [],
        "pending_nodes": [],
        "completed_nodes": [],
        "imagePath1": image_path1,
        "imagePath2": image_path2,
        "doc_only": doc_only,
        "doc_context": doc_context,
        "endMessage": "",
        "turn_messages": [],
        "retrieved_docs": [],
        "memory_text": memory_text,
        "suggestions": [],
        "language": user_lang,
        "intent": user_intent,
        "force_rag": force_rag,
        "topic_hint": topic_hint,
        "current_asr": stored_current_asr,
        "style": arch_flow.get("style", ""),
        "selected_style": arch_flow.get("style", ""),
        "last_style": arch_flow.get("style", ""),
        "arch_stage": arch_flow.get("stage", ""),
        "quality_attribute": arch_flow.get("quality_attribute", "") or topic_hint or last_topic,
        "add_context": arch_flow.get("add_context", ""),
        "tactics_list": arch_flow.get("tactics", []),
        "diagram_history": {int(k): v for k, v in (arch_flow.get("diagram_levels") or {}).items() if v},
        "project_id":             project_id or "",
        "user_id_for_prefs":      user_id,
        "project_context_text":   arch_flow.get("project_context_text", ""),
        "user_style_hint":        arch_flow.get("user_style_hint", ""),
        "project_context_loaded": bool(arch_flow.get("project_context_text", "")),
        "user_style_loaded":      bool(arch_flow.get("user_style_hint", "")),
    }

    # Capture variables needed by the generator closure
    _user_id      = user_id
    _message_id   = message_id
    _session_id   = session_id
    _thread_id    = thread_id
    _user_intent  = user_intent
    _made_asr     = made_asr
    _message      = message
    _arch_flow    = arch_flow
    _asr_key      = asr_key
    _project_id   = project_id

    async def generate():
        _final: dict = {}

        try:
            # graph.astream(stream_mode="updates") emits one event per node (just
            # that node's return dict), with no intermediate token/sub-chain events.
            # This avoids the serialisation overhead of astream_events(version="v2")
            # which copies the full state on every LLM token.
            async for chunk in get_graph().astream(input_state, config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if not isinstance(node_output, dict):
                        continue

                    if node_name == "asr":
                        yield _sse({
                            "type": "partial",
                            "node": "asr",
                            "endMessage": node_output.get("endMessage", ""),
                        })

                    elif node_name == "style_tactics_parallel":
                        yield _sse({
                            "type": "partial",
                            "node": "style_tactics",
                            "style": node_output.get("style", ""),
                            "tactics_md": node_output.get("tactics_md", ""),
                            "endMessage": node_output.get("endMessage", ""),
                        })

                    elif node_name == "diagram_agent":
                        yield _sse({
                            "type": "partial",
                            "node": "diagram",
                            "diagram": node_output.get("diagram", {}),
                        })

                    elif node_name == "unifier":
                        # unifier returns {**state, "endMessage": ...} so the full
                        # accumulated state (diagram, turn_messages, suggestions…)
                        # is available here. unifier -> END, so astream will
                        # terminate naturally on the next iteration.
                        _final = node_output
                        payload = fix_utf8_recursive({
                            "type": "complete",
                            "endMessage": (node_output.get("endMessage", "") or "").strip(),
                            "diagram":    node_output.get("diagram", {}),
                            "messages":   node_output.get("turn_messages", []),
                            "session_id": _session_id,
                            "message_id": _message_id,
                            "thread_id":  _thread_id,
                            "suggestions": node_output.get("suggestions", []),
                            # F2-T4: modo activo y sugerencia de cambio.
                            "mode":             node_output.get("mode", "professional"),
                            "mode_suggestion":  node_output.get("mode_suggestion"),
                        })
                        yield _sse(payload)

            yield "data: [DONE]\n\n"

        except GeneratorExit:
            # Client disconnected mid-stream; let the generator close cleanly.
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log.exception("Streaming pipeline error")
            yield _sse({"type": "error", "message": str(exc)})
            yield "data: [DONE]\n\n"
            return

        # Post-processing after stream closes (arch_flow persistence, memory saves)
        if _final:
            _stream_post_process(
                result=_final,
                user_id=_user_id,
                message_id=_message_id,
                session_id=_session_id,
                user_intent=_user_intent,
                made_asr=_made_asr,
                message=_message,
                arch_flow=_arch_flow,
                asr_key=_asr_key,
                project_id=_project_id,
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



# ===================== /feedback ========================
@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    message_id: int = Form(...),
    thumbs_up: int = Form(...),
    thumbs_down: int = Form(...),
):
    update_feedback(session_id=session_id, message_id=message_id, up=thumbs_up, down=thumbs_down)
    return {"status": "Feedback recorded successfully"}

# ===================== /test (mock) =====================
@app.post("/test")
async def test_endpoint(message: str = Form(...), file: UploadFile = File(None)):
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")

    test_response = {
        "diagram": {"ok": True, "format": "svg", "svg_b64": ""},
        "endMessage": "this is a response to " + message,
        "messages": [
            {"name": "Supervisor", "text": "Mensaje del supervisor"},
            {"name": "researcher", "text": "Mensaje del investigador"},
        ],
    }
    test_response = fix_utf8_recursive(test_response)
    return JSONResponse(content=test_response, media_type="application/json; charset=utf-8")


# ===================== /sessions — ledger endpoints (P1) =====================
from src.ledger import (
    load_ledger as _load_ledger,
    is_phase_complete as _is_phase_complete,
    Phase as _LedgerPhase,
    render_dossier as _render_dossier,
    render_dossier_compact as _render_dossier_compact,
)


@app.get("/sessions/{session_id}/dossier")
def get_session_dossier(
    session_id: str,
    lang: str = "es",
    focus: Optional[str] = None,
    compact: bool = False,
    project_id: Optional[str] = None,
):
    """Render the design dossier as Markdown for a session."""
    user_id = session_id
    pid = (project_id or "").strip() or None
    ledger = _load_ledger(user_id, pid)
    if compact:
        md = _render_dossier_compact(ledger, lang=lang)
    else:
        md = _render_dossier(ledger, lang=lang, focus=focus)
    return Response(content=md, media_type="text/markdown; charset=utf-8")


@app.get("/sessions/{session_id}/ledger")
def get_session_ledger(
    session_id: str,
    project_id: Optional[str] = None,
):
    """Return the raw Design Ledger JSON for a session."""
    user_id = session_id
    pid = (project_id or "").strip() or None
    return _load_ledger(user_id, pid)


@app.get("/sessions/{session_id}/phase")
def get_session_phase(
    session_id: str,
    project_id: Optional[str] = None,
):
    """Return current phase, iteration, pending advance and per-phase completion flags."""
    user_id = session_id
    pid = (project_id or "").strip() or None
    ledger = _load_ledger(user_id, pid)
    return {
        "current_phase":     ledger["current_phase"],
        "current_iteration": ledger["current_iteration"],
        "pending_advance":   ledger["pending_advance"],
        "completion": {
            p.value.lower(): _is_phase_complete(ledger, p)
            for p in [
                _LedgerPhase.ASR, _LedgerPhase.STYLE, _LedgerPhase.TACTICS,
                _LedgerPhase.DIAGRAM, _LedgerPhase.ANALYSIS,
            ]
        },
    }

