# src/main.py

from typing import Optional
from pathlib import Path

import os, re, sqlite3, base64

from dotenv import load_dotenv
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

from fastapi import UploadFile, File, Form, HTTPException, Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


from langchain_core.messages import HumanMessage
from src.graph import graph
from src.rag_agent import create_or_load_vectorstore
from src.memory import (
    init as memory_init,
    get as memory_get,
    set_kv as memory_set,
    load_arch_flow,
    save_arch_flow,
)
from src.services.doc_ingest import extract_pdf_text
memory_init()

# ===================== Detección simple de idioma (ES/EN) ==========================
def detect_lang(q: str) -> str:
    ql = (q or "").lower()
    if re.search(r"[áéíóúñ¿¡]", ql): return "es"
    if re.search(r"\b(what|how|why|when|which|where|who|the|and|or|if|is|are|can|do|does|should|would)\b", ql): return "en"
    ascii_ratio = sum(1 for c in q if ord(c) < 128) / max(1, len(q))
    return "en" if ascii_ratio > 0.97 else "es"


def _merge_context_text(base: str, incoming: str, max_chars: int = 4000) -> str:
    b = (base or "").strip()
    i = (incoming or "").strip()
    if not i:
        return b
    if not b:
        return i[:max_chars]
    if i in b:
        return b[:max_chars]
    merged = (b + "\n\n" + i).strip()
    return merged[-max_chars:]


def _build_context_summary(seed: dict, max_chars: int = 1400) -> str:
    active = seed.get("active_decisions") or {}
    facts = seed.get("context_facts") or {}
    lines = [
        f"Stage: {active.get('arch_stage', '')}",
        f"Quality Attribute: {active.get('quality_attribute', '')}",
        f"Current ASR: {active.get('current_asr', '')}",
        f"Architecture style: {active.get('style', '')}",
        f"Tactics: {active.get('tactics', [])}",
        f"Domain: {facts.get('domain', '')}",
        f"Constraints: {facts.get('constraints', [])}",
        f"NFR focus: {facts.get('nfr_focus', [])}",
        f"Business / Context: {seed.get('add_context', '')}",
    ]
    return "\n".join(lines).strip()[:max_chars]

# ===================== Lifespan ==========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        create_or_load_vectorstore()
        print("[startup] RAG listo")
    except Exception as e:
        print(f"[startup] RAG init omitido: {e}")
    yield
    print("[shutdown] Cerrando app...")

# Una sola instancia de FastAPI
app = FastAPI(title="ArquIA API", lifespan=lifespan)

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
    mentions_tactics = bool(re.search(r"\btactic|\btáctic|\btactica|\btáctica", low))
    has_topic = bool(_extract_topic_from_text(low))
    return mentions_tactics and not has_topic

# ===================== ASR helpers =======================
ASR_HEAD_RE = re.compile(
    r"\b(ASR|Architecture[-\s]?Significant[-\s]?Requirement|Requisit[oa]\s+Significativ[oa]\s+de\s+Arquitectura)\b[:：]?",
    re.I,
)

def _looks_like_make_asr(msg: str) -> bool:
    if not msg: return False
    low = msg.lower()
    return bool(re.search(r"\b(create|make|draft|write|generate|produce|compose)\b.*\b(asr)\b", low)) \
        or bool(re.search(r"\b(crea|haz|redacta|genera|produce)\b.*\b(asr)\b", low))

def _extract_asr_from_message(msg: str) -> str:
    if not msg: return ""
    m = re.search(r"(?:review|evaluate|revisar|evalua\w*)\s+(?:this|este|esta)?\s*asr\s*[:：]\s*(.+)$", msg, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"(?:review|evaluate|revisar|evalua\w*)\s+(?:this|este|esta)?\s*asr[\s,)\-:]*\s*(.+)$", msg, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"\basr\s*[:：]\s*(.+)$", msg, re.I | re.S)
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
        asr = re.split(r"\n\s*#{1,6}\s|\n\s*(?:Rationale|Razonamiento|Conclusiones)\s*[:：]", asr, maxsplit=1)[0]
        return asr.strip()
    m = re.search(r"(?:^|\n)\s*[-*]\s*ASR\s*[:：]\s*(.+)", text, re.I)
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
        "style for this asr",
        "styles for this asr",
        "style for the asr",
        "what style", "which style",
        # ES
        "estilo de arquitectura",
        "estilo arquitectónico",
        "estilos para este asr",
        "qué estilo", "que estilo",
    ]
    # también capturamos frases donde simplemente se combinan "style" y "asr"
    return any(k in low for k in keys) or ("style" in low and "asr" in low)


def _wants_asr(txt: str) -> bool:
    low = (txt or "").lower()
    keys = [
        "asr", "qas", "quality attribute scenario",
        "requisito significativo", "requisito arquitectonico",
        "genera asr", "genera un asr", "genera asrs", "create asr", "make asr",
    ]
    return any(k in low for k in keys) or _looks_like_make_asr(txt)


def _wants_tactics(txt: str) -> bool:
    low = (txt or "").lower()
    keys = [
        "táctica", "tactica", "tácticas", "tacticas",
        "tactic", "tactics",
        "estrategia", "estrategias", "strategy", "strategies",
        "cómo cumplir", "como cumplir",
        "how to satisfy", "how to meet", "how to achieve"
    ]
    return any(k in low for k in keys)

def _wants_deployment(txt: str) -> bool:
    low = (txt or "").lower()
    patterns = [
        r"\bdespliegue\b",
        r"\bdeployment\b",
        r"\bdeployment\s+diagram\b",
        r"\bdiagrama\s+de\s+despliegue\b",
        r"\bplantuml\b",
        r"\bmermaid\b",
    ]
    return any(re.search(p, low) for p in patterns)


# ===================== Health ===========================
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== /message =========================
@app.post("/message")
async def message(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
):
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")

    # Identidad simple por sesión
    user_id = request.headers.get("X-User-Id") or session_id
    arch_flow = load_arch_flow(user_id)

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

    # ➜ FIX: antes se usaba uploaded_pdf_snippets (no existe). Usamos doc_context.
    pdf_context_turn = doc_context  # FIX

    if pdf_context_turn:
        # Persistimos en arch_flow.add_context (append no destructivo)
        af = dict(arch_flow)
        prev_ctx = (af.get("add_context") or "").strip()
        af["add_context"] = (prev_ctx + "\n\n" + pdf_context_turn).strip() if prev_ctx else pdf_context_turn
        save_arch_flow(user_id, af)
        arch_flow = af  # usarlo ya mismo

    # ---- Context Contract v1 (source of truth per turn) ----
    response_lang = detect_lang(message)
    context_facts = {
        "domain": "",
        "constraints": [],
        "nfr_focus": [arch_flow.get("quality_attribute", "")] if arch_flow.get("quality_attribute") else [],
        "open_assumptions": [],
    }
    active_decisions = {
        "arch_stage": arch_flow.get("stage", ""),
        "current_asr": arch_flow.get("current_asr", ""),
        "quality_attribute": arch_flow.get("quality_attribute", ""),
        "style": arch_flow.get("style", ""),
        "tactics": arch_flow.get("tactics", []) or [],
        "decision_log": [],
    }
    turn_context = {
        "user_message": message,
        "intent": "general",
        "references": [],
        "attachments": {
            "images": int(bool(image_path1)) + int(bool(image_path2)),
            "pdfs": int(bool(doc_context.strip())),
        },
        "doc_only": doc_only,
    }
    state_seed = {
        "context_facts": context_facts,
        "active_decisions": active_decisions,
        "turn_context": turn_context,
        "arch_stage": arch_flow.get("stage", ""),
        "quality_attribute": arch_flow.get("quality_attribute", ""),
        "current_asr": arch_flow.get("current_asr", ""),
        "style": arch_flow.get("style", ""),
        "tactics_list": arch_flow.get("tactics", []) or [],
        "add_context": arch_flow.get("add_context", ""),
    }
    memory_text = _build_context_summary(state_seed) or "N/A"

    # --- ASR pegado por el usuario (si lo hay) ---
    asr_in_msg = _extract_asr_from_message(message)
    if asr_in_msg:
        memory_set(user_id, "current_asr", asr_in_msg)

    # --- Config base del grafo ---
    config_base = {"recursion_limit": 20}
    user_lang = response_lang

    # --- Heurísticas locales ---
    topic_hint = _extract_topic_from_text(message) or _extract_topic_from_text(last_topic)
    msg_low = message.lower()
    force_rag = (
        _needs_topic_hint(message) or
        bool(re.search(
            r"\b(add|qas|asr|tactic|táctica|latenc|scalab|throughput|rendim|availability|disponib|diagrama|diagram)\b",
            msg_low
        ))
    )

    if doc_only:
        force_rag = False  # DOC-ONLY desactiva RAG

    requested_outputs = []
    if _wants_asr(message):
        requested_outputs.append("asr")
    if _wants_style(message):
        requested_outputs.append("style")
    if _wants_tactics(message):
        requested_outputs.append("tactics")
    if _wants_deployment(message):
        requested_outputs.append("diagram")

    if not requested_outputs:
        if not arch_flow.get("current_asr"):
            requested_outputs = ["asr"]
        elif _wants_style(message):
            requested_outputs = ["style"]
        elif _wants_tactics(message):
            requested_outputs = ["tactics"]
        elif _wants_deployment(message):
            requested_outputs = ["diagram"]
        else:
            requested_outputs = ["general"]

    if (
        not arch_flow.get("current_asr")
        and any(x in requested_outputs for x in ["style", "tactics"])
        and "asr" not in requested_outputs
    ):
        requested_outputs = ["asr"] + requested_outputs

    ordered = [x for x in ["asr", "style", "tactics", "diagram"] if x in requested_outputs]
    run_intents = ordered if ordered else [requested_outputs[0]]

    def _clear_turn_state(run_config):
        reset_values = {
            "endMessage": "",
            "mermaidCode": "",
            "diagram": {},
            "hasVisitedDiagram": False,
            "turn_messages": [],
            "current_asr": memory_get(user_id, "current_asr", "") or arch_flow.get("current_asr", ""),
        }
        try:
            graph.update_state(run_config, reset_values)
        except Exception as e:
            try:
                graph.update_state(run_config, {"values": reset_values})
            except Exception as e2:
                print(f"[warn] state reset skipped: {e}; fallback failed: {e2}")

    def _invoke_once(forced_intent: str, current_flow: dict, run_config: dict):
        _clear_turn_state(run_config)
        current_asr = memory_get(user_id, "current_asr", "") or current_flow.get("current_asr", "")
        return graph.invoke(
            {
                "messages": turn_messages,
                "userQuestion": message,
                "localQuestion": "",
                "hasVisitedInvestigator": False,
                "hasVisitedCreator": False,
                "hasVisitedEvaluator": False,
                "hasVisitedASR": False,
                "nextNode": "supervisor",
                "imagePath1": image_path1,
                "imagePath2": image_path2,
                "doc_only": doc_only,
                "doc_context": doc_context,
                "endMessage": "",
                "mermaidCode": "",
                "turn_messages": [],
                "retrieved_docs": [],
                "memory_text": memory_text,
                "suggestions": [],
                "language": user_lang,
                "response_language": user_lang,
                "intent": forced_intent,
                "forced_intent": forced_intent,
                "force_rag": force_rag,
                "context_contract_version": "v1",
                "context_facts": context_facts,
                "active_decisions": {
                    "arch_stage": current_flow.get("stage", ""),
                    "current_asr": current_asr,
                    "quality_attribute": current_flow.get("quality_attribute", ""),
                    "style": current_flow.get("style", ""),
                    "tactics": current_flow.get("tactics", []) or [],
                    "decision_log": [],
                },
                "turn_context": {
                    **turn_context,
                    "intent": forced_intent,
                    "forced_intent": forced_intent,
                    "requested_outputs": requested_outputs,
                },
                "pending_questions": [],
                "context_summary": memory_text,
                "topic_hint": topic_hint,
                "current_asr": current_asr,
                "style": current_flow.get("style", ""),
                "selected_style": current_flow.get("style", ""),
                "last_style": current_flow.get("style", ""),
                "arch_stage": current_flow.get("stage", ""),
                "quality_attribute": current_flow.get("quality_attribute", ""),
                "add_context": current_flow.get("add_context", ""),
                "tactics_list": current_flow.get("tactics", []),
            },
            run_config,
        )

    results_by_intent = {}
    last_result = {}
    combined_turn_messages = []
    combined_suggestions = []
    for forced_intent in run_intents:
        if forced_intent in ("style", "tactics") and not (arch_flow.get("current_asr") or memory_get(user_id, "current_asr", "")):
            continue
        run_config = {
            **config_base,
            "configurable": {
                "thread_id": f"{thread_id}:{message_id}:{forced_intent}",
            },
        }
        try:
            result = _invoke_once(forced_intent, arch_flow, run_config)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Graph error: {e}")

        last_result = result
        results_by_intent[forced_intent] = result
        combined_turn_messages.extend(result.get("turn_messages", []) or [])
        for s in (result.get("suggestions", []) or []):
            if s and s not in combined_suggestions:
                combined_suggestions.append(s)

        end_msg_partial = result.get("endMessage", "") or ""
        asr_from_result = _extract_asr_from_result_text(end_msg_partial)
        if asr_from_result:
            memory_set(user_id, "current_asr", asr_from_result)
        elif forced_intent == "asr" and len(end_msg_partial) > 80:
            memory_set(user_id, "current_asr", end_msg_partial.strip())

        if result.get("hasVisitedASR") or forced_intent == "asr":
            arch_flow["current_asr"] = memory_get(user_id, "current_asr", "") or arch_flow.get("current_asr", "")
            arch_flow["quality_attribute"] = result.get("asr_quality_attribute", arch_flow.get("quality_attribute", ""))
            arch_flow["add_context"] = result.get("asr_context", arch_flow.get("add_context", ""))
            arch_flow["stage"] = "ASR"

        style_text = result.get("style") or result.get("selected_style") or result.get("last_style")
        if style_text and (result.get("arch_stage") == "STYLE" or forced_intent == "style"):
            arch_flow["style"] = style_text
            arch_flow["stage"] = "STYLE"

        tactics_json = result.get("tactics_struct") or None
        tactics_md = result.get("tactics_md") or ""
        if forced_intent == "tactics" and (tactics_json or tactics_md):
            arch_flow["tactics"] = tactics_json or arch_flow.get("tactics", [])
            arch_flow["stage"] = "TACTICS"

        updated_ctx = (result.get("add_context") or "").strip()
        if updated_ctx:
            arch_flow["add_context"] = _merge_context_text(arch_flow.get("add_context", ""), updated_ctx, max_chars=4000)

    result = last_result or {}
    user_intent = run_intents[-1] if run_intents else "general"

    # --- Feedback inicial ---
    upsert_feedback(session_id=session_id, message_id=message_id, up=0, down=0)

    # --- Actualiza memoria simple ---
    low = message.lower()
    if "latencia" in low:
        memory_set(user_id, "topic", "latencia")
    elif "escalabilidad" in low:
        memory_set(user_id, "topic", "escalabilidad")
    if "asr" in low:
        memory_set(user_id, "asr_notes", message)

    # ===================== DIAGRAMA =====================
    # Ya no generamos SVG/PNG ni usamos Kroki.
    # Solo devolvemos el script de Mermaid que construye el grafo con ASR + estilo + tácticas.
    diagram_obj = {}

    # Si el usuario pidió explícitamente un diagrama, marcamos el stage
    diagram_requested = "diagram" in run_intents
    if diagram_requested:
        arch_flow["stage"] = "DEPLOYMENT"

    # Persistimos el flujo ADD 3.0 actualizado (ASR, estilo, tácticas, stage, etc.)
    save_arch_flow(user_id, arch_flow)

    def _section_text(intent_key: str) -> str:
        return (results_by_intent.get(intent_key, {}) or {}).get("endMessage", "") or ""

    end_msg = result.get("endMessage", "") or ""
    if len(run_intents) > 1:
        sections = []
        asr_txt = _section_text("asr")
        style_txt = _section_text("style")
        tactics_txt = _section_text("tactics")
        if user_lang == "es":
            if asr_txt:
                sections.append("ASR propuesto:\n" + asr_txt)
            if style_txt:
                sections.append("Estilo(s) recomendado(s):\n" + style_txt)
            if tactics_txt:
                sections.append("Tacticas recomendadas:\n" + tactics_txt)
        else:
            if asr_txt:
                sections.append("Proposed ASR:\n" + asr_txt)
            if style_txt:
                sections.append("Recommended style(s):\n" + style_txt)
            if tactics_txt:
                sections.append("Recommended tactics:\n" + tactics_txt)
        if sections:
            end_msg = "\n\n---\n\n".join(sections)

    mermaid_code = ""
    if diagram_requested:
        mermaid_code = ((results_by_intent.get("diagram") or {}).get("mermaidCode") or result.get("mermaidCode") or "").strip()

    if diagram_requested and mermaid_code:
        if user_lang == "es":
            mermaid_help = (
                "\n\n---\n"
                "Aqui tienes el **script Mermaid** de este diagrama.\n"
                "Puedes copiarlo y pegarlo en Mermaid live (https://mermaid.live), "
                "en un plugin de Mermaid para VS Code o en cualquier renderizador compatible:\n\n"
                "```mermaid\n"
                f"{mermaid_code}\n"
                "```"
            )
        else:
            mermaid_help = (
                "\n\n---\n"
                "Here is the **Mermaid script** for this diagram.\n"
                "You can copy & paste it into the Mermaid live editor (https://mermaid.live), "
                "a VS Code Mermaid plugin, or any compatible renderer:\n\n"
                "```mermaid\n"
                f"{mermaid_code}\n"
                "```"
            )
        end_msg = (end_msg + mermaid_help).strip()

    else:
        end_msg = end_msg.strip()

    # --- Payload al front (no pisamos suggestions si las necesitas) ---
    clean_payload = {
        "endMessage": end_msg,
        "mermaidCode": mermaid_code,
        "diagram": diagram_obj,  # ahora siempre vacío; ya no mandamos SVG
        "messages": combined_turn_messages or result.get("turn_messages", []),
        "session_id": session_id,
        "message_id": message_id,
        "thread_id": thread_id,
        "suggestions": combined_suggestions or result.get("suggestions", []),
    }

    return clean_payload

    # --- Payload al front (no pisamos suggestions si las necesitas) ---
    clean_payload = {
        "endMessage": end_msg,
        "mermaidCode": mermaid_code,
        "diagram": diagram_obj,                # ahora puede venir vacío si user_intent == "diagram"
        "messages": result.get("turn_messages", []),
        "session_id": session_id,
        "message_id": message_id,
        "thread_id": thread_id,
        "suggestions": result.get("suggestions", []),
    }

    return clean_payload


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

    return {
        "mermaidCode": "flowchart LR\nA-->B",
        "endMessage": "this is a response to " + message,
        "messages": [
            {"name": "Supervisor", "text": "Mensaje del supervisor"},
            {"name": "researcher", "text": "Mensaje del investigador"},
        ],
    }
