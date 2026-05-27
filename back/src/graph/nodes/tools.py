import re
import logging
import chromadb
from pathlib import Path
from typing import Annotated, Optional
from langchain_core.tools import tool
# pyrefly: ignore [missing-import]
from langgraph.prebuilt import InjectedState
from src.graph.resources import llm, retriever, _HAS_VERTEX, Image, GenerativeModel, rag_trace_record
from src.graph.state import investigatorSchema, evaluatorSchema, GraphState
from src.graph.consts import (
    EVAL_THEORY_PREFIX, EVAL_VIABILITY_PREFIX,
    EVAL_NEEDS_PREFIX, ANALYZE_PREFIX
)
from src.graph.utils import _clip_text
from src.rag_agent import get_retriever
from evrag.config import EVRAG_CONFIG
from evrag.indexer import EVRAGIndexer

log = logging.getLogger("main")

@tool
def LLM(prompt: str) -> dict:
    """Researcher centrado en ADD/ADD 3.0.
    Devuelve un dict con [definition, useCases, examples] segun investigatorSchema."""
    return llm.with_structured_output(investigatorSchema).invoke(prompt)

@tool
def LLMWithImages(image_path: str) -> str:
    """Analiza diagramas de arquitectura en imagenes (si Vertex AI esta disponible)."""
    if not _HAS_VERTEX:
        return "Image analysis unavailable: Vertex AI SDK not installed."
    try:
        image = Image.load_from_file(image_path)
        generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
        resp = generative_multimodal_model.generate_content([
            ("Identify software-architecture tactics/patterns present. "
             "If the image is a class diagram, list classes/relations and OOD principles."),
            image
        ])
        return str(resp)
    except Exception as e:
        return f"Error analyzing image: {e}"

@tool
def local_RAG(prompt: str, quality_attribute: str = "general") -> str:
    """Responde con documentos locales (RAG) sobre tacticas/ADD/performance.
    Devuelve sintesis breve seguida de un bloque SOURCES para la UI.
    Pasa quality_attribute (e.g., 'escalabilidad', 'latencia') para obtener
    resultados filtrados por el indice de ese atributo de calidad."""
    q = (prompt or "").strip()
    synonyms = []
    if re.search(r"\badd\b", q, re.I):
        synonyms += ["Attribute-Driven Design", "ADD 3.0",
                     "architecture design method ADD", "Bass Clements Kazman ADD",
                     "quality attribute scenarios ADD"]
    if re.search(r"scalab|latenc|throughput|performance|tactic", q, re.I):
        synonyms += ["performance and scalability tactics", "latency tactics",
                     "scalability tactics", "architectural tactics performance"]

    queries = [q] + [f"{q} - {s}" for s in synonyms]
    docs_all = []
    seen_ids = set()

    _retriever = get_retriever(k=8)

    for qq in queries:
        try:
            for d in _retriever.invoke(qq):
                doc_id = f"{d.metadata.get('source_path')}_{d.metadata.get('page')}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs_all.append(d)
                if len(docs_all) >= 8:
                    break
        except Exception as e:
            log.error(f"[local_RAG] ERROR al buscar '{qq}': {e}")
            pass
        if len(docs_all) >= 8:
            break
    rag_trace_record(query=q, docs=docs_all)

    preview = []
    for i, d in enumerate(docs_all[:2], 1):
        snip = (d.page_content or "").replace("\n", " ").strip()
        snip = (snip[:400] + "...") if len(snip) > 400 else snip
        preview.append(f"[{i}] {snip}")

    src_lines = []
    for d in docs_all[:6]:
        title = d.metadata.get("title") or Path(d.metadata.get("source_path", "")).stem or "doc"
        page = d.metadata.get("page_label") or d.metadata.get("page")
        src = d.metadata.get("source_path") or d.metadata.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        line = f"- {title}{page_str} - {src}"
        src_lines.append(_clip_text(line, 60))

    return "\n\n".join(preview) + "\n\nTEXT_SOURCES:\n" + "\n".join(src_lines)

# ===== Evaluator tools =====

@tool
def theory_tool(prompt: str) -> dict:
    """Evalua correccion teorica vs buenas practicas (patrones, tacticas, vistas)."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_THEORY_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def viability_tool(prompt: str) -> dict:
    """Evalua viabilidad (coste, complejidad, operatividad, riesgos)."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_VIABILITY_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def needs_tool(prompt: str) -> dict:
    """Valida alineacion con necesidades/ASRs y traza decisiones a requerimientos."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_NEEDS_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def analyze_tool(image_path: str, image_path2: str) -> str:
    """Compara dos diagramas de arquitectura (si Vertex AI esta disponible)."""
    if not _HAS_VERTEX:
        return "Diagram compare unavailable: Vertex AI SDK not installed."
    try:
        image = Image.load_from_file(image_path)
        image2 = Image.load_from_file(image_path2)
        generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
        resp = generative_multimodal_model.generate_content([ANALYZE_PREFIX, image, image2])
        return str(resp)
    except Exception as e:
        return f"Error analyzing diagrams: {e}"


@tool
def video_RAG(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Busca en contenido de videos (transcripciones y frames) usando EVRAG.
    Detecta escenas relevantes y visualiza diagramas explicados por el profesor.
    """
    evrag_mode = state.get("evrag_mode", "hybrid")
    indexer = EVRAGIndexer()
    
    # Realizar consulta multimodal/híbrida
    results = indexer.query_multimodal(query, top_k=5, mode=evrag_mode)
    
    if not results["frames"] and not results["segments"]:
        return "No relevant video content found for your query."

    output_lines = ["**Video Content Analysis (EVRAG)**"]
    
    # 1. Procesar Segmentos de Texto (Transcripciones)
    if results["segments"]:
        output_lines.append("\nRelevant Transcript Segments:")
        for i, seg in enumerate(results["segments"], 1):
            start_str = _format_seconds(seg["start_time"])
            end_str = _format_seconds(seg["end_time"])
            output_lines.append(f"[{i}] **{seg['video_id']}** ({start_str}-{end_str}): {seg['text']}")

    # 2. Procesar Frames (Imágenes / Descripciones)
    if results["frames"]:
        output_lines.append("\nRelevant Visual Scenes:")
        for i, frame in enumerate(results["frames"], 1):
            f_type = frame.get("type", "visual")
            desc = frame.get("description", "No description available")
            output_lines.append(f"[{i}] Frame at index {frame['frame_index']} (Source: {f_type})")
            if f_type == "descriptive":
                output_lines.append(f"    *Description:* {desc}")

    # Build structured VIDEO_SOURCES for frontend parsing
    _base = Path(__file__).resolve().parents[3]
    frames_dir = _base / "videos" / "frames"
    available_frames = list(frames_dir.glob("*.jpg")) if frames_dir.exists() else []

    video_source_lines = []
    # Usamos tanto frames como segmentos para generar fuentes
    all_sources = []
    
    for seg in results["segments"]:
        video_id = seg["video_id"]
        start = seg["start_time"]
        end = seg["end_time"]
        frame_file = _find_closest_frame(video_id, start, available_frames)
        all_sources.append(f"- {video_id}.mp4 | {start} | {end} | {frame_file} | {video_id}.mp4")

    for frame in results["frames"]:
        video_id = frame["video_id"]
        frame_path = Path(frame["frame_path"])
        # Aproximar tiempo basado en índice (heurística 1 frame cada 2-5 segs)
        time_est = frame["frame_index"] * 5 
        all_sources.append(f"- {video_id}.mp4 | {time_est} | {time_est+5} | {frame_path.name} | {video_id}.mp4")

    # Eliminar duplicados de fuentes y limitar a 6
    unique_sources = list(dict.fromkeys(all_sources))[:6]

    return "\n".join(output_lines) + "\n\nVIDEO_SOURCES:\n" + "\n".join(unique_sources)


def _format_seconds(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _find_closest_frame(video_id: str, start_time: float, available_frames: list) -> str:
    """Find the closest frame file for a given video_id and start_time."""
    matching = [f for f in available_frames if f.name.startswith(f"{video_id}_scene_")]
    if not matching: return ""
    best_file = matching[0].name
    min_delta = float('inf')
    target_frame = start_time * 30
    for f in matching:
        m = re.search(r'_frame_(\d+)', f.name)
        if m:
            frame_num = int(m.group(1))
            delta = abs(frame_num - target_frame)
            if delta < min_delta:
                min_delta = delta
                best_file = f.name
    return best_file
