
import re
import math
from pathlib import Path
from langchain_core.tools import tool
from src.graph.resources import llm, retriever, _HAS_VERTEX, Image, GenerativeModel
from src.graph.state import investigatorSchema, evaluatorSchema
from src.graph.consts import (
    EVAL_THEORY_PREFIX, EVAL_VIABILITY_PREFIX, 
    EVAL_NEEDS_PREFIX, ANALYZE_PREFIX
)
from src.graph.utils import _clip_text

@tool
def LLM(prompt: str) -> dict:
    """Researcher centrado en ADD/ADD 3.0.
    Devuelve un dict con [definition, useCases, examples] según investigatorSchema."""
    return llm.with_structured_output(investigatorSchema).invoke(prompt)

@tool
def LLMWithImages(image_path: str) -> str:
    """Analiza diagramas de arquitectura en imágenes (si Vertex AI está disponible)."""
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
def local_RAG(prompt: str) -> str:
    """Responde con documentos locales (RAG) sobre tácticas/ADD/performance.
    Devuelve síntesis breve seguida de un bloque SOURCES para la UI."""
    q = (prompt or "").strip()
    synonyms = []
    if re.search(r"\badd\b", q, re.I):
        synonyms += ["Attribute-Driven Design", "ADD 3.0",
                     "architecture design method ADD", "Bass Clements Kazman ADD",
                     "quality attribute scenarios ADD"]
    if re.search(r"scalab|latenc|throughput|performance|tactic", q, re.I):
        synonyms += ["performance and scalability tactics", "latency tactics",
                     "scalability tactics", "architectural tactics performance"]

    queries = [q] + [f"{q} — {s}" for s in synonyms]
    docs_all = []
    seen_ids = set()

    for qq in queries:
        try:
            for d in retriever.invoke(qq):
                # Basic dedup by source+page
                doc_id = f"{d.metadata.get('source_path')}_{d.metadata.get('page')}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs_all.append(d)
                if len(docs_all) >= 8:
                    break
        except Exception:
            pass
        if len(docs_all) >= 8:
            break

    # preview solo 2 y cada uno 400 chars
    preview = []
    for i, d in enumerate(docs_all[:2], 1):
        snip = (d.page_content or "").replace("\n", " ").strip()
        snip = (snip[:400] + "…") if len(snip) > 400 else snip
        preview.append(f"[{i}] {snip}")

    # fuentes máximo 6
    src_lines = []
    for d in docs_all[:6]:
        title = d.metadata.get("title") or Path(d.metadata.get("source_path", "")).stem or "doc"
        page = d.metadata.get("page_label") or d.metadata.get("page")
        src = d.metadata.get("source_path") or d.metadata.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        line = f"- {title}{page_str} — {src}"
        src_lines.append(_clip_text(line, 60))

    return "\n\n".join(preview) + "\n\nSOURCES:\n" + "\n".join(src_lines)

# ===== Evaluator tools =====

@tool
def theory_tool(prompt: str) -> dict:
    """Evalúa corrección teórica vs buenas prácticas (patrones, tácticas, vistas)."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_THEORY_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def viability_tool(prompt: str) -> dict:
    """Evalúa viabilidad (coste, complejidad, operatividad, riesgos)."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_VIABILITY_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def needs_tool(prompt: str) -> dict:
    """Valida alineación con necesidades/ASRs y traza decisiones a requerimientos."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_NEEDS_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def analyze_tool(image_path: str, image_path2: str) -> str:
    """Compara dos diagramas de arquitectura (si Vertex AI está disponible)."""
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
