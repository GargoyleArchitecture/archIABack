"""Tools especificas del Modo Tutor (F2-T5).

Solo se registran cuando state["mode"] == "tutor" en el investigador.
Filtran el RAG existente por metadata.kind == "glossary" para responder
con definiciones cortas y citas a documentacion oficial.
"""
from __future__ import annotations
from langchain_core.tools import tool
from src.rag_agent import get_indexed_retriever


def _glossary_search(query: str, k: int = 4) -> list:
    """Helper: retriever filtrado por kind='glossary'."""
    retr = get_indexed_retriever(quality_attribute="general", k=k)
    try:
        # Filtro Mongo-like compatible con Chroma del proyecto.
        retr.search_kwargs = {
            **(retr.search_kwargs or {}),
            "filter": {"kind": {"$eq": "glossary"}},
        }
        return list(retr.invoke(query))
    except Exception:
        return []


@tool
def lookup_glossary(term: str) -> dict:
    """Busca una definicion corta del termino en el corpus de glosario.

    Devuelve {term, definition, source}. Si no hay entradas glossary en el
    corpus, retorna definition vacia y nota explicativa en source.
    """
    docs = _glossary_search(term, k=2)
    if not docs:
        return {
            "term": term,
            "definition": "",
            "source": "(no glossary entries available; fallback to local_RAG)",
        }
    d = docs[0]
    md = d.metadata or {}
    title = md.get("source_title") or md.get("title") or "doc"
    page = md.get("page_label") or md.get("page")
    page_str = f" (p.{page})" if page is not None else ""
    return {
        "term": term,
        "definition": (d.page_content or "").strip()[:500],
        "source": f"{title}{page_str}",
    }


@tool
def cite_documentation(concept: str) -> dict:
    """Devuelve referencia a documentacion oficial sobre el concepto.

    Devuelve {concept, citation, snippet}. Usa el mismo filtro glossary.
    """
    docs = _glossary_search(concept, k=3)
    if not docs:
        return {
            "concept": concept,
            "citation": "(no glossary entries available)",
            "snippet": "",
        }
    items = []
    for d in docs:
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page = md.get("page_label") or md.get("page")
        path = md.get("source_path") or md.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        items.append(f"- {title}{page_str} - {path}")
    return {
        "concept": concept,
        "citation": "\n".join(items),
        "snippet": (docs[0].page_content or "").strip()[:400],
    }
