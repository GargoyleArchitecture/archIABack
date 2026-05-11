"""Nodo `inverse_rag_search` del subgrafo RoutineGenerator (F5-T3).

Consulta el corpus paralelo `bad_code_corpus` (ChromaDB) usando
`target_weakness` como query y deja los top-k hits en `state["raw_snippet"]`
para que `synthesize_challenge` los use como semilla pedagógica.

Fallback graceful en cualquier error → `raw_snippet=""`. El subgrafo sigue
funcionando aunque:
- El corpus no esté construido todavía (directorio inexistente).
- ChromaDB falle al inicializar.
- El retriever lance al consultar.
- `target_weakness` venga vacío.

Esta tolerancia mantiene los tests F5-T1 funcionando sin cambios y
evita que un fallo del RAG inverso degrade la generación de retos.
"""
from __future__ import annotations

import logging

from src.graph.schemas.routine import RoutineState
from src.graph.services.inverse_rag import get_bad_code_retriever

log = logging.getLogger("routine.inverse_rag_search")

_TOP_K = 2


def inverse_rag_search_node(state: RoutineState) -> RoutineState:
    """Busca antipatrones relacionados con `target_weakness`."""
    weakness = (state.get("target_weakness") or "").strip()
    if not weakness:
        log.info("inverse_rag_search: empty target_weakness → raw_snippet=''")
        return {**state, "raw_snippet": ""}

    retriever = get_bad_code_retriever(k=_TOP_K)
    if retriever is None:
        log.info("inverse_rag_search: corpus not available → raw_snippet=''")
        return {**state, "raw_snippet": ""}

    try:
        docs = retriever.invoke(weakness)
    except Exception:
        log.exception(
            "inverse_rag_search: retriever raised; falling back to empty snippet"
        )
        return {**state, "raw_snippet": ""}

    docs = list(docs or [])[:_TOP_K]
    snippet = "\n\n---\n\n".join(d.page_content for d in docs)

    log.info(
        "inverse_rag_search: weakness='%s' hits=%d snippet_len=%d",
        weakness,
        len(docs),
        len(snippet),
    )
    return {**state, "raw_snippet": snippet}
