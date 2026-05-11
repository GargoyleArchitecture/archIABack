"""Tests F5-T3: corpus paralelo `bad_code_corpus` + nodo `inverse_rag_search`.

Todos los tests mockean el retriever para evitar dependencia real de
ChromaDB y embeddings. La validación end-to-end con el corpus construido es
parte del smoke manual (ver `back/build_bad_code_corpus.py`).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.graph.services.inverse_rag import (
    _clear_retriever_cache,
    get_bad_code_retriever,
)
from src.graph.nodes.routine.inverse_rag_search import inverse_rag_search_node


def setup_function():
    """Resetea el singleton entre tests para aislar estado."""
    _clear_retriever_cache()


# ==========================================================================
# Servicio singleton
# ==========================================================================

def test_returns_none_when_corpus_dir_missing(monkeypatch, tmp_path):
    """Si el directorio del corpus no existe, retorna None y NO intenta cargar Chroma."""
    monkeypatch.setenv("BAD_CODE_CHROMA_DIR", str(tmp_path / "does_not_exist"))
    assert get_bad_code_retriever() is None


def test_singleton_caches_subsequent_calls(monkeypatch, tmp_path):
    """Llamar dos veces no reintenta cargar (idempotente vía cache)."""
    monkeypatch.setenv("BAD_CODE_CHROMA_DIR", str(tmp_path / "does_not_exist"))
    first = get_bad_code_retriever()
    second = get_bad_code_retriever()
    assert first is None
    assert second is None  # mismo resultado, sin reintentar carga


# ==========================================================================
# Nodo `inverse_rag_search`
# ==========================================================================

def test_node_returns_empty_when_weakness_blank():
    """target_weakness vacío → raw_snippet='' sin invocar retriever."""
    out = inverse_rag_search_node({"target_weakness": ""})
    assert out["raw_snippet"] == ""


def test_node_returns_empty_when_retriever_unavailable(monkeypatch):
    """Corpus aún no construido → fallback graceful."""
    from src.graph.nodes.routine import inverse_rag_search as mod

    monkeypatch.setattr(mod, "get_bad_code_retriever", lambda k=2: None)
    out = inverse_rag_search_node({"target_weakness": "Caching"})
    assert out["raw_snippet"] == ""


def test_node_concatenates_top_2_hits(monkeypatch):
    """Con retriever válido, el nodo une los top-2 hits con separador."""
    from src.graph.nodes.routine import inverse_rag_search as mod

    fake_retriever = MagicMock()
    fake_retriever.invoke = MagicMock(
        return_value=[
            Document(page_content="bad code A\n\nConcept: x"),
            Document(page_content="bad code B\n\nConcept: y"),
            Document(page_content="bad code C\n\nConcept: z"),  # debe ignorarse (k=2)
        ]
    )
    monkeypatch.setattr(mod, "get_bad_code_retriever", lambda k=2: fake_retriever)

    out = inverse_rag_search_node({"target_weakness": "Caching"})
    assert "bad code A" in out["raw_snippet"]
    assert "bad code B" in out["raw_snippet"]
    assert "bad code C" not in out["raw_snippet"]
    assert "---" in out["raw_snippet"]  # separador entre hits
    fake_retriever.invoke.assert_called_once_with("Caching")


def test_node_falls_back_when_retriever_raises(monkeypatch):
    """Si el retriever lanza, raw_snippet='' (no rompe el subgrafo)."""
    from src.graph.nodes.routine import inverse_rag_search as mod

    fake_retriever = MagicMock()
    fake_retriever.invoke = MagicMock(side_effect=RuntimeError("Chroma 500"))
    monkeypatch.setattr(mod, "get_bad_code_retriever", lambda k=2: fake_retriever)

    out = inverse_rag_search_node({"target_weakness": "Caching"})
    assert out["raw_snippet"] == ""


def test_node_handles_empty_hits(monkeypatch):
    """Retriever devuelve [] (o None) → raw_snippet='' sin crash."""
    from src.graph.nodes.routine import inverse_rag_search as mod

    fake_retriever = MagicMock()
    fake_retriever.invoke = MagicMock(return_value=[])
    monkeypatch.setattr(mod, "get_bad_code_retriever", lambda k=2: fake_retriever)

    out = inverse_rag_search_node({"target_weakness": "Caching"})
    assert out["raw_snippet"] == ""


def test_node_strips_whitespace_from_weakness(monkeypatch):
    """target_weakness con espacios al borde se normaliza antes del query."""
    from src.graph.nodes.routine import inverse_rag_search as mod

    fake_retriever = MagicMock()
    fake_retriever.invoke = MagicMock(
        return_value=[Document(page_content="snippet")]
    )
    monkeypatch.setattr(mod, "get_bad_code_retriever", lambda k=2: fake_retriever)

    inverse_rag_search_node({"target_weakness": "  Caching  "})
    # Verifica que el espacio se removió antes de pasar al retriever
    fake_retriever.invoke.assert_called_once_with("Caching")
