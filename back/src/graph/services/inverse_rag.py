"""F5-T3: Singleton del retriever del corpus paralelo `bad_code_corpus`.

Lazy init: la carga del Chroma + embeddings es costosa, así que el primer
caller paga el costo y los siguientes reusan la instancia. Si el corpus
todavía no está construido (directorio inexistente) o falla la carga, el
singleton queda en `None` y el nodo `inverse_rag_search` cae a fallback
graceful (raw_snippet="").

`_clear_retriever_cache()` es solo para tests; no se exporta vía `__all__`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("inverse_rag")

# El servicio vive en back/src/graph/services/, así que parents[3] es back/.
_BACK_DIR = Path(__file__).resolve().parents[3]
_DEFAULT_DIR = str((_BACK_DIR / "chroma_bad_code").resolve())
_COLLECTION = "bad_code_corpus"

# Cache del retriever: `loaded` distingue "todavía no intenté" de "intenté y
# fallé/no había corpus" (en cuyo caso `value` es None y no debe reintentar).
_retriever_cache: dict = {"value": None, "loaded": False}


def get_bad_code_retriever(k: int = 2):
    """Retorna el retriever del corpus de mal código, o None si no está disponible.

    Idempotente: si ya se intentó cargar (con éxito o no), reutiliza el
    resultado en cache. Para forzar reload (p.ej. tras correr el script de
    ingesta sin reiniciar el proceso), llama `_clear_retriever_cache()`.
    """
    if _retriever_cache["loaded"]:
        return _retriever_cache["value"]

    persist_dir = os.environ.get("BAD_CODE_CHROMA_DIR", _DEFAULT_DIR)
    _retriever_cache["loaded"] = True  # marca el intento (incluso si falla)

    if not Path(persist_dir).exists():
        log.info("inverse_rag corpus dir does not exist: %s", persist_dir)
        _retriever_cache["value"] = None
        return None

    try:
        # Imports diferidos para no pagar el costo si el corpus no existe.
        try:
            from langchain_chroma import Chroma  # type: ignore
        except Exception:  # pragma: no cover
            from langchain_community.vectorstores import Chroma  # type: ignore

        from src.rag_agent import _embeddings as _embeddings_factory

        vdb = Chroma(
            collection_name=_COLLECTION,
            embedding_function=_embeddings_factory(),
            persist_directory=persist_dir,
        )
        retriever = vdb.as_retriever(search_kwargs={"k": k})
        _retriever_cache["value"] = retriever
        log.info(
            "inverse_rag retriever loaded (collection=%s, dir=%s, k=%d)",
            _COLLECTION,
            persist_dir,
            k,
        )
        return retriever
    except Exception:
        log.exception("inverse_rag failed to initialize retriever")
        _retriever_cache["value"] = None
        return None


def _clear_retriever_cache() -> None:
    """Resetea el cache. SOLO para tests."""
    _retriever_cache["value"] = None
    _retriever_cache["loaded"] = False
