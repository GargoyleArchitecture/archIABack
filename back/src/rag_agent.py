# src/rag_agent.py
from __future__ import annotations

import os
from pathlib import Path

# Usa el paquete nuevo si está instalado; si no, cae al community.
try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings

# ================== Paths / Config ==================

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../back
DEFAULT_CHROMA_DIR = str((PROJECT_ROOT / "chroma_db").resolve())

# Modelo de embeddings por env, con default razonable
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Nombre de colección (consistente con el build)
COLLECTION_NAME = "arquia"

# Singleton del vectorstore
_VDB: Chroma | None = None


def _embeddings():
    """Selecciona embeddings según proveedor (Azure/OpenAI)."""
    import os
    # Azure
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        from langchain_openai import AzureOpenAIEmbeddings
        dep = (
            os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        if not dep:
            raise ValueError("Usando Azure: define AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT (deployment de embeddings).")
        return AzureOpenAIEmbeddings(
            azure_deployment=dep,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    # OpenAI (pública/compatibles)
    from langchain_openai import OpenAIEmbeddings
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, chunk_size=10)


def create_or_load_vectorstore() -> Chroma:
    """
    Carga la BD de Chroma ya persistida (construida por build_vectorstore.py).
    Si la carpeta está vacía, la instancia se crea igualmente (sin data).
    """
    global _VDB
    if _VDB is not None:
        return _VDB

    persist_directory = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    print(f"[RAG] persist_directory = {persist_directory}")

    # Solo cargar (el build se hace con back/build_vectorstore.py)
    _VDB = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=persist_directory,
    )
    # Nota: si no hay datos aún, _VDB está “vacío” pero funcional.
    return _VDB


# src/rag_agent.py  (reemplaza tu get_retriever por este)
def get_retriever(title: str | list[str] | None = None, k: int = 6):
    """
    Devuelve un retriever del vector store.
    - Si `title` es string: filtra por igualdad exacta en metadata.title
    - Si `title` es lista: usa $in para cualquiera
    """
    vectorstore = create_or_load_vectorstore()
    if isinstance(title, list) and title:
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": {"title": {"$in": title}}}
        )
    if isinstance(title, str) and title:
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": {"title": {"$eq": title}}}
        )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_indexed_retriever(
    quality_attribute: str | None = None,
    content_type: str | None = None,
    k: int = 6,
):
    """
    Devuelve un retriever filtrado por quality_attribute y/o content_type.

    - Si quality_attribute es None o "general" → no filtra por QA.
    - Si content_type es None o "general"       → no filtra por tipo.
    - Si ambos presentes y válidos              → usa $and en el filtro.
    - Si ninguno                                → retriever sin filtros (igual a get_retriever()).

    Esto permite retrieval indexado por atributo de calidad y tipo de contenido,
    sin romper el comportamiento existente cuando se llama sin parámetros.
    """
    vectorstore = create_or_load_vectorstore()

    has_qa = bool(quality_attribute and quality_attribute != "general")
    has_ct = bool(content_type and content_type != "general")

    if has_qa and has_ct:
        filter_dict = {
            "$and": [
                {"quality_attribute": {"$eq": quality_attribute}},
                {"content_type": {"$eq": content_type}},
            ]
        }
    elif has_qa:
        filter_dict = {"quality_attribute": {"$eq": quality_attribute}}
    elif has_ct:
        filter_dict = {"content_type": {"$eq": content_type}}
    else:
        filter_dict = None

    search_kwargs: dict = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    return vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def rebuild_vectorstore():
    """
    Helper opcional: si quieres reconstruir en caliente, borra el dir
    y vuelve a cargar (pero el pipeline recomendado es usar build_vectorstore.py).
    """
    import shutil
    global _VDB
    persist_directory = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    if os.path.isdir(persist_directory):
        print(f"[RAG] Removing existing Chroma DB at {persist_directory}")
        shutil.rmtree(persist_directory, ignore_errors=True)
    _VDB = None
    return create_or_load_vectorstore()
