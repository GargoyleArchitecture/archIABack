# back/build_bad_code_corpus.py
"""F5-T3: Script CLI de ingesta del corpus de mal código a ChromaDB.

Lee `back/data/bad_code_seed.json` (90 entradas: 30 por categoría) y crea
una colección ChromaDB paralela `bad_code_corpus` consultable por el nodo
`inverse_rag_search` del subgrafo RoutineGenerator.

Uso:
    cd back
    python build_bad_code_corpus.py            # idempotente; refuse si ya tiene items
    python build_bad_code_corpus.py --rebuild  # borra el directorio y reingesta

Ubicación del store: `BAD_CODE_CHROMA_DIR` env (default: `back/chroma_bad_code/`).
Embeddings: reutiliza `_embeddings_factory` de `src/rag_agent.py` (Azure/OpenAI).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv, find_dotenv

# Resolver paths primero (este archivo vive en back/, así que sys.path debe
# incluir back/ para importar `src.rag_agent`).
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(find_dotenv())

# Imports posteriores al sys.path.insert
from langchain_core.documents import Document  # noqa: E402

try:
    from langchain_chroma import Chroma  # noqa: E402
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma  # noqa: E402

from src.rag_agent import _embeddings as _embeddings_factory  # noqa: E402


# ================== Config ==================

SEED_PATH = BASE_DIR / "data" / "bad_code_seed.json"
PERSIST_DIR = Path(
    os.environ.get("BAD_CODE_CHROMA_DIR", str(BASE_DIR / "chroma_bad_code"))
)
COLLECTION_NAME = "bad_code_corpus"
REQUIRED_CATEGORIES = ("acoplamiento", "escalabilidad", "latencia")
MIN_PER_CATEGORY = 30


def _build_documents(items: List[dict]) -> List[Document]:
    """Construye Document objects:
    - page_content combina código + concept + fix_hint para que el embedding
      capture las tres señales y matchee bien con queries de tipo weakness.
    - metadata expone los campos por separado para filtrado posterior.
    """
    docs: List[Document] = []
    for item in items:
        page_content = (
            f"{item['code']}\n\n"
            f"Concept: {item['concept']}\n"
            f"Fix: {item['fix_hint']}"
        )
        metadata = {
            "id": item["id"],
            "category": item["category"],
            "concept": item["concept"],
            "severity": item["severity"],
            "fix_hint": item["fix_hint"],
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def _validate_items(items: List[dict]) -> None:
    """Valida estructura mínima y conteos por categoría (criterio F5-T3)."""
    counts: dict[str, int] = {}
    required_keys = {"id", "category", "concept", "severity", "fix_hint", "code"}
    for it in items:
        missing = required_keys - set(it.keys())
        if missing:
            raise SystemExit(
                f"[bad_code] Item {it.get('id', '?')} missing keys: {missing}"
            )
        counts[it["category"]] = counts.get(it["category"], 0) + 1

    print(f"[bad_code] items: {len(items)}, by category: {counts}")
    for cat in REQUIRED_CATEGORIES:
        if counts.get(cat, 0) < MIN_PER_CATEGORY:
            raise SystemExit(
                f"[bad_code] Category '{cat}' has only {counts.get(cat, 0)} "
                f"items (need >= {MIN_PER_CATEGORY})."
            )


def _corpus_already_populated(persist_dir: Path) -> bool:
    """Heurística simple: el directorio existe y NO está vacío."""
    if not persist_dir.exists():
        return False
    return any(persist_dir.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Borra el directorio existente antes de reingerir.",
    )
    args = parser.parse_args()

    print(f"[bad_code] seed_path       = {SEED_PATH}")
    print(f"[bad_code] persist_dir     = {PERSIST_DIR}")
    print(f"[bad_code] collection_name = {COLLECTION_NAME}")

    if args.rebuild and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print(f"[bad_code] Removed existing dir: {PERSIST_DIR}")

    if _corpus_already_populated(PERSIST_DIR) and not args.rebuild:
        print(
            f"[bad_code] Refusing to overwrite an existing corpus at {PERSIST_DIR}. "
            "Use --rebuild to recreate."
        )
        raise SystemExit(2)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    if not SEED_PATH.exists():
        raise SystemExit(f"[bad_code] Seed file not found: {SEED_PATH}")

    with open(SEED_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    _validate_items(items)
    docs = _build_documents(items)

    print(f"[bad_code] Embedding {len(docs)} documents…")
    emb = _embeddings_factory()
    vdb = Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )
    if hasattr(vdb, "persist"):
        try:
            vdb.persist()
        except Exception:
            pass

    print(f"[bad_code] Persisted {len(docs)} docs to {PERSIST_DIR}")


if __name__ == "__main__":
    main()
