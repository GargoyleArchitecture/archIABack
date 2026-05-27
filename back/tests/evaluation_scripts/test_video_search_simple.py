#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simple para probar busqueda en ChromaDB de videos
"""
import chromadb
from pathlib import Path
from back.evrag.config import EVRAG_CONFIG

chroma_path = Path(__file__).parent / "back" / "videos" / "chroma_db"

print("=" * 70)
print("TEST: Busqueda directa en ChromaDB de videos")
print("=" * 70)

# Conectar
client = chromadb.PersistentClient(path=str(chroma_path))
coll_name = f"{EVRAG_CONFIG['chroma_collection_name']}_transcript"
collection = client.get_collection(coll_name)

print(f"\nTotal segmentos: {collection.count()}")

# Queries de prueba (sin embeddings, solo usando ChromaDB's built-in)
test_queries = [
    "microservices architecture patterns",
    "CQRS event sourcing",
    "scalability and performance",
    "system design patterns",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"[TEST {i}] Query: '{query}'")
    print('-' * 70)

    try:
        # Query directo con texto (ChromaDB hace embedding automatico)
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        if results and results["documents"] and results["documents"][0]:
            for j, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ), 1):
                video_name = meta.get("video_name", "unknown")
                start = meta.get("start_time", 0)
                end = meta.get("end_time", 0)
                relevance = 1 - dist

                # Format timestamps
                start_min = int(start // 60)
                start_sec = int(start % 60)
                end_min = int(end // 60)
                end_sec = int(end % 60)

                print(f"\n[{j}] **{video_name}** ({start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d})")
                print(f"    Relevance: {relevance:.0%}")
                print(f"    Text: {doc}")
        else:
            print("No se encontraron resultados")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("FIN DE TESTS")
print("=" * 70)
