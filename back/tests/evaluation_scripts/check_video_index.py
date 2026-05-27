#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar que videos estan indexados en ChromaDB
"""
import chromadb
from pathlib import Path
import json
from back.evrag.config import EVRAG_CONFIG

# Ruta a ChromaDB de videos
chroma_path = Path(__file__).parent / "back" / "videos" / "chroma_db"

if not chroma_path.exists():
    print("[ERROR] ChromaDB de videos no encontrado")
    exit(1)

print(f"ChromaDB path: {chroma_path}\n")

# --- Verificación de Datasets de Evaluación ---
datasets_path = Path(__file__).parent / "back" / "eval" / "datasets"
if datasets_path.exists():
    json_files = list(datasets_path.glob("*_video_dataset.json"))
    print(f"📊 Datasets de evaluación encontrados: {len(json_files)}")
    for jf in json_files:
        print(f"   - {jf.name}")

    if len(json_files) < 10:
        print(f"\n⚠️  Faltan {10 - len(json_files)} videos por completar el flujo de evaluación.")
else:
    print("\n[WARN] No se encontró la carpeta de datasets de evaluación.")

try:
    # Conectar a ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Listar colecciones
    collections = client.list_collections()
    print(f"Colecciones encontradas: {len(collections)}")
    for col in collections:
        print(f"   - {col.name}")

    # Verificar la coleccion de video transcripts
    try:
        coll_name = f"{EVRAG_CONFIG['chroma_collection_name']}_transcript"
        collection = client.get_collection(coll_name)
        count = collection.count()
        print(f"\n[OK] Coleccion '{coll_name}' encontrada")
        print(f"   Total de segmentos indexados: {count}")

        if count > 0:
            # Obtener algunos ejemplos
            results = collection.get(limit=5, include=["metadatas", "documents"])
            print(f"\nPrimeros {min(5, count)} segmentos:")
            for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
                video_name = meta.get("video_name", "unknown")
                start = meta.get("start_time", 0)
                end = meta.get("end_time", 0)
                snippet = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"\n   [{i}] Video: {video_name}")
                print(f"       Time: {start:.1f}s - {end:.1f}s")
                print(f"       Text: {snippet}")
        else:
            print("\n[WARN] La coleccion esta vacia (no hay videos indexados)")

    except Exception as e:
        print(f"\n[ERROR] Error accediendo a coleccion 'video_transcripts': {e}")

except Exception as e:
    print(f"[ERROR] Error conectando a ChromaDB: {e}")
