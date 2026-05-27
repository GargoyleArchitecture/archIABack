#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar la herramienta video_RAG
"""
import sys
from pathlib import Path

# Agregar el directorio back al path
sys.path.insert(0, str(Path(__file__).parent / "back"))

# Importar la herramienta video_RAG
from src.graph.nodes.tools import video_RAG

print("=" * 70)
print("TEST: Herramienta video_RAG")
print("=" * 70)

# Queries de prueba
test_queries = [
    "What are microservices?",
    "Que es CQRS?",
    "scalability patterns",
    "event driven architecture",
    "latencia y rendimiento"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n[TEST {i}] Query: '{query}'")
    print("-" * 70)

    try:
        # Invocar la herramienta
        result = video_RAG.invoke({"query": query, "top_k": 3})
        print(result)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    print()

print("=" * 70)
print("FIN DE TESTS")
print("=" * 70)
