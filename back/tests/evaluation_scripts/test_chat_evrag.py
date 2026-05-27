#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar que el chat usa EVRAG automaticamente
"""
import requests
import json

API = "http://localhost:8000"

print("=" * 70)
print("TEST: Chat End-to-End con EVRAG")
print("=" * 70)

# Queries de prueba que deberian activar video_RAG
test_queries = [
    "What is CQRS?",
    "Explica los patrones de microservicios",
    "Como mejorar la escalabilidad de un sistema?",
]

session_id = "test_evrag_session"

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"[TEST {i}] Query: '{query}'")
    print('-' * 70)

    # Simular mensaje del usuario
    data = {
        "message": query,
        "session_id": session_id
    }

    try:
        response = requests.post(f"{API}/message", data=data, timeout=60)
        result = response.json()

        print(f"\nRespuesta del sistema:")
        print("-" * 70)

        # Mostrar respuesta principal
        end_message = result.get("endMessage", "")
        print(end_message[:500] + "..." if len(end_message) > 500 else end_message)

        # Verificar si hay resultados de video
        if "Video Results" in end_message or "video" in end_message.lower():
            print("\n[OK] La respuesta incluye resultados de videos!")
        else:
            print("\n[INFO] No se detectaron resultados de video en esta respuesta")

        # Mostrar mensajes internos (para debug)
        internal_msgs = result.get("messages", [])
        if internal_msgs:
            print(f"\n[DEBUG] Nodos internos ejecutados: {len(internal_msgs)}")
            for msg in internal_msgs:
                role = msg.get("name") or msg.get("role", "unknown")
                content_preview = str(msg.get("content", ""))[:80]
                print(f"  - {role}: {content_preview}...")

    except requests.exceptions.Timeout:
        print("[ERROR] Timeout esperando respuesta del backend")
    except Exception as e:
        print(f"[ERROR] {e}")

print("\n" + "=" * 70)
print("FIN DE TESTS")
print("=" * 70)
