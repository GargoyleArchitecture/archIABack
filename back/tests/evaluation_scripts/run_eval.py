#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run RAG Evaluation Pipeline for Layer 1 Books

Este script ejecuta el pipeline de evaluación RAG para evaluar el desempeño
del sistema en los documentos PDF actuales (back/docs/).

Soporta OpenAI y Ollama como providers (configurado en back/eval/config.py).

Uso:
    poetry run python run_eval.py

Requisitos:
    - OpenAI API key (si usas provider="openai")
    - Ollama con llama3.1 (si usas provider="ollama")
    - Vector store construido (back/chroma_db/)
    - Servidor RAG corriendo (solo para evaluación real)
"""

import json
import requests
from pathlib import Path
from datetime import datetime

# Cargar variables de entorno desde .env
from dotenv import load_dotenv
import os

# Cargar .env desde la raíz del proyecto y desde back/src/
load_dotenv(Path(__file__).parent / "back" / "src" / ".env")
load_dotenv(Path(__file__).parent / "back" / ".env")
load_dotenv()  # También carga .env en el directorio actual

from back.eval import EVAL_CONFIG, evaluate_layer_1_books
from back.eval.config import get_total_qa_pairs


# =============================================================================
# RAG INVOKE FUNCTION (HTTP-based)
# =============================================================================

RAG_API_URL = "http://localhost:8000/message"


def rag_invoke_func(question: str, session_id: str) -> dict:
    """
    Invoke RAG system via HTTP API.

    Args:
        question: Question to ask
        session_id: Session identifier

    Returns:
        Dictionary with retrieved_context and generated_answer

    Raises:
        RuntimeError: If API is not available
    """
    try:
        # Preparar payload para el endpoint /message
        payload = {
            "message": question,
            "session_id": session_id,
        }

        # Hacer request al API
        response = requests.post(
            RAG_API_URL,
            data=payload,
            timeout=120,  # 2 minutos timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.status_code} - {response.text}")

        result = response.json()

        # Extraer respuesta del RAG
        # El formato depende de cómo responde el grafo
        generated_answer = result.get("response", result.get("message", ""))

        # Extraer contexto recuperado (si está disponible)
        retrieved_context = result.get("context", result.get("retrieved_docs", []))
        if isinstance(retrieved_context, list):
            retrieved_context = "\n\n".join(
                [doc.get("content", "") for doc in retrieved_context[:3]]
            )

        return {
            "retrieved_context": retrieved_context,
            "generated_answer": generated_answer,
        }

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to RAG API at {RAG_API_URL}. "
            "Make sure the server is running: poetry run uvicorn src.main:app --port 8000"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"RAG API timeout after 120s for question: {question[:50]}...")
    except Exception as e:
        raise RuntimeError(f"Error invoking RAG: {e}")


# =============================================================================
# MOCK RAG FUNCTION (for testing without API)
# =============================================================================

def mock_rag_invoke_func(question: str, session_id: str) -> dict:
    """
    Mock RAG invoke function for testing pipeline without running API.

    Simula respuestas del RAG con precisión variable.

    Args:
        question: Question to ask
        session_id: Session identifier

    Returns:
        Dictionary with mock retrieved_context and generated_answer
    """
    import random

    # Simular precisión del 80%
    accuracy = 0.8

    if random.random() < accuracy:
        # Respuesta "correcta" (similar al ground truth)
        generated_answer = (
            "Según la documentación, " +
            question.replace("¿", "").replace("?", "") +
            " se refiere a un concepto importante en arquitectura de software."
        )
        retrieved_context = "Contexto recuperado del documento PDF relevante..."
    else:
        # Respuesta incorrecta o incompleta
        generated_answer = "No tengo suficiente información para responder esta pregunta."
        retrieved_context = ""

    return {
        "retrieved_context": retrieved_context,
        "generated_answer": generated_answer,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    print("=" * 70)
    print("OLLAMA RAG EVALUATION PIPELINE - LAYER 1 BOOKS")
    print("=" * 70)
    print()

    # Mostrar configuración
    print(f"LLM Provider: {EVAL_CONFIG['llm_provider']}")
    print(f"LLM Model: {EVAL_CONFIG['llm_model']}")
    print(f"Evaluation Mode: {EVAL_CONFIG['eval_mode']}")
    print(f"QA Pairs per Doc: {get_total_qa_pairs()}")
    print(f"Metrics Enabled: {len([k for k, v in EVAL_CONFIG['metrics'].items() if v])}")
    print()

    # Preguntar si usar RAG real o mock
    print("-" * 70)
    print("Select evaluation mode:")
    print("  1. Real RAG (requires API running at localhost:8000)")
    print("  2. Mock RAG (for testing pipeline without API)")
    print("  3. Exit")
    print("-" * 70)

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "3":
        print("Exiting...")
        return

    use_mock = choice == "2"

    if use_mock:
        print("\n⚠️  Running with MOCK RAG (simulated results)")
        rag_func = mock_rag_invoke_func
    else:
        print("\n✓ Running with REAL RAG API")
        print("  Make sure the server is running:")
        print("  poetry run uvicorn src.main:app --port 8000")
        print()
        input("Press Enter to continue or Ctrl+C to cancel...")
        rag_func = rag_invoke_func

    # Ejecutar evaluación
    print("\n" + "=" * 70)
    print("Starting evaluation...")
    print("=" * 70)

    try:
        report = evaluate_layer_1_books(
            rag_invoke_func=rag_func,
            force_regenerate=False,  # Usar datasets cacheados si existen
        )

        # Mostrar resultados
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print()
        print(report.to_markdown())

        # Guardar reporte
        reports_dir = Path(EVAL_CONFIG["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"ollama_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text(report.to_markdown(), encoding="utf-8")

        print(f"\n✓ Report saved to: {report_path}")
        print()

        # Resumen final
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Documents evaluated: {len(report.document_results)}")
        print(f"Overall score: {report.aggregate_metrics.get('overall', 0):.4f}")
        print()

        # Métricas clave
        print("Key Metrics:")
        for metric in ['faithfulness', 'answer_relevance', 'context_precision', 'context_recall']:
            if metric in report.aggregate_metrics:
                score = report.aggregate_metrics[metric]
                print(f"  - {metric}: {score:.4f}")

        print()
        print("=" * 70)
        print("✅ Evaluation complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure Ollama is running: ollama serve")
        print("  - Check model is installed: ollama pull llama3.1")
        print("  - Verify vector store exists: back/chroma_db/")
        print("  - For real RAG: Start server with 'poetry run uvicorn src.main:app --port 8000'")
        raise


if __name__ == "__main__":
    main()
