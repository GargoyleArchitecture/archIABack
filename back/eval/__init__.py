"""
RAG Evaluation Framework for ArchIA

Este paquete proporciona evaluación automatizada de sistemas RAG usando:
- Generación de datasets estilo MiRAGE (multi-agente)
- Métricas híbridas (CCRS + RAGAS)
- Pipeline automatizado con reportes
- Soporte para PDFs y Videos (EVRAG)

Quick Start:
    # Evaluar capa 1 (libros actuales)
    from eval import evaluate_layer_1_books

    def my_rag_func(question, session_id):
        # Tu lógica para invocar el RAG
        return {
            "retrieved_context": "...",
            "generated_answer": "...",
        }

    report = evaluate_layer_1_books(rag_invoke_func=my_rag_func)
    print(report.to_markdown())

    # Evaluar capa 3 (videos)
    from eval import evaluate_layer_3_videos

    report = evaluate_layer_3_videos(rag_invoke_func=my_rag_func)
    print(report.to_markdown())

Command Line:
    poetry run python -m eval --layer layer1_books
    poetry run python -m eval --layer layer3_videos
"""

from .config import (
    EVAL_CONFIG,
    get_eval_mode,
    get_total_qa_pairs,
    set_eval_mode,
    get_enabled_metrics,
)
from .pipeline import (
    RAGEvaluationPipeline,
    EvaluationReport,
    evaluate_layer_1_books,
    evaluate_layer_2_new_docs,
    evaluate_layer_3_videos,
)
from .generators import DatasetGenerator, DocumentDataset, QAPair
from .metrics import HybridEvaluator, DocumentEvaluationResult

__version__ = "1.0.0"

__all__ = [
    # Config
    "EVAL_CONFIG",
    "get_eval_mode",
    "get_total_qa_pairs",
    "set_eval_mode",
    "get_enabled_metrics",

    # Pipeline
    "RAGEvaluationPipeline",
    "EvaluationReport",
    "evaluate_layer_1_books",
    "evaluate_layer_2_new_docs",
    "evaluate_layer_3_videos",

    # Generators
    "DatasetGenerator",
    "DocumentDataset",
    "QAPair",

    # Metrics
    "HybridEvaluator",
    "DocumentEvaluationResult",
]
