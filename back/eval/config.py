"""
Configuration for RAG Evaluation Framework

Configuración centralizada para el framework de evaluación RAG.
Incluye settings para generación de datasets, métricas y pipeline.
"""

from typing import Literal


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVAL_CONFIG = {
    # -------------------------------------------------------------------------
    # LLM Provider: Ollama (local, gratis)
    # Para OpenAI: cambiar a "openai" y poner API key en .env
    # -------------------------------------------------------------------------
    "llm_provider": "ollama",      # "ollama" o "openai"
    "llm_model": "llama3.1",       # Modelos: "llama3.1", "mistral", "gemma2"
    
    # OpenAI (solo si usas provider="openai")
    # "generation_model": "gpt-4o-mini",
    # "evaluation_model": "gpt-4o",
    
    # -------------------------------------------------------------------------
    # Dataset Generation (MiRAGE-style)
    # -------------------------------------------------------------------------
    "qa_pairs_per_doc": 10,  # QA pairs por documento
    
    # -------------------------------------------------------------------------
    # Adversarial Verification
    # -------------------------------------------------------------------------
    "verifier_strictness": "high",  # low, medium, high
    
    # Intentos de verificación antes de descartar QA pair
    "verification_attempts": 2,
    
    # -------------------------------------------------------------------------
    # Evaluation Metrics (Híbrido CCRS + RAGAS)
    # -------------------------------------------------------------------------
    "metrics": {
        # RAGAS metrics
        "faithfulness": True,
        "answer_relevance": True,
        "context_precision": True,
        "context_recall": True,
        
        # CCRS metrics
        "contextual_coherence": True,
        "question_relevance": True,
        "information_density": True,
        "answer_correctness": True,
        "information_recall": True,
    },
    
    # Batch size para evaluación (evitar rate limits)
    "eval_batch_size": 5,
    
    # -------------------------------------------------------------------------
    # Caching & Incremental Updates
    # -------------------------------------------------------------------------
    "cache_enabled": True,  # Reusa QA si documento no cambió
    "cache_ttl_days": 30,   # Caducidad del cache
    
    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    "docs_dir": "back/docs",
    "videos_dir": "back/videos",
    "datasets_dir": "back/eval/datasets",
    "reports_dir": "back/eval/reports",
    
    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    "report_format": "json",  # json, markdown, html
    "include_per_doc_metrics": True,
    "include_aggregate_metrics": True,
    "include_temporal_comparison": True,
}


# =============================================================================
# TYPE ALIASES
# =============================================================================

QuestionType = Literal["factual", "multi_hop", "synthesis"]
StrictnessLevel = Literal["low", "medium", "high"]
ReportFormat = Literal["json", "markdown", "html"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_question_type_distribution() -> dict[QuestionType, int]:
    """
    Returns the configured distribution of question types.
    
    Returns:
        Dictionary mapping question type to count.
    """
    return EVAL_CONFIG["question_types"].copy()


def get_total_qa_pairs() -> int:
    """
    Returns total QA pairs per document.
    
    Returns:
        Total number of QA pairs to generate per document.
    """
    return sum(EVAL_CONFIG["question_types"].values())


def is_metric_enabled(metric_name: str) -> bool:
    """
    Check if a specific metric is enabled.
    
    Args:
        metric_name: Name of the metric to check.
        
    Returns:
        True if metric is enabled, False otherwise.
    """
    return EVAL_CONFIG["metrics"].get(metric_name, False)


def get_enabled_metrics() -> list[str]:
    """
    Returns list of enabled metric names.
    
    Returns:
        List of enabled metric names.
    """
    return [k for k, v in EVAL_CONFIG["metrics"].items() if v]
