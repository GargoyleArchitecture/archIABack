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
    # LLM Provider: Ollama (local, gratis) o OpenAI
    # -------------------------------------------------------------------------
    "llm_provider": "ollama",      # "ollama" o "openai"
    "llm_model": "llama3.1",       # Modelos: "llama3.1", "mistral", "gemma2"
    
    # OpenAI (solo si usas provider="openai")
    # "generation_model": "gpt-4o-mini",
    # "evaluation_model": "gpt-4o",
    
    # -------------------------------------------------------------------------
    # Evaluation Mode: "standard" (10 QA) o "comprehensive" (30 QA)
    # -------------------------------------------------------------------------
    "eval_mode": "standard",  # "standard" o "comprehensive"
    
    # Standard: 10 QA pairs/doc (balanceado costo/calidad)
    "standard": {
        "qa_pairs_per_doc": 10,
        "question_types": {
            "factual": 4,       # 40%
            "multi_hop": 4,     # 40%
            "synthesis": 2,     # 20%
        },
    },
    
    # Comprehensive: 30 QA pairs/doc (MiRAGE-style, máxima calidad)
    "comprehensive": {
        "qa_pairs_per_doc": 30,
        "question_types": {
            "factual": 12,      # 40%
            "multi_hop": 12,    # 40%
            "synthesis": 6,     # 20%
        },
    },
    
    # Agentes para generación (2 = Generator + Verifier)
    "num_agents": 2,
    
    # Temperaturas para generación
    "generation_temperature": 0.7,
    "evaluation_temperature": 0.0,  # Determinístico para evaluación
    
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

def _get_active_mode_config() -> dict:
    """Get configuration for active evaluation mode."""
    mode = EVAL_CONFIG.get("eval_mode", "standard")
    return EVAL_CONFIG.get(mode, EVAL_CONFIG["standard"])


def get_question_type_distribution() -> dict[QuestionType, int]:
    """
    Returns the configured distribution of question types.
    
    Returns:
        Dictionary mapping question type to count.
    """
    mode_config = _get_active_mode_config()
    return mode_config.get("question_types", EVAL_CONFIG["standard"]["question_types"]).copy()


def get_total_qa_pairs() -> int:
    """
    Returns total QA pairs per document.
    
    Returns:
        Total number of QA pairs to generate per document.
    """
    mode_config = _get_active_mode_config()
    return mode_config.get("qa_pairs_per_doc", 10)


def get_eval_mode() -> str:
    """
    Returns current evaluation mode.
    
    Returns:
        "standard" or "comprehensive"
    """
    return EVAL_CONFIG.get("eval_mode", "standard")


def set_eval_mode(mode: str) -> None:
    """
    Set evaluation mode.
    
    Args:
        mode: "standard" or "comprehensive"
    """
    if mode not in ["standard", "comprehensive"]:
        raise ValueError(f"Invalid mode: {mode}. Use 'standard' or 'comprehensive'")
    EVAL_CONFIG["eval_mode"] = mode


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
