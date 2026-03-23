"""
EVRAG Configuration

Configuración para procesamiento de videos con EVRAG.
"""

from typing import Literal


# =============================================================================
# EVRAG CONFIGURATION
# =============================================================================

EVRAG_CONFIG = {
    # -------------------------------------------------------------------------
    # Video Processing
    # -------------------------------------------------------------------------
    "scene_detection_method": "opencv",  # opencv, scenedetect
    "scene_detection_threshold": 30,  # Umbral para detectar cambio de escena (0-100)
    "min_scene_length_sec": 2,  # Longitud mínima de escena en segundos
    "max_frames_per_video": 200,  # Máximo frames a extraer
    
    # -------------------------------------------------------------------------
    # Audio Transcription
    # -------------------------------------------------------------------------
    "transcriber": "whisper_local",  # whisper_local, whisper_api
    "whisper_model": "base",  # tiny, base, small, medium, large
    "language": "es",  # Idioma del audio (es, en)
    
    # -------------------------------------------------------------------------
    # CLIP Embeddings
    # -------------------------------------------------------------------------
    "clip_model": "ViT-B/32",  # ViT-B/32, ViT-B/16, ViT-L/14
    "embed_batch_size": 32,  # Batch size para embeddings
    
    # -------------------------------------------------------------------------
    # QA Generation (MiRAGE-style)
    # -------------------------------------------------------------------------
    "qa_pairs_per_video": 25,  # QA pairs para generar
    "question_types": {
        "factual": 10,       # 40% - Sobre contenido explícito
        "multi_hop": 10,     # 40% - Requiere unir audio + visual
        "synthesis": 5,      # 20% - Comprensión global
    },
    
    # Modelos LLM
    "generation_model": "gpt-4o-mini",  # Generar QA (barato)
    "evaluation_model": "gpt-4o",       # Evaluar (calidad)
    
    # -------------------------------------------------------------------------
    # ChromaDB Storage
    # -------------------------------------------------------------------------
    "chroma_collection_name": "evrag_videos",
    "persist_directory": "back/videos/chroma_db",
    
    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    "videos_dir": "back/videos/raw",
    "processed_dir": "back/videos/processed",
    "frames_dir": "back/videos/frames",
    "transcripts_dir": "back/videos/transcripts",
}


# =============================================================================
# TYPE ALIASES
# =============================================================================

SceneDetectionMethod = Literal["opencv", "scenedetect"]
TranscriberType = Literal["whisper_local", "whisper_api"]
WhisperModel = Literal["tiny", "base", "small", "medium", "large"]
ClipModel = Literal["ViT-B/32", "ViT-B/16", "ViT-L/14"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_total_qa_pairs() -> int:
    """Returns total QA pairs per video."""
    return sum(EVRAG_CONFIG["question_types"].values())


def is_multimodal_evaluation() -> bool:
    """
    Returns True if EVRAG should evaluate both text and visual content.
    
    For your thesis (unificación de atributos), this should be True.
    """
    return True
