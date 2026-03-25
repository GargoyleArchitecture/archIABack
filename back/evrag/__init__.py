"""
EVRAG - Enhanced Video Retrieval-Augmented Generation

Implementación de EVRAG para procesamiento de videos educativos sobre arquitectura de software.

Basado en el paper: "EVRAG: Enhanced Video Retrieval-Augmented Generation for Democratized Social Computing"
- Scene-change detection para frames clave
- Whisper para transcripción de audio
- CLIP para embeddings multimodales
- ChromaDB para almacenamiento vectorial
- Anonimización de transcripts y difuminado de rostros

Costo estimado: ~$1.11 por video de 1 hora (Whisper API + LLM)
"""

from .config import EVRAG_CONFIG
from .video_processor import VideoProcessor, Scene
from .transcriber import AudioTranscriber, TranscriptionResult
from .clip_embedder import CLIPEmbedder
from .indexer import EVRAGIndexer, IndexedVideo
from .privacy import TextAnonymizer, FaceBlurrer, SecureStorage
from .pipeline import EVRAGPipeline, EVRAGResult

__version__ = "1.0.0"
__all__ = [
    "EVRAG_CONFIG",
    "VideoProcessor",
    "Scene",
    "AudioTranscriber",
    "TranscriptionResult",
    "CLIPEmbedder",
    "EVRAGIndexer",
    "IndexedVideo",
    "TextAnonymizer",
    "FaceBlurrer",
    "SecureStorage",
    "EVRAGPipeline",
    "EVRAGResult",
]
