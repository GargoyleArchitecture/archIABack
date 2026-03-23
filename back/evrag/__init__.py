"""
EVRAG - Enhanced Video Retrieval-Augmented Generation

Implementación de EVRAG para procesamiento de videos educativos sobre arquitectura de software.

Basado en el paper: "EVRAG: Enhanced Video Retrieval-Augmented Generation for Democratized Social Computing"
- Scene-change detection para frames clave
- Whisper para transcripción de audio
- CLIP para embeddings multimodales
- ChromaDB para almacenamiento vectorial

Costo estimado: ~$0.77 por video de 1 hora
"""

from .config import EVRAG_CONFIG
from .video_processor import VideoProcessor, Scene
from .transcriber import AudioTranscriber, TranscriptionResult
from .clip_embedder import CLIPEmbedder
from .indexer import EVRAGIndexer, IndexedVideo

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
]
