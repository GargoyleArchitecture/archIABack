"""
Dataset Generators for RAG Evaluation

Generadores de datasets para evaluación RAG usando enfoque multi-agente.
"""

from .dataset_generator import (
    DatasetGenerator,
    DocumentDataset,
    QAPair,
)
from .video_dataset_generator import (
    VideoDataset,
    VideoDatasetGenerator,
    generate_video_datasets,
)
from .video_processor import (
    VideoInfo,
    VideoProcessor,
    Scene,
)

__all__ = [
    # PDF generators
    "DatasetGenerator",
    "DocumentDataset",
    "QAPair",
    # Video generators
    "VideoDataset",
    "VideoDatasetGenerator",
    "generate_video_datasets",
    # Video processor
    "VideoInfo",
    "VideoProcessor",
    "Scene",
]
