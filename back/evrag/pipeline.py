"""
EVRAG Pipeline

Pipeline completo para procesamiento de videos con EVRAG:
1. Procesar video (scene-change detection)
2. Transcribir audio (Whisper)
3. Generar embeddings (CLIP)
4. Indexar en ChromaDB
5. Generar QA pairs para evaluación
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import EVRAG_CONFIG
from .video_processor import VideoProcessor
from .transcriber import AudioTranscriber
from .clip_embedder import CLIPEmbedder
from .indexer import EVRAGIndexer


@dataclass
class EVRAGResult:
    """
    Result of complete EVRAG processing.
    
    Attributes:
        video_path: Path to processed video
        scenes_detected: Number of scenes detected
        frames_extracted: Number of frames extracted
        transcript_length: Length of transcribed text
        indexed: Whether video was indexed in ChromaDB
        processing_time_sec: Total processing time
        processed_at: Timestamp
    """
    video_path: str
    scenes_detected: int = 0
    frames_extracted: int = 0
    transcript_length: int = 0
    indexed: bool = False
    processing_time_sec: float = 0
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_path": self.video_path,
            "scenes_detected": self.scenes_detected,
            "frames_extracted": self.frames_extracted,
            "transcript_length": self.transcript_length,
            "indexed": self.indexed,
            "processing_time_sec": self.processing_time_sec,
            "processed_at": self.processed_at,
        }


class EVRAGPipeline:
    """
    Complete EVRAG processing pipeline.
    
    Orchestrates:
    1. Video processing (scene detection)
    2. Audio transcription (Whisper)
    3. Image embedding (CLIP)
    4. Indexing (ChromaDB)
    
    Usage:
        pipeline = EVRAGPipeline()
        result = pipeline.process_video("back/videos/raw/my_video.mp4")
        
        # Query the indexed video
        results = pipeline.query("What does the video say about latency?")
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize EVRAG pipeline.
        
        Args:
            config: Configuration dictionary (default: EVRAG_CONFIG)
        """
        self.config = config or EVRAG_CONFIG
        
        # Initialize components
        self.video_processor = VideoProcessor(config=self.config)
        self.audio_transcriber = AudioTranscriber(config=self.config)
        
        # CLIP may not be available (Python 3.13 compatibility)
        try:
            self.clip_embedder = CLIPEmbedder(config=self.config)
            self.clip_available = True
        except (ImportError, RuntimeError):
            self.clip_embedder = None
            self.clip_available = False
            print("Note: CLIP not available. EVRAG will work in text-only mode.")
        
        self.indexer = EVRAGIndexer(config=self.config, clip_enabled=self.clip_available)
        
        self.stats = {
            "videos_processed": 0,
            "total_frames_extracted": 0,
            "total_transcript_chars": 0,
        }
    
    def process_video(
        self,
        video_path: Path | str,
        force_reprocess: bool = False,
    ) -> EVRAGResult:
        """
        Process a complete video through EVRAG pipeline.
        
        Args:
            video_path: Path to video file
            force_reprocess: If True, reprocess even if cached
            
        Returns:
            EVRAGResult with processing metadata
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        print(f"\n{'='*60}")
        print(f"EVRAG: Processing {video_path.name}")
        print(f"{'='*60}\n")
        
        # Check if already processed
        processed_info_path = Path(self.config["processed_dir"]) / f"{video_path.stem}_evrag.json"
        
        if processed_info_path.exists() and not force_reprocess:
            import json
            print(f"Using cached EVRAG processing for {video_path.name}")
            data = json.loads(processed_info_path.read_text())
            return EVRAGResult(
                video_path=data["video_path"],
                scenes_detected=data["scenes_detected"],
                frames_extracted=data["frames_extracted"],
                transcript_length=data["transcript_length"],
                indexed=data.get("indexed", False),
                processing_time_sec=data.get("processing_time_sec", 0),
                processed_at=data.get("processed_at", ""),
            )
        
        # Step 1: Process video (scene detection + frame extraction)
        print("Step 1: Processing video (scene detection)...")
        video_info = self.video_processor.process_video(
            video_path=video_path,
            force_reprocess=force_reprocess,
        )
        
        frames = [Path(f) for f in video_info["frame_paths"]]
        
        # Step 2: Transcribe audio
        print("\nStep 2: Transcribing audio (Whisper)...")
        transcription = self.audio_transcriber.process_video_audio(
            video_path=video_path,
            use_cache=not force_reprocess,
        )
        
        # Step 3: Generate CLIP embeddings for frames (if available)
        frame_embeddings = None
        if self.clip_available:
            print("\nStep 3: Generating CLIP embeddings...")
            frame_embeddings = self.clip_embedder.embed_images(frames)
        else:
            print("\nStep 3: Skipping CLIP embeddings (not available)")
        
        # Step 4: Index in ChromaDB
        print("\nStep 4: Indexing in ChromaDB...")
        self.indexer.index_video(
            video_path=video_path,
            frames=frames,
            frame_embeddings=frame_embeddings,
            transcript_text=transcription.text,
            transcript_segments=transcription.segments,
            video_duration_sec=video_info["video_duration_sec"],
        )
        
        # Build result
        result = EVRAGResult(
            video_path=str(video_path),
            scenes_detected=video_info["scenes_detected"],
            frames_extracted=video_info["frames_extracted"],
            transcript_length=len(transcription.text),
            indexed=True,
            processing_time_sec=time.time() - start_time,
        )
        
        # Save result
        processed_info_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        processed_info_path.write_text(json.dumps(result.to_dict(), indent=2))
        
        # Update stats
        self.stats["videos_processed"] += 1
        self.stats["total_frames_extracted"] += len(frames)
        self.stats["total_transcript_chars"] += len(transcription.text)
        
        print(f"\n{'='*60}")
        print(f"EVRAG Processing Complete!")
        print(f"  Scenes detected: {result.scenes_detected}")
        print(f"  Frames extracted: {result.frames_extracted}")
        print(f"  Transcript length: {result.transcript_length} chars")
        print(f"  Processing time: {result.processing_time_sec:.1f}s")
        print(f"{'='*60}\n")
        
        return result
    
    def query(self, query_text: str, top_k: int = 5) -> dict[str, Any]:
        """
        Query indexed videos.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            
        Returns:
            Dictionary with matching frames and transcript segments
        """
        return self.indexer.query_multimodal(
            query=query_text,
            top_k_frames=top_k,
            top_k_segments=top_k,
        )
    
    def get_indexed_videos(self) -> list:
        """
        Get list of all indexed videos.
        
        Returns:
            List of IndexedVideo objects
        """
        return self.indexer.get_indexed_videos()
    
    def get_stats(self) -> dict[str, Any]:
        """
        Returns pipeline statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()
