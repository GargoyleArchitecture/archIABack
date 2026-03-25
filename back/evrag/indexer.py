"""
EVRAG ChromaDB Indexer

Indexa frames y transcripciones en ChromaDB para retrieval multimodal.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class IndexedVideo:
    """
    Metadata for an indexed video.
    
    Attributes:
        video_path: Path to original video
        video_hash: SHA256 hash of video file
        transcript: Transcribed text
        frames: List of frame paths
        frame_embeddings_shape: Shape of frame embeddings array
        duration_sec: Video duration
    """
    video_path: str
    video_hash: str
    transcript: str = ""
    frames: list[str] = field(default_factory=list)
    frame_embeddings_shape: tuple = (0,)
    duration_sec: float = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_path": self.video_path,
            "video_hash": self.video_hash,
            "transcript": self.transcript,
            "frames": self.frames,
            "frame_embeddings_shape": self.frame_embeddings_shape,
            "duration_sec": self.duration_sec,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class EVRAGIndexer:
    """
    EVRAG indexer for multimodal retrieval.
    
    Indexes:
    1. Video frames with CLIP embeddings
    2. Transcript segments with timestamps
    3. Metadata (scenes, duration, etc.)
    
    Enables queries like:
    - "Show me frames about latency optimization"
    - "What does the video say about ASR unification?"
    - "Find the section discussing trade-offs"
    
    Attributes:
        config: Configuration dictionary
        chroma_client: ChromaDB client
        frames_collection: ChromaDB collection for frames
        transcript_collection: ChromaDB collection for transcript
    """
    
    def __init__(self, config: dict[str, Any] | None = None, clip_enabled: bool = True):
        """
        Initialize EVRAG indexer.
        
        Args:
            config: Configuration dictionary (default: EVRAG_CONFIG)
            clip_enabled: Whether CLIP embeddings are available
        """
        from .config import EVRAG_CONFIG
        self.config = config or EVRAG_CONFIG
        self.clip_enabled = clip_enabled
        
        self.chroma_client = None
        self.frames_collection = None
        self.transcript_collection = None
        
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collections."""
        import chromadb
        from chromadb.config import Settings
        
        persist_dir = Path(self.config["persist_directory"])
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initializing ChromaDB at {persist_dir}")
        
        # New ChromaDB API (no chroma_db_impl parameter)
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
        )
        
        # Get or create collections
        collection_name = self.config["chroma_collection_name"]
        
        self.frames_collection = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}_frames",
            metadata={"description": "Video frames with CLIP embeddings"},
        )
        
        self.transcript_collection = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}_transcript",
            metadata={"description": "Video transcript segments"},
        )
        
        print(f"  Frames collection: {self.frames_collection.name}")
        print(f"  Transcript collection: {self.transcript_collection.name}")
    
    def index_video(
        self,
        video_path: Path | str,
        frames: list[Path],
        frame_embeddings: np.ndarray | None = None,
        transcript_text: str = "",
        transcript_segments: list[dict[str, Any]] = "",
        video_duration_sec: float = 0,
    ) -> IndexedVideo:
        """
        Index a complete video in ChromaDB.
        
        Args:
            video_path: Path to video file
            frames: List of frame paths
            frame_embeddings: CLIP embeddings for frames (optional if clip_enabled=False)
            transcript_text: Full transcribed text
            transcript_segments: Segments with timestamps
            video_duration_sec: Video duration
            
        Returns:
            IndexedVideo metadata object
        """
        video_path = Path(video_path)
        
        # Compute video hash
        import hashlib
        video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
        
        print(f"\nIndexing video: {video_path.name}")
        
        # Index frames (if CLIP available)
        if self.clip_enabled and frame_embeddings is not None:
            self._index_frames(
                video_id=video_path.stem,
                frames=frames,
                embeddings=frame_embeddings,
            )
        else:
            print("  Skipping frame indexing (CLIP not available)")
        
        # Index transcript
        self._index_transcript(
            video_id=video_path.stem,
            text=transcript_text,
            segments=transcript_segments,
        )
        
        # Build metadata
        indexed_video = IndexedVideo(
            video_path=str(video_path),
            video_hash=video_hash,
            transcript=transcript_text,
            frames=[str(f) for f in frames],
            frame_embeddings_shape=frame_embeddings.shape,
            duration_sec=video_duration_sec,
        )
        
        # Save metadata
        metadata_path = Path(self.config["processed_dir"]) / f"{video_path.stem}_indexed.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(indexed_video.to_json(), encoding="utf-8")
        
        print(f"  Indexed {len(frames)} frames")
        print(f"  Indexed {len(transcript_segments)} transcript segments")
        
        return indexed_video
    
    def _index_frames(
        self,
        video_id: str,
        frames: list[Path],
        embeddings: np.ndarray,
    ):
        """Index video frames with embeddings."""
        import uuid
        
        ids = []
        metadatas = []
        
        for i, frame_path in enumerate(frames):
            frame_id = f"{video_id}_frame_{i:05d}"
            ids.append(frame_id)
            
            metadatas.append({
                "video_id": video_id,
                "frame_index": i,
                "frame_path": str(frame_path),
                "modality": "visual",
            })
        
        # Add to ChromaDB
        self.frames_collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
    
    def _index_transcript(
        self,
        video_id: str,
        text: str,
        segments: list[dict[str, Any]],
    ):
        """Index transcript segments."""
        from .clip_embedder import CLIPEmbedder
        
        embedder = CLIPEmbedder(config=self.config)
        
        ids = []
        texts = []
        metadatas = []
        
        # Create chunks from segments
        for i, segment in enumerate(segments):
            segment_id = f"{video_id}_segment_{i:05d}"
            segment_text = segment.get("text", "")
            
            if not segment_text.strip():
                continue
            
            ids.append(segment_id)
            texts.append(segment_text)
            
            metadatas.append({
                "video_id": video_id,
                "segment_index": i,
                "start_time": segment.get("start", 0),
                "end_time": segment.get("end", 0),
                "modality": "textual",
            })
        
        if not texts:
            print("  No transcript segments to index")
            return
        
        # Generate embeddings for segments
        embeddings = embedder.embed_texts(texts)
        
        # Add to ChromaDB
        self.transcript_collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts,
        )
    
    def query_multimodal(
        self,
        query: str,
        top_k_frames: int = 5,
        top_k_segments: int = 3,
    ) -> dict[str, Any]:
        """
        Query both frames and transcript.
        
        Args:
            query: Text query
            top_k_frames: Number of frames to return
            top_k_segments: Number of transcript segments to return
            
        Returns:
            Dictionary with frames and transcript results
        """
        from .clip_embedder import CLIPEmbedder
        
        embedder = CLIPEmbedder(config=self.config)
        
        # Embed query
        query_embedding = embedder.embed_texts([query])
        
        # Query frames
        frame_results = self.frames_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k_frames,
            include=["metadatas", "distances"],
        )
        
        # Query transcript
        segment_results = self.transcript_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k_segments,
            include=["metadatas", "documents", "distances"],
        )
        
        # Format results
        return {
            "query": query,
            "frames": [
                {
                    "frame_path": m["frame_path"],
                    "video_id": m["video_id"],
                    "frame_index": m["frame_index"],
                    "distance": float(d),
                }
                for m, d in zip(
                    frame_results["metadatas"][0],
                    frame_results["distances"][0],
                )
            ],
            "segments": [
                {
                    "text": doc,
                    "video_id": m["video_id"],
                    "start_time": m["start_time"],
                    "end_time": m["end_time"],
                    "distance": float(d),
                }
                for doc, m, d in zip(
                    segment_results["documents"][0],
                    segment_results["metadatas"][0],
                    segment_results["distances"][0],
                )
            ],
        }
    
    def get_indexed_videos(self) -> list[IndexedVideo]:
        """
        Get list of all indexed videos.
        
        Returns:
            List of IndexedVideo objects
        """
        processed_dir = Path(self.config["processed_dir"])
        
        if not processed_dir.exists():
            return []
        
        indexed_videos = []
        
        for metadata_file in processed_dir.glob("*_indexed.json"):
            data = json.loads(metadata_file.read_text())
            indexed_videos.append(IndexedVideo(
                video_path=data["video_path"],
                video_hash=data["video_hash"],
                transcript=data.get("transcript", ""),
                frames=data.get("frames", []),
                frame_embeddings_shape=tuple(data.get("frame_embeddings_shape", (0,))),
                duration_sec=data.get("duration_sec", 0),
            ))
        
        return indexed_videos
