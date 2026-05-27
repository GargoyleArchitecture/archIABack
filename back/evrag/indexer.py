"""
EVRAG ChromaDB Indexer

Indexa frames y transcripciones en ChromaDB para retrieval multimodal.
Incluye anonimización automática de PII.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class IndexedVideo:
    """
    Metadata for an indexed video.
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
    """

    def __init__(self, config: dict[str, Any] | None = None, clip_enabled: bool = True):
        """
        Initialize EVRAG indexer.
        """
        from .config import EVRAG_CONFIG
        from .privacy import TextAnonymizer
        
        self.config = config or EVRAG_CONFIG
        self.clip_enabled = clip_enabled
        self.anonymizer = TextAnonymizer(language="es")

        self.chroma_client = None
        self.frames_collection = None
        self.transcript_collection = None
        self.descriptions_collection = None

        self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize ChromaDB client and collections."""
        import chromadb
        from chromadb.config import Settings

        persist_dir = Path(self.config["persist_directory"])
        persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing ChromaDB at {persist_dir}")

        self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))

        collection_name = self.config["chroma_collection_name"]

        self.frames_collection = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}_frames",
            metadata={"description": "Video frames with CLIP embeddings"},
        )

        self.transcript_collection = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}_transcript",
            metadata={"description": "Video transcript segments"},
        )

        self.descriptions_collection = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}_descriptions",
            metadata={"description": "Textual descriptions of video frames"},
        )

    def index_video(
        self,
        video_path: Path | str,
        frames: list[Path],
        frame_embeddings: np.ndarray | None = None,
        transcript_text: str = "",
        transcript_segments: list[dict[str, Any]] = [],
        video_duration_sec: float = 0,
        frame_descriptions: list[str] | None = None,
    ) -> IndexedVideo:
        """
        Index a complete video in ChromaDB with automatic anonymization.
        """
        video_path = Path(video_path)

        # 1. Anonimizar datos antes de indexar
        print(f"\n[Privacy] Anonimizando datos de: {video_path.name}")
        
        # Anonimizar transcripción completa
        anon_transcript_res = self.anonymizer.anonymize(transcript_text)
        clean_transcript_text = anon_transcript_res.anonymized_text
        
        # Anonimizar segmentos de transcripción
        clean_segments = []
        for seg in transcript_segments:
            seg_text = seg.get("text", "")
            seg_clean = self.anonymizer.anonymize(seg_text).anonymized_text
            clean_segments.append({**seg, "text": seg_clean})
            
        # Anonimizar descripciones de frames
        clean_descriptions = None
        if frame_descriptions:
            clean_descriptions = [
                self.anonymizer.anonymize(desc).anonymized_text 
                for desc in frame_descriptions
            ]

        # Compute video hash
        import hashlib
        try:
            video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
        except FileNotFoundError:
            video_hash = "manual_index_no_video"

        print(f"Indexing video: {video_path.name}")

        # Index frames (visual) - No PII here usually
        if self.clip_enabled and frame_embeddings is not None:
            self._index_frames(
                video_id=video_path.stem,
                frames=frames,
                embeddings=frame_embeddings,
            )

        # Index descriptions (clean)
        if clean_descriptions:
            self._index_descriptions(
                video_id=video_path.stem,
                frames=frames,
                descriptions=clean_descriptions,
            )

        # Index transcript (clean)
        self._index_transcript(
            video_id=video_path.stem,
            text=clean_transcript_text,
            segments=clean_segments,
        )

        # Build metadata (clean)
        indexed_video = IndexedVideo(
            video_path=str(video_path),
            video_hash=video_hash,
            transcript=clean_transcript_text,
            frames=[str(f) for f in frames],
            frame_embeddings_shape=frame_embeddings.shape if frame_embeddings is not None else (0,),
            duration_sec=video_duration_sec,
        )

        # Save metadata (clean)
        metadata_path = Path(self.config["processed_dir"]) / f"{video_path.stem}_indexed.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(indexed_video.to_json(), encoding="utf-8")

        return indexed_video

    def _index_frames(self, video_id: str, frames: list[Path], embeddings: np.ndarray):
        """Index video frames with embeddings."""
        ids = [f"{video_id}_frame_{i:05d}" for i in range(len(frames))]
        metadatas = []
        for i, frame_path in enumerate(frames):
            metadatas.append({
                "video_id": video_id,
                "frame_index": i,
                "frame_path": str(frame_path),
                "modality": "visual",
            })
        self.frames_collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def _index_descriptions(self, video_id: str, frames: list[Path], descriptions: list[str]):
        """Index textual descriptions of frames."""
        ids = [f"{video_id}_desc_{i:05d}" for i in range(len(frames))]
        metadatas = []
        for i, frame_path in enumerate(frames):
            metadatas.append({
                "video_id": video_id,
                "frame_index": i,
                "frame_path": str(frame_path),
                "modality": "descriptive",
            })
        self.descriptions_collection.upsert(
            ids=ids,
            documents=descriptions,
            metadatas=metadatas,
        )

    def _index_transcript(self, video_id: str, text: str, segments: list[dict[str, Any]]):
        """Index transcript segments."""
        ids = []
        texts = []
        metadatas = []
        for i, segment in enumerate(segments):
            segment_id = f"{video_id}_segment_{i:05d}"
            segment_text = segment.get("text", "")
            if not segment_text.strip(): continue
            ids.append(segment_id)
            texts.append(segment_text)
            metadatas.append({
                "video_id": video_id,
                "segment_index": i,
                "start_time": segment.get("start", 0),
                "end_time": segment.get("end", 0),
                "modality": "textual",
            })
        if texts:
            self.transcript_collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def query_multimodal(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid" 
    ) -> dict[str, Any]:
        """
        Query multimodal data. Query is also anonymized for consistency.
        """
        # Anonymize query to match indexed data patterns
        clean_query = self.anonymizer.anonymize(query).anonymized_text
        
        results = {"query": clean_query, "frames": [], "segments": []}
        
        # 1. Visual Search (CLIP)
        if mode in ["visual", "hybrid"] and self.clip_enabled:
            from .clip_embedder import CLIPEmbedder
            embedder = CLIPEmbedder(config=self.config)
            q_emb = embedder.embed_texts([clean_query])
            res = self.frames_collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=top_k,
                include=["metadatas", "distances"]
            )
            if res["metadatas"]:
                for m, d in zip(res["metadatas"][0], res["distances"][0]):
                    results["frames"].append({**m, "distance": float(d), "type": "visual"})

        # 2. Descriptive Search
        if mode in ["descriptive", "hybrid"]:
            res = self.descriptions_collection.query(
                query_texts=[clean_query],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            if res["metadatas"]:
                for m, doc, d in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
                    results["frames"].append({**m, "description": doc, "distance": float(d), "type": "descriptive"})

        # 3. Transcript Search
        res = self.transcript_collection.query(
            query_texts=[clean_query],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        if res["metadatas"]:
            for m, doc, d in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
                results["segments"].append({**m, "text": doc, "distance": float(d)})

        # Fallback: if no frames found but segments found (e.g. CLIP disabled, no descriptions db),
        # associate segments with their closest scene frames
        if not results["frames"] and results["segments"]:
            import re
            _base = Path(__file__).resolve().parent.parent
            frames_dir = _base / "videos" / "frames"
            available_frames = list(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
            
            for seg in results["segments"]:
                video_id = seg["video_id"]
                start_time = seg.get("start_time", 0)
                
                # Find closest frame file
                matching = [f for f in available_frames if f.name.startswith(f"{video_id}_scene_")]
                if matching:
                    best_file = matching[0].name
                    min_delta = float('inf')
                    target_frame = start_time * 30
                    for f in matching:
                        m = re.search(r'_frame_(\d+)', f.name)
                        if m:
                            frame_num = int(m.group(1))
                            delta = abs(frame_num - target_frame)
                            if delta < min_delta:
                                min_delta = delta
                                best_file = f.name
                    
                    frame_idx = 0
                    m_idx = re.search(r'_scene_(\d+)', best_file)
                    if m_idx:
                        frame_idx = int(m_idx.group(1))
                    
                    results["frames"].append({
                        "video_id": video_id,
                        "frame_index": frame_idx,
                        "frame_path": str(frames_dir / best_file),
                        "description": seg["text"],
                        "distance": seg["distance"],
                        "type": "descriptive",
                    })

        if mode == "hybrid":
            results["frames"].sort(key=lambda x: x["distance"])
            results["frames"] = results["frames"][:top_k]

        return results

    def get_indexed_videos(self) -> list[IndexedVideo]:
        """Get list of all indexed videos."""
        processed_dir = Path(self.config["processed_dir"])
        if not processed_dir.exists(): return []
        indexed_videos = []
        for metadata_file in processed_dir.glob("*_indexed.json"):
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
            indexed_videos.append(IndexedVideo(
                video_path=data["video_path"],
                video_hash=data["video_hash"],
                transcript=data.get("transcript", ""),
                frames=data.get("frames", []),
                frame_embeddings_shape=tuple(data.get("frame_embeddings_shape", (0,))),
                duration_sec=data.get("duration_sec", 0),
            ))
        return indexed_videos
