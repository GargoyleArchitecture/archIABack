"""
Simple transcript indexer for videos with SRT subtitles.
Uses Sentence Transformers instead of CLIP for text embeddings.
"""

import sys
import json
import hashlib
from pathlib import Path
import re


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT file into segments."""
    content = srt_path.read_text(encoding='utf-8')
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n\d+\n|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    segments = []
    for num, start_str, end_str, text in matches:
        start_sec = _ts_to_sec(start_str)
        end_sec = _ts_to_sec(end_str)
        text = re.sub(r'<[^>]+>', '', text).replace('\n', ' ').strip()
        if text:
            segments.append({"start": start_sec, "end": end_sec, "text": text})
    return segments


def _ts_to_sec(ts: str) -> float:
    ts = ts.replace(',', '.')
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def get_sentence_embeddings():
    """Get sentence transformers (faster than CLIP for text-only)."""
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')


def index_transcripts(video_srt_pairs: list[tuple[Path, Path]]):
    """Index video transcripts in ChromaDB."""
    import chromadb
    
    # Initialize ChromaDB
    chroma_path = Path("back/videos/chroma_db")
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name="video_transcripts",
        metadata={"description": "Video transcript segments with timestamps"},
    )
    
    # Load embedding model
    model = get_sentence_embeddings()
    
    total_segments = 0
    
    for video_path, srt_path in video_srt_pairs:
        print(f"\n{'='*60}")
        print(f"Indexing: {video_path.name}")
        print(f"{'='*60}")
        
        segments = parse_srt(srt_path)
        full_text = " ".join([s["text"] for s in segments])
        
        print(f"  Segments: {len(segments)}")
        print(f"  Text length: {len(full_text)} chars")
        
        # Filter out very short segments
        valid_segments = [s for s in segments if len(s["text"]) > 10]
        
        if not valid_segments:
            print("  No valid segments to index")
            continue
        
        # Generate embeddings
        texts = [s["text"] for s in valid_segments]
        print(f"  Generating embeddings for {len(texts)} segments...")
        embeddings = model.encode(texts, show_progress_bar=True).tolist()
        
        # Index in ChromaDB
        video_id = video_path.stem
        ids = [f"{video_id}_seg_{i:05d}" for i in range(len(valid_segments))]
        
        metadatas = [
            {
                "video_id": video_id,
                "video_name": video_path.name,
                "segment_index": i,
                "start_time": s["start"],
                "end_time": s["end"],
            }
            for i, s in enumerate(valid_segments)
        ]
        
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        total_segments += len(valid_segments)
        
        # Save metadata
        meta = {
            "video_path": str(video_path),
            "video_id": video_id,
            "segments_indexed": len(valid_segments),
            "text_length": len(full_text),
        }
        
        meta_path = Path("back/videos/processed") / f"{video_id}_indexed.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        
        print(f"  Indexed {len(valid_segments)} segments")
    
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"Total segments indexed: {total_segments}")
    print(f"{'='*60}")


def query_videos(query: str, top_k: int = 5):
    """Query indexed videos."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    chroma_path = Path("back/videos/chroma_db")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection("video_transcripts")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    print(f"\nQuery: {query}\n")
    print(f"Top {len(results['documents'][0])} results:\n")
    
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), 1):
        print(f"{i}. [{meta['video_name']}] ({meta['start_time']:.0f}s - {meta['end_time']:.0f}s)")
        print(f"   {doc}")
        print(f"   Relevance: {1 - dist:.2%}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Index and query video transcripts")
    parser.add_argument("--index", action="store_true", help="Index all videos with SRT files")
    parser.add_argument("--query", type=str, help="Query indexed videos")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent  # back/
    
    video_srt_pairs = [
        (base_dir / "videos/raw/Top15_System_Design_Patterns.mp4", base_dir / "videos/raw/Top15_System_Design_Patterns.en.srt"),
        (base_dir / "videos/raw/Event_Driven_Architecture.mp4", base_dir / "videos/raw/Event_Driven_Architecture.en.srt"),
        (base_dir / "videos/raw/Event_Sourcing_CQRS.mp4", base_dir / "videos/raw/Event_Sourcing_CQRS.en.srt"),
        (base_dir / "videos/raw/Master_Software_Architecture_GOTO_2025.mp4", base_dir / "videos/raw/Master_Software_Architecture_GOTO_2025.en.srt"),
        (base_dir / "videos/raw/Scalable_Resilient_Architectures.mp4", base_dir / "videos/raw/Scalable_Resilient_Architectures.en.srt"),
    ]
    
    if args.index:
        # Filter to only existing files
        valid_pairs = [(v, s) for v, s in video_srt_pairs if v.exists() and s.exists()]
        print(f"Found {len(valid_pairs)} videos with SRT files")
        index_transcripts(valid_pairs)
    
    if args.query:
        query_videos(args.query, args.top_k)
    
    if not args.index and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()
