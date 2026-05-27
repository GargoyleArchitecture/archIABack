"""
Process videos using existing SRT subtitles instead of Whisper transcription.
This is much faster than running Whisper on CPU.
"""

import json
import re
from pathlib import Path
from datetime import datetime


def parse_srt_to_segments(srt_path: Path) -> list[dict]:
    """Parse SRT file into segments with timestamps."""
    content = srt_path.read_text(encoding='utf-8')
    
    # Pattern: number \n timestamp --> timestamp \n text
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n\d+\n|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    segments = []
    for num, start_str, end_str, text in matches:
        # Convert timestamp to seconds
        start_sec = _timestamp_to_seconds(start_str)
        end_sec = _timestamp_to_seconds(end_str)
        
        # Clean text
        text = text.strip()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
        
        if text.strip():
            segments.append({
                "start": start_sec,
                "end": end_sec,
                "text": text.strip()
            })
    
    return segments


def _timestamp_to_seconds(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.replace(',', '.')
    parts = ts.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def process_video_with_srt(video_path: Path, srt_path: Path, evrag_pipeline):
    """Process a video using existing SRT subtitles."""
    print(f"\n{'='*60}")
    print(f"EVRAG: Processing {video_path.name} (using SRT)")
    print(f"{'='*60}\n")
    
    # Step 1: Process video (scene detection + frame extraction)
    print("Step 1: Processing video (scene detection)...")
    video_info = evrag_pipeline.video_processor.process_video(
        video_path=video_path,
        force_reprocess=True,
    )
    
    frames = [Path(f) for f in video_info["frame_paths"]]
    print(f"  Scenes detected: {video_info['scenes_detected']}")
    print(f"  Frames extracted: {video_info['frames_extracted']}")
    
    # Step 2: Use existing SRT instead of Whisper
    print(f"\nStep 2: Loading SRT subtitles from {srt_path.name}...")
    segments = parse_srt_to_segments(srt_path)
    
    # Build full text
    full_text = " ".join([seg["text"] for seg in segments])
    
    # --- APLICAR PRIVACIDAD (CENSURA DE NOMBRES CON LLM + REGEX) ---
    try:
        print("\nStep 2.5: Anonymizing transcripts (detecting names with LLM)...")
        from evrag.privacy import PrivacyProcessor
        privacy = PrivacyProcessor()
        privacy_result = privacy.anonymize_transcript(full_text)
        
        if privacy_result["names_found"]:
            print(f"  Names found and redacted: {', '.join(privacy_result['names_found'])}")
        else:
            print("  No names found to redact.")
            
        full_text = privacy_result["anonymized_text"]
        
        # Anonymize each segment using the names found
        import re
        for seg in segments:
            for name in privacy_result["names_found"]:
                if name != "FALLBACK_REGEX_APPLIED":
                    pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
                    seg["text"] = pattern.sub("[REDACTADO]", seg["text"])
                else:
                    seg["text"] = re.sub(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', '[REDACTADO]', seg["text"])
    except Exception as e:
        print(f"  Privacy Warning: Could not anonymize transcript: {e}")
    
    # Convert segments to Whisper-like format
    whisper_segments = []
    for seg in segments:
        whisper_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })
    
    print(f"  Transcript length: {len(full_text)} chars")
    print(f"  Segments: {len(segments)}")
    
    # Step 3: Index in ChromaDB
    print("\nStep 3: Indexing in ChromaDB...")
    evrag_pipeline.indexer.index_video(
        video_path=video_path,
        frames=frames,
        frame_embeddings=None,  # No CLIP for now
        transcript_text=full_text,
        transcript_segments=whisper_segments,
        video_duration_sec=video_info["video_duration_sec"],
    )
    
    # Save processed info
    processed_info_path = Path(evrag_pipeline.config["processed_dir"]) / f"{video_path.stem}_evrag.json"
    processed_info_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "video_path": str(video_path),
        "scenes_detected": video_info["scenes_detected"],
        "frames_extracted": video_info["frames_extracted"],
        "transcript_length": len(full_text),
        "indexed": True,
        "processing_time_sec": 0,
        "processed_at": datetime.now().isoformat(),
        "method": "srt_subtitles"
    }
    
    processed_info_path.write_text(json.dumps(result, indent=2))
    
    print(f"\n{'='*60}")
    print(f"✅ Processing Complete!")
    print(f"  Video: {video_path.name}")
    print(f"  Scenes: {video_info['scenes_detected']}")
    print(f"  Frames: {video_info['frames_extracted']}")
    print(f"  Transcript: {len(full_text)} chars")
    print(f"{'='*60}\n")
    
    return result


def main():
    import sys
    from pathlib import Path
    # Add back/ directory to path
    back_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(back_dir))
    
    # Override working directory for EVRAG config
    import os
    os.chdir(back_dir.parent)  # Change to project root
    
    from evrag.pipeline import EVRAGPipeline
    
    # Initialize pipeline (disable face blur and anonymization for speed)
    pipeline = EVRAGPipeline(
        enable_anonymization=False,
        enable_face_blur=False,
        secure_delete_originals=False,
    )
    
    # Video mappings - use absolute paths
    base_dir = back_dir  # back/
    videos = [
        {
            "video": base_dir / "videos/raw/Top15_System_Design_Patterns.mp4",
            "srt": base_dir / "videos/raw/Top15_System_Design_Patterns.en.srt"
        },
        {
            "video": base_dir / "videos/raw/Event_Driven_Architecture.mp4",
            "srt": base_dir / "videos/raw/Event_Driven_Architecture.en.srt"
        },
        {
            "video": base_dir / "videos/raw/Event_Sourcing_CQRS.mp4",
            "srt": base_dir / "videos/raw/Event_Sourcing_CQRS.en.srt"
        },
        {
            "video": base_dir / "videos/raw/Master_Software_Architecture_GOTO_2025.mp4",
            "srt": base_dir / "videos/raw/Master_Software_Architecture_GOTO_2025.en.srt"
        },
        {
            "video": base_dir / "videos/raw/Scalable_Resilient_Architectures.mp4",
            "srt": base_dir / "videos/raw/Scalable_Resilient_Architectures.en.srt"
        },
    ]
    
    for i, v in enumerate(videos, 1):
        video_path = Path(v["video"])
        srt_path = Path(v["srt"])
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            continue
        
        if not srt_path.exists():
            print(f"❌ SRT not found: {srt_path}")
            continue
        
        print(f"\n[{i}/{len(videos)}] Processing {video_path.name}...")
        
        try:
            process_video_with_srt(video_path, srt_path, pipeline)
        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # List all indexed videos
    print("\n" + "="*60)
    print("INDEXED VIDEOS")
    print("="*60)
    indexed = pipeline.get_indexed_videos()
    for video in indexed:
        print(f"  ✅ {Path(video.video_path).name} - {video.duration_sec:.0f}s, {len(video.frames)} frames")
    
    print(f"\nTotal: {len(indexed)} videos indexed")


if __name__ == "__main__":
    main()
