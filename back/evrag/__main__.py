"""
EVRAG CLI

Procesar videos con EVRAG desde la línea de comandos.

Uso:
    poetry run python -m back.evrag --video back/videos/raw/mi_video.mp4
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="EVRAG: Enhanced Video Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video back/videos/raw/mi_video.mp4
  %(prog)s --video mi_video.mp4 --reprocess
  %(prog)s --query "¿Qué dice sobre latencia?"
  %(prog)s --list
        """,
    )
    
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to process",
    )
    
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing even if cached",
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Query indexed videos",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all indexed videos",
    )
    
    args = parser.parse_args()
    
    # List indexed videos
    if args.list:
        from .pipeline import EVRAGPipeline
        pipeline = EVRAGPipeline()
        
        videos = pipeline.get_indexed_videos()
        
        if not videos:
            print("No videos indexed yet.")
            return
        
        print(f"\nIndexed videos: {len(videos)}\n")
        for video in videos:
            print(f"  - {Path(video.video_path).name}")
            print(f"    Duration: {video.duration_sec:.1f}s")
            print(f"    Frames: {len(video.frames)}")
            print()
        return
    
    # Query videos
    if args.query:
        from .pipeline import EVRAGPipeline
        pipeline = EVRAGPipeline()
        
        print(f"\nQuery: {args.query}\n")
        results = pipeline.query(args.query)
        
        print("Matching frames:")
        for frame in results["frames"]:
            print(f"  - {Path(frame['frame_path']).name} (distance: {frame['distance']:.4f})")
        
        print("\nMatching transcript segments:")
        for seg in results["segments"]:
            print(f"  - [{seg['start_time']:.1f}s - {seg['end_time']:.1f}s] {seg['text'][:100]}...")
        
        return
    
    # Process video
    if not args.video:
        parser.print_help()
        print("\nError: --video is required for processing")
        sys.exit(1)
    
    video_path = Path(args.video)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    from .pipeline import EVRAGPipeline
    
    pipeline = EVRAGPipeline()
    
    result = pipeline.process_video(
        video_path=video_path,
        force_reprocess=args.reprocess,
    )
    
    print(f"\n✅ EVRAG processing complete!")
    print(f"   Video: {result.video_path}")
    print(f"   Scenes: {result.scenes_detected}")
    print(f"   Frames: {result.frames_extracted}")
    print(f"   Transcript: {result.transcript_length} chars")
    print(f"   Time: {result.processing_time_sec:.1f}s")


if __name__ == "__main__":
    main()
