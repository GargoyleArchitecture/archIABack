"""
ArchIA Unified Processor

Procesamiento automático de PDFs (RAG) y videos (EVRAG).

Uso:
    poetry run python -m back.processor --watch
    poetry run python -m back.processor --scan
    poetry run python -m back.processor --pdf file.pdf
    poetry run python -m back.processor --video file.mp4
"""

import argparse
import sys
from pathlib import Path


def process_pdf(pdf_path: Path) -> bool:
    print(f"\n📄 Processing PDF: {pdf_path.name}")
    try:
        from back.rag_agent import create_or_load_vectorstore

        print("   Adding to RAG index...")
        create_or_load_vectorstore()
        print("   ✅ PDF indexed")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def process_video(video_path: Path) -> bool:
    print(f"\n🎬 Processing video: {video_path.name}")
    try:
        from back.evrag import EVRAGPipeline

        pipeline = EVRAGPipeline(
            enable_anonymization=True,
            enable_face_blur=True,
            secure_delete_originals=True,
        )

        result = pipeline.process_video(video_path)
        print(f"   ✅ Video processed: {result.frames_extracted} frames")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def run_watcher():
    from back.watcher import create_watcher_with_handlers

    print("\n🔍 Starting ArchIA Unified Watcher...")
    print("   Monitoring:")
    print("   - back/docs/ (PDFs)")
    print("   - back/videos/raw/ (videos)")
    print("\n   Press Ctrl+C to stop\n")

    watcher = create_watcher_with_handlers(
        rag_index_func=process_pdf,
        evrag_process_func=process_video,
    )

    print("📊 Scanning existing files...")
    watcher.scan_existing_files()
    watcher.run_forever()


def scan_existing():
    from back.watcher import ArchIAWatcher

    watcher = ArchIAWatcher()
    watcher.scan_existing_files()


def main():
    parser = argparse.ArgumentParser(
        description="ArchIA Unified Processor - RAG + EVRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --watch                    # Start file watcher
  %(prog)s --scan                     # Scan existing files
  %(prog)s --pdf documento.pdf        # Process specific PDF
  %(prog)s --video tutorial.mp4       # Process specific video
  %(prog)s --all                      # Process all existing files
        """,
    )
    
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Start file watcher (continuous monitoring)",
    )
    
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan existing files and report",
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        help="Process specific PDF file",
    )
    
    parser.add_argument(
        "--video",
        type=str,
        help="Process specific video file",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all existing files",
    )
    
    args = parser.parse_args()
    
    # Watch mode
    if args.watch:
        run_watcher()
        return
    
    # Scan mode
    if args.scan:
        scan_existing()
        return
    
    # Process specific PDF
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            sys.exit(1)
        process_pdf(pdf_path)
        return
    
    # Process specific video
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)
        process_video(video_path)
        return
    
    # Process all existing
    if args.all:
        from back.watcher import ArchIAWatcher
        
        watcher = ArchIAWatcher()
        existing = watcher.scan_existing_files()
        
        print("\n🔄 Processing all existing files...\n")
        
        # Process PDFs
        for pdf_path in existing["pdfs"]:
            process_pdf(Path(pdf_path))
        
        # Process videos
        for video_path in existing["videos"]:
            process_video(Path(video_path))
        
        return
    
    # No arguments - show help
    parser.print_help()
    print("\nError: No action specified. Use --watch, --scan, --pdf, --video, or --all")
    sys.exit(1)


if __name__ == "__main__":
    main()
