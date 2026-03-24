"""
File Watcher for ArchIA

Detecta cambios en back/docs/ y back/videos/raw/
para procesamiento automático.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


@dataclass
class WatcherStats:
    """Watcher statistics."""
    files_processed: int = 0
    pdfs_indexed: int = 0
    videos_processed: int = 0
    errors: int = 0
    last_activity: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "pdfs_indexed": self.pdfs_indexed,
            "videos_processed": self.videos_processed,
            "errors": self.errors,
            "last_activity": self.last_activity,
        }


class ArchIAFileHandler(FileSystemEventHandler):
    """
    Handler para eventos de archivos.
    
    Solo procesa archivos nuevos (usa manifiesto para evitar reprocesar).
    """
    
    def __init__(
        self,
        on_pdf_detected: Callable | None = None,
        on_video_detected: Callable | None = None,
        watch_docs: bool = True,
        watch_videos: bool = True,
    ):
        """
        Initialize file handler.
        
        Args:
            on_pdf_detected: Callback when new PDF is detected
            on_video_detected: Callback when new video is detected
            watch_docs: Whether to watch docs directory
            watch_videos: Whether to watch videos directory
        """
        super().__init__()
        
        self.on_pdf_detected = on_pdf_detected
        self.on_video_detected = on_video_detected
        self.watch_docs = watch_docs
        self.watch_videos = watch_videos
        
        self.stats = WatcherStats()
        
        # Video extensions to watch for
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        
        # Track processed files to avoid reprocessing
        self.processed_files: set[str] = set()
        self.manifest_path = Path("back/.processed_files.json")
        self._load_manifest()
    
    def _load_manifest(self):
        """Load manifest of already processed files."""
        if self.manifest_path.exists():
            import json
            try:
                data = json.loads(self.manifest_path.read_text())
                self.processed_files = set(data.get("processed_files", []))
                print(f"📋 Loaded manifest: {len(self.processed_files)} files already processed")
            except Exception as e:
                print(f"Warning: Could not load manifest: {e}")
                self.processed_files = set()
        else:
            print("📋 No manifest found - will process new files only")
    
    def _save_manifest(self):
        """Save manifest of processed files."""
        import json
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "processed_files": list(self.processed_files),
            "last_updated": datetime.now().isoformat(),
        }
        self.manifest_path.write_text(json.dumps(data, indent=2))
    
    def _is_already_processed(self, file_path: Path) -> bool:
        """Check if file was already processed."""
        # Use absolute path + modification time as key
        file_key = f"{file_path.absolute()}"
        return file_key in self.processed_files
    
    def _mark_as_processed(self, file_path: Path):
        """Mark file as processed and save manifest."""
        file_key = f"{file_path.absolute()}"
        self.processed_files.add(file_key)
        self._save_manifest()
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Ignore temporary files
        if file_path.name.startswith('~$') or file_path.name.startswith('.'):
            return
        
        # Ignore files in subdirectories we don't care about
        if 'chroma_db' in str(file_path) or 'processed' in str(file_path):
            return
        
        # Wait a bit for file to be fully written
        time.sleep(1)
        
        if file_path.suffix.lower() == '.pdf':
            self._handle_new_pdf(file_path)
        
        elif file_path.suffix.lower() in self.video_extensions:
            self._handle_new_video(file_path)
    
    def on_modified(self, event):
        """Handle file modification events (treat as new file)."""
        # Only process modifications for PDFs and videos
        # This catches cases where files are copied/updated
        self.on_created(event)
    
    def _handle_new_pdf(self, pdf_path: Path):
        """Handle new PDF file."""
        # Check if already processed
        if self._is_already_processed(pdf_path):
            print(f"\n⏭️  PDF already processed: {pdf_path.name}")
            return
        
        print(f"\n📄 New PDF detected: {pdf_path.name}")
        
        self.stats.files_processed += 1
        self.stats.last_activity = datetime.now().isoformat()
        
        if self.on_pdf_detected:
            try:
                self.on_pdf_detected(pdf_path)
                self.stats.pdfs_indexed += 1
                print(f"   ✅ PDF indexed successfully")
                
                # Mark as processed
                self._mark_as_processed(pdf_path)
                
            except Exception as e:
                print(f"   ❌ Error indexing PDF: {e}")
                self.stats.errors += 1
        else:
            print(f"   ⚠️ No PDF handler configured")
    
    def _handle_new_video(self, video_path: Path):
        """Handle new video file."""
        # Check if already processed
        if self._is_already_processed(video_path):
            print(f"\n⏭️  Video already processed: {video_path.name}")
            return
        
        print(f"\n🎬 New video detected: {video_path.name}")
        
        self.stats.files_processed += 1
        self.stats.last_activity = datetime.now().isoformat()
        
        if self.on_video_detected:
            try:
                self.on_video_detected(video_path)
                self.stats.videos_processed += 1
                print(f"   ✅ Video processed successfully")
                
                # Mark as processed
                self._mark_as_processed(video_path)
                
            except Exception as e:
                print(f"   ❌ Error processing video: {e}")
                self.stats.errors += 1
        else:
            print(f"   ⚠️ No video handler configured")
    
    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics."""
        return self.stats.to_dict()


class ArchIAWatcher:
    """
    Main file watcher for ArchIA project.
    
    Monitors:
    - back/docs/ for new PDFs (triggers RAG indexing)
    - back/videos/raw/ for new videos (triggers EVRAG processing)
    
    Usage:
        watcher = ArchIAWatcher()
        watcher.start()
        
        # Keep running...
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()
    """
    
    def __init__(
        self,
        docs_dir: Path | None = None,
        videos_dir: Path | None = None,
        watch_docs: bool = True,
        watch_videos: bool = True,
        auto_start: bool = False,
    ):
        """
        Initialize ArchIA watcher.
        
        Args:
            docs_dir: Directory to watch for PDFs (default: back/docs)
            videos_dir: Directory to watch for videos (default: back/videos/raw)
            watch_docs: Whether to watch docs directory
            watch_videos: Whether to watch videos directory
            auto_start: Whether to start watching immediately
        """
        from back.eval.config import EVAL_CONFIG
        from back.evrag.config import EVRAG_CONFIG
        
        self.docs_dir = docs_dir or Path(EVAL_CONFIG["docs_dir"])
        self.videos_dir = videos_dir or Path(EVRAG_CONFIG["videos_dir"]) / "raw"
        
        self.watch_docs = watch_docs
        self.watch_videos = watch_videos
        
        # Handlers for processing
        self.pdf_handler: Callable | None = None
        self.video_handler: Callable | None = None
        
        # Create file system handler
        self.fs_handler = ArchIAFileHandler(
            on_pdf_detected=self._on_pdf,
            on_video_detected=self._on_video,
            watch_docs=watch_docs,
            watch_videos=watch_videos,
        )
        
        # Create observer
        self.observer = Observer()
        
        # Register directories
        self._setup_watcher()
        
        self._running = False
    
    def _setup_watcher(self):
        """Set up watcher for configured directories."""
        if self.watch_docs and self.docs_dir.exists():
            self.observer.schedule(self.fs_handler, str(self.docs_dir), recursive=False)
            print(f"📁 Watching PDFs in: {self.docs_dir}")
        
        if self.watch_videos and self.videos_dir.exists():
            self.observer.schedule(self.fs_handler, str(self.videos_dir), recursive=False)
            print(f"🎬 Watching videos in: {self.videos_dir}")
    
    def set_pdf_handler(self, handler: Callable):
        """
        Set handler function for new PDFs.
        
        Args:
            handler: Function that takes Path and processes PDF
        """
        self.pdf_handler = handler
    
    def set_video_handler(self, handler: Callable):
        """
        Set handler function for new videos.
        
        Args:
            handler: Function that takes Path and processes video
        """
        self.video_handler = handler
    
    def _on_pdf(self, pdf_path: Path):
        """Internal PDF handler."""
        if self.pdf_handler:
            self.pdf_handler(pdf_path)
    
    def _on_video(self, video_path: Path):
        """Internal video handler."""
        if self.video_handler:
            self.video_handler(video_path)
    
    def start(self):
        """Start watching."""
        self.observer.start()
        self._running = True
        print("\n✅ ArchIA Watcher started")
        print("   Press Ctrl+C to stop\n")
    
    def stop(self):
        """Stop watching."""
        self.observer.stop()
        self.observer.join()
        self._running = False
        print("\n🛑 ArchIA Watcher stopped")
    
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running
    
    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics."""
        return self.fs_handler.get_stats()
    
    def run_forever(self):
        """
        Start watcher and run until interrupted.
        
        Blocks until Ctrl+C is pressed.
        """
        self.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def scan_existing_files(self) -> dict[str, list[str]]:
        """
        Scan existing files in watched directories.
        
        Returns:
            Dictionary with lists of existing PDFs and videos
        """
        result = {
            "pdfs": [],
            "videos": [],
        }
        
        # Scan PDFs
        if self.docs_dir.exists():
            for pdf in self.docs_dir.glob("*.pdf"):
                result["pdfs"].append(str(pdf))
        
        # Scan videos
        if self.videos_dir.exists():
            for video_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
                for video in self.videos_dir.glob(f"*{video_ext}"):
                    result["videos"].append(str(video))
        
        print(f"\n📊 Existing files found:")
        print(f"   PDFs: {len(result['pdfs'])}")
        print(f"   Videos: {len(result['videos'])}")
        
        return result


def create_watcher_with_handlers(
    rag_index_func: Callable | None = None,
    evrag_process_func: Callable | None = None,
) -> ArchIAWatcher:
    """
    Create watcher with default handlers for RAG and EVRAG.
    
    Args:
        rag_index_func: Function to index PDFs in RAG
        evrag_process_func: Function to process videos with EVRAG
        
    Returns:
        Configured ArchIAWatcher instance
    """
    watcher = ArchIAWatcher()
    
    # Set up PDF handler
    if rag_index_func:
        def handle_pdf(pdf_path: Path):
            print(f"   Indexing PDF in RAG...")
            rag_index_func(pdf_path)
        
        watcher.set_pdf_handler(handle_pdf)
    
    # Set up video handler
    if evrag_process_func:
        def handle_video(video_path: Path):
            print(f"   Processing video with EVRAG...")
            evrag_process_func(video_path)
        
        watcher.set_video_handler(handle_video)
    
    return watcher


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ArchIA File Watcher - Monitor docs and videos for automatic processing"
    )
    
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Watch only docs directory (PDFs)",
    )
    
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Watch only videos directory",
    )
    
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan existing files and exit",
    )
    
    args = parser.parse_args()
    
    # Create watcher
    watcher = ArchIAWatcher(
        watch_docs=not args.videos_only,
        watch_videos=not args.docs_only,
    )
    
    # Set up default handlers (placeholder)
    def dummy_pdf_handler(path: Path):
        print(f"   → Would index: {path.name}")
        # TODO: Integrate with build_vectorstore.py
    
    def dummy_video_handler(path: Path):
        print(f"   → Would process: {path.name}")
        # TODO: Integrate with EVRAG pipeline
    
    watcher.set_pdf_handler(dummy_pdf_handler)
    watcher.set_video_handler(dummy_video_handler)
    
    if args.scan:
        # Just scan and exit
        watcher.scan_existing_files()
    else:
        # Run watcher
        print("\n🔍 Starting ArchIA File Watcher...")
        print("   Watching for new PDFs and videos\n")
        watcher.run_forever()
