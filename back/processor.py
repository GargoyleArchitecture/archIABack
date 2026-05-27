"""
ArchIA Unified Processor

Procesamiento automático de PDFs (RAG) y videos (EVRAG).

Uso:
    poetry run python -m processor --watch
    poetry run python -m processor --scan
    poetry run python -m processor --pdf file.pdf
    poetry run python -m processor --video file.mp4
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env en la misma carpeta
load_dotenv(Path(__file__).resolve().parent / ".env")

def check_system_dependencies() -> bool:
    """Verifica si las dependencias críticas (OpenCV, FFmpeg, ChromaDB) están instaladas."""
    missing = []
    try:
        import cv2 # noqa: F401
    except ImportError:
        missing.append("opencv-python")

    import shutil
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg (herramienta de sistema)")

    try:
        import chromadb # noqa: F401
    except ImportError:
        missing.append("chromadb (necesario para base de datos vectorial)")

    if missing:
        print("\n❌ Error: Faltan dependencias críticas para el funcionamiento del sistema.")
        for m in missing:
            print(f"   - {m}")
        return False
    return True

def process_pdf(pdf_path: Path) -> bool:
    print(f"\n📄 Processing PDF: {pdf_path.name}")
    try:
        # Alta Cohesión: En una futura iteración, src.rag_agent debería permitir indexar
        # un archivo individual sin re-escanear todo el directorio.
        from .src.rag_agent import create_or_load_vectorstore

        print(f"   Adding {pdf_path.name} to RAG index...")
        # Si create_or_load_vectorstore soporta paths específicos, pásalo aquí.
        create_or_load_vectorstore()
        print(f"   ✅ {pdf_path.name} indexed")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def process_video(video_path: Path, pipeline: Optional[object] = None) -> bool:
    """
    Procesa un video individual usando el pipeline de EVRAG.
    Permite recibir una instancia de pipeline existente para reutilizar modelos de IA.
    """
    print(f"\n🎬 Processing video: {video_path.name}")
    try:
        from .evrag import EVRAGPipeline

        if pipeline is None:
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
    from .watcher import create_watcher_with_handlers
    from .evrag import EVRAGPipeline

    # Validar presencia de API Key antes de iniciar
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY no encontrada.")
        print("Asegúrate de que el archivo back/.env existe y es correcto.")
        return

    print("\n🔍 Starting ArchIA Unified Watcher...")
    print("   Monitoring:")
    print("   - back/docs/ (PDFs)")
    print("   - back/videos/raw/ (videos)")
    print("\n   Press Ctrl+C to stop\n")

    # Reutilización de recursos: Instanciamos el pipeline una vez para el watcher.
    # Esto evita recargas costosas de Whisper/CLIP por cada archivo nuevo detectado.
    shared_video_pipeline = EVRAGPipeline()

    watcher = create_watcher_with_handlers(
        rag_index_func=process_pdf,
        evrag_process_func=lambda p: process_video(p, pipeline=shared_video_pipeline),
    )

    print("📊 Scanning existing files...")
    watcher.scan_existing_files()
    watcher.run_forever()


def scan_existing():
    from .watcher import ArchIAWatcher

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
        if not check_system_dependencies():
            sys.exit(1)
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
        if not check_system_dependencies():
            sys.exit(1)
        process_pdf(pdf_path)
        return

    # Process specific video
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)
        if not check_system_dependencies():
            sys.exit(1)
        process_video(video_path)
        return

    # Process all existing
    if args.all:
        from .watcher import ArchIAWatcher

        if not check_system_dependencies():
            sys.exit(1)

        watcher = ArchIAWatcher()
        existing = watcher.scan_existing_files()

        print("\n🔄 Processing all existing files...\n")

        # Process PDFs
        for pdf_path in existing["pdfs"]:
            process_pdf(Path(pdf_path))

        # Process videos
        from .evrag import EVRAGPipeline
        # Optimización por lotes: Instancia única para procesar toda la lista
        video_pipeline = EVRAGPipeline()
        for video_path in existing["videos"]:
            process_video(Path(video_path), pipeline=video_pipeline)

        return

    # No arguments - show help
    parser.print_help()
    print("\nError: No action specified. Use --watch, --scan, --pdf, --video, or --all")
    sys.exit(1)


if __name__ == "__main__":
    main()
