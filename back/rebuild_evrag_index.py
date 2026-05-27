import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger("rebuilder")

# Importar componentes de EVRAG (asumiendo que estamos en 'back/')
try:
    from evrag.indexer import EVRAGIndexer
    from evrag.clip_embedder import CLIPEmbedder
    from evrag.config import EVRAG_CONFIG
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from evrag.indexer import EVRAGIndexer
    from evrag.clip_embedder import CLIPEmbedder
    from evrag.config import EVRAG_CONFIG

def rebuild_index():
    base_dir = Path(__file__).parent
    frames_dir = base_dir / "videos" / "frames"
    transcripts_dir = base_dir / "videos" / "transcripts"
    
    indexer = EVRAGIndexer()
    embedder = CLIPEmbedder()
    
    print(f"\n{'='*60}")
    print("🛠️ RECONSTRUYENDO ÍNDICE EVRAG (MULTIMODAL)")
    print(f"{'='*60}\n")

    # 1. PROCESAR TRANSCRIPCIONES
    print("📝 Indexando transcripciones...")
    transcript_files = list(transcripts_dir.glob("*.json"))
    for ts_file in transcript_files:
        try:
            data = json.loads(ts_file.read_text(encoding="utf-8"))
            video_id = ts_file.stem.replace("_transcript", "")
            
            # Formatear segmentos para el indexador
            segments = []
            if isinstance(data, dict) and "segments" in data:
                segments = data["segments"]
            elif isinstance(data, list):
                segments = data
            
            full_text = " ".join([s.get("text", "") for s in segments])
            
            print(f"   - {video_id}: {len(segments)} segmentos")
            indexer._index_transcript(video_id, full_text, segments)
        except Exception as e:
            log.error(f"Error indexando {ts_file.name}: {e}")

    # 2. PROCESAR FRAMES
    print("\n🖼️ Indexando frames visuales (con CLIP)...")
    frame_files = sorted(list(frames_dir.glob("*.jpg")))
    
    # Agrupar por video ID
    videos_frames = {}
    for f in frame_files:
        # Heurística: el video_id es el prefijo antes de '_scene_'
        if "_scene_" in f.name:
            video_id = f.name.split("_scene_")[0]
            if video_id not in videos_frames:
                videos_frames[video_id] = []
            videos_frames[video_id].append(f)
    
    for video_id, frames in videos_frames.items():
        print(f"   - {video_id}: {len(frames)} frames")
        try:
            # Generar embeddings para este lote de frames
            print(f"     Generando embeddings CLIP...")
            embeddings = embedder.embed_images(frames)
            
            # Indexar en ChromaDB
            indexer._index_frames(video_id, frames, embeddings)
            print(f"     ✅ Indexado exitoso")
        except Exception as e:
            log.error(f"Error indexando frames de {video_id}: {e}")

    print(f"\n{'='*60}")
    print("✅ RECONSTRUCCIÓN COMPLETADA")
    print(f"Se han indexado {len(transcript_files)} transcripciones y {len(frame_files)} frames.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    rebuild_index()
