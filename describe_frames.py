import asyncio
from pathlib import Path
from back.evrag.indexer import EVRAGIndexer
from back.evrag.descriptor import FrameDescriptor
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    print("--- Iniciando Descripción de Frames con IA de Visión ---")
    
    indexer = EVRAGIndexer()
    descriptor = FrameDescriptor()
    
    # Obtener videos ya indexados
    videos = indexer.get_indexed_videos()
    
    if not videos:
        print("No se encontraron videos indexados. Corre primero el indexador.")
        return

    for video in videos:
        video_id = Path(video.video_path).stem
        print(f"\nProcesando video: {video_id}")
        
        frames = [Path(f) for f in video.frames]
        descriptions = []
        
        # Verificar cuántos frames ya tienen descripción para no repetir
        # (Por ahora procesaremos todos para asegurar calidad)
        print(f"Describiendo {len(frames)} frames. Esto puede tardar un poco...")
        
        for i, frame_path in enumerate(frames):
            if not frame_path.exists():
                print(f"  [Error] No se encuentra el frame: {frame_path}")
                continue
                
            print(f"  [{i+1}/{len(frames)}] Describiendo: {frame_path.name}")
            description = await descriptor.describe_frame(frame_path)
            descriptions.append(description)
            
            # Guardar una por una para evitar pérdida de datos si falla
            indexer._index_descriptions(video_id, [frame_path], [description])
            
        print(f"✅ Finalizado: {len(descriptions)} descripciones generadas e indexadas.")

if __name__ == "__main__":
    asyncio.run(main())
