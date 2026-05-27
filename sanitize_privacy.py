import asyncio
import json
import os
import shutil
import re
from pathlib import Path
from back.evrag.indexer import EVRAGIndexer
from back.evrag.privacy import TextAnonymizer, ImageAnonymizer

def slugify(text):
    """Limpia nombres de archivos de forma agresiva."""
    # Reemplazar secuencias comunes de mala codificación de Teams
    text = text.replace("Ã³", "o").replace("Ã¡", "a").replace("Ã©", "e").replace("Ã", "i").replace("Ã±", "n")
    text = text.replace(" ", "_").replace("(", "").replace(")", "")
    return re.sub(r'(?u)[^-\w.]', '', text)

async def sanitize_and_purge():
    print("--- PURGA FINAL: RESOLVIENDO COLISIONES Y CORRUPCIÓN (Windows) ---")
    
    indexer = EVRAGIndexer()
    anonymizer = TextAnonymizer(language="es")
    img_anonymizer = ImageAnonymizer()
    
    frames_dir = Path("back/videos/frames")
    processed_dir = Path("back/videos/processed")
    
    name_map = {}

    # 1. Resolver Colisiones y Sanitizar
    print("\n1. Limpiando archivos duplicados y sanitizando nombres...")
    for folder in [frames_dir, processed_dir]:
        for file_path in folder.glob("*"):
            if file_path.is_dir(): continue
            old_name = file_path.name
            new_name = slugify(old_name)
            
            if old_name != new_name:
                new_path = folder / new_name
                if new_path.exists():
                    # Si el archivo limpio ya existe, borramos el corrupto para evitar duplicados
                    try:
                        file_path.unlink()
                        print(f"    [Borrado Duplicado] {old_name}")
                        name_map[old_name] = new_name
                    except: pass
                else:
                    # Si no existe, lo renombramos
                    try:
                        os.rename(str(file_path.absolute()), str(new_path.absolute()))
                        name_map[old_name] = new_name
                    except Exception as e:
                        print(f"    [Error Rename] {old_name}: {e}")

    # 2. Sincronizar ChromaDB
    print("\n2. Sincronizando ChromaDB...")
    for coll_name in ["evrag_videos_frames", "evrag_videos_descriptions"]:
        coll = indexer.chroma_client.get_collection(name=coll_name)
        res = coll.get()
        ids, metas = [], []
        for i in range(len(res['ids'])):
            meta = res['metadatas'][i]
            old_path = meta.get("frame_path", "")
            if old_path:
                old_fname = Path(old_path).name
                new_fname = slugify(old_fname)
                if old_fname != new_fname:
                    new_path = str(Path(old_path).parent / new_fname)
                    meta["frame_path"] = new_path
                    ids.append(res['ids'][i])
                    metas.append(meta)
        if ids:
            coll.update(ids=ids, metadatas=metas)

    # 3. Aplicar Privacidad (Esquemas Inteligentes)
    print("\n3. Aplicando Privacidad a los archivos limpios...")
    desc_coll = indexer.chroma_client.get_collection(name="evrag_videos_descriptions")
    res = desc_coll.get()
    frame_layouts = {}
    for i in range(len(res['ids'])):
        text = res['documents'][i].lower()
        f_path = res['metadatas'][i].get("frame_path", "")
        if f_path:
            f_name = Path(f_path).name
            frame_layouts[f_name] = "gallery" if "galería" in text else ("speaker" if "rostro" in text else "slide")

    for frame_path in frames_dir.glob("*.jpg"):
        if frame_path.name == "test_blurred.jpg": continue
        layout = frame_layouts.get(frame_path.name, "slide")
        try:
            # Usar ruta absoluta limpia
            p = str(frame_path.absolute())
            img_anonymizer.process_image(p, layout_type=layout)
        except Exception as e:
            # Si falla el nombre limpio, algo muy raro pasa con el sistema de archivos
            print(f"    [Error Crítico] {frame_path.name}: {e}")

    # 4. Sincronizar JSONs
    print("\n4. Finalizando metadatos JSON...")
    for json_path in processed_dir.glob("*.json"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "frames" in data:
                data["frames"] = [str(Path(f).parent / slugify(Path(f).name)) for f in data["frames"]]
            
            if "transcript" in data:
                data["transcript"] = anonymizer.anonymize(data["transcript"]).anonymized_text
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except: pass

    print("\n--- LIMPIEZA Y PRIVACIDAD COMPLETADA ---")

if __name__ == "__main__":
    asyncio.run(sanitize_and_purge())
