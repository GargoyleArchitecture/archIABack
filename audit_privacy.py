import asyncio
import json
import os
from pathlib import Path
from back.evrag.indexer import EVRAGIndexer
from back.evrag.privacy import TextAnonymizer, ImageAnonymizer
import chromadb

def get_win_path(p):
    """Convierte una ruta a formato extendido de Windows para evitar Errno 22."""
    abs_p = os.path.abspath(str(p))
    if not abs_p.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_p
    return abs_p

async def audit_and_purge():
    print("--- Iniciando PURGA INTELIGENTE (Versión Ultra-Robusta Windows) ---")
    
    indexer = EVRAGIndexer()
    anonymizer = TextAnonymizer(language="es")
    img_anonymizer = ImageAnonymizer()
    
    # Mapeo de layouts
    frame_layouts = {}
    
    print("\n1. Clasificando escenas desde ChromaDB...")
    descriptions_coll = indexer.chroma_client.get_collection(name="evrag_videos_descriptions")
    results = descriptions_coll.get()
    
    for i in range(len(results['ids'])):
        text = results['documents'][i].lower()
        meta = results['metadatas'][i]
        f_path = meta.get("frame_path", "")
        if f_path:
            frame_layouts[Path(f_path).name] = "gallery" if any(kw in text for kw in ["galería", "cuadrícula", "lista"]) else ("speaker" if "rostro" in text else "slide")

    print("\n2. Anonimizando texto...")
    for coll_name in ["evrag_videos_transcript", "evrag_videos_descriptions"]:
        coll = indexer.chroma_client.get_collection(name=coll_name)
        res = coll.get()
        u_ids, u_docs, u_metas = [], [], []
        for j in range(len(res['ids'])):
            anon_res = anonymizer.anonymize(res['documents'][j])
            if anon_res.entities_removed:
                u_ids.append(res['ids'][j])
                u_docs.append(anon_res.anonymized_text)
                u_metas.append(res['metadatas'][j])
        if u_ids:
            coll.update(ids=u_ids, documents=u_docs, metadatas=u_metas)

    print("\n3. Aplicando Esquemas de Privacidad a las imágenes...")
    frames_dir = Path("back/videos/frames")
    frame_files = list(frames_dir.glob("*.jpg"))
    
    counts = {"slide": 0, "gallery": 0, "speaker": 0}
    for frame_path in frame_files:
        if frame_path.name == "test_blurred.jpg": continue
        layout = frame_layouts.get(frame_path.name, "slide")
        try:
            # USAR RUTA EXTENDIDA PARA EVITAR ERRNO 22
            win_p = get_win_path(frame_path)
            img_anonymizer.process_image(win_p, layout_type=layout)
            counts[layout] += 1
            if sum(counts.values()) % 25 == 0:
                print(f"    {sum(counts.values())}/{len(frame_files)} procesadas...")
        except Exception as e:
            print(f"    [Error Imagen] {frame_path.name}: {e}")

    print("\n4. Sincronizando metadatos JSON...")
    processed_dir = Path("back/videos/processed")
    for json_path in processed_dir.glob("*.json"):
        try:
            win_json_p = get_win_path(json_path)
            with open(win_json_p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "transcript" in data:
                data["transcript"] = anonymizer.anonymize(data["transcript"]).anonymized_text
                with open(win_json_p, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"    [Error JSON] {json_path.name}: {e}")

    print("\n--- PROCESO FINALIZADO CON ÉXITO ---")

if __name__ == "__main__":
    asyncio.run(audit_and_purge())
