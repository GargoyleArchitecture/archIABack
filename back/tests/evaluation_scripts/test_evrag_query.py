from pathlib import Path
# pyrefly: ignore [missing-import]
from evrag.indexer import EVRAGIndexer

def main():
    print("Iniciando indexador EVRAG (cargando base de datos)...")
    indexer = EVRAGIndexer()

    # Algunas consultas interesantes para buscar en la arquitectura de software
    queries = [
        "¿Cuáles son los atributos de calidad más importantes para la arquitectura?",
        "patrones de arquitectura y escalabilidad",
        "diagrama de componentes y despliegue"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"🔎 BUSCANDO: '{query}'")
        print(f"{'='*80}\n")

        # Hacemos la consulta a la base de datos ChromaDB
        results = indexer.query_multimodal(
            query=query,
            top_k_frames=2,     # Los 2 frames más relevantes (si tuvieras CLIP activado)
            top_k_segments=5    # Los 5 fragmentos de transcripción más relevantes
        )

        print("--- 📝 TRANSCRIPCIONES ENCONTRADAS ---")
        if not results["segments"]:
            print("No se encontraron resultados de texto.")
        else:
            for i, seg in enumerate(results["segments"], 1):
                # Extraemos el video (sin el sufijo largo para que sea más legible)
                video_name = seg['video_id'].split('-')[0]

                # Formatear el tiempo (de segundos a minutos:segundos)
                start_min, start_sec = divmod(int(seg['start_time']), 60)
                end_min, end_sec = divmod(int(seg['end_time']), 60)
                time_str = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"

                print(f"\n{i}. 📼 Video: {video_name} {time_str}")
                print(f"   Score de relevancia: {seg['distance']:.4f} (menor es mejor)")
                print(f"   💬 Texto: \"{seg['text'].strip()}\"")

        print("\n--- 🖼️ FRAMES DE VIDEO ---")
        if not results["frames"]:
            print("No se buscaron imágenes porque CLIP no está instalado/activado.")
        else:
            for i, frame in enumerate(results["frames"], 1):
                print(f"{i}. Video: {frame['video_id']}")
                print(f"   Imagen: {frame['frame_path']}")
                print(f"   Score: {frame['distance']:.4f}")

if __name__ == "__main__":
    main()
