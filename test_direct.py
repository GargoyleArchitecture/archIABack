import sys
sys.path.append('back')
from evrag.indexer import EVRAGIndexer

print('Iniciando indexador EVRAG...')
indexer = EVRAGIndexer()
query = '¿Qué es un atributo de calidad?'
print(f'\nBUSCANDO: {query}')

results = indexer.query_multimodal(query=query, top_k=3, mode='hybrid')

print('\n--- TRANSCRIPCIONES ENCONTRADAS ---')
for i, seg in enumerate(results.get('segments', []), 1):
    start_min, start_sec = divmod(int(seg.get('start_time', 0)), 60)
    end_min, end_sec = divmod(int(seg.get('end_time', 0)), 60)
    time_str = f'[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]'
    print(f'{i}. Video: {seg.get("video_id")} {time_str}')
    print(f'   Score: {seg.get("distance", 0):.4f}')
    print(f'   Texto: {seg.get("text", "").strip()}')
