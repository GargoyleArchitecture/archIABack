import sys
sys.path.append('.')
from evrag.indexer import EVRAGIndexer
indexer = EVRAGIndexer()
results = indexer.query_multimodal('latencia', top_k=5, mode='hybrid')
print('Results length frames:', len(results['frames']))
print('Results length segments:', len(results['segments']))
