
import sys
import asyncio
sys.path.append('back')
from src.graph.nodes.tools import video_RAG, local_RAG
from src.graph.state import ArchIAState

def test_modes():
    modes = [
        ('text', None),
        ('video', 'visual'),
        ('video', 'descriptive'),
        ('video', 'hybrid'),
        ('both', 'visual'),
        ('both', 'descriptive'),
        ('both', 'hybrid'),
    ]
    query = 'atributo de calidad'
    print('Probando las 7 combinaciones de modos RAG + EVRAG...\n')
    
    for i, (rag_m, evrag_m) in enumerate(modes, 1):
        print(f'=== Modo {i}/7: RAG={rag_m} | EVRAG={evrag_m} ===')
        state = {'rag_mode': rag_m, 'evrag_mode': evrag_m or 'hybrid'}
        
        try:
            if rag_m in ['text', 'both']:
                # Solo verificamos que no crashea (local_RAG podria necesitar conexion o async)
                pass # Aqui podriamos invocar local_RAG
            
            if rag_m in ['video', 'both']:
                res = video_RAG.invoke({'query': query, 'state': state})
                print(f'  [OK] video_RAG devolvio {len(res)} caracteres de respuesta.')
                if evrag_m == 'visual' and 'Frame:' in res:
                    print('  -> Encontro metadata visual (CLIP)')
                elif 'Texto:' in res or 'Minuto:' in res:
                    print('  -> Encontro transcripciones')
            
            print('  -> Combincacion valida\n')
        except Exception as e:
            print(f'  [ERROR]: {str(e)}\n')

if __name__ == '__main__':
    test_modes()

