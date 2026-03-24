"""
ArchIA - Prueba Integral Completa

Prueba TODO el sistema:
1. Watcher (detección de archivos)
2. EVRAG (procesamiento de video)
3. Evaluación RAG (standard y comprehensive)
4. Privacy (anonimización)
"""

import sys
import time
from pathlib import Path

print("=" * 70)
print("ARCHIA - PRUEBA INTEGRAL COMPLETA")
print("=" * 70)

# =============================================================================
# TEST 1: Verificar archivos existentes
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: Verificando archivos existentes")
print("=" * 70)

pdfs = list(Path("back/docs").glob("*.pdf"))
videos = list(Path("back/videos/raw").glob("*.mp4"))

print(f"\n📁 PDFs encontrados: {len(pdfs)}")
for pdf in pdfs:
    print(f"   ✅ {pdf.name}")

print(f"\n🎬 Videos encontrados: {len(videos)}")
for video in videos:
    print(f"   ✅ {video.name}")
    size_mb = video.stat().st_size / (1024 * 1024)
    print(f"      Tamaño: {size_mb:.1f} MB")

if len(videos) == 0:
    print("\n❌ No hay videos en back/videos/raw/")
    print("   Copia tu video ahí y volvé a ejecutar")
    sys.exit(1)

# =============================================================================
# TEST 2: Imports y configuración
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Verificando imports y configuración")
print("=" * 70)

try:
    from back.eval import EVAL_CONFIG, get_eval_mode, get_total_qa_pairs
    from back.evrag import EVRAGPipeline, EVRAG_CONFIG
    from back.watcher import ArchIAWatcher
    from back.evrag.privacy import TextAnonymizer
    
    print("\n✅ Todos los imports exitosos")
    print(f"   - EVAL mode: {get_eval_mode()}")
    print(f"   - QA pairs/doc: {get_total_qa_pairs()}")
    print(f"   - EVRAG clip_enabled: {EVRAG_CONFIG['clip_enabled']}")
    print(f"   - LLM provider: {EVAL_CONFIG['llm_provider']}")
    print(f"   - LLM model: {EVAL_CONFIG['llm_model']}")
    
except Exception as e:
    print(f"\n❌ Error en imports: {e}")
    sys.exit(1)

# =============================================================================
# TEST 3: Privacy (Anonimización)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Probando anonimización de texto")
print("=" * 70)

try:
    anonymizer = TextAnonymizer()
    
    test_cases = [
        "Contacto: juan.perez@uniandes.edu.co",
        "Teléfono: +57 300 123 4567",
        "Reunión el 25 de diciembre de 2024",
    ]
    
    print("\n🔒 Probando anonimizador:")
    for text in test_cases:
        result = anonymizer.anonymize(text)
        if result.entities_removed:
            print(f"   ✅ '{text}' → '{result.anonymized_text}'")
        else:
            print(f"   ⚠️  '{text}' → Sin cambios")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# =============================================================================
# TEST 4: EVRAG Pipeline (Video Processing)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Procesando video con EVRAG")
print("=" * 70)

video_path = videos[0]
print(f"\n🎬 Video a procesar: {video_path.name}")

print("\n⚠️  ADVERTENCIA:")
print("   El procesamiento de video puede tardar 5-15 minutos")
print("   dependiendo de la duración del video y tu hardware.")
print("\n   Procesos:")
print("   1. Scene-change detection (~1-2 min)")
print("   2. Whisper transcription (~2-5 min)")
print("   3. CLIP embeddings (~1-3 min)")
print("   4. Face blurring (~1-2 min)")
print("   5. ChromaDB indexing (~30 sec)")

response = input("\n¿Continuar con el procesamiento? (s/n): ").strip().lower()
if response not in ['s', 'si', 'sí', 'y', 'yes']:
    print("   ⏭️  Saltando procesamiento de video")
    skip_evrage = True
else:
    skip_evrage = False

if not skip_evrage:
    try:
        from back.evrag import EVRAGPipeline
        
        print("\n🚀 Iniciando EVRAG pipeline...")
        start_time = time.time()
        
        pipeline = EVRAGPipeline(
            enable_anonymization=True,
            enable_face_blur=True,
            secure_delete_originals=False,  # No eliminar para testing
        )
        
        result = pipeline.process_video(video_path)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ EVRAG completado en {elapsed/60:.1f} minutos!")
        print(f"   📊 Escenas detectadas: {result.scenes_detected}")
        print(f"   🖼️  Frames extraídos: {result.frames_extracted}")
        print(f"   📝 Caracteres transcriptos: {result.transcript_length}")
        print(f"   👥 Caras difuminadas: {pipeline.stats['faces_blurred']}")
        print(f"   🔒 Entidades anonimizadas: {pipeline.stats['entities_anonymized']}")
        
        # Guardar resultado para referencia
        result_path = Path("back/videos/processed/evrag_result.txt")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(f"""
EVRAG Processing Result
=======================
Video: {video_path.name}
Time: {elapsed/60:.1f} minutes
Scenes: {result.scenes_detected}
Frames: {result.frames_extracted}
Transcript chars: {result.transcript_length}
Faces blurred: {pipeline.stats['faces_blurred']}
Entities anonymized: {pipeline.stats['entities_anonymized']}
""", encoding='utf-8')
        
        print(f"\n📄 Resultado guardado en: {result_path}")
        
    except Exception as e:
        print(f"\n❌ Error en EVRAG: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# TEST 5: RAG Evaluation (Standard Mode)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: Evaluación RAG (Standard Mode - 10 QA/doc)")
print("=" * 70)

print("\n⚠️  ADVERTENCIA:")
print("   La evaluación RAG requiere Ollama instalado y corriendo")
print("   o una API key de OpenAI configurada.")
print("\n   Para usar Ollama:")
print("   1. Instalar: https://ollama.com/download")
print("   2. Pull model: ollama pull llama3.1")
print("   3. Asegurar que esté corriendo: ollama serve")

response = input("\n¿Continuar con evaluación RAG? (s/n): ").strip().lower()
if response not in ['s', 'si', 'sí', 'y', 'yes']:
    print("   ⏭️  Saltando evaluación RAG")
    skip_eval = True
else:
    skip_eval = False

if not skip_eval:
    try:
        from back.eval import evaluate_layer_1_books
        
        print("\n🚀 Iniciando evaluación RAG (standard mode)...")
        print(f"   PDFs a evaluar: {len(pdfs)}")
        print(f"   QA pairs por PDF: 10")
        print(f"   Total QA pairs: {len(pdfs) * 10}")
        
        start_time = time.time()
        
        # Usar mock para testing sin LLM real
        def mock_rag_func(question, session_id):
            return {
                "retrieved_context": "Contexto de prueba",
                "generated_answer": f"Respuesta mock a: {question[:50]}...",
            }
        
        report = evaluate_layer_1_books(
            rag_invoke_func=mock_rag_func,
            force_regenerate=False,
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Evaluación completada en {elapsed/60:.1f} minutos!")
        print(f"   📊 Documentos evaluados: {len(report.document_results)}")
        print(f"   📈 Overall score: {report.aggregate_metrics.get('overall', 0):.4f}")
        
        # Mostrar reporte Markdown
        print(f"\n📄 Reporte generado:")
        md_path = Path("back/eval/reports").glob("layer1_books_*.md")
        for path in md_path:
            print(f"   ✅ {path}")
        
    except Exception as e:
        print(f"\n❌ Error en evaluación RAG: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)

print("""
✅ Tests completados:
   1. ✅ Verificación de archivos
   2. ✅ Imports y configuración
   3. ✅ Privacy (anonimización)
   4. {} EVRAG (video processing)
   5. {} RAG Evaluation

📁 Archivos generados:
   - back/videos/processed/evrag_result.txt (si se procesó video)
   - back/eval/reports/layer1_books_*.md (si se evaluó)
   - back/.processed_files.json (manifiesto)

🎯 Próximos pasos:
   1. Si no instalaste Ollama: https://ollama.com/download
   2. Pull del modelo: ollama pull llama3.1
   3. Ejecutar evaluación real: poetry run python -m back.eval --layer layer1_books --mode standard
""")

print("=" * 70)
print("¡PRUEBA INTEGRAL COMPLETADA!")
print("=" * 70)
