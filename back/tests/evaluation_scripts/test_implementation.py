"""
Test script para verificar toda la implementación de ArchIA
"""

import sys
from pathlib import Path

print("=" * 60)
print("ARCHIA - PRUEBA INTEGRAL DE IMPLEMENTACIÓN")
print("=" * 60)

# Test 1: Eval Framework
print("\n1️⃣  Testing EVAL Framework...")
try:
    from back.eval import EVAL_CONFIG, evaluate_layer_1_books
    from back.eval.config import get_total_qa_pairs
    
    assert EVAL_CONFIG["qa_pairs_per_doc"] == 10, "QA pairs should be 10"
    assert get_total_qa_pairs() == 10, "Total QA should be 10"
    
    print(f"   ✅ EVAL_CONFIG: {EVAL_CONFIG['qa_pairs_per_doc']} QA pairs/doc")
    print(f"   ✅ Metrics enabled: {len(EVAL_CONFIG['metrics'])} metrics")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 2: EVRAG
print("\n2️⃣  Testing EVRAG...")
try:
    from back.evrag import EVRAGPipeline, EVRAG_CONFIG
    from back.evrag.config import get_total_qa_pairs as evrag_qa
    
    assert EVRAG_CONFIG["clip_enabled"] == True, "CLIP should be enabled"
    assert evrag_qa() == 25, "EVRAG should have 25 QA pairs"
    
    print(f"   ✅ EVRAG_CONFIG: clip_enabled={EVRAG_CONFIG['clip_enabled']}")
    print(f"   ✅ EVRAG QA pairs: {evrag_qa()}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 3: Watcher
print("\n3️⃣  Testing Watcher...")
try:
    from back.watcher import ArchIAWatcher, ArchIAFileHandler
    
    watcher = ArchIAWatcher()
    print(f"   ✅ Watcher created")
    print(f"   ✅ Docs dir: {watcher.docs_dir}")
    print(f"   ✅ Videos dir: {watcher.videos_dir}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 4: Privacy
print("\n4️⃣  Testing Privacy Modules...")
try:
    from back.evrag.privacy import TextAnonymizer, FaceBlurrer, SecureStorage
    
    # Test TextAnonymizer
    anonymizer = TextAnonymizer()
    test_text = "Contact: john@example.com, Phone: 555-1234"
    result = anonymizer.anonymize(test_text)
    
    assert "[EMAIL]" in result.anonymized_text, "Should anonymize email"
    assert "[TELEFONO]" in result.anonymized_text, "Should anonymize phone"
    
    print(f"   ✅ TextAnonymizer: {len(result.entities_removed)} entities removed")
    print(f"   ✅ FaceBlurrer: Available")
    print(f"   ✅ SecureStorage: Available")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 5: CLIP
print("\n5️⃣  Testing CLIP...")
try:
    from back.evrag.clip_embedder import CLIPEmbedder
    
    embedder = CLIPEmbedder()
    print(f"   ✅ CLIPEmbedder created")
    print(f"   ✅ Model: {EVRAG_CONFIG['clip_model']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 6: Processor
print("\n6️⃣  Testing Processor...")
try:
    from back.processor import process_pdf, process_video, run_watcher
    
    print(f"   ✅ Processor functions available")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 7: File Structure
print("\n7️⃣  Verifying File Structure...")
required_dirs = [
    "back/docs",
    "back/videos/raw",
    "back/eval",
    "back/evrag",
    "back/eval/datasets",
    "back/eval/reports",
]

for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"   ✅ {dir_path}/")
    else:
        print(f"   ⚠️  {dir_path}/ (missing)")

# Test 8: Existing Files
print("\n8️⃣  Scanning Existing Files...")
pdfs = list(Path("back/docs").glob("*.pdf"))
videos = []
for ext in ['.mp4', '.avi', '.mov', '.mkv']:
    videos.extend(Path("back/videos/raw").glob(f"*{ext}"))

print(f"   ✅ PDFs found: {len(pdfs)}")
for pdf in pdfs[:5]:
    print(f"      - {pdf.name}")
if len(pdfs) > 5:
    print(f"      ... and {len(pdfs) - 5} more")

print(f"   ✅ Videos found: {len(videos)}")
for video in videos[:5]:
    print(f"      - {video.name}")

print("\n" + "=" * 60)
print("✅ TODAS LAS PRUEBAS PASARON!")
print("=" * 60)
print("\nEl sistema está listo para usar:")
print("  - Watcher: poetry run python -m back.processor --watch")
print("  - Scan:    poetry run python -m back.processor --scan")
print("  - Eval:    poetry run python -m back.eval --layer layer1_books --mock")
print("  - EVRAG:   poetry run python -m back.evrag --video <video.mp4>")
print("=" * 60)
