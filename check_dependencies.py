"""
ArchIA - Verificador de Dependencias

Verifica TODAS las dependencias necesarias y muestra qué falta.
"""

import sys
import subprocess
from pathlib import Path

print("=" * 70)
print("ARCHIA - VERIFICADOR COMPLETO DE DEPENDENCIAS")
print("=" * 70)

def check_package(name, import_name=None):
    """Check if package is installed."""
    import_name = import_name or name
    try:
        __import__(import_name)
        print(f"✅ {name}")
        return True
    except ImportError:
        print(f"❌ {name} - FALTA")
        return False

def check_command(cmd):
    """Check if command is available."""
    try:
        result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
        version = result.stdout.split("\n")[0] or result.stderr.split("\n")[0]
        print(f"✅ {cmd}: {version}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"❌ {cmd} - FALTA")
        return False

def check_file(path, description):
    """Check if file/directory exists."""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} - FALTA: {path}")
        return False

print("\n" + "=" * 70)
print("1. PYTHON Y PAQUETES BASE")
print("=" * 70)
print(f"Python: {sys.version}")
check_package("poetry")
check_package("numpy")
check_package("cv2", "cv2")  # OpenCV
check_package("PIL", "PIL")  # Pillow
check_package("tqdm")

print("\n" + "=" * 70)
print("2. LANGCHAIN Y LLMs")
print("=" * 70)
check_package("langchain-core")
check_package("langchain-openai")
check_package("langchain-ollama")
check_package("langchain-google-vertexai")
check_package("langgraph")

print("\n" + "=" * 70)
print("3. RAG Y VECTOR STORE")
print("=" * 70)
check_package("chromadb")
check_package("unstructured")
check_package("pymupdf", "fitz")
check_package("pypdf")

print("\n" + "=" * 70)
print("4. EVRAG - VIDEO PROCESSING")
print("=" * 70)
check_package("whisper")  # OpenAI Whisper
check_package("torch")
check_package("clip")  # CLIP embeddings

print("\n" + "=" * 70)
print("5. COMANDOS EXTERNOS")
print("=" * 70)
check_command("ffmpeg")
check_command("ollama")

print("\n" + "=" * 70)
print("6. DIRECTORIOS")
print("=" * 70)
check_file("back/docs", "Directorio de PDFs")
check_file("back/videos/raw", "Directorio de videos")
check_file("back/eval", "Módulo de evaluación")
check_file("back/evrag", "Módulo EVRAG")
check_file("back/watcher.py", "File watcher")

print("\n" + "=" * 70)
print("7. ARCHIVOS DE CONFIGURACIÓN")
print("=" * 70)
check_file("back/.env", "Environment variables (API keys)")
check_file("pyproject.toml", "Poetry dependencies")

print("\n" + "=" * 70)
print("8. OLLAMA MODELOS")
print("=" * 70)
try:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    if "llama3.1" in result.stdout:
        print("✅ Ollama modelo llama3.1 instalado")
    else:
        print("❌ Ollama modelo llama3.1 - FALTA")
        print("   Ejecutar: ollama pull llama3.1")
except Exception as e:
    print(f"❌ Error verificando Ollama: {e}")

print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print("""
Si hay algún ❌, instalalo con:

# Paquetes Python:
poetry add <nombre-paquete>

# FFmpeg (Windows):
winget install ffmpeg

# Ollama + modelo:
ollama pull llama3.1

# Si todo está ✅, podés ejecutar:
poetry run python test_full_system.py
""")
print("=" * 70)
