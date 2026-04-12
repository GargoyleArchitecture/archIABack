# EVRAG Setup Guide

## ✅ CLIP Disponible!

CLIP ahora funciona con Python 3.13 usando torch CPU.

**Dependencias instaladas:**
- torch (CPU) - 2.11.0
- torchvision (CPU) - 0.26.0
- openai-clip - 1.0.1
- pillow - 12.1.1

**Costo: $0** (todo local, sin APIs)

## Configuración Actual

EVRAG funciona con **todas las características**:

- ✅ Scene-change detection (OpenCV)
- ✅ Extracción de frames
- ✅ Transcripción de audio (Whisper API o local)
- ✅ **CLIP embeddings** (torch CPU, Python 3.13)
- ✅ ChromaDB indexing (texto + visual)

## Uso Básico

```python
from back.evrag import EVRAGPipeline

pipeline = EVRAGPipeline()

# Procesar video (con CLIP completo!)
result = pipeline.process_video("back/videos/raw/mi_video.mp4")

# Query multimodal (texto + frames)
results = pipeline.query("¿Qué dice sobre latencia?")
# → Retorna frames relevantes + segmentos de transcripto
```

## Costos Estimados

| Componente | Costo |
|------------|-------|
| Scene detection | $0 (local) |
| CLIP embeddings | $0 (local) |
| Whisper API | $0.36/hora (o $0 con local) |
| QA generation | $0.15 (gpt-4o-mini) |
| Evaluación | $0.60 (gpt-4o) |
| **Total por video 1hr** | **~$1.11** |

## Siguientes Pasos

1. **Para tu video de 1 hora:**
   - Sube el video a `back/videos/raw/`
   - Ejecuta: `poetry run python -m back.evrag --video mi_video.mp4`
   - EVRAG procesará:
     - Detección de escenas (~100-200 frames clave)
     - Transcripción de audio (Whisper)
     - Embeddings de frames (CLIP)
     - Indexación en ChromaDB

2. **Para evaluar el video:**
   - Los QA pairs se generarán considerando:
     - Contenido visual (frames)
     - Contenido textual (transcripción)
   - Evaluación multimodal completa
