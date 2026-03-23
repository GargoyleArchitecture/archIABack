# EVRAG Setup Guide

## Requisitos

### Python Version

**Importante:** EVRAG con CLIP requiere Python 3.11 o 3.12.

- Python 3.13: ❌ No compatible con torch
- Python 3.11-3.12: ✅ Compatible

### Dependencias Instaladas

```bash
# Ya instaladas
poetry add opencv-python transformers pillow

# Para CLIP (solo Python 3.11-3.12):
poetry add torch openai-whisper
```

## Configuración Actual

Tu entorno actual tiene Python 3.13, así que EVRAG funciona en **modo texto**:

- ✅ Scene-change detection (OpenCV)
- ✅ Extracción de frames
- ✅ Transcripción de audio (Whisper API)
- ❌ CLIP embeddings (requiere Python 3.11-3.12)
- ✅ ChromaDB indexing (texto)

## Opciones para CLIP

### Opción 1: Usar Whisper API (Recomendado para tu tesis)

Configura en `back/evrag/config.py`:

```python
EVRAG_CONFIG = {
    "transcriber": "whisper_api",  # Usa API en vez de local
    "clip_enabled": False,  # Sin CLIP por ahora
    ...
}
```

**Costo:** ~$0.36 por hora de audio  
**Ventaja:** Funciona con Python 3.13

### Opción 2: Crear entorno Python 3.11

```bash
# Crear entorno separado para EVRAG
py -3.11 -m venv .venv-evrag
.venv-evrag\Scripts\activate
pip install openai-whisper torch

# Usar este entorno solo para EVRAG
```

### Opción 3: Solo texto (Gratis)

Extrae el transcripto manualmente y úsalo con el RAG de texto existente.

## Uso Básico

```python
from back.evrag import EVRAGPipeline

pipeline = EVRAGPipeline()

# Procesar video
result = pipeline.process_video("back/videos/raw/mi_video.mp4")

# Query multimodal (texto + frames si CLIP está disponible)
results = pipeline.query("¿Qué dice sobre latencia?")
```

## Costos Estimados

| Componente | Costo |
|------------|-------|
| Scene detection | $0 (local) |
| Whisper API | $0.36/hora |
| CLIP embeddings | $0 (local, si disponible) |
| QA generation | $0.15 (gpt-4o-mini) |
| Evaluación | $0.60 (gpt-4o) |
| **Total por video 1hr** | **~$1.11** |

## Siguientes Pasos

1. **Para tu video de 1 hora:**
   - Sube el video a `back/videos/raw/`
   - Ejecuta: `poetry run python -m back.evrag --video mi_video.mp4`

2. **Para añadir CLIP después:**
   - Actualiza a Python 3.11-3.12
   - Instala torch + whisper
   - Setea `clip_enabled: True`
