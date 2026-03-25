# ArchIA - Sistema Unificado de Procesamiento

## 🎯 Visión General

Sistema unificado para procesamiento automático de:
- **PDFs** → RAG indexing (ChromaDB)
- **Videos** → EVRAG processing (scene detection + CLIP + Whisper)

---

## 🚀 Uso Rápido

### 1. **Modo Watcher (Recomendado)**

Monitoreo continuo automático:

```bash
# Iniciar watcher (detecta nuevos archivos automáticamente)
poetry run python -m back.processor --watch
```

El watcher:
- ✅ Detecta nuevos PDFs en `back/docs/` → Los indexa en RAG
- ✅ Detecta nuevos videos en `back/videos/raw/` → Los procesa con EVRAG
- ✅ Aplica anonimización y difuminado de rostros
- ✅ Elimina videos originales después de procesar

### 2. **Comandos Específicos**

```bash
# Escanear archivos existentes
poetry run python -m back.processor --scan

# Procesar PDF específico
poetry run python -m back.processor --pdf documento.pdf

# Procesar video específico
poetry run python -m back.processor --video tutorial.mp4

# Procesar TODO lo existente
poetry run python -m back.processor --all
```

---

## 📁 Estructura de Directorios

```
back/
├── docs/                    # ← Coloca PDFs aquí
│   ├── libro1.pdf
│   └── libro2.pdf
│
├── videos/
│   ├── raw/                 # ← Coloca videos aquí
│   │   ├── tutorial.mp4
│   │   └── clase1.avi
│   ├── processed/           # Videos procesados (metadatos)
│   ├── frames/              # Frames extraídos
│   └── transcripts/         # Transcripciones
│
├── chroma_db/               # Vector store RAG
├── eval/                    # Evaluación framework
└── evrag/                   # EVRAG pipeline
```

---

## 🔄 Flujo Automático (Watcher)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHIA WATCHER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  back/docs/ (PDFs)                                              │
│     │                                                            │
│     ├─→ Nuevo PDF detectado                                     │
│     │                                                            │
│     └─→ RAG Indexing                                            │
│          └─→ ChromaDB                                           │
│                                                                  │
│  back/videos/raw/ (Videos)                                      │
│     │                                                            │
│     ├─→ Nuevo video detectado                                   │
│     │                                                            │
│     └─→ EVRAG Pipeline                                          │
│          ├─→ Scene detection                                    │
│          ├─→ Whisper transcription → Anonimizar                 │
│          ├─→ CLIP embeddings                                    │
│          ├─→ Face blurring                                      │
│          ├─→ ChromaDB indexing                                  │
│          └─→ Secure delete original                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔒 Privacidad y Seguridad

El watcher incluye por defecto:

### Para Videos:
- ✅ **Anonimización de transcripts**: Elimina emails, teléfonos, IDs
- ✅ **Difuminado de rostros**: Automático en todos los frames
- ✅ **Eliminación segura**: Overwrite + delete del video original
- ✅ **Verificación BitLocker**: Confirma cifrado del disco
- ✅ **Log de accesos**: Auditoría de procesamiento

### Para PDFs:
- ✅ Indexación en ChromaDB
- ✅ Evaluación automática con RAGAS + CCRS

---

## 📊 Comandos de Evaluación

```bash
# Evaluar RAG (texto)
poetry run python -m back.eval --layer layer1_books --mock

# Evaluar video procesado
poetry run python -m back.evrag --query "¿Qué dice sobre latencia?"

# Ver estadísticas del watcher
poetry run python -c "from back.watcher import ArchIAWatcher; w = ArchIAWatcher(); print(w.get_stats())"
```

---

## 🎯 Casos de Uso

### Caso 1: Agregar nuevo libro
```bash
# 1. Copiar PDF a docs/
cp nuevo_libro.pdf back/docs/

# 2. El watcher lo detecta y procesa automáticamente
# O manualmente:
poetry run python -m back.processor --pdf back/docs/nuevo_libro.pdf
```

### Caso 2: Agregar nuevo video
```bash
# 1. Copiar video a videos/raw/
cp clase.mp4 back/videos/raw/

# 2. El watcher lo detecta y procesa automáticamente
# O manualmente:
poetry run python -m back.processor --video back/videos/raw/clase.mp4
```

### Caso 3: Procesamiento masivo inicial
```bash
# Procesar todo lo existente
poetry run python -m back.processor --all
```

---

## 🛠️ Configuración

### Watcher Settings

En `back/watcher.py`:

```python
watcher = ArchIAWatcher(
    docs_dir=Path("back/docs"),
    videos_dir=Path("back/videos/raw"),
    watch_docs=True,
    watch_videos=True,
)
```

### EVRAG Privacy Settings

En `back/evrag/pipeline.py`:

```python
pipeline = EVRAGPipeline(
    enable_anonymization=True,      # Anonimizar transcripts
    enable_face_blur=True,          # Difuminar rostros
    secure_delete_originals=True,   # Eliminar originales
)
```

---

## 📈 Monitoreo y Logs

### Ver Logs de Acceso
```bash
# Logs de procesamiento de videos
cat back/videos/.access_logs/access_*.log
```

### Ver Estadísticas
```python
from back.watcher import ArchIAWatcher

watcher = ArchIAWatcher()
stats = watcher.get_stats()

print(f"Files processed: {stats['files_processed']}")
print(f"PDFs indexed: {stats['pdfs_indexed']}")
print(f"Videos processed: {stats['videos_processed']}")
print(f"Errors: {stats['errors']}")
```

---

## 🐛 Solución de Problemas

### Watcher no detecta archivos
```bash
# Verificar que los directorios existen
dir back\docs
dir back\videos\raw

# Verificar permisos
# Asegurarse de que Python tiene acceso de lectura
```

### Error al procesar PDF
```bash
# Verificar que el PDF no está corrupto
# Rebuild vectorstore manualmente
poetry run python back/build_vectorstore.py
```

### Error al procesar video
```bash
# Verificar formato de video soportado
# Formatos: mp4, avi, mov, mkv, wmv, flv

# Procesar manualmente para ver error detallado
poetry run python -m back.processor --video mi_video.mp4
```

---

## 📚 Referencias

- **RAG Evaluation**: `back/eval/README.md`
- **EVRAG**: `back/evrag/README.md`
- **Privacy**: `back/evrag/privacy.py`

---

## 💰 Costos Estimados

| Componente | Costo |
|------------|-------|
| **RAG (PDFs)** | $0 (local) |
| **EVRAG (Videos)** | ~$1.11/hr (Whisper API + LLM) |
| **CLIP Embeddings** | $0 (local) |
| **Face Blurring** | $0 (local) |
| **Watcher** | $0 (local) |

---

## 🎓 Para Tu Tesis

Este sistema unificado cumple con:

1. ✅ **Procesamiento automático** de documentos y videos
2. ✅ **Anonimización** de datos sensibles
3. ✅ **Privacidad** (difuminado de rostros)
4. ✅ **Seguridad** (eliminación segura, BitLocker)
5. ✅ **Evaluación** (RAGAS + CCRS para texto, EVRAG para video)
6. ✅ **Trazabilidad** (logs de acceso, auditoría)

**Todo en un solo lugar!** 🎯
