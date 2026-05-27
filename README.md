# archIABack

## Requisitos previos

- Python 3.11 o superior (Detectado: 3.13.11)
- pip
- Graphviz (instalado a nivel de sistema)

---

## 🛠️ Configuración inicial

Este proyecto requiere **Python 3.11+**. En su sistema, el comando a utilizar es `python`.

### 1. Verificar Python

**Verificar versión de Python:**

```bash
python --version
```

Si no está instalado (Fedora/RHEL):

```bash
sudo dnf install python3.11
```

**Instalar pip para Python 3.11:**

```bash
python3.11 -m ensurepip --upgrade
```

### 2. Instalar Graphviz

```bash
sudo dnf install graphviz
```

**Verificar versión de Graphviz:**

```bash
dot -V
```

### 3. Instalar Poetry

```bash
python3.11 -m pip install poetry
```

**Verificar versión de Poetry:**

```bash
python3.11 -m poetry --version
```

### 4. Instalar las dependencias del proyecto

Desde la carpeta `back/`:

```bash
cd back
python3.11 -m poetry env use python3.11
python3.11 -m poetry install
```

### 5. Configurar el archivo `.env`

Dentro de la carpeta `back/`, crear el archivo `.env` con API Key:

```
OPENAI_API_KEY="Tu API Key va aquí"
```

### 6. Correr el servidor

```bash
python3.11 -m poetry run uvicorn src.main:app --port 8000
```

El servidor queda disponible en **http://localhost:8000**. La documentación interactiva de la API está en **http://localhost:8000/docs**.

---

## 🚀 Procesamiento Automático (Watcher)

El sistema incluye un monitor que detecta nuevos archivos y los procesa automáticamente (RAG para PDFs y EVRAG para videos).

```bash
# Iniciar el monitor automático
python3.11 -m poetry run python -m back.processor --watch
```

Esto anonimiza datos sensibles, difumina rostros en videos y actualiza la base de datos vectorial de forma transparente.

---

## Administración del Módulo RAG (ChromaDB)

El sistema integra un motor de Generación Aumentada por Recuperación (RAG) para fundamentar las recomendaciones arquitectónicas en documentación técnica de referencia.

### 1. Gestión de la Base de Conocimientos (Vector Store)

Para inicializar o actualizar el almacén de datos vectoriales con nuevos documentos:

1. Ubicar los archivos PDF en el directorio `back/docs/`.
2. Ejecutar el proceso de indexación:
   ```bash
   cd back
   python3.11 -m poetry run python build_vectorstore.py
   ```

### 2. Auditoría y Visualización (ChromaDB Explorer)

Se proporciona una herramienta de exploración para validar la carga de datos y el comportamiento de la búsqueda semántica.

**Linux/Mac:**

1. Iniciar el servicio de exploración:
   ```bash
   cd back
   python3.11 -m poetry run python chroma_web.py
   ```
2. Acceder a la consola web en: [**http://localhost:8001**](http://localhost:8001)

**Windows:**

- Ejecutar el script de acceso directo: `back/start_chroma_web.bat`

### Capacidades del Explorador

- **Métricas de Estado**: Visualización del volumen de fragmentos (chunks) y fuentes registradas.
- **Pruebas de Recuperación**: Motor de búsqueda semántica para verificar la relevancia de los resultados.
- **Inspección de Metadatos**: Validación de la trazabilidad (fuente y página) de los segmentos almacenados.

## 🛡️ Pipeline de Privacidad y Sanitización EVRAG

Para garantizar la protección de Información de Identificación Personal (PII) durante el procesamiento multimodal (EVRAG), el sistema implementa un robusto pipeline de anonimización:

### 1. Anonimización Visual (Privacy by Design)
- **Enmascaramiento Dinámico:** El sistema detecta el layout de la escena (`Slide`, `Gallery`, `Speaker`) y aplica bloques geométricos de color oscuro para censurar listas de participantes y avatares.
- **Supresión Definitiva:** Se utiliza enmascaramiento sólido en lugar de desenfoque para garantizar la irrecuperabilidad de los datos sensibles en las imágenes procesadas.

### 2. Anonimización de Texto
- Sustituye técnicas clásicas por el modelo generativo **GPT-4o-mini**, el cual extrae con alta precisión semántica las entidades PII (nombres propios, instituciones) directamente de las transcripciones crudas. Posteriormente, expresiones regulares de alta velocidad inyectan identificadores anónimos (`[REDACTADO]`) antes de la indexación vectorial.

### 3. Resiliencia y Sanitización de Sistema
- **Sincronización ChromaDB:** Resuelve colisiones de nombres y sincroniza los metadatos de los vectores de forma coherente con el estado real de los archivos de imagen.
- **Tolerancia a Fallos OS:** Implementa rutas extendidas (`\\?\`) y lógica de reintentos para evadir limitaciones de longitud de ruta y bloqueos del sistema de archivos en Windows (Errno 22).

---

## 📊 Evaluación Académica Avanzada (MiRAGE Framework)

El sistema incluye un framework de evaluación de vanguardia basado en el paper *Seeing Through the MiRAGE*, que permite comparar objetivamente diferentes modos de RAG y EVRAG.

### 1. Ejecución de la Evaluación
Para generar el reporte comparativo de métricas (InfoF1, CiteF1, Faithfulness, etc.):

```bash
# Desde la raíz del proyecto
poetry run python evaluate_advanced.py
```

### 2. Métricas Implementadas
- **InfoF1 (MiRAGE):** Mide la veracidad granular mediante la descomposición de respuestas en afirmaciones atómicas (subclaims).
- **CiteF1 (MiRAGE):** Evalúa la precisión y cobertura de las citas bibliográficas y de video.
- **RAGAS Faithfulness:** Mide qué tan fiel es la respuesta al contexto proporcionado.
- **CCRS Correctness/Relevancy:** Evalúa la precisión técnica y la utilidad del contexto recuperado.

### 3. Resultados
Los resultados se exportan automáticamente a `evaluation_results_advanced.json`, proporcionando una base empírica para la sección de resultados de la tesis.

---

## 📦 Compartir Bases de Datos de Forma Externa (ZIP)

Dado que las bases de datos vectoriales y de procesamiento contienen archivos binarios pesados (e información indexada), **no se deben subir al repositorio de GitHub**. Estos directorios están configurados en el `.gitignore`.

Para compartir el estado actual del backend y evitar tener que re-indexar los documentos PDF y procesar los videos desde cero (lo cual consume tiempo y recursos de API), debes comprimir en un archivo ZIP los siguientes directorios ubicados dentro de la carpeta `back/`:

### 📂 Carpetas a incluir en el archivo ZIP (ej. `bases_de_datos.zip`):
1. **`back/chroma_db/`** — Base de datos vectorial del RAG de Texto (documentos PDF indexados).
2. **`back/videos/chroma_db/`** — Base de datos vectorial del EVRAG de Video (embeddings visuales de los keyframes).
3. **`back/videos/processed/`** — Metadatos generados (transcripciones de audio, resúmenes descriptivos de escenas y archivos de control JSON).
4. **`back/videos/frames/`** — Keyframes extraídos de los videos (crucial para que el buscador visual y la interfaz muestren las miniaturas correspondientes).
5. **`back/feedback_db/`** — SQLite que almacena las calificaciones y feedback de los usuarios.
6. **`back/state_db/`** — SQLite que persiste el estado de las conversaciones del agente de LangGraph.

### 🚫 Carpetas a EXCLUIR para ahorrar espacio:
- **`back/videos/raw/`** — Videos originales en MP4 (son muy pesados y no se necesitan para que las búsquedas y visualizaciones de fotogramas clave funcionen).
- **`back/videos/audio/`** y **`back/videos/transcripts/`** — Archivos de audio temporales y caché interna de transcripciones.

### 📥 Instrucciones para restaurar en otra máquina:
1. Clonar el repositorio normalmente.
2. Descargar el archivo ZIP con las bases de datos pre-computadas desde la pestaña de **Releases** de este repositorio de GitHub.
3. Extraer el contenido del ZIP directamente dentro de la carpeta `back/` de modo que se ubiquen en sus rutas correspondientes (`back/chroma_db`, `back/videos/...`, etc.).
4. Ejecutar el backend normalmente (`poetry run uvicorn src.main:app --port 8000`). El sistema detectará las bases de datos indexadas y funcionará inmediatamente sin requerir procesos de carga previos o credenciales API activas de forma obligatoria para búsquedas locales.

---

## 📚 Documentación Técnica

Para detalles sobre la arquitectura del sistema, el grafo de estados y los agentes especializados, consulta:

- [Documentación de Arquitectura (v4)](./docs_back/Arquitectura.md)

