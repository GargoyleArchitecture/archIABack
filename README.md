# archIABack

## Requisitos previos

- Python 3.11.x
- pip
- Graphviz (instalado a nivel de sistema)

---

## Configuración inicial (primera vez)

### 1. Instalar Python 3.11

**Verificar versión de Python:**

```bash
python3.11 --version
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

Dentro de la carpeta `back/`, crear el archivo `.env` con  API Key:

```
OPENAI_API_KEY="Tu API Key va aquí"
```

### 6. Correr el servidor

```bash
python3.11 -m poetry run uvicorn src.main:app --port 8000
```

El servidor queda disponible en **http://localhost:8000**. La documentación interactiva de la API está en **http://localhost:8000/docs**.

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

---

## Endpoints (post-Fase 11)

| Verbo | Path | Auth | Descripción |
|---|---|---|---|
| POST | `/message` | Bearer (opt) | Chat conversacional. Form-data con `message`, `session_id`, `mode`, `user_id`. Respuesta SSE. |
| POST | `/generate-routine` | `X-Internal-Token` | Genera un reto pedagógico via subgrafo `RoutineGeneratorGraph` (F5-T2). |
| GET | `/health` | — | Liveness check. |
| GET | `/diagram/export` | Bearer | Exporta el último diagrama del thread (SVG/DOT/drawio). |

Variables de entorno relevantes:

| Variable | Default | Propósito |
|---|---|---|
| `INTERNAL_API_TOKEN` | — | Token compartido con Negocio para `/generate-routine` y sync de perfil |
| `BUSINESS_API_BASE_URL` | `http://localhost:3000` | Backend Negocio para sync (`PUT /internal/users/:id/profile`) y hidratación inversa (`GET …`) |
| `BUSINESS_API_PROFILE_PATH` | `/internal/users/{userId}/profile` | Template del path |
| `PROFILE_SYNC_ENABLED` | `true` | Habilita el sync IA → Negocio |
| `PROFILE_SYNC_TIMEOUT` | `15s` | Por intento HTTP |
| `SHADOW_AGENT_EVERY_N_TURNS` | `4` | Cadencia del Shadow Agent (F3-T2) |
| `SHADOW_AGENT_WINDOW_SIZE` | `6` | Ventana de mensajes a evaluar |
| `DECAY_RATE_DEFAULT` | `0.05` | Tasa de olvido default por concepto (F3-T4) |
| `MODE_SUGGESTION_THRESHOLD` | `0.7` | Confianza mínima para sugerir cambio de modo (F2-T4) |
| `ENABLE_PYTHON_REPL` | `false` | Habilita la tool `python_repl_tool` en modo professional (F2-T6) |

---

## Subgrafo `RoutineGeneratorGraph` (F5-T1)

Documentación viva en [`back/docs_back/graph.md`](./back/docs_back/graph.md). Resumen:

```
select_weakness → inverse_rag_search → synthesize_challenge → validate_difficulty
                                              ↑                       │
                                              └── regen (≤2) ─────────┘
```

Regeneración del corpus RAG inverso (F5-T3):
```powershell
cd back
python build_bad_code_corpus.py --rebuild
```

---

## Tests

```powershell
cd back
poetry run pytest -q                           # Suite completa
poetry run pytest tests/test_telemetry.py -q   # F11-T6: emisor de eventos
python -m tests.load.shadow_agent_load --dry-run --users 50 --turns 6 --seed 42  # F11-T2
```

---

## Documentación adicional

- [`back/docs_back/graph.md`](./back/docs_back/graph.md) — GraphState, subgrafos, telemetría.
- [`back/docs_back/store.md`](./back/docs_back/store.md) — LangGraph Store + namespaces (F1-T4).
- [`../docs/architecture.md`](../docs/architecture.md) — arquitectura global del sistema.
- [`../docs/endpoints.md`](../docs/endpoints.md) — inventario consolidado de endpoints.
- [`../docs/observability.md`](../docs/observability.md) — telemetría y métricas.
- [`../docs/Tracking_Maestro.md`](../docs/Tracking_Maestro.md) — historial completo de fases.
