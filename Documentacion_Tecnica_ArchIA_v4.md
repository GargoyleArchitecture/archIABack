# ArchIA — Documentación Técnica del Sistema (v4)

> **Versión**: 4.0
> **Fecha**: Abril 2026
> **Tipo de documento**: Especificación Técnica de Arquitectura Completa
> **Reemplaza**: v3 (Marzo 2025)

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Dependencias del Sistema](#2-dependencias-del-sistema)
3. [Arquitectura del Flujo — El Grafo de Estados](#3-arquitectura-del-flujo--el-grafo-de-estados)
4. [Nodos Especializados](#4-nodos-especializados)
5. [Herramientas (Tools)](#5-herramientas-tools)
6. [Estado del Grafo](#6-estado-del-grafo)
7. [Sistema de Quality Attributes (QA)](#7-sistema-de-quality-attributes-qa)
8. [Sistema de Diagramas](#8-sistema-de-diagramas)
9. [Sistema RAG](#9-sistema-rag)
10. [Servicios Core](#10-servicios-core)
11. [API REST](#11-api-rest)
12. [Persistencia](#12-persistencia)
13. [Jerarquía de Archivos](#13-jerarquía-de-archivos)
14. [Variables de Entorno](#14-variables-de-entorno)
15. [Changelog v3 → v4](#15-changelog-v3--v4)

---

## 1. Introducción

### 1.1 Propósito de ArchIA

**ArchIA** es un asistente inteligente especializado en diseño de arquitectura de software, fundamentado en la metodología **ADD 3.0** (Attribute-Driven Design). Asiste a arquitectos y equipos de desarrollo durante las fases tempranas del SDLC:

| Fase del SDLC              | Contribución de ArchIA                                                            |
| -------------------------- | --------------------------------------------------------------------------------- |
| **Análisis de Requisitos** | Extracción de ASRs (Quality Attribute Scenarios) en formato canónico de 6 partes |
| **Diseño Arquitectónico**  | Recomendación de estilos arquitectónicos y tácticas con soporte RAG               |
| **Documentación**          | Generación de diagramas DOT con 3 niveles de detalle y exportación multi-formato  |
| **Validación**             | Evaluación de ASRs, comparación de diagramas y consulta RAG en documentación PDF  |

### 1.2 Paradigma de Interacción

ArchIA opera mediante un **Grafo de Estados (State Graph)** orquestado por **LangGraph**. Cada mensaje del usuario inicia un flujo donde un Clasificador detecta idioma e intención, un Supervisor planifica la ejecución, y nodos especializados procesan la tarea antes de consolidar una respuesta final.

```
START → boot → classifier → supervisor → [router → nodo(s) especializado(s)] → unifier → END
```

### 1.3 Stack Tecnológico

| Componente         | Tecnología                                              |
| ------------------ | ------------------------------------------------------- |
| **Backend**        | FastAPI 0.128+ / Uvicorn                                |
| **Orquestación**   | LangGraph con MemorySaver (checkpointer en RAM)         |
| **LLM**            | Multi-proveedor: Azure OpenAI / OpenAI / Ollama         |
| **RAG**            | ChromaDB con OpenAI Embeddings (`text-embedding-3-small`) |
| **Diagramas**      | Graphviz (DOT) → SVG / draw.io XML                     |
| **Persistencia**   | SQLite (estado de sesión + feedback)                    |
| **PDF Extraction** | PyMuPDF (fitz) + pypdf (fallback)                       |
| **Vision**         | Vertex AI Gemini (opcional, para análisis de imágenes)  |

> **Nota v4**: El checkpointer fue revertido de `SqliteSaver` a `MemorySaver`. El estado de sesión de LangGraph vive en RAM; la persistencia de largo plazo (arch_flow, ASR, diagramas) se maneja exclusivamente en `state_db/memory.db` via el módulo `memory.py`.

---

## 2. Dependencias del Sistema

El sistema requiere los siguientes componentes instalados en el entorno de ejecución. Para instrucciones de instalación paso a paso, referirse al [README.md](README.md).

### 2.1 Runtime

| Dependencia | Versión mínima | Rol |
| ----------- | -------------- | --- |
| **Python**  | 3.11.x         | Lenguaje de ejecución del backend |
| **Poetry**  | 2.x            | Gestor de dependencias y entorno virtual |
| **Graphviz** | Cualquiera    | Renderizado de diagramas DOT a SVG (binario `dot` en PATH) |

### 2.2 Dependencias Python (principales)

Definidas en `back/requirements.txt` y gestionadas por Poetry:

| Paquete | Rol |
| ------- | --- |
| `fastapi` + `uvicorn` | Servidor HTTP y framework de la API REST |
| `langgraph` | Orquestación del grafo de estados |
| `langchain-core` + `langchain-community` | Abstracciones LLM y herramientas |
| `langchain-openai` | Integración con OpenAI y Azure OpenAI |
| `chromadb` | Motor de vectorstore para RAG |
| `pymupdf` + `pypdf` | Extracción de texto desde PDFs |
| `graphviz` | Binding Python para renderizado DOT |
| `python-multipart` | Soporte multipart/form-data en FastAPI |
| `pydantic` | Validación de schemas |
| `httpx` | Cliente HTTP asíncrono |

### 2.3 Servicios Externos

| Servicio | Requerido | Descripción |
| -------- | --------- | ----------- |
| **OpenAI API** | Sí (o alternativa) | Proveedor LLM principal. Requiere `OPENAI_API_KEY`. |
| **Azure OpenAI** | No | Alternativa a OpenAI público. Ver [Sección 14](#14-variables-de-entorno). |
| **Ollama** | No | Alternativa local sin API key. Requiere servidor Ollama corriendo. |
| **Vertex AI (GCP)** | No | Opcional. Habilita análisis de imágenes via Gemini en el Evaluator. |

---

## 3. Arquitectura del Flujo — El Grafo de Estados

### 3.1 Ciclo de Procesamiento

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CICLO DE PROCESAMIENTO v4                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  1. ENTRADA       → Usuario envía mensaje + archivos opcionales (PDF/img)   │
│  2. BOOT          → Reset de buffers de turno, preserva estado de sesión    │
│  3. CLASSIFIER    → Detecta idioma (ES/EN), clasifica intención y QA        │
│  4. SUPERVISOR    → Planifica multi-intent, decide nodo(s) a activar        │
│  5. ROUTER        → Enruta a nodo especializado (incluye QA-aware routing)  │
│  6. EJECUCIÓN     → Nodo(s) especializado(s) procesan la tarea              │
│  7. UNIFICACIÓN   → Consolida outputs de múltiples nodos                    │
│  8. RESPUESTA     → Genera respuesta final con sugerencias de seguimiento   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Diagrama del Grafo

```mermaid
graph TD
    START([START]) --> boot[Boot Node]
    boot --> classifier[Classifier]
    classifier --> supervisor[Supervisor]
    supervisor --> router{Router}

    router --> investigator[Investigator]
    router --> asr[ASR Node]
    router --> evaluator[Evaluator]
    router --> diagram_agent[Diagram Agent]

    router --> style[Style Node]
    router --> style_latency[Style Latency]
    router --> style_scalability[Style Scalability]
    router --> style_availability[Style Availability]

    router --> tactics[Tactics Node]
    router --> tactics_latency[Tactics Latency]
    router --> tactics_scalability[Tactics Scalability]
    router --> tactics_availability[Tactics Availability]

    investigator --> supervisor
    asr --> supervisor
    evaluator --> supervisor
    diagram_agent --> supervisor
    style --> supervisor
    style_latency --> supervisor
    style_scalability --> supervisor
    style_availability --> supervisor
    tactics --> supervisor
    tactics_latency --> supervisor
    tactics_scalability --> supervisor
    tactics_availability --> supervisor

    supervisor --> unifier[Unifier]
    unifier --> END([END])
```

> **Cambio v4**: Todos los nodos especializados retornan al **Supervisor** (no al Unifier directamente). El Supervisor decide si hay más nodos pendientes o si ya puede ir al Unifier. Esto permite el scheduler multi-intent.

### 3.3 Nodo Boot (`workflow.py::boot_node`)

Resetea los buffers de turno en cada nueva iteración sin perder el estado persistente de la sesión:

```python
def boot_node(state: GraphState) -> GraphState:
    return {
        **state,
        "hasVisitedInvestigator": False,
        "hasVisitedEvaluator": False,
        "hasVisitedASR": False,
        "hasVisitedDiagram": False,
        "diagram": {},
        "endMessage": "",
        "requested_nodes": [],
        "pending_nodes": [],
        "completed_nodes": [],
    }
```

Preserva: `current_asr`, `selected_style`, `tactics_struct`, `diagram_history`, `last_asr`.

### 3.4 Nodo Classifier (`nodes/classifier.py`)

Clasifica la entrada del usuario antes de que el Supervisor tome decisiones.

**LLM Call**: `llm.with_structured_output(ClassifyOut).invoke(prompt)`

**Output Schema (`ClassifyOut`)**:
```python
class ClassifyOut(TypedDict):
    language: Literal["en", "es"]
    intent: Literal[
        "greeting", "smalltalk", "architecture",
        "diagram", "asr", "tactics", "style", "other"
    ]
    use_rag: bool
    quality_attribute: str  # latencia | escalabilidad | disponibilidad | general
```

**Resolución de Quality Attribute** (prioridad en cascada):
1. QA clasificado junto al intent por el LLM
2. Si resulta "general", invoca `resolve_quality_attribute()` (LLM adicional via `index_resolver.py`)
3. QA previo del estado (continuidad entre turnos)
4. Default: `"general"`

**Override por keywords** (post-clasificación LLM):
- Palabras clave de estilo → `intent = "style"`
- Palabras clave de tácticas → `intent = "tactics"`
- Palabras clave de diagrama (solo si no es asr/style/tactics) → `intent = "diagram"`

**Patrones de Followup** (`FOLLOWUP_PATTERNS`):
- `explain_tactics` — regex para "explica las tácticas"
- `make_asr` — regex para "crea/genera un ASR"
- `component_view`, `deployment_view`, `functional_view`
- `compare`, `checklist`

**State escrito**: `language`, `intent`, `force_rag`, `resolved_index`, `quality_attribute`

### 3.5 Nodo Supervisor (`nodes/supervisor.py`)

Coordinador principal. Analiza la intención, planifica la ejecución multi-intent y decide el siguiente nodo.

**Funciones clave**:
- `detect_lang(text)` — Detección de idioma (ES/EN) por keywords
- `classify_followup(question)` — Clasificación de intención por patrones regex
- `_infer_requested_nodes(uq, state, forced)` — Planificador multi-intent
- `_augment_completed_nodes(state, completed)` — Sincroniza nodos completados desde turn_messages y flags
- `makeSupervisorPrompt(state)` — Genera el prompt para el fallback LLM
- `supervisor_node(state)` — Nodo principal

**Planificación Multi-Intent**:

El supervisor construye un plan (`requested_nodes`) en el primer paso del turno y lo ejecuta secuencialmente via `pending_nodes`:

```
Ejemplo: "Genera un ASR y dame estilos y tácticas"
  requested_nodes = ["asr", "style", "tactics"]
  
  Turno 1: next_node = "asr"          (prioridad ASR si falta)
  Turno 2: next_node = "style"        (primer pending)
  Turno 3: next_node = "tactics"      (siguiente pending)
  Turno 4: next_node = "unifier"      (todos completados)
```

**Prioridad ASR**: Si ASR está en el plan y aún no se completó, siempre se ejecuta primero (estilos y tácticas dependen del ASR).

**Fallback LLM**: Si no hay plan explícito y la intención es general, usa `llm.with_structured_output(supervisorSchema)` para decidir el siguiente nodo.

**Guard anti-unifier vacío**: Si `next_node == "unifier"` pero ningún nodo ha sido visitado en el turno, redirige a `investigator`.

**State leído**: `userQuestion`, `language`, `intent`, `hasVisited*`, `completed_nodes`, `pending_nodes`, `requested_nodes`, `diagram`, `doc_only`

**State escrito**: `nextNode`, `localQuestion`, `intent`, `language`, `requested_nodes`, `pending_nodes`, `completed_nodes`

### 3.6 Router (`workflow.py::router`)

Función de enrutamiento condicional que traduce `state["nextNode"]` al nodo LangGraph correspondiente:

```python
def router(state: GraphState) -> str:
    # Si hay SVG listo → unifier (el supervisor ya lo detectó)
    if state["nextNode"] == "unifier":
        return "unifier"

    # ASR: si debe pasar por RAG primero
    if state["nextNode"] == "asr" and not state.get("hasVisitedASR"):
        if not state.get("hasVisitedInvestigator") and not state.get("doc_only") and state.get("force_rag"):
            return "investigator"
        return "asr"

    # QA-aware routing para estilos y tácticas
    qa = normalize_qa(state.get("quality_attribute") or state.get("resolved_index") or "")
    if state["nextNode"] == "style":
        return style_node_name_for_qa(qa)   # e.g. "style_latency"
    if state["nextNode"] == "tactics":
        return tactics_node_name_for_qa(qa) # e.g. "tactics_escalabilidad"

    # Nodos con guards de visita
    if state["nextNode"] == "investigator" and not state["hasVisitedInvestigator"]:
        return "investigator"
    if state["nextNode"] == "diagram_agent" and not state.get("hasVisitedDiagram"):
        return "diagram_agent"
    if state["nextNode"] == "evaluator" and not state["hasVisitedEvaluator"]:
        return "evaluator"

    return "unifier"
```

Los nodos dinámicos QA-aware se registran automáticamente desde `config/indices.json`.

---

## 4. Nodos Especializados

### 4.1 Investigator (`nodes/investigator.py`)

**Propósito**: Búsqueda RAG en documentación PDF para fundamentación técnica.

**Patrón**: ReAct Agent con herramientas

**LLM Call**: `create_react_agent(llm, tools=[local_RAG, LLM, LLMWithImages])`
- Recursion limit: 12 pasos

**Herramientas disponibles**:
| Herramienta     | Función                                               |
| --------------- | ----------------------------------------------------- |
| `local_RAG`     | Búsqueda semántica en ChromaDB por QA + content_type  |
| `LLM`           | Llamada directa al LLM (output estructurado)          |
| `LLMWithImages` | Análisis de imágenes via Vertex AI Gemini             |

**Guards (condiciones de skip)**:
- Si `intent == "diagram"` → skip
- Si `intent ∈ ("greeting", "smalltalk")` → respuesta rápida sin RAG
- Si `doc_only == True` → no se invoca RAG externo; trabaja solo con `doc_context`

**State escrito**: `messages`, `hasVisitedInvestigator`, `add_context`

### 4.2 ASR Node (`nodes/asr.py`)

**Propósito**: Generación de Quality Attribute Scenarios según ADD 3.0.

**LLM Call**: `llm.invoke(prompt)` — Una sola llamada para generar el ASR completo.

**Formato de salida obligatorio**:
```
ASR complete: [descripción en una línea]

- Source: [origen del estímulo]
- Stimulus: [evento/condición que activa]
- Environment: [contexto operacional]
- Artifact: [componente/sistema afectado]
- Response: [comportamiento esperado]
- Response Measure: [métricas: p95/p99, RPS, % disponibilidad, etc.]
```

**Integración RAG**: `get_indexed_retriever(quality_attribute=qa, content_type="asr", k=6)` para obtener ejemplos de referencia.

**Detección de QA**: Heurística — "scalab" → scalability, "latenc" → latency, else performance.

**State escrito**: `last_asr`, `current_asr`, `quality_attribute`, `arch_stage="ASR"`, `endMessage`, `hasVisitedASR`, `turn_messages`, `asr_sources_list`

### 4.3 Style Nodes (`nodes/styles/`)

**Propósito**: Recomendación de estilos arquitectónicos basada en el ASR actual.

**Estructura modular**:
```
styles/
├── style.py              # Nodo genérico + factory make_style_qa_node()
├── common.py             # Implementación compartida style_node_impl()
├── latency_style.py      # Nodo especializado para latencia
├── scalability_style.py  # Nodo especializado para escalabilidad
└── availability_style.py # Nodo especializado para disponibilidad
```

**LLM Call**: `llm.invoke(prompt)` — Genera JSON con candidatos de estilo.

**Output JSON**:
```json
{
    "style_1": { "name": "Microservices", "impact": "..." },
    "style_2": { "name": "Event-Driven",  "impact": "..." },
    "best_style": "style_1",
    "rationale": "..."
}
```

**Integración RAG**: Queries como `"{qa} architecture style"`, `"Bass Clements Kazman architecture styles"` con `content_type="estilos"`.

**Factory pattern**: `make_style_qa_node(qa_id)` genera nodos dinámicos que delegan a `style_node_impl(state, qa_override=qa_id)`.

**State escrito**: `style`, `selected_style`, `last_style`, `arch_stage="STYLE"`, `quality_attribute`, `memory_text`, `endMessage`

### 4.4 Tactics Nodes (`nodes/tactics/`)

**Propósito**: Generación de 3 tácticas arquitectónicas en formato JSON estructurado.

**Estructura modular**:
```
tactics/
├── tactics.py              # Nodo genérico + factory make_tactics_qa_node()
├── common.py               # Implementación compartida tactics_node_impl()
├── latency_tactics.py      # Nodo especializado para latencia
├── scalability_tactics.py  # Nodo especializado para escalabilidad
└── availability_tactics.py # Con catálogo restringido de Fault Detection
```

**LLM Call**: `llm.invoke(prompt)` — Genera 3 tácticas con JSON.

**Output JSON Schema**:
```json
[
    {
        "name": "TacticName",
        "purpose": "Descripción del propósito",
        "rationale": "Justificación",
        "risks": ["riesgo1"],
        "tradeoffs": ["tradeoff1"],
        "categories": ["fault-detection"],
        "traces_to_asr": "Relación con el ASR",
        "expected_effect": "Efecto esperado",
        "success_probability": 0.82,
        "rank": 1
    }
]
```

**Extracción y reparación de JSON** (pipeline de fallbacks):
1. `extract_json_array(raw)` — Extrae del markdown fence
2. `_json_only_repair_pass()` — Reparación via LLM si falla el parse
3. `build_json_from_markdown()` — Fallback regex desde markdown
4. `normalize_tactics_json()` — Normaliza a exactamente 3 items con campos requeridos

**Caso especial — `availability_tactics.py`**:
- Define `_FAULT_DETECTION_CATALOG` con 9 tácticas específicas: Ping/Echo, Heartbeat, Monitor (watchdog), Timestamp, Sanity checking, Condition monitoring, Voting, Exception detection, Self-test
- `restrict_to_preferred_tactics=True` — Constraint duro para usar solo tácticas del catálogo
- Canonicaliza nombres y rellena con tácticas fallback si el LLM no produce suficientes

**Integración RAG**: `content_type="tacticas"` con queries específicas por QA.

**State escrito**: `tactics_md`, `tactics_struct` (lista de dicts), `tactics_list` (solo nombres), `arch_stage="TACTICS"`, `quality_attribute`, `endMessage`, `intent="tactics"`

### 4.5 Diagram Orchestrator (`nodes/diagram.py`)

**Propósito**: Generación de diagramas arquitectónicos en formato DOT con 3 niveles de detalle y exportación multi-formato.

**Funciones clave**:
- `diagram_orchestrator_node(state)` — Nodo principal
- `_sanitize_dot(raw)` — Extrae DOT válido del output del LLM; preserva UTF-8
- `_llm_nl_to_dot(natural_prompt, level, expand)` — Convierte lenguaje natural a DOT via LLM
- `_render_dot_svg_b64(dot_code)` — Renderiza DOT a SVG base64 via Graphviz binary
- `_infer_level_from_user_request(user_q)` — Infiere nivel de detalle del prompt del usuario
- `_resolve_diagram_level(state, user_q)` — Resuelve nivel desde state override o prompt

**LLM Call**: `llm.invoke([SystemMessage, HumanMessage])` — Genera código DOT.

**System Prompts por nivel** (definidos en `consts.py`):
| Nivel | Constante | Descripción |
| ----- | --------- | ----------- |
| 1 (OVERVIEW)  | `DOT_SYSTEM_OVERVIEW` | 5-15 nodos, subsistemas de alto nivel |
| 2 (MEDIUM)    | `DOT_SYSTEM`          | ~30 nodos, componentes visibles |
| 3 (DETAILED)  | `DOT_SYSTEM`          | Todos los componentes, complejidad completa |
| Expansión     | `DOT_SYSTEM_EXPAND`   | Reutiliza DOT anterior como base |

**Modo Expansión**:
- Detecta diagrama previo en `diagram_history` y lo reutiliza como base
- Usa `DOT_SYSTEM_EXPAND` con `EXPAND_TARGET_MEDIUM` o `EXPAND_TARGET_DETAILED`
- Invalida niveles superiores cuando se regenera un nivel inferior

**Pipeline de renderizado**:
```
LLM Output → _sanitize_dot() → parse_dot_to_model() → build_diagram_model()
    → render_dot()        (DOT limpio)
    → render_svg_b64()    (SVG base64 via Graphviz binary)
    → render_dot_drawio() (draw.io XML / mxGraph)
```

**State escrito**:
```python
diagram = {
    "ok": bool,
    "svg_b64": str,        # SVG codificado en base64
    "dot": str,            # DOT renderizado limpio
    "dot_raw": str,        # DOT crudo del LLM
    "dot_drawio": str,     # XML draw.io
    "node_count": int,
    "edge_count": int,
    "level": int,          # 1, 2, 3
    "detail_level": str,   # "overview", "medium", "detailed"
    "error": str | None
}
diagram_history = {
    1: "DOT nivel 1",
    2: "DOT nivel 2",
    3: "DOT nivel 3"
}
```

### 4.6 Evaluator (`nodes/evaluator.py`)

**Propósito**: Evaluación y crítica de ASRs, arquitecturas y diagramas.

**Funciones clave**:
- `evaluator_node(state)` — Nodo principal
- `_pick_asr_to_evaluate(state)` — Extrae ASR del estado o del mensaje del usuario
- `_book_snippets_for_eval(retriever, concern_hint)` — Obtiene contexto RAG para evaluación
- `getEvaluatorPrompt(image_path1, image_path2)` — Genera prompt de evaluación

**Dos modos de operación**:

| Modo | Cuándo | LLM Call |
| ---- | ------ | -------- |
| **ASR Evaluation** | Cuando hay ASR para evaluar | `llm.invoke(eval_prompt)` directo |
| **General**        | Evaluación abierta          | `create_react_agent(llm, tools=[theory, viability, needs, analyze])` |

**Estructura del prompt de evaluación**:
- **Verdict**: Good / Weak / Invalid
- **Gaps**: Campos QAS faltantes o vagos
- **Quality**: Medibilidad, precisión, realismo
- **Risks & Tactics**: Riesgos potenciales y mitigaciones
- **Rewrite**: ASR mejorado
- **References**: Fuentes de snippets

**Triggers de evaluación** (detección en supervisor):
```python
EVAL_TRIGGERS = [
    "evaluate this asr", "check this asr", "review this asr",
    "evalúa este asr", "evalua este asr", "revisa este asr",
    "es bueno este asr", "mejorar este asr", "critique this asr"
]
```

**State escrito**: `messages`, `hasVisitedEvaluator`, `turn_messages`

### 4.7 Unifier (`nodes/unifier.py`)

**Propósito**: Consolidación final de outputs de múltiples nodos especializados en una respuesta coherente.

**Funciones clave**:
- `unifier_node(state)` — Nodo principal
- `_last_ai_by(state, name)` — Recupera último mensaje AI por nombre de nodo
- `_last_turn_by(state, name)` — Recupera último mensaje de turno
- `_split_sections(text)` — Parsea secciones Answer/References/Next
- `_extract_rag_sources_from(text)` — Extrae fuentes para atribución

**LLM Call**: `llm.invoke(prompt)` — Sintetiza respuesta final.

**Modos de output**:

| Modo | Condición | Comportamiento |
| ---- | --------- | -------------- |
| **Composite** | `len(requested_nodes) >= 2` | Combina ASR + estilos + tácticas + diagrama |
| **Diagram**   | `intent == "diagram"`       | Muestra SVG con sugerencias de seguimiento |
| **Style**     | `intent == "style"`         | Destaca recomendación de estilo |
| **Tactics**   | `intent == "tactics"`       | Muestra tácticas + fuentes |
| **ASR**       | `intent == "asr"`           | Muestra ASR con sugerencias de refinamiento |
| **Greeting**  | `intent == "greeting"`      | Respuesta de saludo directa |
| **Default**   | Otros                       | Sintetiza outputs de investigator + evaluator |

**State escrito**: `endMessage` (respuesta final), `suggestions` (chips de seguimiento), `intent`

---

## 5. Herramientas (Tools)

Definidas en `nodes/tools.py` y usadas por el Investigator y el Evaluator:

```python
@tool
def LLM(prompt: str) -> dict:
    """Llamada directa al LLM con output estructurado (investigatorSchema)."""
    return llm.with_structured_output(investigatorSchema).invoke(prompt)

@tool
def LLMWithImages(image_path: str) -> str:
    """Análisis de imágenes via Vertex AI Gemini 1.0 Pro Vision."""
    model = GenerativeModel("gemini-1.0-pro-vision")
    return model.generate_content([prompt, image])

@tool
def local_RAG(prompt: str, quality_attribute: str = "general") -> str:
    """Búsqueda semántica en ChromaDB filtrada por QA y content_type.
    Retorna: preview + bloque SOURCES con paths y números de página."""
    retriever = get_indexed_retriever(quality_attribute=quality_attribute, k=8)
    ...

@tool
def theory_tool() -> dict:
    """Evalúa corrección teórica vs mejores prácticas."""

@tool
def viability_tool() -> dict:
    """Evalúa factibilidad, costo y riesgos."""

@tool
def needs_tool() -> dict:
    """Valida alineamiento con requisitos."""

@tool
def analyze_tool(image_path, image_path2) -> str:
    """Compara dos diagramas arquitectónicos (requiere Vertex AI)."""
```

---

## 6. Estado del Grafo

### 6.1 GraphState (TypedDict)

```python
class GraphState(TypedDict):
    # === Control de flujo ===
    messages: Annotated[list[AnyMessage], add_messages]  # Historial de conversación
    userQuestion: str       # Pregunta original del usuario
    localQuestion: str      # Pregunta refinada para el worker
    nextNode: str           # Siguiente nodo a ejecutar

    # === Planificación multi-intent ===
    requested_nodes: list   # Nodos solicitados por el usuario en este turno
    pending_nodes: list     # Nodos pendientes de ejecución
    completed_nodes: list   # Nodos completados en este turno

    # === Flags de visita ===
    hasVisitedInvestigator: bool
    hasVisitedEvaluator: bool
    hasVisitedASR: bool
    hasVisitedDiagram: bool

    # === Clasificación ===
    language: str           # "es" | "en"
    intent: str             # greeting | architecture | diagram | asr | tactics | style | general
    force_rag: bool         # Si se debe forzar búsqueda RAG
    quality_attribute: str  # latencia | escalabilidad | disponibilidad | general
    resolved_index: str     # Índice QA resuelto para RAG

    # === Contexto arquitectónico ===
    current_asr: str        # ASR vigente de la sesión
    last_asr: str           # Último ASR generado en el turno
    arch_stage: str         # ASR | STYLE | TACTICS | DEPLOYMENT
    selected_style: str     # Estilo seleccionado
    last_style: str         # Último estilo generado
    style: str              # Texto del estilo

    tactics_list: list      # Lista de nombres de tácticas
    tactics_struct: list    # Estructura JSON de tácticas
    tactics_md: str         # Tácticas en formato markdown

    # === RAG y documentos ===
    doc_context: str        # Contexto de documentos subidos (PDF)
    add_context: str        # Contexto adicional del investigator
    doc_only: bool          # Modo DOC-ONLY (solo documentos, sin RAG externo)
    force_rag: bool         # Forzar búsqueda RAG
    asr_sources_list: list  # Fuentes usadas para ASR
    retrieved_docs: list    # Documentos recuperados del vectorstore

    # === Memoria liviana ===
    memory_text: str        # Snapshot del estado arquitectónico para contexto

    # === Diagramas ===
    diagram: dict           # Datos del diagrama del turno actual
    diagram_history: dict   # DOT por nivel {1: "...", 2: "...", 3: "..."}

    # === Imágenes ===
    imagePath1: str         # Ruta de imagen 1 subida
    imagePath2: str         # Ruta de imagen 2 subida

    # === Turno actual ===
    turn_messages: list     # Mensajes generados en el turno actual
    endMessage: str         # Respuesta final para el usuario
    suggestions: list       # Sugerencias de seguimiento (chips)
```

### 6.2 Schemas de Respuesta de Nodos

```python
class supervisorResponse(TypedDict):
    localQuestion: str   # Pregunta refinada para el worker
    nextNode: Literal[
        "investigator", "evaluator", "diagram_agent",
        "tactics", "asr", "style", "unifier"
    ]

class investigatorSchema(TypedDict):
    definition: str      # Definición del concepto
    useCases: str        # Casos de uso
    examples: str        # Ejemplos

class evaluatorResponse(TypedDict):
    positiveAspects: str
    negativeAspects: str
    suggestions: str

class ClassifyOut(TypedDict):
    language: Literal["en", "es"]
    intent: str
    use_rag: bool
    quality_attribute: str
```

**Tactics Schema** (valida JSON de salida):
```python
TACTIC_ITEM_SCHEMA = {
    "required": ["name", "rationale", "categories", "success_probability", "rank"],
    "optional": ["purpose", "risks", "tradeoffs", "traces_to_asr", "expected_effect"]
}
TACTICS_ARRAY_SCHEMA = { "type": "array", "minItems": 3, "maxItems": 3 }
```

---

## 7. Sistema de Quality Attributes (QA)

### 7.1 QA Registry (`graph/qa_registry.py`)

Gestión centralizada de atributos de calidad con normalización de keywords:

**Funciones clave**:
- `normalize_qa(value)` — Mapea variantes → ID canónico (ej: "rendimiento" → "latencia")
- `supported_qas()` — Lista QAs disponibles desde `config/indices.json`
- `style_node_name_for_qa(qa_id)` — Resuelve nombre del nodo de estilo dinámico
- `tactics_node_name_for_qa(qa_id)` — Resuelve nombre del nodo de tácticas dinámico
- `qa_to_focus_label(qa_id)` — Label en inglés para prompts

### 7.2 Index Resolver (`graph/index_resolver.py`)

Clasifica la pregunta del usuario → índice QA para filtrado RAG:

1. Carga config desde `config/indices.json`
2. Usa LLM para clasificar pregunta contra descripciones de QA
3. Fallback a matching de substrings sobre keywords
4. Default: `"general"` si no hay match

### 7.3 Configuración de QAs (`config/indices.json`)

```json
{
    "quality_attributes": [
        {
            "id": "escalabilidad",
            "description": "Capacidad del sistema para manejar crecimiento",
            "keywords_en": ["throughput", "scale", "elastic", "sharding", "distributed"],
            "keywords_es": ["escalabilidad", "crecimiento", "carga", "particionamiento"]
        },
        {
            "id": "latencia",
            "description": "Tiempo de solicitud a respuesta",
            "keywords_en": ["response time", "real-time", "p95", "p99", "performance"],
            "keywords_es": ["latencia", "rendimiento", "tiempo de respuesta"]
        },
        {
            "id": "disponibilidad",
            "description": "Operabilidad del sistema cuando se requiere",
            "keywords_en": ["uptime", "fault tolerance", "resilience", "redundancy", "failover"],
            "keywords_es": ["disponibilidad", "tolerancia a fallos", "resiliencia"]
        }
    ],
    "content_types": ["asr", "estilos", "tacticas"]
}
```

### 7.4 Routing QA-Aware

Al agregar un nuevo QA en `config/indices.json`, el workflow registra sus nodos automáticamente sin modificar `workflow.py`:

```
Para QA "latencia":
  → style_latency     (usa style_node_impl con qa_override="latencia")
  → tactics_latency   (usa tactics_node_impl con qa_override="latencia")

Para QA "escalabilidad":
  → style_escalabilidad
  → tactics_escalabilidad

Para QA "disponibilidad":
  → style_disponibilidad
  → tactics_disponibilidad (con catálogo restringido de Fault Detection)
```

---

## 8. Sistema de Diagramas

### 8.1 Representación Intermedia (IR) (`services/diagram_ir.py`)

Modelo agnóstico al renderizador:

```python
class DiagramNode:
    id: str
    label: str
    kind: NodeKind  # SERVICE, DATABASE, QUEUE, CACHE, GATEWAY,
                    # LOADBALANCER, CDN, CLIENT, EXTERNAL, CLUSTER, GENERIC

class DiagramEdge:
    source: str
    target: str
    label: str
    kind: EdgeKind  # SYNC, ASYNC, DATA, DEPENDS

class DiagramModel:
    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    groups: list[DiagramGroup]  # Clusters jerárquicos
    detail_level: DetailLevel   # OVERVIEW, MEDIUM, DETAILED

class DiagramLevel(Enum):
    OVERVIEW  = 1   # 5-15 nodos
    MEDIUM    = 2   # ~30 nodos
    DETAILED  = 3   # Todos los componentes
```

### 8.2 Pipeline de Generación

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. Usuario solicita diagrama                                        │
│  2. _resolve_diagram_level() → determina nivel (1, 2, 3)            │
│  3. Construye prompt con: ASR + Style + Tactics + Contexto           │
│  4. _llm_nl_to_dot() → LLM genera código DOT                        │
│  5. _sanitize_dot() → extrae DOT válido, preserva UTF-8             │
│  6. parse_dot_to_model() → DiagramModel (IR)                        │
│  7. build_diagram_model() → modelo con nivel de abstracción         │
│  8. Rendering paralelo:                                              │
│     ├→ render_dot()        → DOT limpio                              │
│     ├→ render_svg_b64()    → SVG base64 (via Graphviz binary)       │
│     └→ render_dot_drawio() → XML draw.io (mxGraph)                  │
│  9. Almacena en diagram_history[level] para expansión futura        │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.3 Expansión Progresiva (Progressive Disclosure)

El sistema mantiene `diagram_history` con el DOT de cada nivel generado:

- **Nivel 1 → 2**: Expansión de subsistemas a componentes
- **Nivel 2 → 3**: Expansión completa con todos los detalles
- **Re-generación**: Si se regenera nivel 1, se invalidan niveles 2 y 3
- **Cross-level**: `build_expanded_view()` puede enfocar un nodo específico del overview

### 8.4 Renderizado Multi-Formato (`services/diagram_render.py`)

| Formato       | Método              | Descripción                                           |
| ------------- | ------------------- | ----------------------------------------------------- |
| **DOT**       | `render_dot()`      | DOT con estilos de legibilidad                        |
| **SVG**       | `render_svg_b64()`  | Via binario `dot` de Graphviz (subprocess)            |
| **draw.io**   | `render_drawio()`   | XML nativo mxGraph con posiciones computadas          |
| **dot_drawio**| `render_dot_drawio()` | DOT simplificado compatible con import draw.io      |

### 8.5 Constantes DOT (`consts.py`)

- `DOT_SYSTEM` — Prompt general para generación de DOT (niveles 2 y 3)
- `DOT_SYSTEM_OVERVIEW` — Prompt para nivel 1 (alto nivel, pocos nodos)
- `DOT_SYSTEM_EXPAND` — Prompt para expansión (reutiliza DOT anterior como base)
- `EXPAND_TARGET_MEDIUM` / `EXPAND_TARGET_DETAILED` — Descripciones de expansión

---

## 9. Sistema RAG

### 9.1 RAG Agent (`rag_agent.py`)

**Integración ChromaDB**:
- Collection: `"arquia"`
- Embeddings: `text-embedding-3-small` (configurable via `OPENAI_EMBED_MODEL`)
- Singleton pattern: `_VDB` (lazy loading en primer acceso)
- Soporte dual: Azure OpenAI Embeddings o OpenAI público (autodetectado)

**Funciones clave**:
```python
def create_or_load_vectorstore() -> Chroma:
    """Carga o crea el vectorstore singleton de ChromaDB."""

def get_retriever(title: str | list[str] | None = None, k: int = 6):
    """Retriever con filtrado opcional por título de documento.
    - str: filtra por igualdad exacta (metadata.title == title)
    - list: filtra por cualquiera ($in)
    """

def get_indexed_retriever(quality_attribute=None, content_type=None, k=6):
    """Retriever filtrado por QA y/o content_type.
    - Ambos presentes: usa $and
    - Solo uno: filtra por ese
    - Ninguno: sin filtros (igual a get_retriever())
    """

def rebuild_vectorstore():
    """Elimina el directorio Chroma y reconstruye desde cero."""
```

**Metadata almacenada por documento**:
- `source_path` — Ruta del archivo fuente
- `title` — Título del documento
- `page_label` — Número de página
- `quality_attribute` — QA asociado
- `content_type` — Tipo (asr, estilos, tacticas)

### 9.2 RAG Tracing (`resources.py`)

Sistema de trazabilidad de consultas RAG en memoria, por sesión:

```python
def rag_trace_set_session(session_id) -> None  # Fija sesión activa
def rag_trace_reset(session_id)      -> None  # Inicializa contadores
def rag_trace_record(*, query, docs, session_id) -> None  # Registra hit
def rag_trace_get(session_id)        -> dict  # Recupera métricas
```

Estructura de traza:
```python
{
    "attempted": bool,
    "hit_count": int,
    "queries": ["query1", "query2"],
    "sources": ["Title (p.3) — /path/to/doc.pdf"]
}
```

### 9.3 Ingesta de Documentos (`services/doc_ingest.py`)

**Pipeline de extracción PDF**:
1. **PyMuPDF (fitz)**: Método primario, itera páginas con preservación de estructura
2. **pypdf**: Fallback si PyMuPDF falla
3. Normalización de whitespace + truncamiento a `max_chars` (default 8000)

### 9.4 Construcción del Vectorstore (`build_vectorstore.py`)

Script CLI para poblar ChromaDB desde PDFs:
- Lee PDFs de `back/docs/`
- Chunks via LangChain text splitters
- Persiste en `back/chroma_db/`
- Almacena metadata: source_path, title, page_label, quality_attribute, content_type

---

## 10. Servicios Core

### 10.1 LLM Factory (`services/llm_factory.py`)

Abstracción multi-proveedor con autodetección:

| Proveedor        | Clase             | Condición de activación                        |
| ---------------- | ----------------- | ---------------------------------------------- |
| **Azure OpenAI** | `AzureChatOpenAI` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` presentes |
| **OpenAI**       | `ChatOpenAI`      | `OPENAI_API_KEY` o `OPENAI_BASE_URL` presentes |
| **Ollama**       | `ChatOllama`      | Fallback si ninguna de las anteriores          |

El proveedor puede forzarse con `ROS_LG_LLM_PROVIDER`.

**Aliases de modelos Azure**:
```python
AZURE_ALIASES = {
    "gpt4omini": "gpt-4o-mini",
    "gpt4o":     "gpt-4o",
    "o3mini":    "o3-mini",
    "gpt41":     "gpt-4.1",
    "gpt41mini": "gpt-4.1-mini",
    "gpt5":      "gpt-5",
    "gpt5mini":  "gpt-5-mini"
}
```

**Aliases de modelos Ollama**:
```python
OLLAMA_ALIASES = {
    "llama3.2:3b":        "llama3.2:3b",
    "llama3.1:8b":        "llama3.1:8b",
    "mistral:7b-instruct":"mistral:7b-instruct",
    "qwen2.5:7b-instruct":"qwen2.5:7b-instruct",
    "deepseek-r1:7b":     "deepseek-r1:7b"
}
```

**Parámetros configurables**: `temperature` (default 0.0), `max_tokens`, `timeout`, `max_retries` (default 2).

### 10.2 Memory Manager (`memory.py`)

**Persistencia SQLite** en `back/state_db/memory.db`:

```python
# Tabla: memory (user_id, key, value, updated_at)

def init()                             # Crea la tabla si no existe
def set_kv(user_id, key, value)        # Insert/update con timestamp
def get(user_id, key, default="")      # Retrieve por user_id + key
def load_arch_flow(user_id) -> dict    # Carga estado ADD 3.0 completo
def save_arch_flow(user_id, flow)      # Persiste estado ADD 3.0
```

**Esquema `arch_flow`** (estado ADD 3.0 por usuario):
```python
{
    "stage": "",                        # ASR | STYLE | TACTICS | DEPLOYMENT
    "quality_attribute": "",            # latencia | escalabilidad | disponibilidad
    "add_context": "",                  # Dominio / driver de negocio
    "current_asr": "",                  # ASR oficial de la sesión
    "style": "",                        # Estilo arquitectónico elegido
    "tactics": [],                      # Lista de tácticas aceptadas
    "deployment_diagram_puml": "",      # PlantUML del despliegue final
    "deployment_diagram_svg_b64": "",   # SVG base64 del despliegue final
}
```

`load_arch_flow` siempre retorna todas las llaves con sus defaults, haciendo el struct seguro para acceso directo.

### 10.3 Sesión HTTP (`resources.py`)

Sesión HTTP con retry automático:
```python
Retry(total=3, backoff_factor=0.5, status_forcelist=(502, 503, 504))
HTTPAdapter(pool_connections=10, pool_maxsize=10)
```

---

## 11. API REST

### 11.1 Endpoints

| Endpoint          | Método | Propósito |
| ----------------- | ------ | --------- |
| `/`               | GET    | Health / info raíz |
| `/health`         | GET    | Health check |
| `/message`        | POST   | Chat principal (multipart: message, session_id, image1, image2) |
| `/diagram/export` | GET    | Exportar diagrama generado (query: session_id, format, level, focus) |
| `/diagrams`       | GET    | Exportar grafo del workflow LangGraph (query: format=dot\|svg) |
| `/feedback`       | POST   | Registrar feedback thumbs up/down por mensaje |
| `/test`           | POST   | Endpoint de test con respuestas mock |

### 11.2 Endpoint `/message` — Flujo Detallado

```
1.  Recibe: message (str), session_id (str), image1/image2 (files opcionales)
2.  Detecta idioma (ES/EN)
3.  Si hay PDF → extrae texto via doc_ingest (max 8000 chars) → almacena en doc_context
4.  Si hay imagen → guarda en back/images/ → almacena path
5.  Carga arch_flow desde memory (SQLite) para construir memory_text
6.  Detecta heurísticas: force_rag, user_intent, topic_hint
7.  Limpia buffers de turno en el estado del grafo (sin borrar historial)
8.  Invoca LangGraph: graph.invoke(state, config={"thread_id": session_id})
9.  Extrae resultado: endMessage, diagram, suggestions, turn_messages
10. Persiste arch_flow actualizado (ASR, style, tactics, last_diagram)
11. Aplica fix_utf8_recursive() al payload de respuesta
12. Retorna JSONResponse con charset=utf-8
```

### 11.3 Response del `/message`

```json
{
    "endMessage": "Respuesta final consolidada",
    "diagram": {
        "ok": true,
        "dot": "digraph {...}",
        "svg_b64": "base64...",
        "dot_drawio": "<mxGraphModel>...</mxGraphModel>",
        "node_count": 12,
        "edge_count": 15,
        "level": 1,
        "detail_level": "overview"
    },
    "messages": [
        {"name": "investigator", "text": "..."},
        {"name": "asr", "text": "..."}
    ],
    "session_id": "abc123",
    "message_id": 3,
    "thread_id": "abc123",
    "suggestions": ["Expandir diagrama", "Evaluar ASR", "Generar tácticas"]
}
```

### 11.4 Endpoint `/diagram/export`

```
Query params:
  - session_id    (required) : Sesión que produjo el diagrama
  - format        : svg | dot | dot_drawio | drawio (default: svg)
  - detail_level  : overview | detailed (legacy)
  - level         : 1 | 2 | 3 (preferido sobre detail_level)
  - focus         : ID de nodo del overview para expandir (requiere level=3)

Retorna: archivo adjunto en el formato solicitado
```

### 11.5 Middleware

| Middleware   | Propósito |
| ------------ | --------- |
| **CORS**     | Orígenes: `localhost:5173`, `127.0.0.1:5173`. Credentials habilitados. Todos los métodos y headers. |
| **UTF-8**    | Asegura `charset=utf-8` en responses JSON. Corrige doble-encoding Latin-1/UTF-8. |
| **Lifespan** | Startup: inicializa RAG vectorstore. Shutdown: cierre graceful. |

### 11.6 Sistema de Feedback

Persistido en `back/feedback_db/feedback.db` (SQLite):

```sql
CREATE TABLE message_feedback (
    session_id   TEXT NOT NULL,
    message_id   INTEGER NOT NULL,
    thumbs_up    INTEGER DEFAULT 0,
    thumbs_down  INTEGER DEFAULT 0,
    PRIMARY KEY (session_id, message_id)
)
```

Cada respuesta del `/message` genera un `message_id` incremental por sesión. El endpoint `/feedback` permite actualizar thumbs up/down.

---

## 12. Persistencia

### 12.1 Capas de Persistencia

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        CAPAS DE PERSISTENCIA v4                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │  ChromaDB      │  │  SQLite        │  │  SQLite        │             │
│  │  (chroma_db/)  │  │  (state_db/)   │  │  (feedback_db/)│             │
│  ├────────────────┤  ├────────────────┤  ├────────────────┤             │
│  │ Embeddings     │  │ Estado ADD 3.0 │  │ Likes/Dislikes │             │
│  │ documentos     │  │ por usuario    │  │ por mensaje    │             │
│  │ (asr, estilos, │  │ (arch_flow,    │  │                │             │
│  │  tácticas)     │  │  ASR, estilos) │  │                │             │
│  └────────────────┘  └────────────────┘  └────────────────┘             │
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐                                 │
│  │  RAM           │  │  Filesystem    │                                 │
│  │  (MemorySaver) │  │  (docs_uploads │                                 │
│  ├────────────────┤  │   + images/)   │                                 │
│  │ Checkpoints    │  ├────────────────┤                                 │
│  │ del grafo      │  │ PDFs subidos   │                                 │
│  │ por thread_id  │  │ Imágenes       │                                 │
│  │ (volátil)      │  │ del usuario    │                                 │
│  └────────────────┘  └────────────────┘                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Cambios respecto a v3

| Aspecto | v3 | v4 |
| ------- | -- | -- |
| **Checkpointing LangGraph** | `SqliteSaver` (persistente) | `MemorySaver` (RAM, volátil) |
| **Memoria ADD 3.0** | SQLite persistente | SQLite persistente (sin cambios) |
| **RAG tracing** | No documentado | `rag_trace_record()` en `resources.py` |
| **Embeddings Azure** | No soportado | Autodetección Azure/OpenAI en `rag_agent.py` |
| **UTF-8** | Parcial | Middleware + `fix_utf8_recursive()` en todas las respuestas |
| **Nodo boot** | Incluido en v3 | Igual, documentado con código real |
| **Retorno de nodos** | Algunos → Unifier | Todos → Supervisor (scheduler multi-intent) |

---

## 13. Jerarquía de Archivos

```
archIABack/
├── README.md                              # Guía de instalación y setup
├── Documentacion_Tecnica_ArchIA_v4.md     # Este documento
│
└── back/                                  # Backend Python
    ├── .env                               # Variables de entorno (no commitear)
    ├── requirements.txt                   # Dependencias Python (referencia)
    ├── langgraph.json                     # Configuración LangGraph CLI
    ├── build_vectorstore.py               # Script de indexación RAG
    ├── chroma_web.py                      # UI web para ChromaDB (puerto 8001)
    │
    ├── config/
    │   └── indices.json                   # Configuración QA + content types
    │
    ├── docs/                              # PDFs de referencia para RAG
    │   ├── Software Architecture in practice.pdf
    │   ├── Software Systems Architecture cap*.pdf
    │   └── ...
    │
    ├── chroma_db/                         # Vectorstore persistido (generado)
    ├── state_db/                          # SQLite de memoria de sesión (generado)
    │   └── memory.db
    ├── feedback_db/                       # SQLite de feedback (generado)
    │   └── feedback.db
    ├── images/                            # Imágenes subidas por usuarios (generado)
    ├── docs_uploads/                      # PDFs subidos por usuarios (generado)
    │
    └── src/
        ├── main.py                        # Entry point FastAPI (puerto 8000)
        ├── memory.py                      # Gestión de memoria SQLite (arch_flow)
        ├── rag_agent.py                   # Integración ChromaDB
        ├── quoting.py                     # Utilidades de citado
        │
        ├── graph/                         # Paquete del Grafo LangGraph
        │   ├── __init__.py                # Exporta instancia 'graph'
        │   ├── workflow.py                # Definición del flujo (boot → ... → END)
        │   ├── state.py                   # TypedDicts (GraphState + schemas)
        │   ├── consts.py                  # System prompts DOT + constantes
        │   ├── resources.py               # LLM, RAG tracer, HTTP client, builder
        │   ├── utils.py                   # Token counting, JSON extraction, helpers
        │   ├── qa_registry.py             # Gestión y normalización de QAs
        │   ├── index_resolver.py          # Resolución QA → índice RAG
        │   │
        │   └── nodes/                     # Nodos especializados
        │       ├── classifier.py          # Clasificador idioma + intención + QA
        │       ├── supervisor.py          # Planificador multi-intent + router lógico
        │       ├── investigator.py        # ReAct Agent con RAG
        │       ├── asr.py                 # Generación ASR (ADD 3.0)
        │       ├── evaluator.py           # Evaluador de ASRs y arquitecturas
        │       ├── diagram.py             # Orquestador de diagramas DOT
        │       ├── unifier.py             # Consolidador de respuestas
        │       ├── tools.py               # Herramientas: local_RAG, LLM, LLMWithImages
        │       │
        │       ├── styles/                # Nodos de estilos arquitectónicos
        │       │   ├── style.py           # Nodo base + factory make_style_qa_node()
        │       │   ├── common.py          # Implementación compartida
        │       │   ├── latency_style.py
        │       │   ├── scalability_style.py
        │       │   └── availability_style.py
        │       │
        │       └── tactics/               # Nodos de tácticas arquitectónicas
        │           ├── tactics.py         # Nodo base + factory make_tactics_qa_node()
        │           ├── common.py          # Implementación compartida (core)
        │           ├── latency_tactics.py
        │           ├── scalability_tactics.py
        │           └── availability_tactics.py  # Con catálogo restringido
        │
        ├── services/                      # Servicios core
        │   ├── llm_factory.py             # Factory multi-proveedor (Azure/OpenAI/Ollama)
        │   ├── doc_ingest.py              # Extracción de texto PDF
        │   ├── diagram_ir.py              # Representación Intermedia de diagramas
        │   ├── diagram_render.py          # DOT → SVG / draw.io rendering
        │   └── diagram_export.py          # Exportación del grafo del workflow
        │
        └── utils/
            └── json_helpers.py            # Extracción y reparación de JSON
```

---

## 14. Variables de Entorno

Todas las variables se leen desde `back/.env` (cargado automáticamente al iniciar).

### 14.1 LLM — OpenAI (Público)

| Variable | Requerida | Descripción |
| -------- | --------- | ----------- |
| `OPENAI_API_KEY` | Sí (si usa OpenAI) | API Key de OpenAI |
| `OPENAI_BASE_URL` | No | URL base para servidores compatibles (ej: proxies) |
| `OPENAI_MODEL` | No | Modelo a usar (default: `gpt-5-mini`) |
| `OPENAI_EMBED_MODEL` | No | Modelo de embeddings (default: `text-embedding-3-small`) |

### 14.2 LLM — Azure OpenAI

| Variable | Requerida | Descripción |
| -------- | --------- | ----------- |
| `AZURE_OPENAI_API_KEY` | Sí (si usa Azure) | API Key de Azure |
| `AZURE_OPENAI_ENDPOINT` | Sí (si usa Azure) | Endpoint del recurso Azure |
| `AZURE_OPENAI_API_VERSION` | No | Versión de API (default: `2024-12-01-preview`) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | No | Deployment de chat por defecto |
| `AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI` | No | Deployment específico para gpt-4o-mini |
| `AZURE_OPENAI_DEPLOYMENT_GPT4O` | No | Deployment específico para gpt-4o |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` | Sí (si usa Azure) | Deployment de embeddings |

### 14.3 LLM — Ollama (Local)

| Variable | Requerida | Descripción |
| -------- | --------- | ----------- |
| `OLLAMA_BASE_URL` | No | URL del servidor Ollama (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | No | Modelo a usar (default: `llama3.2:3b`) |

### 14.4 Proveedor General

| Variable | Requerida | Descripción |
| -------- | --------- | ----------- |
| `ROS_LG_LLM_PROVIDER` | No | Fuerza proveedor: `azure` \| `openai` \| `ollama` |
| `ROS_LG_LLM_MODEL` | No | Fuerza modelo específico |

### 14.5 RAG / ChromaDB

| Variable | Requerida | Descripción |
| -------- | --------- | ----------- |
| `CHROMA_DIR` | No | Directorio del vectorstore (default: `back/chroma_db/`) |

### 14.6 Sistema

| Variable | Requerida | Descripción |
| -------- | --------- | ----------- |
| `GRAPHVIZ_ENGINE` | No | Motor de Graphviz (default: `dot`) |
| `LOG_LEVEL` | No | Nivel de logging (default: `INFO`) |

### 14.7 Ejemplo de `.env` Mínimo (OpenAI)

```env
OPENAI_API_KEY=sk-proj-...
```

### 14.8 Ejemplo de `.env` para Azure

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://mi-recurso.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini-deployment
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small-deployment
```

---

## 15. Changelog v3 → v4

### Cambios de Arquitectura

| Área | Cambio |
| ---- | ------ |
| **Checkpointer** | Revertido de `SqliteSaver` a `MemorySaver`. El estado del grafo LangGraph vive en RAM; la persistencia de largo plazo usa `memory.py`. |
| **Retorno de nodos** | Todos los nodos (incluyendo ASR, style, tactics) retornan al Supervisor, no al Unifier directamente. Esto habilita el scheduler multi-intent. |
| **Routing de diagramas** | El Supervisor detecta un SVG listo (`diagram.ok == True`) y redirige al Unifier directamente, sin re-procesar. |

### Nuevas Funcionalidades

| Feature | Descripción |
| ------- | ----------- |
| **RAG Tracing** | `rag_trace_record()` registra queries, hits y fuentes por sesión para debugging y auditoría. |
| **Azure Embeddings** | `rag_agent.py` detecta automáticamente si está configurado Azure y usa `AzureOpenAIEmbeddings`. |
| **UTF-8 Fix global** | `fix_utf8_recursive()` aplicado a todas las respuestas del `/message`. Middleware UTF-8 en todas las responses JSON. |
| **Tactics Schema v2** | Nuevo schema con campos: `purpose`, `risks`, `tradeoffs`, `traces_to_asr`, `expected_effect`. |
| **DOT Sanitizer mejorado** | `_sanitize_dot()` ahora preserva caracteres UTF-8 al extraer DOT del LLM. |
| **Guard anti-unifier vacío** | El Supervisor redirige a `investigator` si se intenta ir al Unifier sin haber visitado ningún nodo. |

### Correcciones (PRs recientes)

| PR | Corrección |
| -- | ---------- |
| #40 | Fix: `rag_trace_record` faltante en `resources.py` |
| #38 | Fix: encoding UTF-8 en constantes y comentarios de nodos |
