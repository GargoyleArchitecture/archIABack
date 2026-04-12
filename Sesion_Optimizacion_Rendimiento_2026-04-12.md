# Sesión de Optimización de Rendimiento — ArchIA Backend
**Fecha:** 12 de abril de 2026  
**Rama:** `fix/ResponseTimesAudit`  
**Autor:** Manuel Gómez  
**Período:** 15:36 – 18:07 (≈ 2.5 horas)

---

## Contexto y motivación

El backend de ArchIA presentaba tiempos de respuesta extremadamente altos para flujos completos de diseño arquitectónico (ASR → Estilos → Tácticas → Diagrama). Según el análisis previo documentado en `Audit_Tiempos_Respuesta_ArchIA_Backend.md`, un prompt típico como el de TravelHub ejecutaba **8 nodos**, realizaba **6–8 llamadas LLM bloqueantes** y entre **10–15 búsquedas RAG**, lo que resultaba en tiempos de respuesta de **45–80 segundos** solo en invocaciones al LLM.

Los tres cuellos de botella identificados fueron:

| Prioridad | Problema | Impacto estimado |
|-----------|----------|-----------------|
| P1 | Llamadas LLM secuenciales y bloqueantes | 60–70% de la latencia total |
| P2 | Sin streaming al cliente (el usuario espera sin feedback) | 20–40% de la latencia percibida |
| P3 | Búsquedas RAG vectoriales secuenciales dentro de cada nodo | 15–20% de la latencia |

Esta sesión abordó los tres problemas a través de cuatro commits incrementales.

---

## Resumen de commits

| Hash | Hora | Descripción |
|------|------|-------------|
| `b8cbed4` | 15:36 | Paralelización de queries RAG con `ThreadPoolExecutor` dentro de los nodos |
| `db9f37b` | 16:01 | Caché con `lru_cache` para el clasificador, estilos y tácticas + separación System/Human message |
| `61ce8bc` | 17:29 | Nuevo nodo `style_tactics_parallel` — ejecución concurrente de estilos y tácticas |
| `a8d7b4b` | 18:07 | Migración a Streaming SSE en `/message` + `render_svg_async` + caché de SVG |

---

## Detalle de cada cambio

### Commit 1 — `b8cbed4`: Paralelización de queries RAG con `ThreadPoolExecutor`

**Archivos modificados:**
- `back/src/graph/nodes/styles/common.py`
- `back/src/graph/nodes/tactics/common.py`

**Qué se hizo:**  
Antes de este cambio, los nodos de estilos y tácticas ejecutaban múltiples queries RAG al vectorstore de forma **secuencial** mediante un loop `for q in queries`. Cada query bloqueaba hasta completarse antes de lanzar la siguiente.

Se reemplazó el loop secuencial por un `ThreadPoolExecutor` que lanza todas las queries en paralelo:

```python
# ANTES (secuencial):
for q in queries:
    for d in _retriever.invoke(q):
        ...

# DESPUÉS (paralelo):
with ThreadPoolExecutor(max_workers=len(queries)) as executor:
    futures = {executor.submit(_retriever.invoke, q): q for q in queries}
    for future in as_completed(futures):
        for d in future.result():
            ...
```

**Impacto:** Las 4–6 búsquedas vectoriales que antes se ejecutaban de forma secuencial ahora se ejecutan concurrentemente. Reducción estimada de ~60–70% en el tiempo de recuperación RAG por nodo.

---

### Commit 2 — `db9f37b`: Caché LRU + separación System/Human message

**Archivos modificados:**
- `back/src/graph/nodes/classifier.py`
- `back/src/graph/nodes/styles/common.py`
- `back/src/graph/nodes/tactics/common.py`

#### 2a. Caché en el clasificador (`classifier.py`)

Se añadieron dos funciones privadas decoradas con `@lru_cache` que envuelven las operaciones más costosas del nodo clasificador:

```python
@lru_cache(maxsize=256)
def _classify_cached(msg: str, qa_opts_str: str) -> tuple:
    """Retorna (language, intent, use_rag, quality_attribute). Cacheado por (msg, qa_opts_str)."""
    ...

@lru_cache(maxsize=128)
def _resolve_qa_cached(msg: str) -> str:
    """Wrapper cacheado alrededor de resolve_quality_attribute (determinístico, temperature=0.0)."""
    ...
```

**Razonamiento:** El clasificador es invocado en cada turno. Cuando el usuario repite o hace preguntas similares, el modelo se invoca con el mismo texto de entrada, incurriendo en el mismo costo LLM. Como el clasificador es determinístico (temperatura baja), el caché es seguro y elimina llamadas redundantes al modelo.

#### 2b. Caché de recuperación RAG para estilos y tácticas

Se extrajo la lógica de búsqueda vectorial de los nodos `style_node_impl` y `tactics_node_impl` hacia funciones privadas cacheadas:

```python
@lru_cache(maxsize=64)
def _fetch_styles_rag(qa: str, resolved_index: str, k: int = 6) -> str:
    """Retorna book_snippets string. Cacheado por (qa, resolved_index, k)."""
    ...

@lru_cache(maxsize=64)
def _fetch_tactics_rag(qa: str, resolved_index: str, k: int = 6) -> tuple:
    """Retorna (book_snippets, src_meta). Cacheado por (qa, resolved_index, k)."""
    ...
```

**Razonamiento:** Los índices RAG son estáticos durante la sesión. Para un mismo atributo de calidad (`qa`) e índice resuelto (`resolved_index`), los documentos recuperados serán siempre los mismos. El caché elimina todas las queries a ChromaDB en turnos subsecuentes que trabajen sobre el mismo QA, un patrón muy frecuente en conversaciones de diseño arquitectónico.

#### 2c. Separación de System / Human message y extracción de prompts como constantes de módulo

Se refactorizaron los prompts de estilos y tácticas para adoptar la estructura estándar de la API de OpenAI / Anthropic con mensajes separados:

```python
# ANTES: un único string concatenado
prompt = f"""You are a software architect...\n{directive}\n{qa}\n..."""
result = llm.invoke(prompt)

# DESPUÉS: System message fijo + Human message variable
_STYLES_SYSTEM = (
    "You are a software architect applying ADD 3.0.\n\n"
    "..."  # instrucciones estáticas — candidatas a prompt caching
)

result = llm.invoke([
    SystemMessage(content=_STYLES_SYSTEM),
    HumanMessage(content=human_content),  # contexto dinámico del turno
])
```

**Razonamiento:** Al separar las instrucciones estáticas (rol, formato de salida, reglas) del contenido dinámico (ASR, contexto del proyecto, snippets RAG), se habilita el **prompt caching del proveedor LLM**. El `SystemMessage` con contenido idéntico entre llamadas puede ser cacheado a nivel de API, reduciendo costos y latencia en llamadas repetidas para el mismo tipo de nodo. Adicionalmente, tener los prompts como constantes de módulo (`_STYLES_SYSTEM`, `_TACTICS_SYSTEM`) mejora la mantenibilidad y facilita ajustes sin tocar la lógica de invocación.

---

### Commit 3 — `61ce8bc`: Nuevo nodo `style_tactics_parallel`

**Archivos creados:**
- `back/src/graph/nodes/style_tactics_parallel.py` *(nuevo)*

**Archivos modificados:**
- `back/src/graph/nodes/supervisor.py`
- `back/src/graph/workflow.py`
- `back/src/graph/state.py`
- `back/src/graph/resources.py`
- `back/src/graph/nodes/styles/common.py`
- `back/src/graph/nodes/tactics/common.py`
- `test_travelhub_timing.py` *(nuevo)*

**Qué se hizo:**  
Este fue el cambio más estructural de la sesión. Antes del cambio, el flujo de trabajo del grafo ejecutaba los nodos de estilos y tácticas de forma estrictamente secuencial:

```
asr → supervisor → style_scalability → supervisor → tactics_scalability → supervisor → ...
```

Cada nodo esperaba a que el anterior completara su llamada LLM (8–18 s cada uno) antes de comenzar. En total, estilo + tácticas sumaban entre **16–30 s** de latencia secuencial.

Se creó el nodo `style_tactics_parallel_node` que ejecuta ambos en paralelo usando `ThreadPoolExecutor(max_workers=2)`:

```python
def style_tactics_parallel_node(state: GraphState) -> GraphState:
    """Ejecuta style y tactics concurrentemente y combina sus resultados."""
    ...
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="st_par") as pool:
        future_style  = pool.submit(style_fn, state_for_style)
        future_tactics = pool.submit(tactics_fn, state_for_tactics)

        for future in as_completed([future_style, future_tactics]):
            ...

    return _merge_states(input_state, style_result, tactics_result, ...)
```

**Detalles de implementación importantes:**

1. **Deep copy del estado por hilo:** Cada hilo recibe una copia profunda del estado (`copy.deepcopy(dict(state))`) para evitar condiciones de carrera en mutaciones directas al diccionario.

2. **Builders QA-aware:** Las funciones `_build_style_fn(qa)` y `_build_tactics_fn(qa)` encapsulan la lógica de selección por atributo de calidad, incluyendo la configuración de `preferred_tactics` para QAs específicos como disponibilidad.

3. **Merge de estado con reglas explícitas:** La función `_merge_states()` define reglas claras de qué campos provienen de cada nodo:
   - `style / selected_style / last_style / suggestions` → resultado de `style`
   - `tactics_md / tactics_struct / tactics_list` → resultado de `tactics`
   - `turn_messages` → concatenación de ambos deltas sobre el estado de entrada
   - `arch_stage / intent / endMessage` → resultado de `tactics` (o de `style` si tactics falló)

4. **Resiliencia ante fallos parciales:** Si un nodo falla, el resultado del otro se preserva y se registra un `WARNING`. Solo si ambos fallan se re-lanza la excepción.

5. **Trade-off documentado:** En el mismo turno donde se generan estilo y tácticas en paralelo, el nodo de tácticas no puede ver el estilo recién calculado (no existe aún). Usa `last_style` del turno anterior. El prompt de tácticas ya contempla este escenario con "Selected architecture style (if any): ...".

**Reducción de latencia estimada:** Se eliminan ~8–18 s de llamadas LLM secuenciales, reduciendo el segmento estilo+tácticas de ~16–30 s a ~8–18 s (el tiempo del más lento de los dos).

---

### Commit 4 — `a8d7b4b`: Streaming SSE en `/message` + mejoras en Graphviz

**Archivos modificados:**
- `back/src/main.py`
- `back/src/services/diagram_render.py`

**Archivos añadidos al repositorio:**
- `Audit_Tiempos_Respuesta_ArchIA_Backend.md`
- `Audit_Tiempos_Respuesta_ArchIA2.md`

#### 4a. Migración a Streaming SSE en `/message` (`main.py`)

Este fue el cambio más visible para el frontend. El endpoint `/message` pasó de ser un endpoint bloqueante que retornaba un `JSONResponse` único al finalizar el grafo completo, a ser un endpoint de **Server-Sent Events (SSE)** que envía chunks de progreso a medida que cada nodo completa su ejecución.

**Antes:**
```python
@app.post("/message")
async def message(...):
    ...
    result = graph.invoke({...}, config)   # bloqueante, espera END
    ...
    return JSONResponse(content=clean_payload, ...)
```

**Después:**
```python
@app.post("/message")
async def message(...):
    ...
    async def generate():
        async for chunk in graph.astream(input_state, config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if node_name == "asr":
                    yield _sse({"type": "partial", "node": "asr", "endMessage": ...})
                elif node_name == "style_tactics_parallel":
                    yield _sse({"type": "partial", "node": "style_tactics", ...})
                elif node_name == "diagram_agent":
                    yield _sse({"type": "partial", "node": "diagram", ...})
                elif node_name == "unifier":
                    yield _sse({"type": "complete", ...})
                    yield "data: [DONE]\n\n"
        ...
        _stream_post_process(result=_final, ...)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**Protocolo de eventos SSE definido:**

| `type` | Cuándo se emite | Campos principales |
|--------|-----------------|-------------------|
| `partial` (`node: "asr"`) | Al terminar el nodo ASR | `endMessage` |
| `partial` (`node: "style_tactics"`) | Al terminar el nodo paralelo | `style`, `tactics_md`, `endMessage` |
| `partial` (`node: "diagram"`) | Al terminar el agente de diagrama | `diagram` |
| `complete` | Al terminar el nodo `unifier` | `endMessage`, `diagram`, `messages`, `session_id`, `message_id`, `thread_id`, `suggestions` |
| `error` | Si hay excepción en el pipeline | `message` |

El stream siempre termina con el sentinel `data: [DONE]\n\n`.

**Extracción de `_stream_post_process()`:**  
Toda la lógica de post-procesamiento que actualizaba memoria, `arch_flow` y persistía en base de datos se extrajo a la función privada `_stream_post_process()`. Esto separa claramente la generación del stream de los efectos secundarios, ejecutándose después del cierre del stream (cuando `_final` contiene el estado completo del nodo `unifier`).

**Helper `_sse(data: dict) -> str`:**  
Función auxiliar mínima para serializar cualquier dict al formato SSE:
```python
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
```

**Elección de `stream_mode="updates"`:**  
Se eligió `graph.astream(..., stream_mode="updates")` en lugar de `astream_events(version="v2")`. Con `stream_mode="updates"`, LangGraph emite un evento por nodo (solo el dict de retorno de ese nodo), evitando la sobrecarga de serialización del estado completo en cada token LLM que ocurre con `astream_events`.

#### 4b. `render_svg_async` y caché de SVG (`diagram_render.py`)

Se realizaron dos mejoras en el módulo de renderizado de diagramas:

**1. Caché de SVG renderizados:**

```python
# Caché compartido entre render_svg() y render_svg_async()
_SVG_CACHE: dict[int, bytes] = {}
```

Tanto `render_svg()` como la nueva `render_svg_async()` consultan y pueblan este caché usando `hash((dot_string, engine))` como clave. El render de Graphviz es determinístico: el mismo DOT con el mismo motor siempre produce el mismo SVG, por lo que el caché es completamente seguro.

**2. Nueva función `render_svg_async()`:**

```python
async def render_svg_async(dot_string: str, engine: str = "dot") -> bytes:
    """Render DOT→SVG no bloqueante usando asyncio subprocess."""
    key = hash((dot_string, engine))
    if key in _SVG_CACHE:
        return _SVG_CACHE[key]

    proc = await asyncio.create_subprocess_exec(
        engine, "-Tsvg",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=...),
        timeout=30,
    )
    ...
    # Fallback al path síncrono si el proceso async falla
    return render_svg(dot_string, engine=engine)
```

Esta función reemplaza la llamada `subprocess.run()` bloqueante (que en un contexto `async` bloquea el event loop de FastAPI) por `asyncio.create_subprocess_exec()`, que es completamente no bloqueante. El endpoint `diagram_export` fue actualizado para usar esta versión:

```python
# ANTES:
svg_bytes = render_svg_from_dot(dot_str, engine=engine)

# DESPUÉS (endpoint es ahora async):
svg_bytes = await render_svg_async_from_dot(dot_str, engine=engine)
```

El endpoint `diagram_export` también fue cambiado de `def` a `async def` para poder usar `await`.

---

## Flujo de datos post-optimización

El flujo completo de un mensaje con intent `style+tactics` (el más costoso) queda así:

```
Cliente (SSE)
    │
    ├── [evento partial: asr]        ← ~10-15 s (primer feedback visible)
    │
    ├── [evento partial: style_tactics]  ← +8-18 s (paralelo, no secuencial)
    │
    ├── [evento partial: diagram]    ← +10-20 s
    │
    └── [evento complete]            ← payload final + diagrama completo
        └── data: [DONE]
```

Antes del streaming, el cliente no recibía nada hasta el evento `complete`.

---

## Archivos relevantes modificados

| Archivo | Tipo de cambio |
|---------|---------------|
| `back/src/graph/nodes/classifier.py` | `lru_cache` en clasificación y resolución de QA |
| `back/src/graph/nodes/styles/common.py` | `lru_cache` en RAG, `ThreadPoolExecutor`, separación System/Human message |
| `back/src/graph/nodes/tactics/common.py` | `lru_cache` en RAG, `ThreadPoolExecutor`, separación System/Human message |
| `back/src/graph/nodes/style_tactics_parallel.py` | **Nuevo:** nodo de ejecución paralela style+tactics |
| `back/src/graph/nodes/supervisor.py` | Integración del nuevo nodo paralelo en el router |
| `back/src/graph/workflow.py` | Registro del nuevo nodo en el grafo LangGraph |
| `back/src/graph/state.py` | Ajustes de estado para el nodo paralelo |
| `back/src/graph/resources.py` | Ajustes de recursos compartidos |
| `back/src/main.py` | Migración completa a SSE streaming, extracción de `_stream_post_process` |
| `back/src/services/diagram_render.py` | `render_svg_async`, `_SVG_CACHE`, `diagram_export` como `async def` |

---

## Consideraciones y deuda técnica

1. **`_SVG_CACHE` en memoria:** El caché de SVG es un dict en memoria de proceso. No persiste entre reinicios del servidor. Para entornos multi-worker (Gunicorn/Uvicorn con múltiples procesos), cada worker tendrá su propio caché independiente. No es un problema funcional, pero el calentamiento del caché ocurre por separado en cada worker.

2. **`lru_cache` y la longitud del mensaje:** Los cachés de `lru_cache` usan el mensaje completo del usuario como clave. Mensajes muy largos (e.g., contextos de documentos) generarán muchas claves distintas. El `maxsize` configurado (64–256 entradas) es conservador y debería funcionar bien en producción.

3. **Trade-off de paralelismo style/tactics:** En el primer turno donde se solicita estilo + tácticas simultáneamente, las tácticas se generan sin ver el estilo del turno actual (solo `last_style` del turno anterior). Esto es un trade-off explícitamente documentado en `style_tactics_parallel.py`. En conversaciones de múltiples turnos, el impacto es mínimo porque el estilo del turno anterior suele ser relevante.

4. **Sincronización del nuevo nodo con el supervisor:** Si se añaden nuevos QAs con `preferred_tactics` en el futuro, el builder `_build_tactics_fn()` en `style_tactics_parallel.py` debe actualizarse para reflejar esa lógica (hay un comentario `NOTE` al respecto en el código).

5. **Compatibilidad del frontend:** El endpoint `/message` ahora retorna `text/event-stream` en lugar de `application/json`. El frontend debe estar adaptado para consumir SSE. El evento `complete` contiene exactamente los mismos campos que el `JSONResponse` anterior, por lo que la migración del cliente debería ser aditiva (agregar manejo de `partial` es opcional).
