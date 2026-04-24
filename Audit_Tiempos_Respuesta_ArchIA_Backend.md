# Audit de Tiempos de Respuesta — ArchIA Backend

## Flujo que ejecuta el prompt de TravelHub

El prompt solicita ASR + estilos + tacticas + diagrama en una sola respuesta. Eso activa el scheduler multi-intent del supervisor (supervisor.py:165-193), creando la cadena completa:

```text
boot -> classifier -> supervisor -> asr -> supervisor
     -> style_scalability -> supervisor
     -> tactics_scalability -> supervisor
     -> diagram_agent -> supervisor
     -> unifier -> END
```

8 nodos ejecutados, 6-8 llamadas LLM bloqueantes, 10-15 busquedas RAG, 1 render Graphviz.

## Bottlenecks por Prioridad

### P1 — LLM calls secuenciales y bloqueantes (~60-70% de la latencia total)

**Donde:** todas las invocaciones con `.invoke()` en cada nodo.

Cada nodo llama al LLM de forma sincrona y espera la respuesta completa antes de continuar. Para el prompt de TravelHub, el pipeline ejecuta al menos:

| Nodo | Archivo | Costo estimado |
| --- | --- | --- |
| classifier | classifier.py | ~2-4 s |
| asr_node | asr.py:127 | ~8-15 s |
| style_scalability | styles/common.py:132 | ~8-12 s |
| tactics_scalability | tactics/common.py | ~10-18 s (prompt ~800 tokens) |
| diagram_orchestrator | diagram.py:136 | ~10-20 s |
| unifier | unifier.py | ~5-10 s |

Costo total estimado: 45-80 segundos solo en llamadas LLM.

El problema raiz es que asr -> style -> tactics son secuenciales por diseno en el supervisor (supervisor.py:182-196), cuando style y tactics podrian ejecutarse en paralelo una vez que el ASR esta listo. El grafo actual no tiene nodos en paralelo.

### P2 — Sin streaming al cliente (~20-40% de latencia percibida)

**Donde:** main.py:590 — `graph.invoke(...)` espera el resultado completo antes de responder.

```python
result = graph.invoke({...}, config)  # bloquea hasta END
```

El usuario espera sin ver nada hasta que todos los nodos terminan. Con streaming, la primera respuesta visible podria llegar en ~10-15 s (cuando asr_node termina) en lugar de 60-80 s.

LangGraph soporta `graph.astream_events()` que permite enviar chunks por SSE/streaming HTTP. FastAPI tambien soporta `StreamingResponse`. Ninguno esta implementado.

### P3 — RAG con busquedas vectoriales secuenciales (~15-20% de la latencia)

**Donde:** nodos asr, style y tactics hacen multiples queries al vectorstore en loop.

Ejemplo en tactics (tactics/common.py):

```python
for q in queries:          # 4+ queries secuenciales
    for d in _retriever.invoke(q):
        ...
```

Para el prompt de TravelHub (scalability), se ejecutan ~12-16 busquedas vectoriales en ChromaDB de forma completamente secuencial. No hay `asyncio.gather`, no hay batch queries, ni cache de resultados. Cada busqueda tarda ~200-800 ms dependiendo del tamano del indice.

### P4 — Sin cache de ningun tipo (~40-60% de computo redundante)

**Donde:** todo el sistema. Solo existe `@lru_cache(maxsize=1)` en qa_registry.py:50.

Para el mismo prompt (o prompts muy similares sobre TravelHub/escalabilidad), el sistema re-ejecuta integramente:

- Embeddings de la query (ChromaDB recalcula)
- Clasificacion del mensaje (classifier LLM call)
- Generacion del ASR de escalabilidad
- Seleccion de estilos para escalabilidad
- Tacticas para escalabilidad
- Generacion DOT + render SVG

El classifier y supervisor son deterministas (temperature=0.0): sus salidas para el mismo input son identicas. Un cache LRU sobre el hash del mensaje para classifier, o un prompt cache sobre los system prompts de tactics/styles, reduciria significativamente el costo en preguntas frecuentes.

OpenAI soporta Prompt Caching automatico para prompts > 1024 tokens. Los prompts de tactics y diagram superan ese umbral pero no estan estructurados para maximizar el prefijo cacheado (el contexto variable se inyecta en el medio del prompt en lugar del final).

### P5 — Render Graphviz como proceso externo bloqueante (~5-15 s)

**Donde:** diagram.py:220+

El render SVG invoca el binario `dot` de Graphviz como subproceso. Para diagramas complejos (nivel 2-3, con 15+ proveedores PMS como en el prompt de TravelHub), esto puede tomar 5-15 segundos. No hay cache del DOT->SVG, no se usa `asyncio.create_subprocess_exec`, y el proceso bloquea el event loop de FastAPI (siendo llamado desde un contexto async).

### P6 — Un solo worker uvicorn (~impacto en concurrencia, no en latencia individual)

**Donde:** pyproject.toml y configuracion de arranque.

Con un solo worker uvicorn, una segunda peticion durante los ~60-80 s de respuesta de TravelHub queda bloqueada completamente. El `graph.invoke()` es sincrono y ocupa el thread pool de FastAPI. No hay `await` real en la ruta critica: solo el wrapper `async def message()` con una llamada sincrona dentro.

### P7 — Crecimiento ilimitado del estado y prompts (~5-10%, empeora progresivamente)

**Donde:** main.py:525-533 y acumulacion en state.py

El campo `memory_text` se construye incluyendo `add_context` (todo el PDF o contexto previo sin limite real), `current_asr`, tacticas, etc. Ese bloque se inyecta en cada prompt subsiguiente. En sesiones largas o cuando el usuario sube un PDF, el prompt de tactics puede superar 2000-3000 tokens extras, aumentando el tiempo de generacion linealmente.

No hay truncacion del historial de mensajes en el estado del grafo: `messages` crece turno a turno, incrementando el costo de serializacion del checkpointer (resources.py:113 — MemorySaver).

## Resumen de impacto estimado

| Prioridad | Bottleneck | Impacto en latencia | Dificultad |
| --- | --- | --- | --- |
| P1 | LLM calls secuenciales (style + tactics en serie) | -20-30 s con paralelizacion | Media |
| P2 | Sin streaming al cliente | -40-60 s de latencia percibida | Baja |
| P3 | RAG queries secuenciales | -5-10 s | Baja |
| P4 | Sin cache (LLM ni prompt caching) | -20-50% en queries repetidas | Media |
| P5 | Graphviz bloqueante sin cache | -5-15 s en diagramas complejos | Baja |
| P6 | Un solo worker | Bloqueo total en concurrencia | Muy baja |
| P7 | State/prompt crecimiento ilimitado | Empeora progresivamente | Baja |

## Quick wins recomendados por orden de ROI

1. Streaming SSE: cambiar `graph.invoke` -> `graph.astream_events` + `StreamingResponse`. Maximo impacto en latencia percibida, bajo esfuerzo.
2. Paralelizar style + tactics: ejecutarlos en un nodo `Send()` paralelo de LangGraph una vez que el ASR esta confirmado.
3. `asyncio.gather` para RAG queries: las 4+ queries por nodo se pueden disparar en paralelo.
4. Prompt caching: mover el contexto variable al final del prompt para maximizar el prefijo cacheado en OpenAI/Anthropic.
5. Graphviz async: usar `asyncio.create_subprocess_exec` + cache DOT->SVG por hash.
6. Multiples workers: `uvicorn --workers 4` o `gunicorn -w 4 -k uvicorn.workers.UvicornWorker`.