# Audit de Tiempos de Respuesta — ArchIA Backend (v2)

**Baseline medido:** 154 s (prompt TravelHub completo: ASR + estilos + tácticas + diagrama)
**Fecha:** 2026-04-12
**Estado de la iteración anterior:** P1, P3 y P4 implementados.

---

## Estado de implementaciones anteriores

| ID | Descripción | Estado | Archivo |
|----|-------------|--------|---------|
| P1 | style + tactics en paralelo (ThreadPoolExecutor) | ✅ Implementado | `nodes/style_tactics_parallel.py` |
| P3 | RAG queries paralelas dentro de cada nodo | ✅ Implementado | `nodes/styles/common.py:32-50`, `nodes/tactics/common.py:44-58` |
| P4 | `@lru_cache(maxsize=64)` en `_fetch_styles_rag` y `_fetch_tactics_rag` | ✅ Implementado | ambos `common.py` |
| P2 | Streaming SSE al cliente | ❌ Pendiente | `main.py:590` |
| P5 | Graphviz async + cache DOT→SVG | ❌ Pendiente | `services/diagram_render.py:200` |
| P6 | Múltiples workers uvicorn | ❌ Pendiente | `pyproject.toml` / arranque |
| P7 | Truncación de `memory_text` y `messages` | ❌ Pendiente | `main.py:525-533` |

---

## Flujo real del pipeline TravelHub (post-P1)

```text
boot → classifier → supervisor(1)
     → asr → supervisor(2)
     → style_tactics_parallel  ← style + tactics corren en paralelo aquí
     → supervisor(3)
     → diagram_agent → supervisor(4)
     → unifier → END
```

**Nota importante descubierta en code review:** el `unifier_node` NO llama al LLM en el flujo multi-intent (`len(requested_nodes) >= 2`). Para el prompt TravelHub, el unifier hace concatenación directa de bloques sin coste LLM (`unifier.py:70-128`). El LLM call del unifier (`unifier.py:327`) solo se activa en el path `investigator`/`evaluator`.

### Estimación del coste real por nodo (pipeline TravelHub)

| Nodo | Tipo de coste | Estimado |
|------|--------------|----------|
| classifier | LLM call (temp=0.0) | 2–5 s |
| asr | RAG + LLM call | 10–20 s |
| style_tactics_parallel | max(style LLM, tactics LLM) en paralelo | 15–25 s |
| diagram_agent | LLM DOT generation + subprocess.run(dot) | 20–40 s |
| unifier | Concatenación directa (sin LLM) | < 0.5 s |
| supervisor × 4 | Heurística (sin LLM en fast path) | < 1 s total |
| **Total estimado** | | **47–91 s** |

La diferencia entre el estimado (~47-91 s) y el medido (154 s) indica que los LLM calls están tomando el extremo alto del rango, probablemente por latencia del modelo en el prompt de TácticasTravelHub (~800 tokens de prompt más respuesta larga).

---

## Nuevos Bottlenecks Identificados

### N1 — P2: Sin streaming al cliente (mayor impacto en UX, ya documentado)

**Donde:** `main.py:590`

```python
result = graph.invoke({...}, config)  # bloquea hasta END — 154 s sin feedback visual
```

El usuario no recibe nada hasta que el pipeline completo termina. Con `graph.astream_events()` + `StreamingResponse` de FastAPI, la primera respuesta visible (ASR) llegaría al cliente en ~12-20 s. Los resultados de style+tactics llegarían en ~30-45 s. El diagrama en cuanto termine Graphviz.

**Impacto:** No reduce el tiempo total de cómputo, pero transforma la latencia *percibida* de 154 s a ~12-20 s para el primer chunk.

---

### N2 — P5: Graphviz sincrónico bloquea el event loop (no implementado)

**Donde:** `services/diagram_render.py:200`

```python
result = subprocess.run(
    [engine, "-Tsvg"],
    input=dot_string,
    capture_output=True,
    text=True,
)
```

`subprocess.run()` es bloqueante. En el contexto `async def message()` de FastAPI, esto satura el thread pool durante 5-15 s para el diagrama TravelHub (15+ nodos PMS, arquitectura compleja). Además no hay cache: el mismo DOT string se re-renderiza en cada request.

**Dos mejoras independientes:**

1. **Cache DOT→SVG por hash:** `@lru_cache` sobre `hash(dot_string)` — render instantáneo en requests repetidos o reload del mismo diagrama.
2. **`asyncio.create_subprocess_exec`:** libera el event loop durante el render, permitiendo que otras requests se procesen en paralelo.

**Impacto:** −5–15 s de tiempo real + elimina el bloqueo de concurrencia.

---

### N3 — diagram_agent podría correr en paralelo con style_tactics_parallel

**Donde:** `workflow.py:119-125` y `supervisor.py:186-196`

El `diagram_agent` solo requiere el ASR (`current_asr` / `last_asr`) para generar el DOT. No depende del estilo ni de las tácticas recién generadas. Sin embargo, el grafo actual lo ejecuta secuencialmente *después* de `style_tactics_parallel`.

Actualmente la cadena es:

```
asr(~15s) → style_tactics_parallel(~20s) → diagram_agent(~30s) = 65s secuenciales
```

Si `diagram_agent` corriera en paralelo con `style_tactics_parallel`:

```
asr(~15s) → parallel(style_tactics ~20s, diagram_agent ~30s) = 15s + 30s = 45s
```

Ahorro estimado: **~20 s** (el tiempo de `style_tactics_parallel`, que quedaría solapado con el render del diagrama).

**Implementación:** extender `style_tactics_parallel_node` para que también despache `diagram_agent` en un tercer hilo del `ThreadPoolExecutor`, o crear un nodo `full_parallel_node` que ejecute los tres concurrentemente. El merge del estado requiere manejar el campo `diagram` (dict) además de los campos de style/tactics existentes.

**Impacto:** −15–20 s de tiempo real.

---

### N4 — classifier no tiene cache (LLM determinista a temperature=0.0)

**Donde:** `nodes/classifier.py` (invocado en cada request)

El classifier usa `temperature=0.0`, por lo que su salida es determinista para el mismo input. Para el prompt TravelHub, siempre clasificará de la misma manera (`intent=asr`, `quality_attribute=escalabilidad`, `language=es`). Sin embargo se re-ejecuta en cada request sin ningún tipo de cache.

**Implementación:** `@lru_cache` o `functools.lru_cache` sobre `hash(message_text)` en la función interna del classifier, con `maxsize=256` y TTL opcional (usando `cachetools.TTLCache` si se quiere expiración).

**Impacto:** −2–5 s en el primer nodo del pipeline.

---

### N5 — deepcopy costoso del estado completo en style_tactics_parallel

**Donde:** `nodes/style_tactics_parallel.py:185-186`

```python
state_for_style   = copy.deepcopy(dict(state))
state_for_tactics = copy.deepcopy(dict(state))
```

Si el estado incluye `add_context` (texto de PDF, potencialmente miles de caracteres), `messages` acumulado en la sesión, y `memory_text` largo, `copy.deepcopy` puede tardar 0.5–2 s por llamada (2 llamadas = hasta 4 s extra).

**Implementación:** copia superficial del estado + deepcopy selectivo solo de los campos mutables que cada sub-nodo escribe. Style escribe: `style`, `selected_style`, `last_style`, `memory_text`, `suggestions`, `turn_messages`. Tactics escribe: `tactics_md`, `tactics_struct`, `tactics_list`, `arch_stage`, `quality_attribute`, `current_asr`, `endMessage`, `intent`, `messages`, `turn_messages`. Los campos de solo lectura (`add_context`, `doc_context`, `userQuestion`, `resolved_index`, etc.) pueden compartirse sin copia.

**Impacto:** −0.5–4 s según el tamaño del estado (mayor impacto en sesiones con PDF subido).

---

### N6 — Estructura del prompt no maximiza el prompt caching de OpenAI

**Donde:** `nodes/tactics/common.py:225-269`

El prompt de tácticas tiene el siguiente orden:

```
{directive}              ← variable (lang)
"You are an expert..."  ← ESTÁTICO
PROJECT CONTEXT:        ← variable (ctx)
ASR:                    ← variable (asr_text)
quality attribute:      ← variable (qa)
Selected style:         ← variable (style_text)
GROUNDING:              ← semi-estático (RAG snippets, varía por QA)
INSTRUCCIONES FIJAS     ← ESTÁTICO (>600 tokens)
```

OpenAI aplica prompt caching automático para prefijos estáticos de >1024 tokens. Con el orden actual, el contenido variable (`ctx`, `asr_text`, `qa`, `style_text`) aparece antes del bloque de instrucciones estático, rompiendo el prefijo cacheado.

**Implementación:** reorganizar el prompt poniendo todo el contenido estático primero y todo el contenido variable (ASR, qa, style, ctx) al final:

```
"You are an expert..."  ← ESTÁTICO  ┐
INSTRUCCIONES FIJAS                   ├── prefijo cacheado >1024 tokens
GROUNDING (RAG)                       ┘
PROJECT CONTEXT:        ← variable   ┐
ASR:                                  ├── al final, no rompen el prefijo
Selected style:                       ┘
```

Aplica igual a `styles/common.py` y `nodes/asr.py`.

**Impacto:** −20–50% de coste LLM y latencia en queries repetidas (misma estructura de QA).

---

### N7 — P7: memory_text crece sin límite (ya documentado, no implementado)

**Donde:** `main.py:525-533`

El campo `memory_text` acumula `add_context` (PDF completo sin truncar) + ASR anterior + tácticas anteriores. Se inyecta en el prompt de tácticas con un cap de `[:2000]` dentro del nodo, pero el campo en sí crece indefinidamente en la sesión. En sesiones largas con PDF, puede superar 10,000 caracteres, incrementando el coste de serialización del checkpointer (`MemorySaver`) en cada nodo.

**Implementación:** truncar `memory_text` antes de guardarlo en la sesión (e.g., `memory_text = memory_text[-3000:]`). No afecta la calidad: los nodos ya aplican su propio cap interno de 2000 chars al construir el prompt.

**Impacto:** previene degradación progresiva en sesiones largas (empeoraría ~5–15% por turno sin truncación).

---

### N8 — P6: Un solo worker uvicorn (ya documentado, no implementado)

**Donde:** configuración de arranque

Con un solo worker, cualquier segunda petición durante los 154 s del prompt TravelHub queda bloqueada completamente. `graph.invoke()` es síncrono y ocupa el único thread disponible.

**Implementación:** `uvicorn src.main:app --workers 4` o `gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.main:app`. Alternativamente, convertir `graph.invoke` a `await graph.ainvoke` (requiere que LangGraph compile el grafo como async).

**Impacto:** no mejora latencia individual, pero elimina el bloqueo total en concurrencia.

---

## Resumen de impacto estimado (nueva iteración)

| ID | Bottleneck | Impacto en latencia | Dificultad |
|----|-----------|---------------------|------------|
| N1 | Sin streaming SSE (P2) | UX: 154s → ~12s para primer chunk | Media |
| N2 | Graphviz sincrónico + sin cache (P5) | −5–15 s reales | Baja |
| N3 | diagram_agent secuencial (podría ser paralelo con style_tactics) | −15–20 s reales | Media |
| N4 | Classifier sin cache (determinista) | −2–5 s | Baja |
| N5 | deepcopy completo del estado | −0.5–4 s | Baja |
| N6 | Prompt structure rompe OpenAI cache prefix | −20–50% en queries repetidas | Baja |
| N7 | memory_text sin truncación (P7) | Previene degradación progresiva | Muy baja |
| N8 | Un solo worker (P6) | Elimina bloqueo en concurrencia | Muy baja |

---

## Quick wins recomendados por orden de ROI

1. **N4 — Classifier cache:** `@lru_cache` sobre `hash(message)` en el nodo classifier. Esfuerzo mínimo, −2-5 s inmediatos en cada request.

2. **N5 — deepcopy selectivo en style_tactics_parallel:** reemplazar `copy.deepcopy(dict(state))` por una copia superficial + deepcopy solo de los campos que cada hilo escribe. Bajo esfuerzo, impacto variable según estado.

3. **N6 — Reestructurar prompts para OpenAI cache prefix:** mover todo el contenido variable (ASR, qa, style, ctx) al final del prompt en `tactics/common.py`, `styles/common.py` y `asr.py`. Bajo esfuerzo, alto impacto acumulado en uso real repetido.

4. **N2 — Graphviz async + cache DOT→SVG:** reemplazar `subprocess.run` por `asyncio.create_subprocess_exec` + `@lru_cache` sobre `hash(dot_string)`. Bajo esfuerzo, −5-15 s reales.

5. **N3 — Paralelizar diagram_agent con style_tactics_parallel:** extender el `ThreadPoolExecutor` en `style_tactics_parallel_node` para despachar diagram_agent como tercer hilo. Esfuerzo medio, −15-20 s reales (el mayor ahorro de tiempo de cómputo disponible).

6. **N1 — Streaming SSE (P2):** cambiar `graph.invoke` → `graph.astream_events` + `StreamingResponse`. Mayor impacto percibido de todo el plan. Esfuerzo medio-alto por coordinación con el frontend.

7. **N8 — Múltiples workers:** `uvicorn --workers 4`. Esfuerzo mínimo, habilita concurrencia real.

8. **N7 — Truncar memory_text:** una línea de código, previene degradación en sesiones largas.
