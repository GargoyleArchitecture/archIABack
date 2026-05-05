# LangGraph Store API - Memoria Cross-Thread

- Implementacion actual: `langgraph.store.memory.InMemoryStore`.
- Ciclo de vida: instancia unica por proceso, creada en el `lifespan` de FastAPI
  (ver `src/main.py`) y publicada via `set_store()` en `src/graph/resources.py`.
- Persistencia: ephemeral. Los datos se pierden al reiniciar el proceso.
  Cuando LangGraph publique un backend SQLite oficial (o se adopte
  `langgraph-checkpoint-postgres` tambien para Store), reemplazar
  `make_inmemory_store()` en `src/graph/resources.py` sin tocar los call-sites.

## Namespaces estandarizados

- `("user", user_id, "profile")` - perfil tecnico mantenido por el Shadow Agent (F3-T2).
- `("user", user_id, "routines")` - retos generados por el Routine Generator (F5-T1).

## Helpers

```python
from src.graph import get_store

store = get_store()

# Escritura
await store.aput(
    ("user", user_id, "profile"),
    key="profile",
    value={
        "strengths": ["Modularidad"],
        "weaknesses": ["Caching"],
        "evaluated_concepts": [],
        "updated_at": "2026-04-28T10:00:00Z",
    },
)

# Lectura
item = await store.aget(("user", user_id, "profile"), key="profile")
profile = item.value if item else {}
```

`Store` es accesible desde cualquier nodo del grafo porque se inyecta en
`builder.compile(store=...)`. Dentro de un nodo asincrono:

```python
async def my_node(state, *, store):
    item = await store.aget(("user", state["user_id"], "profile"), key="profile")
    profile = item.value if item else {}
    ...
```

## Verificacion cross-thread

Ver `back/tests/test_store_cross_thread.py`. El test escribe bajo el namespace
del usuario y lee desde otro contexto sin reusar el `thread_id` del checkpointer,
demostrando que el Store es independiente de la conversacion.
