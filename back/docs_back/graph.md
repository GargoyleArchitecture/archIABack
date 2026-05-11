# Grafo LangGraph — Backend IA

**Fase:** F11-T4
**Última actualización:** 2026-05-08
**LangGraph:** 1.0.x

Documentación viva del grafo principal del agente conversacional y del subgrafo de generación de retos.

---

## GraphState (estado compartido)

Definido en `src/graph/state.py` como `TypedDict`. Campos relevantes post-Fase 11:

| Campo | Tipo | Origen / Mutación | Notas |
|---|---|---|---|
| `messages` | `list` | Append por cada turno | Histórico canónico del thread |
| `intent` | `Literal[...]` | `classifier_node` | greeting, smalltalk, architecture, diagram, asr, tactics, style, general |
| `language` | `str` | `classifier_node` | Idioma detectado |
| `force_rag` | `bool` | `classifier_node` | Bypass de la heurística de uso de RAG |
| `quality_attribute` | `str` | `classifier_node` o nodo `asr` | Para tactics/style |
| `mode` | `Literal["tutor","professional"]` | API → `boot_node` (F2-T1, F2-T2) | Default `professional` |
| `mode_suggestion` | `Literal["tutor","professional"] \| None` | `classifier_node` (F2-T4) | Forward al Frontend |
| `user_id` | `str` | API → `boot_node` (F2-T1) | Separado de `user_id_for_prefs` (legacy) |
| `user_profile` | `dict` | `boot_node` lee Store (F3-T3) | Decay aplicado en lectura (F3-T4) |
| `turn_count_since_eval` | `int` | `boot_node` incrementa; `unifier` resetea | Cadencia Shadow Agent (F3-T2) |
| `resolved_index` | `str` | `classifier_node` | Nombre del índice de RAG a usar |

---

## Grafo principal

```
                            ┌────────────────┐
        ┌──────────────────▶│   boot_node    │  (F3-T3: hidrata perfil)
        │                   └───────┬────────┘
        │                           ▼
        │                   ┌────────────────┐
        │                   │   classifier   │  (F2-T4: mode_suggestion)
        │                   └───────┬────────┘
        │                           ▼
        │                   ┌────────────────┐
        │                   │     router     │
        │                   └────────────────┘
        │            ┌──────────┬───┴───┬────────────┐
        │            ▼          ▼       ▼            ▼
        │      investigator  tactics  style       (otros)
        │            │          │       │
        │            └──────────┴───┬───┘
        │                           ▼
        │                   ┌────────────────┐
        │                   │    unifier     │
        │                   ├────────────────┤
        │                   │ async →        │  (F3-T2: fire_shadow_eval)
        │                   │   shadow_eval  │
        │                   └───────┬────────┘
        │                           ▼
        │                          END
```

**Inyección de prompts por modo (F2-T3):** los nodos `unifier`, `investigator`, `asr`, `tactics/common`, `styles/common` aplican `apply_mode_prompt(state, base_prompt)` justo antes del `llm.invoke`. Sin acumulación entre turnos.

**Tools gated por modo (F2-T5 / F2-T6):**
- En `mode='tutor'`: `lookup_glossary`, `cite_documentation` (solo si corpus glossary existe).
- En `mode='professional'`: `python_repl_tool` (env `ENABLE_PYTHON_REPL=true`), `local_rag_advanced`.

---

## Subgrafo `RoutineGeneratorGraph` (F5-T1)

```
[select_weakness] → [inverse_rag_search] → [synthesize_challenge] → [validate_difficulty]
                                                  ▲                        │
                                                  │                        ▼
                                                  └──── regen (≤2) ────┐  END
                                                                       │
                                                              if final_difficulty > ceiling
                                                                  regen_count < 2
```

- `select_weakness` (F5-T1): si `target_weakness` viene → usa el provided; si no → el de menor mastery; fallback genérico `"general software architecture"`.
- `inverse_rag_search` (F5-T3): consulta `bad_code_corpus` con `target_weakness`. 5 ramas de fallback graceful (weakness vacío, retriever None, error, hits vacíos, hits válidos).
- `synthesize_challenge`: LLM con `with_structured_output(RoutineOutput)`. Inyecta `REDUCE_SCOPE` cuando `regen_count >= 1`.
- `validate_difficulty` (F5-T4): `ceiling = round(mastery*5) + 2`. Regen si `final_difficulty > ceiling` y `regen_count < 2`. Aceptar siempre tras 2 regens.

`RoutineOutput` (Pydantic) en `src/graph/schemas/routine.py` con validaciones de min_length y rango difficulty 1..5.

---

## Persistencia del grafo

| Componente | Backend | Persistencia | Recuperación |
|---|---|---|---|
| Checkpointer | `AsyncSqliteSaver` (F1-T3) | `back/state_db/graph_checkpoints.db` | Reanuda por `thread_id` tras reinicio del proceso |
| Store cross-thread | `InMemoryStore` (F1-T4) | RAM (efímero) | Hidratación inversa desde Negocio (F3-T6) al primer acceso de un `user_id` |

---

## Telemetría del grafo (F11-T6)

Eventos emitidos por `src/services/telemetry.py:emit()` (logger `archia.telemetry`):

| Evento | Nodo origen | Payload |
|---|---|---|
| `mode_suggested` | `classifier_node` | `{user_id, current_mode, suggestion}` cuando `mode_suggestion != null` |
| `routine_generated` | `routine_generator` (servicio) | `{user_id, target_weakness, difficulty, concepts_count, regen_count, trace_id}` |

---

## Regeneración del corpus RAG inverso

```powershell
# Setup env: API keys de embeddings (Azure/OpenAI) en .env
cd archIABack/back
.\..\.venv\Scripts\python.exe build_bad_code_corpus.py --rebuild
```

El script:
1. Lee `back/data/bad_code_seed.json` (90 entries: 30 acoplamiento + 30 escalabilidad + 30 latencia).
2. Valida schema y conteos por categoría.
3. Embede usando `_embeddings_factory` de `src/rag_agent.py` (mismo provider que el RAG directo).
4. Persiste en `back/bad_code_corpus/` (ChromaDB).

Sin `--rebuild` el script se niega a sobreescribir un directorio con contenido. Documentado en F5-T3.

---

## Comandos útiles

```powershell
# Smoke imports
python -c "from src.graph import build_graph; print('ok')"

# Validar el grafo
python back/scripts/verify_checkpointer.py --write
python back/scripts/verify_checkpointer.py --read

# Suite completa
pytest -q

# Solo telemetría
pytest tests/test_telemetry.py -q

# Load test (dry-run)
python -m tests.load.shadow_agent_load --dry-run --users 50 --turns 6 --seed 42
```
