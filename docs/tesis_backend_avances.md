# Avances del backend inteligente — estado actual

Este documento describe el estado del backend conversacional de ArchIA en su
versión vigente. Cubre el esquema de fases ADD 3.0, el supervisor de
LangGraph, el nodo de intake/diagnóstico, los flujos de confirmación de ASR,
el enrutamiento de usuarios retornantes, el Design Ledger y la deuda técnica
documentada explícitamente en el código.

Todo el contenido se deriva directamente de las rutas `back/src/graph/` y
`back/src/ledger/`.

---

## 1. Esquema de fases ADD 3.0

La fuente canónica de las fases del proceso de diseño es el enum `Phase`
definido en `back/src/ledger/types.py:7-23`:

```python
class Phase(str, Enum):
    INTRO          = "intro"
    DIAGNOSIS      = "diagnosis"
    ASR_TABLE      = "asr_table"
    STYLE_TABLE    = "style_table"
    TACTICS_TABLE  = "tactics_table"
    TECH_PROPOSALS = "tech_proposals"
    DIAGRAM        = "diagram"
    ANALYSIS       = "analysis"
    DONE           = "done"


PHASE_ORDER: list[Phase] = [
    Phase.INTRO, Phase.DIAGNOSIS, Phase.ASR_TABLE, Phase.STYLE_TABLE,
    Phase.TACTICS_TABLE, Phase.TECH_PROPOSALS, Phase.DIAGRAM,
    Phase.ANALYSIS, Phase.DONE,
]
```

El tipo `PhaseLiteral` (`back/src/ledger/types.py:25-28`) replica los nueve
valores como `Literal` para anotaciones de TypedDict.

### Fuente de verdad y sincronización

La fuente de verdad del estado de fase es el campo `current_phase` del
`DesignLedger` persistido en SQLite (`back/src/ledger/types.py:79-90`). El
campo `state.current_phase` del grafo es un espejo de lectura; así lo
documenta el comentario en `back/src/graph/state.py:212-220`:

```python
# ── Ledger hydration (P2) ─────────────────────────────────────────────
# Refreshed by context_loader on every turn. Never reset by boot_node.
ledger: dict                  # full DesignLedger blob; {} before first successful load
ledger_active: dict           # output of compute_active_view(ledger); {} when empty
design_dossier_md: str        # render_dossier(ledger, lang=language); "" before load
current_phase: PhaseLiteral   # ADD 3.0 phase, single source of truth (mirror of ledger["current_phase"])
ledger_dossier_compact: str   # render_dossier_compact(ledger, lang); "" before load
ledger_phase_prompt: str      # render_phase_prompt(ledger, lang); "" before load
ledger_pending_advance: dict  # mirror of ledger["pending_advance"]; {} when None
```

La sincronización ocurre en `context_loader_node`
(`back/src/graph/nodes/context_loader.py:141-172`), que se ejecuta en cada
turno antes del classifier:

```python
if need_ledger:
    try:
        ledger = load_ledger(user_id, project_id or None)
        lang   = (state.get("language") or "es") or "es"
        active = compute_active_view(ledger)

        updates["ledger"]                 = ledger
        updates["design_dossier_md"]      = render_dossier(ledger, lang=lang)
        updates["ledger_dossier_compact"] = render_dossier_compact(ledger, lang=lang)
        updates["ledger_phase_prompt"]    = render_phase_prompt(ledger, lang=lang)
        updates["ledger_pending_advance"] = ledger.get("pending_advance") or {}

        raw_phase = ledger.get("current_phase") or "intro"
        mapped_phase = _LEGACY_PHASE_MAP.get(raw_phase, raw_phase)
```

Tras la carga, los nodos que escriben al ledger invocan
`_refresh_ledger_state` (`back/src/graph/nodes/_ledger_helpers.py:17-36`)
para re-sincronizar `state.current_phase`, `state.ledger`,
`state.ledger_active`, `state.design_dossier_md` y los renders compactos en
el mismo turno, sin esperar al próximo `context_loader`.

### Campos asociados a la fase en `GraphState`

Además de `current_phase` (espejo del ledger), el `GraphState` mantiene un
campo de enrutamiento independiente para preservar el progreso del grafo
entre turnos (`back/src/graph/state.py:231-235`):

```python
# ── Routing phase (BUG-013) ──────────────────────────────────────────────
# Tracks which graph-routing phase the session has reached so boot_node can
# avoid resetting completed_nodes / hasVisitedASR on subsequent turns.
# Distinct from current_phase (ADD 3.0 ledger phase).
routing_phase: Literal["intake", "asr", "style", "tactics", "tech", "done"]
```

`routing_phase` no es la fase ADD 3.0: complementa a `current_phase`
indicando qué nodos del grafo ya han producido salida en la sesión y debe
evitar reentradas espurias en `boot_node`.

---

## 2. Phase-gate en el supervisor

El supervisor implementa el gate en `back/src/graph/nodes/supervisor.py`,
apoyándose en dos diccionarios constantes definidos en
`back/src/graph/consts.py:242-253`:

```python
PHASE_INT: dict[str, int] = {
    "intro": 0, "diagnosis": 1, "asr_table": 2, "style_table": 3,
    "tactics_table": 4, "tech_proposals": 5, "diagram": 6, "analysis": 7, "done": 8,
}

FUNNEL_INTENT_MIN_PHASE: dict[str, str] = {
    "asr":     "asr_table",
    "style":   "style_table",
    "tactics": "tactics_table",
    "tech":    "tech_proposals",
    "diagram": "style_table",
}
```

La verificación del gate vive en `supervisor_node`
(`back/src/graph/nodes/supervisor.py:385-413`):

```python
# ─── M1: Gate de fase ADD 3.0 ───────────────────────────────────────────
current_phase = (state.get("current_phase") or "intro")
intent_raw = (state.get("intent") or "")
min_phase_key = FUNNEL_INTENT_MIN_PHASE.get(intent_raw)

if min_phase_key and PHASE_INT.get(current_phase, 0) < PHASE_INT[min_phase_key]:
    if intent_raw == "diagram":
        block_text = _build_diagram_block_message(current_phase, state_lang)
    else:
        block_text = _build_block_message(current_phase, min_phase_key, state_lang)
    _sugs_es = ["Sí, continuemos", "Quiero cambiar el contexto del sistema"]
    _sugs_en = ["Yes, let's continue", "I want to change the system context"]
    _completed_safe = _augment_completed_nodes(state, list(state.get("completed_nodes") or []))
    return {
        **state,
        "endMessage": block_text,
        "nextNode": "unifier",
        "intent": "intake",
        "language": state_lang,
        "suggestions": _sugs_es if state_lang == "es" else _sugs_en,
        "requested_nodes": [],
        "pending_nodes": [],
        "completed_nodes": _completed_safe,
        "phase_redirect_hint": "",
    }
```

### Comportamiento ante solicitudes fuera de fase

El gate **no rechaza silenciosamente** ni redirige a un nodo de trabajo. En
su lugar:

1. Construye un mensaje explicativo bilingüe mediante `_build_block_message`
   o `_build_diagram_block_message` (`supervisor.py:117-163`) que nombra la
   fase actual, la fase solicitada y la tarea pendiente, todo apoyado en los
   diccionarios `PHASE_DISPLAY` y `PHASE_NEXT_TASK` (`consts.py:255-277`).
2. Fuerza `nextNode = "unifier"` para que el grafo termine el turno con un
   solo mensaje informativo.
3. Inyecta `suggestions` bilingües con dos chips (`"Sí, continuemos"` /
   `"Quiero cambiar el contexto del sistema"`) que el frontend puede
   renderizar como acciones rápidas.
4. Preserva `completed_nodes` (a través de `_augment_completed_nodes`,
   `supervisor.py:67-90`) para evitar que el supervisor reactive nodos ya
   ejecutados al desbloquearse la fase.

### Estado de implementación

El gate está **activo en producción** sin flag condicional. Solo se evita
en dos casos:

- Modo *tutor*: bypass total documentado en `supervisor.py:288-295` (fuera
  del alcance de este documento).
- Fases `intro` o `diagnosis`: el supervisor encamina directamente a
  `intake` sin pasar por el funnel (`supervisor.py:297-298`).

---

## 3. Nodo de intake / diagnóstico

El nodo de intake **sí existe** en
`back/src/graph/nodes/intake_node.py` (1090 líneas) y está registrado en el
grafo en `back/src/graph/workflow.py:226`.

### Propósito y responsabilidad

El nodo conduce el diagnóstico ADD 3.0 sobre la base de un guión fijo de
ocho campos definidos en `back/src/graph/nodes/intake_validators.py:114-165`:

```python
INTAKE_SCRIPT = [
    {"field": "campo_0_requerimiento", ...},
    {"field": "campo_1_alcance_funcional", ...},
    {"field": "campo_2_fuente", ..., "optional": True},
    {"field": "campo_3_estimulo", ..., "optional": True},
    {"field": "campo_4_ambientes", ...},
    {"field": "campo_5_prioridad_qa", ...},
    {"field": "campo_6_restricciones", ...},
    {"field": "campo_7_decisiones", ...},
]
```

Etiquetas legibles para feedback al usuario en
`back/src/graph/nodes/intake_node.py:239-248`:

```python
_FIELD_LABELS: dict[int, dict[str, str]] = {
    0: {"es": "requerimiento",        "en": "requirement"},
    1: {"es": "alcance funcional",     "en": "functional scope"},
    2: {"es": "fuente del estímulo",  "en": "stimulus source"},
    3: {"es": "estímulo",             "en": "stimulus"},
    4: {"es": "ambientes",            "en": "environments"},
    5: {"es": "prioridad QA",         "en": "QA priority"},
    6: {"es": "restricciones",        "en": "constraints"},
    7: {"es": "decisiones previas",   "en": "prior decisions"},
}
```

El nodo persiste los campos validados en `state.intake_fields`,
`state.intake_current_field` (índice 0–8; 8 = guion completo) e
`state.intake_complete`, declarados en `back/src/graph/state.py:222-225`.

Adicionalmente, parsea las métricas de operación normal/sobrecarga de
`campo_4_ambientes` mediante `_extract_baseline`
(`intake_node.py:193-236`) y las publica como `normal_operation_baseline`
en el estado, para que `asr_node` rechace ASRs cuyo Response Measure caiga
dentro del baseline.

### Validación en dos capas

El nodo combina validación determinista (regex sobre vocabulario técnico,
métricas, fuentes de estímulo) con una capa semántica vía LLM en
`extract_and_validate_fields` (`intake_validators.py`). El método central
es `_process_intake_turn` (`intake_node.py:344-420`), que recolecta
extracciones y validaciones de todos los campos pendientes en una sola
invocación, garantizando "fail-open" si el LLM falla.

### Condición de transición a la siguiente fase

Cuando los ocho campos están validados, `intake_node` ejecuta la
transición `diagnosis → asr_table` directamente sobre el ledger. Existen
cuatro ramas según la trayectoria del diálogo:

- **Rama A (`intake_complete=True`)**: post-intake, clasifica la intención
  del arquitecto en `HAVE_ASRS` / `WANT_PROPOSE` / `AMBIGUOUS`
  (`intake_node.py:545-705`). Solo `WANT_PROPOSE` ejecuta la transición y
  emite `nextNode = "asr"` en el mismo turno.
- **Rama B**: detecta `current_index >= 8` en el turno actual y avanza al
  asr_table inmediatamente (`intake_node.py:710-760`).
- **Rama C**: primer turno donde el usuario entrega todos los campos de una
  vez (`intake_node.py:802-856`).
- **Rama D**: turno normal con campos en progreso; cuando los completa,
  publica intake_v1 en el ledger y muestra una pregunta de permiso antes
  de proponer ASRs (`intake_node.py:917-953`).

### Integración en el grafo

`intake_node` está registrado y conectado mediante una arista condicional
en `back/src/graph/workflow.py:226-233`:

```python
builder.add_node("intake", intake_node)

def _intake_next(state: GraphState) -> str:
    # A2 sets nextNode="asr": route through supervisor so asr_node runs this turn.
    # All other branches (B, C, D, A1, A3) use nextNode="unifier" → direct to unifier.
    return "supervisor" if state.get("nextNode") == "asr" else "unifier"

builder.add_conditional_edges("intake", _intake_next, {"supervisor": "supervisor", "unifier": "unifier"})
```

El supervisor lo invoca al inicio de cada turno mientras la fase activa sea
`intro` o `diagnosis` (`supervisor.py:297-298`):

```python
if (state.get("current_phase") or "") in ("intro", "diagnosis") and (state.get("mode") or "professional") != "tutor":
    return {**state, "nextNode": "intake", "localQuestion": ""}
```

---

## 4. Flujo de confirmación y rechazo de ASR

### Confirmación: implementado

El classifier detecta la intención `asr_confirm` cuando el usuario está en
`current_phase == "asr_table"` y hay candidatos vigentes. La lógica vive
en `back/src/graph/nodes/classifier.py:202-230`:

```python
_has_existing_asr = bool(state.get("current_asr") or state.get("last_asr"))
_in_asr_phase = (state.get("current_phase") or "") == "asr_table"
_not_tutor = (state.get("mode") or "professional") != "tutor"
if _has_existing_asr and _in_asr_phase and _not_tutor:
    asr_confirm_triggers = [
        "confirmo", "lo confirmo", "acepto este asr", "acepto ese asr",
        ...
        "tomo ese", "tomo el asr", "me quedo con", "elijo", "voy con",
        "ese me sirve", "perfecto ese", "ok ese", "de acuerdo", "dale",
        ...
    ]
    _asr_id_matches = re.findall(r"\b[Aa](\d+)\b", msg)
    _bare_id = re.match(r"^\s*[Aa]\d+\s*$", msg)
    if any(k in low for k in asr_confirm_triggers) or _asr_id_matches or _bare_id:
        intent = "asr_confirm"
        if _asr_id_matches:
            _selected_ids = [f"A{n}" for n in _asr_id_matches]
            state["selected_asrs"] = list(dict.fromkeys(_selected_ids))
```

El supervisor enruta la intención directamente al nodo dedicado
(`supervisor.py:418-427`) y el router del grafo lo redirige
(`workflow.py:133-134`). El nodo `asr_confirm_node`
(`back/src/graph/nodes/asr_confirm.py:65-269`) realiza:

1. Resuelve los IDs humanos seleccionados contra `state.asr_candidates` con
   un mapa `candidate_id → row`.
2. Para cada candidato confirmado, expande las seis partes canónicas del
   escenario (`Source`, `Stimulus`, `Environment`, `Artifact`, `Response`,
   `Response Measure`) vía `_expand_asr_to_six_part` y persiste una nueva
   decisión `kind="asr"` en el ledger con `append_decision`.
3. Publica el conjunto multi-selección en `state.selected_asrs` con
   estructura `[ULID_A1, "A1", ULID_A2, "A2", ...]`.
4. Inicializa `state.selected_qa_queue` con la lista ordenada de QAs únicos
   (`asr_confirm.py:212-218`) para que tech_node itere por QA.
5. Ejecuta `transition_phase(asr_table → style_table)` y refresca el estado
   con `_refresh_ledger_state`.

### Rechazo: parcialmente implementado

El classifier reconoce la intención `asr_reject`
(`classifier.py:216-232`):

```python
asr_reject_triggers = [
    "rechazo", "ese no", "no me convence", "otro asr", "otro distinto",
    "reject", "another asr", "different asr", "not that one",
]
...
elif any(k in low for k in asr_reject_triggers) or is_asr_regenerate_request(msg):
    intent = "asr_reject"
```

El supervisor reacciona a esa intención forzando una nueva ejecución de
`asr_node` (`supervisor.py:494-499`):

```python
explicit_regen = is_asr_regenerate_request(uq) or (state.get("intent") == "asr_reject")
must_run_asr = ("asr" in requested_nodes) and (explicit_regen or not _asr_already_done)

if must_run_asr:
    next_node = "asr"
    pending_nodes = [n for n in pending_nodes if n != "asr"]
```

**Lo que falta**: no existe un `asr_reject_node` dedicado análogo a
`asr_confirm_node`. La función `reject_decision`
(`back/src/ledger/store.py:226-258`) está implementada y exportada en
`back/src/ledger/__init__.py:30`, pero **ningún nodo la invoca**. Como
consecuencia, los candidatos previos generados por `asr_node` (que ya se
persistieron en el ledger con `append_decision`) permanecen con
`status="active"`. La regeneración apila nuevas decisiones `kind="asr"`
sin marcar las anteriores como rechazadas: el supersession del ledger
exime a los ASRs explícitamente (`store.py:80-82`):

```python
# Issue 2a fix: ASRs are an unordered SELECTION SET, not a single-winner
# decision. Two ASRs picked together (e.g. "A1,A2") are both legitimate
# drivers for the design loop, and the user may later add/inspect more.
# ... Short-circuit here to keep all ASRs active concurrently;
# dedupe-by-candidate-id is handled by the asr_confirm node before write.
if kind == "asr":
    return None
```

El punto de entrada natural para cerrar el ciclo está en ambos extremos:
el classifier ya emite la intención y el supervisor ya redirige a
`asr_node`. Falta intercalar una llamada a `reject_decision` (o un nodo
intermedio `asr_reject_node`) que marque los candidatos previos como
rechazados antes de regenerar, e idealmente registre el motivo del rechazo
del arquitecto en el ledger.

---

## 5. Enrutamiento de usuarios retornantes

El supervisor distingue al usuario retornante por dos señales combinadas:

- **Intención no específica**: classifier emite `intent ∈
  {general, greeting, smalltalk}`.
- **Fase activa avanzada**: `current_phase` (espejo del ledger) está fuera
  de `intro` / `diagnosis`.

La rama vive en `back/src/graph/nodes/supervisor.py:344-383`:

```python
# ─── Orientación para usuarios que regresan ──────────────────────────────
_returning_intent = (state.get("intent") or "") in ("general", "greeting", "smalltalk")
_has_phase_context = (state.get("current_phase") or "intro") not in ("intro", "diagnosis")

if _returning_intent and _has_phase_context:
    _phase_now   = state.get("current_phase") or "intro"
    _task_hint   = PHASE_NEXT_TASK.get(_phase_now, {}).get(state_lang, "")
    _phase_label = PHASE_DISPLAY.get(_phase_now, {}).get(state_lang, _phase_now)
    _compact     = (state.get("ledger_dossier_compact") or "").strip()

    if state_lang == "es":
        _lines = [f"Bienvenido de vuelta. Estamos en la fase de **{_phase_label}**."]
        if _compact:
            _lines.append(_compact)
        if _task_hint:
            _lines.append(f"La siguiente tarea es: *{_task_hint}*. ¿Continuamos?")
    else:
        _lines = [f"Welcome back. We're in the **{_phase_label}** phase."]
        if _compact:
            _lines.append(_compact)
        if _task_hint:
            _lines.append(f"Next up: *{_task_hint}*. Shall we continue?")
```

### Comportamiento actual

El nodo construye un mensaje de bienvenida bilingüe que combina tres
elementos:

1. **Etiqueta de fase** desde `PHASE_DISPLAY` (`consts.py:255-265`).
2. **Resumen compacto** del Design Dossier desde
   `state.ledger_dossier_compact`, producido por
   `render_dossier_compact` (`ledger/render.py:311-355`) con la fase
   actual, iteración, ASR activo, estilo elegido y conteo de tácticas.
3. **Próxima tarea** desde `PHASE_NEXT_TASK` (`consts.py:267-277`), p. ej.
   "seleccionar el estilo arquitectónico para los ASRs".

El supervisor fuerza `nextNode = "unifier"` y mantiene
`completed_nodes` ampliado con `_augment_completed_nodes` para no
disparar nodos ya ejecutados en la sesión anterior.

### Distinción de proyecto nuevo

Existe una rama complementaria que protege contra el opuesto: un usuario
que vuelve para *iniciar un proyecto nuevo* sobre un checkpoint con fase
avanzada (`supervisor.py:306-338`). Usa los regex `_NEW_PROJECT_GREETING_RE`
y `_NEW_PROJECT_DESIGN_RE` (`supervisor.py:94-115`) y exige que el mensaje
tenga al menos 20 palabras y contenga saludo + intención de diseño.
Cuando dispara, resetea el estado y marca `new_project_flow = True` para
que el `context_loader` evite restaurar el ledger anterior
(`context_loader.py:160-166`).

### Limitaciones

El bloque depende de que `context_loader_node` haya hidratado
`ledger_dossier_compact` en el turno actual. Si el ledger está vacío
(`compact == ""`), el mensaje se limita a la etiqueta de fase y la tarea
pendiente, sin reflejar las decisiones tomadas.

---

## 6. Estado actual del Design Ledger

El paquete `back/src/ledger/` expone los siguientes módulos:

- `types.py`: `Phase`, `PhaseLiteral`, `DecisionKind`, `DesignLedger`,
  `Decision`, `DecisionRef`, `PhaseTransition`, `PHASE_ORDER`,
  `LEDGER_SCHEMA_VERSION`, `empty_ledger`, excepciones
  `LedgerValidationError` y `LedgerConcurrencyError`.
- `store.py`: persistencia atómica sobre SQLite con `BEGIN IMMEDIATE` y
  control de versión optimista.
- `validate.py`: validaciones de forma, parents y QA-match.
- `render.py`: tres renders del ledger en Markdown bilingüe.

### Estructura de `DesignLedger`

Definida en `back/src/ledger/types.py:79-104`:

```python
class DesignLedger(TypedDict):
    version: int
    project_id: str
    user_id: str
    current_phase: PhaseLiteral
    current_iteration: int
    phase_history: list[PhaseTransition]
    pending_advance: Optional[PhaseTransition]
    decisions: list[Decision]
    project_context: dict[str, Any]
    user_style_hint: str
```

Las decisiones admiten siete `kind`s
(`back/src/ledger/types.py:32`): `asr`, `style`, `tactic`, `tech`,
`diagram`, `analysis`, `constraint`. La validación de payloads y parents
requeridos vive en `validate.py:14-33`.

### API pública de escritura

`back/src/ledger/store.py` expone cinco operaciones de escritura:
`save_ledger`, `append_decision`, `reject_decision`, `transition_phase`,
`stage_pending_advance`, `clear_pending_advance`. Todas reintentan ante
`LedgerConcurrencyError` con un máximo de dos intentos y reciclan la
versión optimista mediante `expected_version`.

### Vistas

`compute_active_view` (`store.py:330-336`) reduce la lista append-only de
decisiones a un dict `{kind: última_decisión_activa}`. La excepción
explícita son los ASRs: `get_all_active_asrs` (`store.py:339-344`) devuelve
**todas** las decisiones `kind="asr"` con `status="active"`, porque los
ASRs forman un conjunto de selección, no un winner-takes-all.

### Integración con el sistema de fases

El ledger almacena la fase activa en `current_phase` y la expone a los
nodos a través de tres rutas:

1. **Hidratación de `GraphState`**: `context_loader_node` carga el ledger
   y publica `state.current_phase`, `state.ledger_active`,
   `state.design_dossier_md`, `state.ledger_dossier_compact` y
   `state.ledger_phase_prompt` (`context_loader.py:141-179`).
2. **Migración legacy**: el mapeo `_LEGACY_PHASE_MAP`
   (`context_loader.py:25-33`) traduce valores en mayúsculas de ledgers
   anteriores (`INTAKE`, `ASR`, `STYLE`, …) a la nomenclatura snake_case
   actual antes de hidratar el estado.
3. **Refresh post-escritura**: `_refresh_ledger_state`
   (`back/src/graph/nodes/_ledger_helpers.py:17-36`) recarga el ledger
   inmediatamente después de `transition_phase` o `append_decision`,
   evitando un round-trip extra de turnos.

### Validación de transiciones de fase

`validate_transition` (`validate.py:117-154`) verifica tres invariantes:

- `transition.from_phase` debe igualar `ledger.current_phase`.
- `transition.iteration` debe ser exactamente `current_iteration + 1`.
- Si la transición salta fases (p. ej. `diagnosis → diagram`),
  `skipped_phases` debe enumerar las omitidas según `PHASE_ORDER`.

### Mecanismo de supersession

Definido en `back/src/ledger/store.py:62-102`. Cuando se invoca
`append_decision`, la función `_apply_supersession` busca otra decisión
con el mismo `kind` y los mismos `parents` ya activa. Si la encuentra:

```python
for d in ledger["decisions"]:
    if d["kind"] == kind and d["status"] == "active":
        existing_parent_ids = frozenset(r["id"] for r in (d.get("parents") or []))
        if existing_parent_ids == new_parent_ids:
            d["status"] = "superseded"
            d["superseded_by"] = new_decision["id"]
            superseded_id = d["id"]
            break

if superseded_id:
    for d in ledger["decisions"]:
        if d["status"] == "active":
            parent_ids = {r["id"] for r in (d.get("parents") or [])}
            if superseded_id in parent_ids:
                d["parent_status"] = "parent_superseded"
```

Propaga `parent_status = "parent_superseded"` a las decisiones hijas
activas que dependían de la decisión reemplazada. **Los ASRs están
exentos por diseño**: el short-circuit `if kind == "asr": return None` se
documenta en `store.py:74-82` como un cambio explícito para soportar
selección múltiple (un mismo proyecto puede tener varios ASRs activos en
paralelo, cada uno con su propio QA).

### Rechazo

`reject_decision` (`store.py:226-258`) marca la decisión objetivo con
`status="rejected"` y propaga `parent_status="parent_rejected"` a sus
hijos activos. La función está exportada en `__init__.py` pero, como se
documenta en §4, no es invocada por ningún nodo del grafo en el estado
actual del backend.

### Render del Dossier

Tres funciones públicas en `back/src/ledger/render.py`:

- `render_dossier` (línea 120): vista completa del ledger usada por
  `state.design_dossier_md`.
- `render_dossier_compact` (línea 311): one-liner por decisión activa,
  usado por la bienvenida del usuario retornante y prompts del supervisor.
- `render_phase_prompt` (línea 358): footer "Próximo paso:" que el unifier
  añade al final de turnos en transición.

---

## 7. Deuda técnica vigente

Lista de elementos explícitamente señalados en comentarios del código o
descritos como "no implementados" en el propio repositorio:

```
[LEDGER/store.py:352]   `is_phase_complete(INTRO)` retorna True sin verificación —
                        comentario "single-step phase; M6 will refine" indica que
                        la lógica real está pendiente.

[LEDGER/store.py:378]   `is_phase_complete(TECH_PROPOSALS)` retorna False
                        incondicionalmente — comentario "completed by M5 (tech
                        kind not introduced in M2)" señala que la condición de
                        cierre de la fase de tecnologías aún no se evalúa.

[LEDGER/store.py]       `is_phase_complete` no se invoca desde ningún nodo del
                        grafo; las transiciones se disparan por eventos (intake,
                        asr_confirm, style_confirm, tactics_confirm) sin
                        consultar al ledger si la fase realmente terminó.

[LEDGER/store.py]       `reject_decision` está implementada, validada y exportada
                        pero ningún nodo la llama. El intent `asr_reject`
                        regenera ASRs sin marcar los anteriores como rechazados.

[INTAKE/intake_node.py:437] Comentario "M6 has not been implemented yet" sobre
                        la transición `intro → diagnosis`: la lógica fina del
                        ciclo de bienvenida queda como TODO implícito.

[CONTEXT_LOADER/context_loader.py:25-33] `_LEGACY_PHASE_MAP` mantiene compat con
                        ledgers persistidos en formato pre-M2 (uppercase). Es un
                        shim de migración que debería poder retirarse cuando se
                        confirme que no quedan checkpoints antiguos en producción.

[LEDGER/*]              Todos los call-sites del ledger en nodos atrapan
                        excepciones de validación/concurrencia y registran
                        "(nonfatal)". Esto enmascara errores silenciosos: una
                        transición fallida deja el estado del grafo desincronizado
                        del ledger persistido (asr_node compensa con un self-heal
                        en `asr.py:957-980`, pero no es general).

[SUPERVISOR/supervisor.py:344-383] La bienvenida del usuario retornante depende
                        de `ledger_dossier_compact`. Cuando el ledger está vacío
                        pero `current_phase` ya avanzó, el mensaje no resume
                        decisiones tomadas — solo nombra la fase.
```
