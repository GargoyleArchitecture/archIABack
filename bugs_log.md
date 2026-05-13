# Archia Bug Log — Branch: fix/SyncSemana15

> **Protocol**: Every deviation from the expected ADD 3.0 flow or technical error in the LangGraph state is logged here before any code is written. This file is the primary "todo list" for this debugging session.

---

## BUG-001 — `style_payload.tradeoffs` Always Empty: Missing `rationale` Key in LLM JSON Schema

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/styles/common.py` |
| **Lines** | Prompt JSON schema (lines 363–378), `_build_style_payload` call (line 425) |

### Observed Behavior
The ledger `style` decision is written with an empty `tradeoffs` field. When the tactics node builds its `HARD-BINDING` prompt via `_build_dossier_design_binding`, the injected block shows:
```
Active Style: Microservices  (id: ...)
Style Tradeoffs:              ← ALWAYS EMPTY
```

### Expected Behavior
The binding block should contain the chosen style's trade-off reasoning so the tactics LLM can honour the ADD 3.0 style → tactics constraint chain.

### Root Cause
`_build_style_payload` is called with `rationale=data.get("rationale", "")`. However, the LLM JSON schema defined in the prompt only produces `style_1`, `style_2`, and `best_style` — there is **no top-level `rationale` key**:
```json
{
  "style_1": { "name": "...", "justification": "...", "tradeoff": "..." },
  "style_2": { "name": "...", "justification": "...", "tradeoff": "..." },
  "best_style": "style_1"
}
```
`data.get("rationale", "")` always returns `""`. The per-style `tradeoff` field (which contains the real trade-off text) is correctly parsed for the Markdown table but never fed into the ledger payload.

### Fix
In `style_node_impl`, derive `rationale` from the **chosen style's own `tradeoff` field**:
```python
best_key = (data.get("best_style") or "").strip()
_chosen_data = style2 if best_key == "style_2" else style1
rationale = _chosen_data.get("tradeoff", "").strip()
```

### Resolution Status
✅ Fixed in this commit

---

## BUG-002 — M1 Phase Gate Resets `completed_nodes = []`, Breaking Session Continuity

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/supervisor.py` |
| **Lines** | 296–307 (M1 phase gate return block) |

### Observed Behavior
When the supervisor's M1 gate fires (user requests a step before the required ADD 3.0 phase is reached), the returned state sets `"completed_nodes": []`. On the **next turn**, `_augment_completed_nodes` must re-derive completed nodes from secondary signals (`routing_phase`, `hasVisitedASR`, etc.). If `boot_node` has reset `hasVisitedASR = False` and `routing_phase` was lost, the supervisor may re-trigger ASR generation even though an ASR already exists in the ledger.

### Expected Behavior
`completed_nodes` must survive the phase gate. Phase blocking is a redirect, not a session reset.

### Root Cause
Hard-coded `"completed_nodes": []` in the M1 gate return dict (supervisor.py line 305):
```python
return {
    **state,
    "endMessage": block_text,
    ...
    "completed_nodes": [],   # ← BUG: wipes the list unconditionally
    ...
}
```

### Fix
Replace `"completed_nodes": []` with the augmented list:
```python
"completed_nodes": _augment_completed_nodes(state, list(state.get("completed_nodes") or [])),
```

### Resolution Status
✅ Fixed in this commit

---

## BUG-003 — `_apply_supersession` Does Not Propagate `parent_status="parent_superseded"` to Children

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/ledger/store.py` |
| **Lines** | `_apply_supersession` (lines 62–73), `append_decision` (lines 159–193) |

### Observed Behavior
After the user regenerates an ASR (ASR-2 supersedes ASR-1), the old style and tactics decisions that were children of ASR-1 remain with `status="active"` and `parent_status="ok"`. `compute_active_view` iterates all `status="active"` decisions and returns the last per kind, so it may return **Style-1** (linked to the superseded ASR-1) as the active style even though no new style has been generated yet for ASR-2.

Result: the tactics node gets a binding block that cites ASR-1's response measure together with ASR-2's scenario — a logically inconsistent design chain.

### Expected Behavior
When decision D is marked `status="superseded"`, all active decisions that list D's ID in their `parents` should have their `parent_status` updated to `"parent_superseded"`. This ensures `compute_active_view` can distinguish orphaned children from legitimately active decisions.

### Root Cause
`_apply_supersession` in `store.py` only mutates the superseded decision itself — it never iterates the rest of the ledger to propagate `parent_status` to children. Compare with `reject_decision`, which correctly propagates `parent_status="parent_rejected"` to children.

### Fix
After marking the old decision superseded, propagate `parent_status` to all active children that reference the superseded ID in their `parents` list.

### Resolution Status
✅ Fixed in this commit

---

## BUG-004 — `_fetch_styles_rag` / `_fetch_tactics_rag` Use Process-Level `@lru_cache` That Never Invalidates

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/styles/common.py` (line 45), `back/src/graph/nodes/tactics/common.py` (line 43) |

### Observed Behavior
Both RAG functions are decorated with `@lru_cache(maxsize=64)`. If the ChromaDB is rebuilt (new PDFs added/removed), RAG results for already-cached keys remain stale until the process restarts.

### Expected Behavior
RAG grounding should reflect the current vector store state. Cache invalidation should trigger on vectorstore rebuild.

### Root Cause
`rebuild_vectorstore()` in `rag_agent.py` resets `_VDB` but does not call `.cache_clear()` on the cached RAG functions.

### Fix
Call `.cache_clear()` on both cached functions inside `rebuild_vectorstore()`.

### Resolution Status
✅ Fixed in this commit

---

---

## BUG-021 — `"tech"` kind absent from `_VALID_KINDS` — tech decisions silently dropped

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/ledger/validate.py` |
| **Lines** | `_VALID_KINDS` (line 12), `_REQUIRED_PAYLOAD_KEYS` (line 14), `_REQUIRED_PARENT_KINDS` (line 25) |

### Observed Behavior
`tech_node_impl` appends a decision of `kind: "tech"` to the ledger. No visible error is raised and the node continues normally. However, the tech decision is **never actually written to the ledger**.

### Expected Behavior
`validate_decision` should accept `kind: "tech"` (it is listed in `DecisionKind` in `types.py`) and persist the tech selection. The design dossier should reflect the chosen technologies so that on re-login the full decision chain — ASR → Style → Tactic → Tech — is visible.

### Root Cause
`_VALID_KINDS = {"asr", "style", "tactic", "diagram", "analysis", "constraint"}` did not include `"tech"`. When `validate_decision` checked `if kind not in _VALID_KINDS`, it raised `LedgerValidationError("Unknown kind: 'tech'")`. This exception was caught silently in `tech_node_impl`'s `except LedgerValidationError` block with only a `_tech_log.warning(...)`.

`_REQUIRED_PARENT_KINDS` also lacked a `"tech"` key — `validate_parents` fell back to `[]` via `.get("tech", [])`.

### Fix
Added `"tech"` to `_VALID_KINDS`, added `"tech": {"items"}` to `_REQUIRED_PAYLOAD_KEYS`, added `"tech": []` to `_REQUIRED_PARENT_KINDS`.

### Resolution Status
✅ Fixed — `validate.py` updated on branch `fix/SyncSemana15`

---

## BUG-022 — `style_candidates` state field never populated by `style_node_impl`

| Field | Details |
|---|---|
| **Status** | 🟡 Pending |
| **File** | `back/src/graph/nodes/styles/common.py` (`style_node_impl`) |
| **State field** | `GraphState.style_candidates: list[dict]` |

### Observed Behavior
After the Styles table is generated (S1/S2), `state["style_candidates"]` remains `[]`. Only `state["style"]`, `state["selected_style"]`, and `state["last_style"]` are written.

### Expected Behavior
`style_candidates` should be populated with the two candidate style dicts (name, justification, tradeoff, id keys) so that: (a) the design dossier can render them, (b) any future "which styles were considered?" query can be answered from state without re-querying the LLM, (c) consistency with `asr_candidates` / `tactics_candidates` data model.

### Root Cause
`style_node_impl` was implemented before the `style_candidates` field was added to `GraphState` (introduced as part of the ADD 3.0 candidates/selections model for multi-turn confirmation). The node was never updated to populate it.

### Resolution Status
✅ Fixed — `style_node_impl` updated to write `state["style_candidates"]` on branch `fix/SyncSemana15`

---

## BUG-023 — `@lru_cache` on RAG fetch functions — stale results across sessions / after vectorstore rebuild

| Field | Details |
|---|---|
| **Status** | 🟡 Pending (out of scope for simulation sprint unless triggered) |
| **Files** | `back/src/graph/nodes/styles/common.py:46`, `back/src/graph/nodes/tactics/common.py:43`, `back/src/graph/nodes/tech/common.py:31` |

### Observed Behavior
If the server starts with an empty or partially-built ChromaDB, the first call to `_fetch_styles_rag(qa, qa, k=6)` returns empty snippets. This result is cached permanently in the process. All subsequent calls with the same key return the same empty string — even after `rebuild_vectorstore()` resets `_VDB`. The user sees "GROUNDING: (none)" in every generation for the rest of the process lifetime.

### Expected Behavior
RAG fetch functions should always query the live vectorstore. At minimum, empty results should not be cached.

### Root Cause
`@lru_cache(maxsize=64)` is a process-level, time-unlimited cache. `rebuild_vectorstore()` resets `_VDB = None` but does not call `.cache_clear()` on the cached RAG functions in style/tactics/tech nodes.

### Resolution Status
✅ Fixed — `rebuild_vectorstore()` in `rag_agent.py` now calls `_fetch_tech_rag.cache_clear()` alongside the existing style/tactics clears, on branch `fix/SyncSemana15`

---

## BUG-024 — `_fetch_styles_rag` always called with `resolved_index == qa` — redundant cache key slot

| Field | Details |
|---|---|
| **Status** | 🟡 Pending (cosmetic — low priority) |
| **File** | `back/src/graph/nodes/styles/common.py:310` |

### Observed Behavior
`_fetch_styles_rag(qa, qa, k=6)` always passes the same value as both `qa` and `resolved_index`. The cache key `(qa, qa, 6)` is always redundant — the second parameter provides no additional discrimination.

### Expected Behavior
Either simplify the function signature to `_fetch_styles_rag(qa, k)`, or call with `_fetch_styles_rag(qa, state.get("resolved_index") or qa, k=6)` to allow separate caching when the classifier resolved a different index.

### Root Cause
`resolve_qa_for_style()` already incorporates `resolved_index` into the returned `qa`. The caller passes the fully-resolved value as both args. The second arg was kept to allow filtering by raw classifier output but was never wired up.

### Resolution Status
✅ Fixed — call site changed to `_fetch_styles_rag(qa, state.get("resolved_index") or qa, k=6)` on branch `fix/SyncSemana15`

---

## BUG-025 — Stale LangGraph checkpoint bypasses intake — M1 gate fires on new project intro

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **Files** | `back/src/graph/nodes/supervisor.py`, `back/src/graph/nodes/context_loader.py`, `back/src/graph/state.py` |
| **Trigger** | User sends a fresh project introduction while LangGraph checkpoint has `current_phase = "style_table"` from a prior session |

### Observed Behavior
User sends "Hola. Quiero diseñar la arquitectura de un sistema de pagos..." — a long project description containing "stack tecnológico actual". Archia responds with the M1 phase gate message: "Estamos en la fase de selección de estilo. Para llegar a propuesta de tecnologías primero necesitamos completar la fase actual." Intake never runs.

### Expected Behavior
The system detects a new project introduction and routes to `intake_node`, showing the ADD 3.0 introduction and first diagnostic question.

### Root Cause
Three-layer failure:
1. `context_loader` reloads the persisted SQLite ledger on every turn → `current_phase = "style_table"` (prior session) overrides state
2. `classifier` runs the full pipeline (bypassed fast intake-path because phase is not "intro"/"diagnosis"). The phrase `"stack tecnológico actual"` matches `tech_triggers` → `intent = "tech"`
3. `supervisor` M1 gate: `FUNNEL_INTENT_MIN_PHASE["tech"] = "tech_proposals"`, `PHASE_INT["style_table"] = 3 < 5` → gate fires, blocking message generated

No heuristic existed to detect "user is starting a new project" vs "user is continuing mid-session".

### Fix
- `state.py`: added `new_project_flow: bool` to `GraphState`
- `supervisor.py`: added `_is_new_project_intro()` heuristic (greeting + design intent + ≥20 words). Before the M1 gate, if `current_phase > "diagnosis"` and heuristic fires: resets all ADD 3.0 state, sets `new_project_flow=True`, routes to intake
- `context_loader.py`: when `new_project_flow=True` and ledger phase is beyond "diagnosis", skips `current_phase` override and `_mirror_legacy` call so stale decisions don't bleed into the new intake flow

**Limitation:** Prior-session SQLite ledger is never reset; ledger write-backs during the new project flow fail silently. State-level ADD 3.0 progression works; ledger persistence for the new project does not. A `reset_ledger_for_new_project()` API endpoint is needed for a complete fix (future work).

### Resolution Status
✅ Fixed on branch `fix/SyncSemana15`

---

## BUG-026 — `style_node_impl` never calls `transition_phase` — ledger stuck at `style_table` forever

| Field | Details |
|---|---|
| **Status** | 🔴 Pending |
| **File** | `back/src/graph/nodes/styles/common.py` |
| **Line** | `append_decision` call at line 448; missing `transition_phase` call after it |

### Observed Behavior
The style node generates the S1/S2 table and writes a `kind: "style"` decision to the ledger via `append_decision`. On the next turn the user selects "S1" and requests tactics. Archia responds with the M1 gate message:

> "Estamos en la fase de selección de estilo. Para llegar a selección de tácticas primero necesitamos completar la fase actual."

The session is permanently blocked from advancing to tactics.

### Expected Behavior
After successfully writing the `kind: "style"` ledger decision, the ledger phase should advance from `style_table` to `tactics_table`. Subsequent turns with `intent = "tactics"` should pass the M1 gate and reach the tactics node.

### Root Cause
`style_node_impl` calls `append_decision` (line 448) but never calls `transition_phase`. The ledger's `current_phase` field is only updated via explicit `transition_phase` calls — writing a decision does not advance the phase automatically. `context_loader` always reads `current_phase` directly from `ledger["current_phase"]`, so after `style_node_impl` runs:

- `ledger["current_phase"]` stays `"style_table"`
- Every turn: `context_loader` restores `current_phase = "style_table"` into state
- Every turn: `FUNNEL_INTENT_MIN_PHASE["tactics"] = "tactics_table"`, `PHASE_INT["style_table"] = 3 < 4` → M1 gate fires

Contrast with `intake_node`, which explicitly calls `transition_phase` from `"diagnosis"` to `"asr_table"` in its A2 branch. The style node has no equivalent call.

Also note: the style node asks `"¿Cuál estilo quieres usar? Indícame el ID."` (line 503) as if the user must confirm — but there is no `style_confirm` node, no classifier intent for style confirmation, and no phase-advance triggered by the user's selection. The auto-selected `best_style` is already written to the ledger before the question is even shown.

### Resolution Status
✅ Fixed — `style_node_impl` now calls `transition_phase("style_table" → "tactics_table")` after `append_decision` succeeds; same fix applied proactively to `tactics_node_impl` (`"tactics_table" → "tech_proposals"`). Both guarded with phase equality check for idempotency. Branch `fix/SyncSemana15`.

---

## BUG-027 — `intake_node` Step A1 fires generic question ignoring rich context already in first user message

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/intake_node.py` (M6 intro handler) |
| **Trigger** | User's first message already contains system name, QAs, baseline, constraints — intake node ignores all of it |

### Observed Behavior
User sends a complete first message containing:
- System name and purpose (payment system for high-demand e-commerce)
- Current stack (Node.js 18 monolith, PostgreSQL 15, Redis)
- Quality attributes: checkout p99 > 5s at peak, full crashes during traffic spikes
- Quality targets: p99 < 800ms, 99.95% availability
- Normal operation baseline: 1,200 TPS, 500k DAU
- Peak conditions: 3,000 TPS (Black Friday)
- Constraints: 12 engineers, $8,000/month cloud, PCI-DSS

Archia responds with the ADD 3.0 intro (correct — Phase 0) and then immediately asks:

> "¿Cuál es el requerimiento principal del sistema que deseas diseñar? Describe el objetivo principal, los componentes involucrados y las expectativas de calidad."

This is a verbatim re-ask of everything the user just provided.

### Expected Behavior
Per `ArchIA_Flujo_Interaccion.md`, Phase 1 — DIAGNÓSTICO, rule "Regla de preguntas inteligentes":

> ArchIA no repregunta información que el usuario ya proporcionó.

The intake node must:
1. Parse the incoming `userQuestion` and map each extractable field to the corresponding `GraphState` slot.
2. Identify only what is genuinely missing (in this case: stakeholders — not explicitly named).
3. Ask at most one follow-up question about the missing field.

### Root Cause
`intake_node` appears to be structured as a scripted A1→A2→… step machine. Step A1 unconditionally emits its generic "describe your system" question without first checking whether `userQuestion` already satisfies the required intake fields (`system_name`, `quality_attribute`, `normal_operation_baseline`, `constraints`). The `userQuestion` is treated as turn-level raw input only after A1 fires — the initial message is never pre-processed for field extraction before the A1 question is generated.

### Impact
- Every session that starts with a detailed description wastes at least one round-trip.
- The spec explicitly lists this anti-pattern as a disqualifying behavior.
- The intake node's A1 → A2 internal state machine must be preceded by a pre-extraction pass over `userQuestion`.

### Resolution Status
✅ Fixed — M6 intro handler in `intake_node.py` now calls `_process_intake_turn` on the user's first message and skips to the first MISSING required field instead of always asking campo_0.

---

## BUG-028 — `intake_node` asks ASR 6-part field ("fuente del estímulo") during DIAGNOSIS phase

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/intake_validators.py` (INTAKE_SCRIPT), `back/src/graph/nodes/intake_node.py` (`current_index` calculation) |
| **Trigger** | After user provides system context, intake asks "indica si proviene de: usuario/paciente/médico, sistema externo, evento interno, tiempo/timer/cron" |

### Observed Behavior (confirmed across 2 turns)
**Turn 1:** After user re-confirms system context, intake responds:
> "→ fuente del estímulo: No se identificó la fuente del estímulo. Indica si proviene de: usuario/paciente/médico, sistema externo/API externa, evento interno, tiempo/timer/cron."

**Turn 2:** After user answers "usuario final que inicia el flujo de checkout", intake responds:
> "→ estímulo: Se ha nombrado el trigger [...] falta indicar qué componente del sistema recibe o procesa ese evento. Por favor reenvíe la especificación COMPLETA del campo 'campo_3_estimulo'."

Pattern confirmed: **intake is iterating through ASR 6-part fields one-by-one across multiple turns**, treating the user as the source of ASR generation rather than generating ASRs from context. This will continue for all 6 fields (stimulus source → stimulus → environment → artifact → response → response measure) before leaving DIAGNOSIS.

"Fuente del estímulo" and "campo_3_estimulo" are the 1st and 2nd fields of the ASR 6-part format. They belong exclusively to Phase 2 — ASR_TABLE, generated by the ASR node from the context already collected in DIAGNOSIS.

Additionally: "usuario/paciente/médico" is a healthcare-domain example hardcoded in the prompt — inappropriate for a payment system.

Note: user also had to send their message twice in this turn, suggesting the first submission may have been silently dropped or returned a blank response (possible separate bug — insufficient data to log separately yet).

### Expected Behavior
During DIAGNOSIS, intake collects only: system name, QAs, baseline, constraints, stakeholders. It does NOT drill into ASR sub-fields. The 6-part ASR format is generated by the ASR node in Phase 2, not by the intake node in Phase 1.

If the intake prompt includes an ASR template, it is leaking Phase 2 logic into Phase 1.

### Root Cause
The intake node's prompt template appears to include ASR 6-part field validation as part of its field-completeness check, iterating field-by-field across turns. Either:
(a) The intake prompt was erroneously extended with ASR field prompts, or
(b) A shared "missing fields" template is used by both intake and ASR nodes, and the wrong one is being invoked.

Also: hardcoded "usuario/paciente/médico" in the options list is medical domain bleed that fires in all non-healthcare sessions.

### Resolution Status
✅ Fixed — `INTAKE_SCRIPT` entries at index 2 (`campo_2_fuente`) and 3 (`campo_3_estimulo`) marked `"optional": True`; `current_index` calculation updated to skip optional fields so intake completes on the 6 required fields only.

---

## BUG-030 — Intake enforces arbitrary 8-word minimum on user answers — rejects technically complete responses

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/intake_validators.py` (`validate_field`, `_REPROMPT_ERRORS`) |
| **Trigger** | User answers "Checkout API (interacción síncrona)." — 4 words, fully complete for the question asked |

### Observed Behavior
Archia rejects the answer with:
> "La respuesta es demasiado corta. Necesita al menos 8 palabras con vocabulario técnico concreto (servicio, módulo, API, componente, etc.)."

Then re-asks the same question and again instructs to "reenviar la especificación COMPLETA del campo".

### Expected Behavior
Input validation for ASR fields must be semantic, not word-count-based. "Checkout API (interacción síncrona)" is a complete, precise, technically correct answer to "which component receives the stimulus". A minimum word count enforces verbosity over precision — the opposite of good architecture documentation.

Additionally, this validation is being applied to ASR fields during DIAGNOSIS — a phase where these fields should not be asked at all (BUG-028). So the word count validator compounds an already-wrong behavior.

### Root Cause
The intake prompt or a pre-LLM validation layer is applying a heuristic `len(words) >= 8` check to user answers before passing them to the LLM. This was likely intended to catch empty or trivially short answers ("yes", "no", "API") but it fires on technically complete short answers like component names.

### Resolution Status
✅ Fixed — `validate_field` index 3 (`campo_3_estimulo`) now requires ≥5 tokens instead of ≥8. `_REPROMPT_ERRORS[3]` updated to reflect the new threshold.

---

## BUG-031 — Frontend shows indefinite loading state; messages require double-send to process

| Field | Details |
|---|---|
| **Status** | 🔴 Pending |
| **File** | Frontend (WebSocket / streaming layer) |
| **Trigger** | Observed on multiple turns during intake flow |

### Observed Behavior
After sending a message, the frontend enters a loading state and stays there for an extended period (visible as a spinner/blank area). In multiple turns the user had to send the same message twice to get a response — the second submission triggers the reply.

Screenshot shows two identical user bubbles ("Checkout API (interacción síncrona).") sent in sequence, confirming the pattern.

### Expected Behavior
Each submitted message should be processed exactly once. The frontend should show a clear streaming progress indicator and not require duplicate submissions.

### Root Cause (hypothesis)
Possible causes: (a) WebSocket connection drops silently and the frontend doesn't retry automatically, (b) the intake node's extended processing time (iterating through ASR fields via LLM calls) exceeds a frontend timeout threshold, causing the UI to reset to idle while the backend is still processing, (c) the first message is consumed but the response is dropped before reaching the frontend.

### Resolution Status
🔴 Pending

---

## BUG-034 — Supervisor→ASR→Supervisor→Unifier infinite loop — response never delivered

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/asr.py` (re-render early return, lines 306–320) |
| **Severity** | 🔴 Critical — user receives no response; server loops until timeout or restart |

### Observed Behavior
Backend log shows the following cycle repeating at least 4 times without termination:
```
supervisor: current_phase='asr_table' intent='asr' nextNode='asr'
asr_node: re-rendering existing ASR (no explicit request to change)
supervisor: current_phase='asr_table' intent='asr' nextNode='unifier'
asr_node: re-rendering existing ASR (no explicit request to change)
supervisor: current_phase='asr_table' intent='asr' nextNode='unifier'
...
```

Note the anomalous first supervisor call: `nextNode='supervisor'` (self-loop) before the main cycle begins. Frontend shows blank loading state; user had to send the message 3 times with no result.

### Expected Behavior
The graph should flow: supervisor → asr → unifier → END. After the unifier runs, the graph must terminate and stream the response to the frontend.

### Root Cause (hypothesis)
Two plausible causes:
1. **Unifier does not signal END** — after `unifier` runs, the LangGraph conditional edge function evaluates some completion condition (e.g., `endMessage` empty, `completed_nodes` mismatch) and routes back to supervisor instead of END. Supervisor still sees `intent='asr'` and re-routes to asr.
2. **ASR node routing** — `asr_node` routes to supervisor (not directly to unifier), and the supervisor's `nextNode='unifier'` is a second pass deciding to skip the asr execution, but the graph's edge function still calls asr again before reaching unifier.

The "re-rendering existing ASR" message on every iteration confirms the node is running fresh each time rather than being cached.

### Resolution Status
✅ Fixed — re-render early return in `asr.py` now sets `"hasVisitedASR": True` and appends `"asr"` to `completed_nodes`, preventing the router from re-firing the asr branch on subsequent supervisor calls.

---

## BUG-035 — ASR node re-renders stale ASR from prior session's LangGraph checkpoint

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/context_loader.py` (lines 160–169) |
| **Severity** | 🔴 Critical — new project silently inherits old session's ASR decisions |

### Observed Behavior
Backend log: `context_loader: ledger hydrated phase=style_table decisions=3` — the SQLite design ledger has 3 decisions from a prior session (ASR + style + tactics). Simultaneously, asr_node logs `re-rendering existing ASR (no explicit request to change)`, meaning it detected an existing ASR and chose to re-display it rather than generate a new one.

### Expected Behavior
When `new_project_flow=True`, the asr_node should generate fresh ASR candidates from the new intake context. It must not detect or re-render ASR decisions from the prior session.

### Root Cause
The `new_project_flow` fix in context_loader sets `ledger_active={}` (empty) and skips `_mirror_legacy`. However, asr_node's "existing ASR" detection does NOT read from `ledger_active` — it reads from LangGraph checkpoint state fields such as `state["asr"]`, `state["selected_asrs"]`, or `state["asr_candidates"]`. These fields are stored in the LangGraph SQLite checkpoint and **survive across sessions** because LangGraph's checkpointer restores the full prior state on every turn. The `new_project_flow` mechanism never clears these checkpoint state fields.

### Fix Direction
When supervisor detects `new_project_flow=True` and resets ADD 3.0 scalars, it must explicitly clear all ASR/style/tactics/tech state fields in the returned state dict so the LangGraph checkpoint snapshot is overwritten with empty values before the asr_node runs.

### Resolution Status
✅ Fixed — `context_loader.py` guard changed from `if _new_project_flow and mapped_phase not in ("intro", "diagnosis"):` to `if _new_project_flow:`, unconditionally skipping `_mirror_legacy` and stale ledger restoration whenever a new project flow is active.

---

## BUG-033 — After DIAGNOSIS completes, intake asks permission to generate ASRs instead of auto-transitioning

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/intake_node.py` (Rama B, `current_index >= 8` block) |
| **Severity** | 🟢 Low — wastes one turn, does not block the flow |

### Observed Behavior
After saving the final intake field ("Guardé: ambientes"), Archia asks:
> "Ya tengo toda la información necesaria. ¿Quieres que proponga los ASRs o ya tienes alguno definido?"

### Expected Behavior
Per `ArchIA_Flujo_Interaccion.md`, the transition criterion from DIAGNOSIS to ASR_TABLE is `normal_operation_baseline` populated. Once met, ArchIA generates and presents the ASR candidate table automatically — it does not ask for permission. The "ya tienes alguno definido?" option is not in the spec.

### Resolution Status
✅ Fixed — Rama B in `intake_node.py` now auto-advances: saves `intake_v1` to the ledger, calls `transition_phase("diagnosis"→"asr_table")`, sets `nextNode="asr"`, and routes directly to ASR generation in the same turn.

---

## BUG-032 — Intake demands maintenance windows + RTO/RPO during DIAGNOSIS — misrepresents ADD 3.0 methodology

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/intake_validators.py` (`_ADD3_CRITERIA[4]`, `_REPROMPT_ERRORS[4]`, `build_repair_prompt` template 4) |
| **Trigger** | After user provides normal + peak conditions with concrete metrics, intake rejects response as incomplete |

### Observed Behavior
Intake rejects the environment answer (1,200 TPS normal / 3,000 TPS peak / p99 < 800ms / 99.95% availability) because it lacks:
- Maintenance window dates/times and duration
- Throughput permitted during maintenance
- Downtime percentage during maintenance
- RTO and RPO values

And again appends: "No envíe sólo la parte que falta: reenvíe todo el campo con las tres condiciones completas."

### Expected Behavior
During ADD 3.0 DIAGNOSIS, the intake collects system context: what the system does, who uses it, what problems exist, and what the normal operation baseline is. RTO and RPO are architectural quality requirements that **emerge from** the ADD 3.0 process — they are outputs of the ASR → style → tactics chain, not inputs to diagnosis. Requiring the user to supply RTO/RPO at intake time inverts the methodology.

Maintenance window specifications are operational constraints that may be relevant, but requiring exact numeric values (throughput during maintenance, downtime %) at intake stage is over-specified. The intake spec (`ArchIA_Flujo_Interaccion.md`) lists `normal_operation_baseline` as the required field — not a 3-condition environment matrix with recovery objectives.

### Root Cause
The intake environment prompt was written to the granularity of an ASR scenario — collecting all 3 condition types (normal/peak/maintenance) with full numeric specs — rather than the simpler "what is normal operation today?" that the spec requires. RTO/RPO requirements were added without methodological grounding.

### Resolution Status
✅ Fixed — `_ADD3_CRITERIA[4]` now requires only TWO conditions (normal + overload/peak); maintenance is explicitly optional. `_REPROMPT_ERRORS[4]` and `build_repair_prompt` template 4 updated accordingly.

---

## BUG-029 — Intake instructs user to re-send full specification already provided — violates no-repeguntar rule

| Field | Details |
|---|---|
| **Status** | ✅ Fixed |
| **File** | `back/src/graph/nodes/intake_validators.py` (`build_repair_prompt`, `_UNIFIED_PROMPT_V2`) |
| **Trigger** | Same turn as BUG-028 |

### Observed Behavior
Archia tells the user:

> "Reescribe tu respuesta indicando quién genera el estímulo y su contexto en tu sistema [...] Por favor, reenvía la especificación COMPLETA del campo, incluyendo las partes que ya enviaste en mensajes anteriores."

This is the spec's anti-pattern example made literal: asking the user to resend already-provided content.

### Expected Behavior
Archia identifies ONLY the missing field and asks a single, targeted question for it. It never asks the user to repeat already-provided information.

### Root Cause
The intake prompt template, when requesting missing field data, appends a boilerplate instruction to "reenviar la especificación COMPLETA". This instruction was likely written assuming the user had only provided a partial spec and needed to complete it — but it fires unconditionally even after the user has already provided everything, making the user re-do work already done.

### Resolution Status
✅ Fixed — `_es_reenvia`/`_en_resend` variables and their string concatenation removed from all `build_repair_prompt` templates. `_UNIFIED_PROMPT_V2` instruction #6 rewritten to say "write a concise repair prompt telling the user what is missing — do NOT ask them to re-send what they already provided".

---

## Summary Table

| ID | File | Severity | Status |
|---|---|---|---|
| BUG-001 | `styles/common.py` | 🔴 High — style→tactics binding broken | ✅ Fixed |
| BUG-002 | `supervisor.py` | 🔴 High — session state corruption on phase gate | ✅ Fixed |
| BUG-003 | `ledger/store.py` | 🔴 High — stale ASR→style chain after regeneration | ✅ Fixed |
| BUG-004 | `styles/common.py`, `tactics/common.py` | 🟡 Low — RAG cache never invalidated | ✅ Fixed |
| BUG-021 | `ledger/validate.py` | 🔴 Critical — tech decisions silently dropped | ✅ Fixed |
| BUG-022 | `styles/common.py` | 🟡 Medium — `style_candidates` never populated | ✅ Fixed |
| BUG-023 | `rag_agent.py` | 🟡 Medium — `_fetch_tech_rag` not cleared on vectorstore rebuild | ✅ Fixed |
| BUG-024 | `styles/common.py` | 🟢 Low — redundant cache key in `_fetch_styles_rag` | ✅ Fixed |
| BUG-025 | `supervisor.py`, `context_loader.py`, `state.py` | 🔴 Critical — new project intro hijacked by M1 gate | ✅ Fixed |
| BUG-026 | `styles/common.py`, `tactics/common.py` | 🔴 Critical — phase stuck at `style_table` after style written; tactics permanently blocked | ✅ Fixed |
| BUG-027 | `intake_node.py` | 🔴 High — intake A1 fires generic re-ask question ignoring rich context already in first message | ✅ Fixed |
| BUG-028 | `intake_validators.py` | 🔴 High — intake node asks ASR 6-part field ("fuente del estímulo") during DIAGNOSIS phase — wrong phase for this information | ✅ Fixed |
| BUG-029 | `intake_validators.py` | 🔴 High — intake instructs user to "reenviar la especificación COMPLETA incluyendo partes ya enviadas" — direct violation of no-repeguntar rule | ✅ Fixed |
| BUG-030 | `intake_validators.py` | 🔴 High — intake enforces arbitrary 8-word minimum on user answers; rejects technically complete short answers | ✅ Fixed |
| BUG-031 | Frontend / WebSocket | 🟡 Medium — frontend shows indefinite loading state; messages sometimes require double-send to process | 🔴 Pending |
| BUG-032 | `intake_validators.py` | 🔴 High — intake demands maintenance-window + RTO/RPO during DIAGNOSIS, misrepresenting ADD 3.0 methodology | ✅ Fixed |
| BUG-033 | `intake_node.py` | 🟢 Low — after DIAGNOSIS completes, intake asks permission to generate ASRs instead of auto-transitioning to ASR_TABLE | ✅ Fixed |
| BUG-034 | `graph/nodes/asr.py` | 🔴 Critical — supervisor→asr→supervisor→unifier loop cycles indefinitely; response never delivered to frontend | ✅ Fixed |
| BUG-035 | `graph/nodes/context_loader.py` | 🔴 Critical — asr_node finds stale ASR in LangGraph checkpoint from prior session; re-renders old ASR instead of generating new one | ✅ Fixed |
