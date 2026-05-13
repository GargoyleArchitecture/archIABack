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
