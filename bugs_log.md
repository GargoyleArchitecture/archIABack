# Archia Bug Log — Branch: claude/debug-archia-assistant-Jm1or

> **Protocol**: Every deviation from the expected ADD 3.0 flow or technical error in the LangGraph state is logged here before any code is written. This file is the primary "todo list" for this debugging session.

---

## BUG-001 — `style_payload.tradeoffs` Always Empty: Missing `rationale` Key in LLM JSON Schema

| Field | Details |
|---|---|
| **Status** | 🔴 Pending |
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

### Fix (planned)
In `style_node_impl`, derive `rationale` from the **chosen style's own `tradeoff` field**:
```python
best_key = (data.get("best_style") or "").strip()
_chosen_data = style2 if best_key == "style_2" else style1
rationale = _chosen_data.get("tradeoff", "").strip()
```
Pass this corrected `rationale` to `_build_style_payload`.

---

## BUG-002 — M1 Phase Gate Resets `completed_nodes = []`, Breaking Session Continuity

| Field | Details |
|---|---|
| **Status** | 🔴 Pending |
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

### Fix (planned)
Replace `"completed_nodes": []` with:
```python
"completed_nodes": _augment_completed_nodes(state, list(state.get("completed_nodes") or [])),
```

---

## BUG-003 — `_apply_supersession` Does Not Propagate `parent_status="parent_superseded"` to Children

| Field | Details |
|---|---|
| **Status** | 🔴 Pending |
| **File** | `back/src/ledger/store.py` |
| **Lines** | `_apply_supersession` (lines 62–73), `append_decision` (lines 159–193) |

### Observed Behavior
After the user regenerates an ASR (ASR-2 supersedes ASR-1), the old style and tactics decisions that were children of ASR-1 remain with `status="active"` and `parent_status="ok"`. `compute_active_view` iterates all `status="active"` decisions and returns the last per kind, so it may return **Style-1** (linked to the superseded ASR-1) as the active style even though no new style has been generated yet for ASR-2.

Result: the tactics node gets a binding block that cites ASR-1's response measure together with ASR-2's scenario — a logically inconsistent design chain.

### Expected Behavior
When decision D is marked `status="superseded"`, all active decisions that list D's ID in their `parents` should have their `parent_status` updated to `"parent_superseded"`. This ensures `compute_active_view` can distinguish orphaned children from legitimately active decisions.

### Root Cause
`_apply_supersession` in `store.py` only mutates the superseded decision itself — it never iterates the rest of the ledger to propagate `parent_status` to children:
```python
def _apply_supersession(ledger, new_decision):
    for d in ledger["decisions"]:
        if d["kind"] == kind and d["status"] == "active":
            if existing_parent_ids == new_parent_ids:
                d["status"] = "superseded"
                d["superseded_by"] = new_decision["id"]
                return d["id"]
    return None
    # ← children of the now-superseded decision are never touched
```
Compare with `reject_decision`, which correctly propagates `parent_status="parent_rejected"` to children.

### Fix (planned)
After `_apply_supersession` marks a decision superseded, propagate `parent_status` to all active children:
```python
def _apply_supersession(ledger, new_decision):
    superseded_id = None
    for d in ledger["decisions"]:
        if d["kind"] == kind and d["status"] == "active":
            if existing_parent_ids == new_parent_ids:
                d["status"] = "superseded"
                d["superseded_by"] = new_decision["id"]
                superseded_id = d["id"]
                break
    # Propagate to children
    if superseded_id:
        for d in ledger["decisions"]:
            if d["status"] == "active":
                parent_ids = {r["id"] for r in (d.get("parents") or [])}
                if superseded_id in parent_ids:
                    d["parent_status"] = "parent_superseded"
    return superseded_id
```

---

## BUG-004 — `_fetch_styles_rag` / `_fetch_tactics_rag` Use Process-Level `@lru_cache` That Never Invalidates

| Field | Details |
|---|---|
| **Status** | 🟡 Low Priority / Design Issue |
| **File** | `back/src/graph/nodes/styles/common.py` (line 45), `back/src/graph/nodes/tactics/common.py` (line 43) |

### Observed Behavior
Both RAG functions are decorated with `@lru_cache(maxsize=64)`. Cache key is `(qa, resolved_index, k)`. If the ChromaDB is rebuilt (new PDFs added/removed), RAG results for already-cached keys remain stale until the process restarts.

### Expected Behavior
RAG grounding should reflect the current vector store state. Cache invalidation should trigger on vectorstore rebuild.

### Root Cause
Python `functools.lru_cache` has no TTL or manual invalidation hook. `rebuild_vectorstore()` in `rag_agent.py` resets `_VDB` but does not call `_fetch_styles_rag.cache_clear()` or `_fetch_tactics_rag.cache_clear()`.

### Fix (planned)
Call `.cache_clear()` on both cached functions inside `rebuild_vectorstore()`. For multi-project scenarios, consider keying the cache on `(project_id, qa, resolved_index, k)`.

---

## Summary Table

| ID | File | Severity | Status |
|---|---|---|---|
| BUG-001 | `styles/common.py` | 🔴 High — style→tactics binding broken | Pending |
| BUG-002 | `supervisor.py` | 🔴 High — session state corruption on phase gate | Pending |
| BUG-003 | `ledger/store.py` | 🔴 High — stale ASR→style chain after regeneration | Pending |
| BUG-004 | `styles/common.py`, `tactics/common.py` | 🟡 Low — RAG cache never invalidated | Pending |
