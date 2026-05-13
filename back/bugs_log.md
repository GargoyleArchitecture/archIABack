# Archia — Bug Log · Branch: fix/SyncSemana15

> **Protocol**: Every deviation from expected ADD 3.0 flow is logged HERE before any code is written.
> Format: Observed → Expected → Root Cause → Resolution Status.
> This file is the primary todo list for this branch.

---

## BUG-021 · `"tech"` kind absent from `_VALID_KINDS` — tech decisions silently dropped

| Field | Detail |
|---|---|
| **Severity** | Critical — silent data loss |
| **File** | `archIABack/back/src/ledger/validate.py:13` |
| **Also affects** | `archIABack/back/src/ledger/validate.py:24` (`_REQUIRED_PARENT_KINDS`) |

**Observed Behavior:**
`tech_node_impl` appends a decision of `kind: "tech"` to the ledger. The call never raises a visible error to the user and the node continues normally. However, the tech decision is **never actually written to the ledger**.

**Expected Behavior:**
`validate_decision` should accept `kind: "tech"` (it is listed in `DecisionKind` in `types.py`) and persist the tech selection. The design dossier should reflect the chosen technologies so that on re-login the full decision chain — ASR → Style → Tactic → Tech — is visible.

**Root Cause:**
`_VALID_KINDS = {"asr", "style", "tactic", "diagram", "analysis", "constraint"}` does not include `"tech"`. When `validate_decision` checks `if kind not in _VALID_KINDS`, it raises `LedgerValidationError("Unknown kind: 'tech'")`. This exception is caught silently in `tech_node_impl`'s `except LedgerValidationError` block with only a `_tech_log.warning(...)` — the caller never sees the failure.

`_REQUIRED_PARENT_KINDS` also lacks a `"tech"` key. `validate_parents` would fall back to `[]` (no required parents) via `.get("tech", [])`, which is actually acceptable given tech is the terminal node. But the key should be added for clarity: `["asr", "style", "tactic"]`.

**Resolution Status:** `Fixed — validate.py: added "tech" to _VALID_KINDS and _REQUIRED_PAYLOAD_KEYS / _REQUIRED_PARENT_KINDS`

---

## BUG-022 · `style_candidates` state field never populated by `style_node_impl`

| Field | Detail |
|---|---|
| **Severity** | Medium — state model inconsistency |
| **File** | `archIABack/back/src/graph/nodes/styles/common.py` (`style_node_impl`) |
| **State field** | `GraphState.style_candidates: list[dict]` |

**Observed Behavior:**
After the Styles table is generated (S1/S2), `state["style_candidates"]` remains `[]` or whatever boot_node initialized it to. Only `state["style"]`, `state["selected_style"]`, and `state["last_style"]` are written.

**Expected Behavior:**
`style_candidates` should be populated with the two candidate style dicts (name, justification, tradeoff, id keys) so that: (a) the design dossier can render them, (b) any future "which styles were considered?" query can be answered from state without re-querying the LLM, (c) consistency with `asr_candidates` / `tactics_candidates` data model.

**Root Cause:**
`style_node_impl` was implemented before the `style_candidates` field was added to `GraphState` (the field was introduced as part of the ADD 3.0 candidates/selections model for multi-turn confirmation). The node was never updated to populate it.

**Resolution Status:** `Pending`

---

## BUG-023 · `@lru_cache` on RAG fetch functions — stale results across sessions / after vectorstore rebuild

| Field | Detail |
|---|---|
| **Severity** | Medium — degraded RAG quality in long-running processes |
| **Files** | `styles/common.py:46`, `tactics/common.py:43`, `tech/common.py:31` |

**Observed Behavior:**
If the server starts with an empty or partially-built ChromaDB (e.g., PDFs not yet indexed), the first call to `_fetch_styles_rag(qa, qa, k=6)` returns empty snippets. This result is cached permanently in the process. All subsequent calls with the same `(qa, qa, 6)` key return the same empty string — even after `rebuild_vectorstore()` resets `_VDB`. The user sees "GROUNDING: (none)" in every style/tactics/tech generation for the rest of the process lifetime.

**Expected Behavior:**
RAG fetch functions should always query the live vectorstore. Caching is acceptable for performance, but the cache must be invalidated when `_VDB` is rebuilt or when the cache TTL expires. At minimum, the empty-result case should not be cached.

**Root Cause:**
`@lru_cache(maxsize=64)` is a process-level, time-unlimited cache. It captures the ChromaDB query result at function call time. `rebuild_vectorstore()` in `rag_agent.py` resets `_VDB = None` (so new queries use the fresh store), but the LRU cache in style/tactics/tech nodes is independent — it holds stale Document-level results, not references to `_VDB`.

**Resolution Status:** `Pending` (requires cache invalidation strategy — out of scope for this simulation sprint unless triggered)

---

## BUG-024 · `_fetch_styles_rag` always called with `resolved_index == qa` — redundant cache key slot

| Field | Detail |
|---|---|
| **Severity** | Low — API design smell, no functional impact |
| **File** | `archIABack/back/src/graph/nodes/styles/common.py:310` |

**Observed Behavior:**
`_fetch_styles_rag(qa, qa, k=6)` always passes the same value as both `qa` and `resolved_index`. The cache key `(qa, qa, 6)` is therefore always redundant — the second parameter provides no additional discrimination.

**Expected Behavior:**
Either (a) simplify the function signature to `_fetch_styles_rag(qa, k)` since `resolved_index` has no independent value when always identical to `qa`, or (b) call with `_fetch_styles_rag(qa, state.get("resolved_index") or qa, k=6)` to allow separate caching when the classifier resolved a different index than the QA override.

**Root Cause:**
`resolve_qa_for_style()` already incorporates `resolved_index` into the returned `qa`. The caller passes the fully-resolved value as both args. The second arg was probably kept to allow filtering by the raw classifier output but was never wired up.

**Resolution Status:** `Pending` (cosmetic fix — low priority)

---

## Simulation Trace

| Turn | User Input (fed to Archia) | Observed | Expected | Δ |
|---|---|---|---|---|
| — | — | — | — | Logging begins after first Archia response |

---

*Last updated: 2026-05-13 · by Shadow Debugger on branch fix/SyncSemana15*
