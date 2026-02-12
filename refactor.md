# Refactoring Summary: `back/src/graph` Package

## 1. Overview and Goal

The primary goal of this refactoring was to modularize the monolithic `back/src/graph.py` file into a clean, maintainable Python package structure. The original file contained over 2000 lines of mixed concerns (state definitions, node logic, graph wiring, constants, and utility functions), making it difficult to navigate, test, and extend.

The new structure adheres to **Separation of Concerns (SoC)** and follows standard Python packaging conventions, making the codebase more robust and scalable for future ADD 3.0 enhancements.

## 2. New Package Structure

The `back/src/graph.py` file has been replaced by the `back/src/graph/` directory. Here is the breakdown of the new files and their responsibilities:

### Core Modules

| File               | Responsibility                                                                                                                                                                                     |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`__init__.py`**  | Marks the directory as a Python package and exposes the compiled `graph` object for external use (e.g., by `main.py`).                                                                             |
| **`workflow.py`**  | Defines the LangGraph structure. It imports all nodes, defines the `StateGraph`, wires the nodes with edges (including conditional routing), and compiles the graph.                               |
| **`state.py`**     | Centralized definition of the `GraphState` TypedDict and other schemas (e.g., `supervisorSchema`, `evaluatorSchema`). This ensures a single source of truth for the graph's memory.                |
| **`consts.py`**    | Stores strict constants, regex patterns, and lengthy prompt templates (e.g., `MERMAID_SYSTEM`, `TACTICS_HEADINGS`). Removes clutter from logic files.                                              |
| **`utils.py`**     | Contains shared utility functions like token counting, text clipping (`_clip_text`), and JSON/Mermaid sanitization (`_sanitize_mermaid`, `extract_json_array`).                                    |
| **`resources.py`** | Manages shared resources like the LLM client, Retriever, and Checkpointer. **Crucially**, it now handles lazy loading of these resources to prevent startup crashes when `.env` hasn't loaded yet. |

### Node Modules (`back/src/graph/nodes/`)

Each major "agent" or processing step in the graph now has its own file in `nodes/`. This makes it easy to work on a specific agent without affecting others.

| File                  | Node Name                   | Responsibility                                                                                                                  |
| :-------------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **`classifier.py`**   | `classifier_node`           | Detects language (EN/ES) and high-level intent (Greeting vs. Architecture).                                                     |
| **`supervisor.py`**   | `supervisor_node`           | The router/orchestrator. Decides which specialized worker node to call next based on the user's request and conversation state. |
| **`investigator.py`** | `researcher_node`           | Performs RAG searches and synthesis for architectural questions ("What is a microkernel?").                                     |
| **`asr.py`**          | `asr_node`                  | Generates structured Quality Attribute Scenarios (ASRs) grounded in ADD 3.0.                                                    |
| **`style.py`**        | `style_node`                | Recommends architectural styles (e.g., Layered, Microservices) based on the ASR.                                                |
| **`tactics.py`**      | `tactics_node`              | Suggests specific architectural tactics (e.g., Circuit Breaker, Autoscaling) to satisfy the given ASR and Style.                |
| **`creator.py`**      | `creator_node`              | Generates the actual Mermaid.js diagram code.                                                                                   |
| **`diagram.py`**      | `diagram_orchestrator_node` | Prepares the context for diagram generation (a specialized orchestrator for the diagram flow).                                  |
| **`evaluator.py`**    | `evaluator_node`            | Critiques ASRs or diagrams using tools like "Theory", "Viability", and "Needs".                                                 |
| **`unifier.py`**      | `unifier_node`              | Synthesizes the final response to the user, combining outputs from different nodes into a coherent message.                     |
| **`tools.py`**        | (Various Tools)             | Defines LangChain `Example` tools (Functions) used by agents, such as `local_RAG` or `theory_tool`.                             |

## 3. Key Technical Improvements

### 1. Lazy Resource Loading & Environment Handling

**Problem:** The previous `graph.py` (and initial refactor) eagerly initialized the OpenAI client and Retriever at import time. This caused `uvicorn` to crash immediately on startup because it imported `graph` before loading the `.env` file containing `OPENAI_API_KEY`.
**Solution:**

- Moved resource initialization to `resources.py`.
- Explicitly loaded `.env` at the top of `resources.py`.
- Wrapped the Retriever in a **lazy proxy** (`_LazyRetriever`) so it only initializes on the first _function call_, well after the app has fully started.

### 2. Improved Checkpointing Strategy (SqliteSaver vs. MemorySaver)

**Problem:** The initial implementation attempted to use `SqliteSaver.from_conn_string("path/to/db.sqlite")`.

- **Technical Issue:** In the installed version of `langgraph`, `SqliteSaver.from_conn_string(...)` returns a **context manager** (an object meant to be used in a `with` statement), not a standalone checkpointer instance.
- **Consequence:** When this context manager object was passed to `graph.compile(checkpointer=saver)`, the graph compilation failed or behaved unpredictably at runtime because it expected a persistent object, not a context manager that closes its connection immediately. This led to errors like `AttributeError: __enter__` or invalid state persistence.
- **Why it was "bad":** Using a context manager incorrectly meant the database connection was never properly established or maintained for the lifespan of the application, causing state to be lost or the app to crash on the first request.

**Solution:**

- Switched to **`MemorySaver`** (`langgraph.checkpoint.memory.MemorySaver`).
- **Why it's an improvement:**
  1.  **Simplicity & Stability:** `MemorySaver` is a straightforward class that holds state in RAM. It doesn't require context management (`with` blocks), making it compatible with the global `graph` object pattern used in `workflow.py`.
  2.  **Reliability:** It eliminates the risk of "closed cursor" errors or file-locking contention (SQLite is file-based) during high-concurrency async operations in FastAPI.
  3.  **Correctness:** It matches the requirements of `StateGraph.compile()`, ensuring user conversation state is actually preserved during the session (though it resets on restart, which is often acceptable for ephemeral chat sessions).

_Note: The project still uses SQLite in other areas (e.g., `src/main.py` for storing user feedback), so SQLite is not removed entirely—only for the LangGraph execution state._

### 3. Circular Import Prevention

**Problem:** `evaluator.py` needed a utility `_looks_like_eval` that was defined in `supervisor.py`, but also imported `utils.py`.
**Solution:**

- Clarified the location of functions. `_looks_like_eval` remains in `supervisor.py` (where it belongs logically) and is imported explicitly by `evaluator.py`, avoiding circular dependencies through `utils.py`.

### 4. Explicit Import Exports

**Problem:** `utils.py` tried to import `TACTICS_ARRAY_SCHEMA` from `consts.py`, but it was actually in `state.py`.
**Solution:**

- Moved strict constants to `consts.py` and schema definitions to `state.py`.
- Fixed all imports to point to the correct source of truth.

## 4. How to Run

The application is run exactly as before, but now relies on the clean package structure:

```bash
# 1. Start the backend
cd back
poetry run uvicorn src.main:app --port 8000
```
