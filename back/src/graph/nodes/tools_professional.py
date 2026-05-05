"""Tools especificas del Modo Profesional (F2-T6).

- python_repl_tool: ejecuta codigo Python en subproceso con timeout 5s y
  AST guard que bloquea imports peligrosos. NO es un sandbox completo;
  ver limitaciones documentadas en el docstring.
- local_rag_advanced: retorna top-K con scores explicitos, ordenados por
  similitud semantica (descendente).

Solo se registran cuando state["mode"] == "professional" en el investigador.
"""
from __future__ import annotations
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from langchain_core.tools import tool
from src.rag_agent import get_indexed_retriever


# ---- python_repl_tool ----------------------------------------------------

_BANNED_TOP_LEVEL_NAMES = {"os", "subprocess", "socket"}
_BANNED_BUILTINS = {"__import__", "exec", "eval", "open"}


def _ast_guard(code: str):
    """Devuelve None si el codigo es aceptable; mensaje de error si no."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                root = (n.name or "").split(".")[0]
                if root in _BANNED_TOP_LEVEL_NAMES:
                    return f"Import of {root!r} is not allowed."
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _BANNED_TOP_LEVEL_NAMES:
                return f"Import from {root!r} is not allowed."
        if isinstance(node, ast.Name) and node.id in _BANNED_BUILTINS:
            return f"Use of builtin {node.id!r} is not allowed."
    return None


@tool
def python_repl_tool(code: str) -> str:
    """Ejecuta codigo Python en un subproceso aislado.

    Restricciones:
    - Timeout 5 segundos (proceso terminado al exceder).
    - AST guard rechaza `import os`, `import subprocess`, `import socket`,
      y los builtins __import__, exec, eval, open.
    - CWD aislado en directorio temporal, eliminado al terminar.

    LIMITACIONES (no es un sandbox completo):
    - NO bloquea acceso a red a nivel SO; un import alternativo podria
      hacer requests si no esta en la lista negra. Para produccion real,
      usar Docker / RestrictedPython.
    - Habilitado solo si env ENABLE_PYTHON_REPL == "true".
    """
    if os.getenv("ENABLE_PYTHON_REPL", "false").lower() != "true":
        return "python_repl_tool is disabled (set ENABLE_PYTHON_REPL=true to enable)."

    err = _ast_guard(code)
    if err:
        return f"REPL_BLOCKED: {err}"

    sandbox = tempfile.mkdtemp(prefix="archia-repl-")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        out = (proc.stdout or "").strip()
        err_out = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return f"EXIT={proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err_out}"
        return out or "(no stdout)"
    except subprocess.TimeoutExpired:
        return "REPL_TIMEOUT: execution exceeded 5 seconds and was terminated."
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


# ---- local_rag_advanced --------------------------------------------------

@tool
def local_rag_advanced(query: str, quality_attribute: str = "general", k: int = 6) -> str:
    """RAG avanzado: top-K con scores explicitos, ordenado por similitud.

    A diferencia de `local_RAG`, este expone los scores y el orden semantico
    para razonamiento downstream. Devuelve JSON con:
    {results: [{rank, score, snippet, source}, ...], total: N}.
    """
    try:
        retr = get_indexed_retriever(quality_attribute=quality_attribute, k=k)
        # Acceso directo al vectorstore para usar similarity_search_with_score.
        vs = getattr(retr, "vectorstore", None)
        if vs is None or not hasattr(vs, "similarity_search_with_score"):
            # Fallback: invoke sin score, score=0.0
            docs = list(retr.invoke(query))
            scored = [(d, 0.0) for d in docs]
        else:
            scored = vs.similarity_search_with_score(query, k=k)
        # Chroma scores: distancia, menor = mas similar. Convertimos a similitud.
        scored.sort(key=lambda t: t[1])
        results = []
        for i, (d, dist) in enumerate(scored[:k], start=1):
            md = d.metadata or {}
            title = md.get("source_title") or md.get("title") or "doc"
            page = md.get("page_label") or md.get("page")
            path = md.get("source_path") or md.get("source") or ""
            page_str = f" (p.{page})" if page is not None else ""
            results.append({
                "rank": i,
                "score": float(1.0 / (1.0 + float(dist))),
                "snippet": (d.page_content or "").strip()[:300],
                "source": f"{title}{page_str} - {path}".strip(),
            })
        return json.dumps(
            {"query": query, "total": len(results), "results": results},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"query": query, "error": str(e), "results": [], "total": 0}
        )
