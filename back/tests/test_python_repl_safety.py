"""Tests del AST guard y subprocess del python_repl_tool (F2-T6)."""
import os

from src.graph.nodes.tools_professional import _ast_guard, python_repl_tool


def test_ast_guard_blocks_import_os():
    assert _ast_guard("import os") is not None


def test_ast_guard_blocks_import_subprocess():
    assert _ast_guard("import subprocess") is not None


def test_ast_guard_blocks_import_socket():
    assert _ast_guard("import socket") is not None


def test_ast_guard_blocks_from_os_import():
    assert _ast_guard("from os import path") is not None


def test_ast_guard_blocks_dunder_import():
    assert _ast_guard("__import__('os')") is not None


def test_ast_guard_blocks_eval():
    assert _ast_guard("eval('1+1')") is not None


def test_ast_guard_blocks_exec():
    assert _ast_guard("exec('print(1)')") is not None


def test_ast_guard_blocks_open():
    assert _ast_guard("open('x')") is not None


def test_ast_guard_allows_safe_arithmetic():
    assert _ast_guard("x = 1 + 1\nprint(x)") is None


def test_ast_guard_allows_safe_imports():
    assert _ast_guard("import math\nprint(math.pi)") is None


def test_ast_guard_reports_syntax_error():
    out = _ast_guard("def (")
    assert out is not None and "SyntaxError" in out


def test_repl_disabled_by_default():
    os.environ.pop("ENABLE_PYTHON_REPL", None)
    out = python_repl_tool.invoke({"code": "print('hi')"})
    assert "disabled" in out.lower()


def test_repl_blocks_dangerous_when_enabled():
    os.environ["ENABLE_PYTHON_REPL"] = "true"
    try:
        out = python_repl_tool.invoke({"code": "import os\nprint(os.getcwd())"})
        assert "REPL_BLOCKED" in out
    finally:
        os.environ["ENABLE_PYTHON_REPL"] = "false"


def test_repl_executes_safe_code_when_enabled():
    os.environ["ENABLE_PYTHON_REPL"] = "true"
    try:
        out = python_repl_tool.invoke({"code": "print(1 + 2)"})
        assert "3" in out
    finally:
        os.environ["ENABLE_PYTHON_REPL"] = "false"
