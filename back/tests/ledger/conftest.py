"""Shared fixtures for ledger tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.memory as memory_module

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect memory DB to a temp file; create the table fresh for each test."""
    db_path = tmp_path / "test_memory.db"
    monkeypatch.setattr(memory_module, "DB_PATH", db_path)
    memory_module.init()
    yield db_path


@pytest.fixture
def sample_ledger():
    """Factory: load a named fixture from fixtures/{name}.json."""
    def _load(name: str) -> dict:
        path = FIXTURES_DIR / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return _load
