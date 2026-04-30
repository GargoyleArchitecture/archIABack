"""Tests F3-T3: hidratacion del perfil en boot_node."""
import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from src.graph.workflow import boot_node


# ---- helpers -------------------------------------------------------------

class _FakeStoreItem:
    def __init__(self, value):
        self.value = value


class _FakeStore:
    def __init__(self, initial=None):
        self._data = dict(initial or {})

    async def aget(self, ns, key):
        if (ns, key) in self._data:
            return _FakeStoreItem(self._data[(ns, key)])
        return None

    async def aput(self, ns, key, value):
        self._data[(ns, key)] = value


def _patch_store(monkeypatch, store):
    """Sustituye el holder de Store en src.graph.resources."""
    from src.graph import resources

    monkeypatch.setattr(resources, "_store_holder", {"instance": store})


def _base_state(**overrides):
    base = {
        "user_id": "u1",
        "turn_count_since_eval": 0,
        "messages": [],
    }
    base.update(overrides)
    return base


# ---- tests ---------------------------------------------------------------

def test_boot_creates_empty_profile_when_missing(monkeypatch):
    store = _FakeStore()
    _patch_store(monkeypatch, store)

    out = asyncio.run(boot_node(_base_state(user_id="u1")))

    assert out["user_profile"]["user_id"] == "u1"
    assert out["user_profile"]["evaluated_concepts"] == []
    assert out["user_profile"]["strengths"] == []
    assert out["user_profile"]["weaknesses"] == []
    # Counter incremented
    assert out["turn_count_since_eval"] == 1
    # And the empty profile was written to the store for future reads.
    persisted = asyncio.run(store.aget(("user", "u1", "profile"), "profile"))
    assert persisted is not None
    assert persisted.value["user_id"] == "u1"


def test_boot_hydrates_existing_profile_with_decay(monkeypatch):
    last_seen = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    pre = {
        "user_id": "u1",
        "strengths": ["Modularidad"],
        "weaknesses": [],
        "evaluated_concepts": [
            {"name": "Modularidad", "mastery": 1.0, "last_seen_at": last_seen},
        ],
        "confidence": 0.7,
    }
    store = _FakeStore(initial={(("user", "u1", "profile"), "profile"): pre})
    _patch_store(monkeypatch, store)

    out = asyncio.run(boot_node(_base_state(user_id="u1", turn_count_since_eval=3)))

    profile = out["user_profile"]
    concept = profile["evaluated_concepts"][0]
    # Decay applied on read
    assert concept["mastery"] < 1.0
    # Original preserved as reference
    assert concept["mastery_original"] == 1.0
    # Strengths preserved
    assert "Modularidad" in profile["strengths"]
    # Counter incremented
    assert out["turn_count_since_eval"] == 4


def test_boot_does_not_mutate_store_value(monkeypatch):
    """Decay en lectura NO debe alterar el valor persistido."""
    last_seen = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    pre = {
        "user_id": "u1",
        "strengths": [],
        "weaknesses": [],
        "evaluated_concepts": [
            {"name": "X", "mastery": 1.0, "last_seen_at": last_seen}
        ],
    }
    store = _FakeStore(initial={(("user", "u1", "profile"), "profile"): pre})
    _patch_store(monkeypatch, store)

    asyncio.run(boot_node(_base_state(user_id="u1")))
    # Despues de la hidratacion, el Store sigue con mastery=1.0 sin decay
    persisted = asyncio.run(store.aget(("user", "u1", "profile"), "profile"))
    assert persisted.value["evaluated_concepts"][0]["mastery"] == 1.0


def test_boot_empty_user_id_skips_hydration(monkeypatch):
    store = _FakeStore()
    _patch_store(monkeypatch, store)

    out = asyncio.run(boot_node(_base_state(user_id="")))
    assert out["user_profile"] == {}
    # Nothing written
    assert asyncio.run(store.aget(("user", "", "profile"), "profile")) is None


def test_boot_resets_turn_buffers(monkeypatch):
    store = _FakeStore()
    _patch_store(monkeypatch, store)

    state = _base_state(
        user_id="u1",
        hasVisitedInvestigator=True,
        hasVisitedEvaluator=True,
        endMessage="leftover",
        requested_nodes=["x"],
        pending_nodes=["y"],
        completed_nodes=["z"],
        diagram={"ok": True},
    )
    out = asyncio.run(boot_node(state))
    assert out["hasVisitedInvestigator"] is False
    assert out["hasVisitedEvaluator"] is False
    assert out["endMessage"] == ""
    assert out["requested_nodes"] == []
    assert out["pending_nodes"] == []
    assert out["completed_nodes"] == []
    assert out["diagram"] == {}
