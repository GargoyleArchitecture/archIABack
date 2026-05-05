"""Tests F3-T4: curva de olvido (apply_forgetting_curve, apply_decay_to_profile)."""
import math
from datetime import datetime, timezone, timedelta

from src.graph.services.decay import (
    apply_forgetting_curve,
    apply_decay_to_profile,
    DECAY_RATE_DEFAULT,
)


def test_no_last_seen_returns_unchanged_mastery():
    c = {"name": "X", "mastery": 0.8}
    out = apply_forgetting_curve(c, now=datetime.now(timezone.utc))
    assert out["mastery"] == 0.8
    assert out["mastery_original"] == 0.8


def test_delta_days_zero_no_decay():
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    c = {"name": "X", "mastery": 0.9, "last_seen_at": now.isoformat()}
    out = apply_forgetting_curve(c, now=now)
    assert out["mastery"] == 0.9
    assert out["delta_days"] == 0.0


def test_delta_days_30_decay_default_rate():
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    last = now - timedelta(days=30)
    c = {"name": "X", "mastery": 1.0, "last_seen_at": last.isoformat()}
    out = apply_forgetting_curve(c, now=now)
    expected = 1.0 * math.exp(-0.05 * 30)  # ~0.2231
    assert abs(out["mastery"] - round(expected, 4)) < 1e-4
    assert out["mastery_original"] == 1.0


def test_per_concept_decay_rate_overrides_default():
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    last = now - timedelta(days=30)
    c = {
        "name": "X",
        "mastery": 1.0,
        "last_seen_at": last.isoformat(),
        "decay_rate": 0.01,
    }
    out = apply_forgetting_curve(c, now=now)
    expected = math.exp(-0.01 * 30)
    assert abs(out["mastery"] - round(expected, 4)) < 1e-4


def test_negative_delta_clamped_to_zero():
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    last = now + timedelta(days=5)  # futuro (clock skew)
    c = {"name": "X", "mastery": 0.7, "last_seen_at": last.isoformat()}
    out = apply_forgetting_curve(c, now=now)
    assert out["mastery"] == 0.7
    assert out["delta_days"] == 0.0


def test_lastseenat_camel_case_also_supported():
    """Compatibilidad con la notacion del contrato OpenAPI (lastSeenAt)."""
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    last = now - timedelta(days=10)
    c = {"name": "X", "mastery": 1.0, "lastSeenAt": last.isoformat()}
    out = apply_forgetting_curve(c, now=now)
    expected = math.exp(-0.05 * 10)
    assert abs(out["mastery"] - round(expected, 4)) < 1e-4


def test_invalid_lastseenat_returns_unchanged():
    c = {"name": "X", "mastery": 0.8, "last_seen_at": "garbage-not-a-date"}
    out = apply_forgetting_curve(c, now=datetime.now(timezone.utc))
    assert out["mastery"] == 0.8


def test_apply_decay_to_profile_processes_all_concepts():
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    profile = {
        "user_id": "u1",
        "evaluated_concepts": [
            {
                "name": "A",
                "mastery": 1.0,
                "last_seen_at": (now - timedelta(days=10)).isoformat(),
            },
            {"name": "B", "mastery": 0.5},
        ],
    }
    out = apply_decay_to_profile(profile, now=now)
    assert len(out["evaluated_concepts"]) == 2
    a = out["evaluated_concepts"][0]
    assert a["mastery"] < 1.0
    assert a["mastery_original"] == 1.0
    b = out["evaluated_concepts"][1]
    assert b["mastery"] == 0.5


def test_apply_decay_does_not_mutate_input():
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    profile = {
        "evaluated_concepts": [
            {
                "name": "A",
                "mastery": 1.0,
                "last_seen_at": (now - timedelta(days=30)).isoformat(),
            }
        ]
    }
    original_mastery = profile["evaluated_concepts"][0]["mastery"]
    _ = apply_decay_to_profile(profile, now=now)
    assert profile["evaluated_concepts"][0]["mastery"] == original_mastery


def test_apply_decay_to_empty_profile_safe():
    assert apply_decay_to_profile({}) == {}
    assert apply_decay_to_profile(None) == {}


def test_default_decay_rate_constant():
    assert isinstance(DECAY_RATE_DEFAULT, float)
    assert DECAY_RATE_DEFAULT > 0.0
