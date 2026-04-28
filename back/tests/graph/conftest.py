import pytest
from src.ledger.types import empty_ledger


@pytest.fixture
def mock_ledger_empty():
    return empty_ledger("proj-test", "user-test")


@pytest.fixture
def mock_ledger_with_asr(mock_ledger_empty):
    ledger = dict(mock_ledger_empty)
    ledger["current_phase"] = "ASR"
    ledger["current_iteration"] = 1
    ledger["decisions"] = [{
        "id": "01TEST000000000000ASR0001",
        "kind": "asr", "phase": "ASR", "iteration": 1,
        "qa": "latencia",
        "parents": [],
        "payload": {
            "summary": "p95 < 200ms at 5k RPS",
            "source": "", "stimulus": "", "environment": "",
            "artifact": "", "response": "",
            "response_measure": "p95<200ms@5kRPS", "domain": "e-commerce",
        },
        "rationale": "test rationale", "sources": [],
        "status": "active", "parent_status": "ok",
        "superseded_by": None, "rejection_reason": None,
        "created_at": "2024-01-01T00:00:00Z", "created_by_node": "asr_node",
    }]
    return ledger


@pytest.fixture
def mock_ledger_full(mock_ledger_with_asr):
    ledger = dict(mock_ledger_with_asr)
    ledger["decisions"] = list(ledger["decisions"]) + [
        {
            "id": "01TEST000000000000STY0001",
            "kind": "style", "phase": "STYLE", "iteration": 2,
            "qa": "latencia",
            "parents": [{"id": "01TEST000000000000ASR0001", "kind": "asr"}],
            "payload": {"chosen": "Event-Driven"},
            "rationale": "test style rationale", "sources": [],
            "status": "active", "parent_status": "ok",
            "superseded_by": None, "rejection_reason": None,
            "created_at": "2024-01-01T01:00:00Z", "created_by_node": "style_node",
        },
        {
            "id": "01TEST000000000000TAC0001",
            "kind": "tactic", "phase": "TACTICS", "iteration": 3,
            "qa": "latencia",
            "parents": [
                {"id": "01TEST000000000000ASR0001", "kind": "asr"},
                {"id": "01TEST000000000000STY0001", "kind": "style"},
            ],
            "payload": {"items": [
                {"name": "CQRS", "rationale": "...", "categories": ["data"], "success_probability": 0.9, "rank": 1},
            ]},
            "rationale": "test tactic rationale", "sources": [],
            "status": "active", "parent_status": "ok",
            "superseded_by": None, "rejection_reason": None,
            "created_at": "2024-01-01T02:00:00Z", "created_by_node": "tactics_node",
        },
    ]
    return ledger


@pytest.fixture
def base_state():
    """Minimal valid GraphState for unit-testing context_loader."""
    return {
        "project_id": "proj-test",
        "user_id_for_prefs": "user-test",
        "project_context_loaded": True,
        "user_style_loaded": True,
        "language": "es",
        "current_asr": "",
        "quality_attribute": "",
        "style": "",
        "selected_style": "",
        "last_style": "",
        "tactics_struct": [],
        "tactics_list": [],
        "ledger": {},
        "ledger_active": {},
        "design_dossier_md": "",
        "current_phase": "",
        "ledger_dossier_compact": "",
        "ledger_phase_prompt": "",
        "ledger_pending_advance": {},
    }
