"""Tests del schema UserProfileEvaluation (F3-T1)."""
import pytest
from pydantic import ValidationError

from src.graph.schemas.profile import (
    ConceptScore,
    UserProfileEvaluation,
    UserProfileSchema,
)


# ---- ConceptScore (de F2-T1, validacion de bordes) -----------------------

def test_concept_score_mastery_in_range():
    c = ConceptScore(name="X", mastery=0.5)
    assert c.mastery == 0.5


def test_concept_score_mastery_out_of_range_high():
    with pytest.raises(ValidationError):
        ConceptScore(name="X", mastery=1.5)


def test_concept_score_mastery_out_of_range_low():
    with pytest.raises(ValidationError):
        ConceptScore(name="X", mastery=-0.1)


# ---- UserProfileEvaluation -----------------------------------------------

def test_evaluation_low_confidence_allows_empty_lists():
    """Senial debil: confidence < 0.5, listas vacias permitidas."""
    eval_ = UserProfileEvaluation(confidence=0.3)
    assert eval_.strengths == []
    assert eval_.weaknesses == []
    assert eval_.evaluated_concepts == []


def test_evaluation_high_confidence_requires_non_empty():
    """Confidence >= 0.5 con todo vacio debe rechazarse."""
    with pytest.raises(ValidationError):
        UserProfileEvaluation(confidence=0.7)


def test_evaluation_high_confidence_with_strengths_only_ok():
    eval_ = UserProfileEvaluation(
        confidence=0.7,
        strengths=[ConceptScore(name="Modularidad", mastery=0.8)],
    )
    assert eval_.confidence == 0.7
    assert len(eval_.strengths) == 1


def test_evaluation_high_confidence_with_evaluated_concepts_only_ok():
    eval_ = UserProfileEvaluation(
        confidence=0.6,
        evaluated_concepts=[ConceptScore(name="X", mastery=0.5)],
    )
    assert eval_.confidence == 0.6


def test_evaluation_confidence_out_of_range():
    with pytest.raises(ValidationError):
        UserProfileEvaluation(confidence=1.2)


def test_evaluation_delta_from_previous_optional():
    eval_ = UserProfileEvaluation(confidence=0.3)
    assert eval_.delta_from_previous == {}


def test_evaluation_serializes_round_trip():
    payload = {
        "strengths": [{"name": "A", "mastery": 0.8, "evidence": "evi"}],
        "weaknesses": [{"name": "B", "mastery": 0.2}],
        "evaluated_concepts": [{"name": "A", "mastery": 0.8}],
        "confidence": 0.7,
        "delta_from_previous": {"A": 0.1},
    }
    eval_ = UserProfileEvaluation.model_validate(payload)
    dumped = eval_.model_dump()
    assert dumped["confidence"] == 0.7
    assert dumped["delta_from_previous"] == {"A": 0.1}
    assert dumped["strengths"][0]["name"] == "A"


# ---- UserProfileSchema (de F2-T1, smoke) ---------------------------------

def test_user_profile_schema_strips_empty_strings_in_strengths():
    p = UserProfileSchema(
        user_id="u1",
        strengths=["A", "", "  ", "B"],
        weaknesses=[],
        evaluated_concepts=[],
        confidence=0.4,
    )
    assert p.strengths == ["A", "B"]
