"""Tests heuristica de auto-routing de modo (F2-T4)."""
from src.graph.nodes.classifier import suggest_mode


def test_tutor_trigger_explicame_from_professional():
    assert suggest_mode("Explicame que es un microservicio", "professional") == "tutor"


def test_tutor_trigger_no_entiendo():
    assert suggest_mode("No entiendo el patron CQRS", "professional") == "tutor"


def test_professional_trigger_implementa_from_tutor():
    assert suggest_mode("Dame el codigo para implementar circuit breaker", "tutor") == "professional"


def test_professional_trigger_diagrama():
    assert suggest_mode("Quiero un diagrama de despliegue para microservicios", "tutor") == "professional"


def test_no_suggestion_when_already_in_target_mode_tutor():
    assert suggest_mode("Explicame que es un microservicio", "tutor") is None


def test_no_suggestion_when_already_in_target_mode_professional():
    assert suggest_mode("Dame el codigo para implementar circuit breaker", "professional") is None


def test_no_suggestion_when_neutral_message():
    assert suggest_mode("hola", "professional") is None


def test_english_tutor_trigger():
    assert suggest_mode("explain what is a saga pattern", "professional") == "tutor"


def test_english_professional_trigger():
    assert suggest_mode("give me the code to design the API", "tutor") == "professional"


def test_no_suggestion_when_both_signals_balanced():
    """Si hay 1 trigger en cada lado (ambos 0.5), ninguno supera al otro."""
    msg = "explicame e implementa"
    assert suggest_mode(msg, "professional") is None
