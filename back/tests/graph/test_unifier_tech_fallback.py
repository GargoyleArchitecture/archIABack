"""F20-T1 — unifier rama `intent=tech` ya no produce los literales
"No technology content." ni "Referencias / None" frente al usuario.

Antes de F20-T1, cuando el supervisor enrutaba una pregunta educativa a
intent=tech pero el nodo `tech_advisor` no había aportado contenido y
`endMessage` estaba vacío, el unifier emitía un mensaje degenerado:

    No technology content.

    ---

    ### Referencias

    None

Tras F20-T1 el unifier emite un mensaje útil que redirige la conversación
y, cuando no hay fuentes, omite por completo el bloque "Referencias".
"""
import asyncio
from unittest.mock import patch

from langchain_core.messages import AIMessage
from src.graph.nodes.unifier import unifier_node


_PATCH_FINALIZE = "src.graph.nodes.unifier._finalize_turn"


def _state_tech(**kw):
    base = {
        "intent": "tech",
        "language": "es",
        "requested_nodes": [],
        "messages": [],
        "turn_messages": [],
        "user_style_hint": "",
        "project_context_text": "",
        "memory_text": "",
        "userQuestion": "Explícame el load balancer",
        "diagram": {},
        "suggestions": [],
        "turn_count_since_eval": 0,
        "endMessage": "",
    }
    base.update(kw)
    return base


def _run(state):
    with patch(_PATCH_FINALIZE, side_effect=lambda s, t: s):
        return asyncio.run(unifier_node(state))


def test_degenerate_state_does_not_emit_no_technology_content_literal_es():
    """tech_advisor vacio + endMessage vacio + lang=es: mensaje util en
    espanol, NO el literal "No technology content."."""
    out = _run(_state_tech(language="es"))

    end = out["endMessage"]
    assert "No technology content" not in end, (
        "Regresion F20-T1: el unifier no debe entregar el literal 'No "
        "technology content.' al usuario."
    )
    # El fallback util pide mas contexto y ofrece dos caminos (educativo o
    # decision de stack). Verificamos por tokens estables (no acentuados).
    end_low = end.lower()
    assert "contexto" in end_low
    assert "recomendarte" in end_low or "tecnolog" in end_low
    # Y NO debe incluir el bloque "Referencias / None" cuando no hay refs.
    assert "### Referencias" not in end
    assert "None" not in end.split("\n\n")[-1]


def test_degenerate_state_does_not_emit_no_technology_content_literal_en():
    """Mismo caso pero en inglés — fallback localizado."""
    out = _run(_state_tech(language="en"))

    end = out["endMessage"]
    assert "No technology content" not in end
    assert "more context" in end.lower()
    assert "### References" not in end


def test_degenerate_state_offers_redirect_suggestions():
    """Las followups deben proponer al usuario el camino educativo y el
    camino de decisión de stack — no las default tech-followups."""
    out = _run(_state_tech(language="es"))

    suggestions = out.get("suggestions") or []
    joined = " || ".join(suggestions)
    # No deben ser las suggestions default de "diagrama de componentes".
    assert "diagrama de componentes" not in joined.lower()
    # Sí deben ofrecer el camino educativo o el camino de decisión.
    assert any(
        "ejemplo mínimo" in s.lower() or "atributo de calidad" in s.lower()
        for s in suggestions
    )


def test_tech_advisor_content_preserved_no_regression():
    """Cuando tech_advisor SÍ aporta contenido, se usa tal cual y NO se
    inyecta el fallback. (Garantía de no-regresión del happy path.)"""
    advisor_text = "Recomiendo NGINX como load balancer porque ..."
    state = _state_tech(
        messages=[AIMessage(content=advisor_text, name="tech_advisor")],
        language="es",
    )
    out = _run(state)

    end = out["endMessage"]
    assert advisor_text in end
    assert "más contexto" not in end.lower()


def test_refs_block_emitted_when_sources_present():
    """Cuando tech_sources aporta fuentes, el bloque "Referencias" se emite."""
    advisor_text = "Usa Envoy como L7 LB."
    sources_text = "SOURCES:\n- envoyproxy.io/docs\n- cncf.io/blog"
    state = _state_tech(
        messages=[
            AIMessage(content=advisor_text, name="tech_advisor"),
            AIMessage(content=sources_text, name="tech_sources"),
        ],
        language="es",
    )
    out = _run(state)

    end = out["endMessage"]
    assert advisor_text in end
    assert "### Referencias" in end
    assert "envoyproxy.io/docs" in end


def test_refs_block_omitted_when_sources_yield_only_none():
    """Si `_extract_rag_sources_from` no encuentra fuentes utiles, el
    bloque "Referencias" se omite — antes se emitía "### Referencias\\n\\nNone"."""
    advisor_text = "Algo de contenido tech util"
    # tech_sources sin sección SOURCES → _extract_rag_sources_from devuelve "".
    state = _state_tech(
        messages=[
            AIMessage(content=advisor_text, name="tech_advisor"),
            AIMessage(content="Texto sin secciones de fuentes.", name="tech_sources"),
        ],
        language="es",
    )
    out = _run(state)

    end = out["endMessage"]
    assert advisor_text in end
    assert "### Referencias" not in end
    assert "None" not in end
