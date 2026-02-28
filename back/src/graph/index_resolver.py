"""
index_resolver.py

Resuelve el índice (atributo de calidad) a usar para retrieval RAG,
a partir de la pregunta del usuario y la configuración de índices.

Para añadir un nuevo atributo de calidad, sólo editar back/config/indices.json.
No se requieren cambios en este módulo.
"""

import json
from pathlib import Path
from langchain_core.messages import HumanMessage

# Ruta al config: back/config/indices.json
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "indices.json"


def _load_indices_config() -> dict:
    """Carga la configuración de índices desde indices.json."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"quality_attributes": [], "content_types": []}


def resolve_quality_attribute(question: str, llm) -> str:
    """
    Usa LLM para determinar qué atributo de calidad (índice) corresponde
    a la pregunta del usuario.

    Retorna el 'id' del atributo (e.g., "escalabilidad", "latencia")
    o "general" si no hay match claro o si ocurre un error.
    """
    config = _load_indices_config()
    qa_list = config.get("quality_attributes", [])

    if not qa_list:
        return "general"

    qa_descriptions = "\n".join(
        f'- "{qa["id"]}": {qa["description"]}'
        for qa in qa_list
    )
    valid_ids = [qa["id"] for qa in qa_list] + ["general"]
    ids_str = ", ".join(f'"{i}"' for i in valid_ids)

    prompt = (
        f"Eres un clasificador de preguntas de arquitectura de software.\n\n"
        f"Índices de atributos de calidad disponibles:\n{qa_descriptions}\n"
        f'- "general": si la pregunta no se relaciona claramente con ningún atributo específico\n\n'
        f'Pregunta del usuario: "{question}"\n\n'
        f"¿Con qué índice de atributo de calidad se relaciona más esta pregunta?\n"
        f"Responde ÚNICAMENTE con uno de estos valores: {ids_str}\n"
        f"Sin explicaciones, sin puntuación adicional, sólo el id."
    )

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        answer = result.content.strip().strip('"').strip("'").lower()

        if answer in valid_ids:
            return answer

        # Intento de match parcial
        for id_ in valid_ids:
            if id_ in answer:
                return id_

        return "general"

    except Exception:
        return "general"


def get_available_indices() -> list[dict]:
    """
    Retorna la lista de atributos de calidad disponibles desde la config.
    Útil para listar índices sin abrir el JSON directamente.
    """
    config = _load_indices_config()
    return config.get("quality_attributes", [])


def get_content_types() -> list[dict]:
    """
    Retorna la lista de tipos de contenido disponibles desde la config.
    """
    config = _load_indices_config()
    return config.get("content_types", [])
