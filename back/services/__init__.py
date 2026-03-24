"""
ArchIA Services

Servicios auxiliares:
- LLM Factory (Ollama, OpenAI, Vertex)
"""

from back.services.llm_factory import get_llm, get_local_model, get_openai_model

__all__ = ["get_llm", "get_local_model", "get_openai_model"]
