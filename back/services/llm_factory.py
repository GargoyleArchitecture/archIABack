"""
LLM Factory for ArchIA

Soporta múltiples backends:
- OpenAI (default)
- Ollama (local, gratis)
- Google Vertex AI
"""

import os
from typing import Any


def get_llm(provider: str = "ollama", model: str | None = None, **kwargs) -> Any:
    """
    Get LLM instance from specified provider.
    
    Args:
        provider: "openai", "ollama", or "vertex"
        model: Model name
        **kwargs: Additional arguments
        
    Returns:
        LLM instance
    """
    if provider == "ollama":
        return _get_ollama(model or "llama3.1", **kwargs)
    elif provider == "openai":
        return _get_openai(model, **kwargs)
    elif provider == "vertex":
        return _get_vertex(model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _get_ollama(model: str = "llama3.1", **kwargs) -> Any:
    """
    Get Ollama LLM (local, free).
    
    Args:
        model: Model name (e.g., "llama3.1", "mistral", "gemma2")
        **kwargs: Additional arguments
        
    Returns:
        ChatOllama instance
    """
    try:
        from langchain_ollama import ChatOllama
        
        llm = ChatOllama(
            model=model,
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            temperature=kwargs.get("temperature", 0.0),
        )
        
        print(f"🤖 Using Ollama: {model}")
        return llm
        
    except ImportError:
        print("❌ langchain-ollama not installed")
        print("   Install: poetry add langchain-ollama")
        raise


def _get_openai(model: str | None = None, **kwargs) -> Any:
    """
    Get OpenAI LLM.
    
    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4o-mini")
        **kwargs: Additional arguments
        
    Returns:
        ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    llm = ChatOpenAI(
        model=model or "gpt-4o-mini",
        api_key=api_key,
        temperature=kwargs.get("temperature", 0.0),
    )
    
    print(f"☁️ Using OpenAI: {llm.model_name}")
    return llm


def _get_vertex(model: str | None = None, **kwargs) -> Any:
    """
    Get Google Vertex AI LLM.
    
    Args:
        model: Model name
        **kwargs: Additional arguments
        
    Returns:
        ChatVertexAI instance
    """
    from langchain_google_vertexai import ChatVertexAI
    
    llm = ChatVertexAI(
        model=model or "gemini-1.5-flash",
        temperature=kwargs.get("temperature", 0.0),
    )
    
    print(f"🔵 Using Vertex AI: {llm.model}")
    return llm


# =============================================================================
# Convenience Functions
# =============================================================================

def get_chat_model(provider: str = "ollama", **kwargs) -> Any:
    """Get chat model from specified provider."""
    return get_llm(provider=provider, **kwargs)


def get_local_model() -> Any:
    """Get local Ollama model (default: llama3.1)."""
    return get_llm(provider="ollama", model="llama3.1")


def get_openai_model() -> Any:
    """Get OpenAI model (requires API key)."""
    return get_llm(provider="openai")
