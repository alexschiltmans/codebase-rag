"""Factory for selecting and creating LLM clients based on configured provider."""

from typing import Any

from codebase_rag.config import Config
from codebase_rag.llm.ollama_client import OllamaClient
from codebase_rag.llm.openai_compat_client import OpenAICompatClient


def create_llm_client(
    model_name: str | None = None,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_tokens: int = 1024,
    timeout: int = 120,
    **provider_kwargs: Any,
) -> OllamaClient | OpenAICompatClient:
    """Create an LLM client based on the configured provider.

    Args:
        model_name: Model to use; defaults to config.llm_model_name
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        **provider_kwargs: Provider-specific arguments (top_k for Ollama, etc.)

    Returns:
        OllamaClient or OpenAICompatClient depending on LLM_PROVIDER

    Raises:
        ValueError: If LLM_PROVIDER is invalid (already caught in config validation)
    """
    config = Config.get_instance()

    if config.provider == "ollama":
        return OllamaClient(
            model_name=model_name,
            base_url=config.ollama_base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            num_ctx=provider_kwargs.get("num_ctx"),
            top_k=provider_kwargs.get("top_k", 40),
        )
    if config.provider == "openai-compat":
        context_window = provider_kwargs.get("num_ctx") or config.ollama_num_ctx
        return OpenAICompatClient(
            model_name=model_name,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            context_window=context_window,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {config.provider}")
