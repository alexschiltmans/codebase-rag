"""Ollama LLM client using LangChain's ChatOllama."""

import logging
from collections.abc import Iterator
from typing import Any

import requests
from langchain_ollama import ChatOllama

from ..config import Config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for the Ollama LLM API using ChatOllama.

    Wraps LangChain's ChatOllama to provide a simple interface for text generation
    with connection and model availability checks.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        config = Config.get_instance()

        self.model_name = model_name or config.llm_model_name
        self.base_url = base_url or config.ollama_base_url
        self.temperature = temperature
        self.timeout = timeout

        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_predict=max_tokens,
        )
        logger.info("Initialized OllamaClient (ChatOllama) for model '%s' at %s", self.model_name, self.base_url)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The text prompt to send to the model.

        Returns:
            The generated text response.
        """
        logger.debug("Calling Ollama with prompt length %d", len(prompt))
        message = self._llm.invoke(prompt, **kwargs)
        text = str(message.content)
        logger.debug("Received response of length %d", len(text))
        return text

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream a response for the given prompt as it's generated.

        Args:
            prompt: The text prompt to send to the model.

        Yields:
            Successive text chunks of the generated response.
        """
        logger.debug("Streaming from Ollama with prompt length %d", len(prompt))
        for chunk in self._llm.stream(prompt, **kwargs):
            text = str(chunk.content)
            if text:
                yield text

    def check_connection(self) -> dict[str, Any]:
        """Check the connection to the Ollama service."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                return {
                    "status": "connected",
                    "version": version_info.get("version", "unknown"),
                    "model": self.model_name,
                    "url": self.base_url,
                }
            return {
                "status": "error",
                "message": f"Ollama responded with status code {response.status_code}",
                "url": self.base_url,
            }
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": f"Cannot connect to Ollama at {self.base_url}", "url": self.base_url}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Error checking Ollama connection: {e}", "url": self.base_url}

    def check_model_availability(self) -> dict[str, Any]:
        """Check if the configured model is available in Ollama."""
        connection_status = self.check_connection()
        if connection_status["status"] != "connected":
            return connection_status

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                if self.model_name in model_names:
                    return {"status": "available", "model": self.model_name, "all_models": model_names}
                return {
                    "status": "not_found",
                    "message": f"Model '{self.model_name}' not found in Ollama",
                    "suggested_action": f"Run 'ollama pull {self.model_name}'",
                    "available_models": model_names,
                }
            return {"status": "error", "message": f"Failed to get model list: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Error checking model availability: {e}"}
