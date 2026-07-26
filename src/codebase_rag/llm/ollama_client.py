"""Ollama LLM client using LangChain's ChatOllama."""

import logging
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import requests
from langchain_ollama import ChatOllama

from ..config import Config

logger = logging.getLogger(__name__)

# Reserved for the model's own output so the prompt budget doesn't crowd it out.
GENERATION_MARGIN_TOKENS = 256
# Conservative chars-per-token estimate for code-heavy English text; no tokenizer dependency.
CHARS_PER_TOKEN = 4
# Below this, the budget can't hold the template, a question, and one context chunk.
MIN_PROMPT_BUDGET_CHARS = 2000


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
        num_ctx: int | None = None,
    ) -> None:
        config = Config.get_instance()

        self.model_name = model_name or config.llm_model_name
        self.base_url = base_url or config.ollama_base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx if num_ctx is not None else config.ollama_num_ctx

        reservation_tokens = self.max_tokens + GENERATION_MARGIN_TOKENS
        self.prompt_budget_chars = (self.num_ctx - reservation_tokens) * CHARS_PER_TOKEN
        if self.prompt_budget_chars < MIN_PROMPT_BUDGET_CHARS:
            raise ValueError(
                f"OLLAMA_NUM_CTX={self.num_ctx} leaves a prompt budget of "
                f"{self.prompt_budget_chars} chars after reserving {reservation_tokens} tokens "
                f"({self.max_tokens} for generation + {GENERATION_MARGIN_TOKENS} margin) — "
                f"raise OLLAMA_NUM_CTX or lower max_tokens."
            )

        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_predict=max_tokens,
            num_ctx=self.num_ctx,
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
        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueError covers response.json()'s json.JSONDecodeError: a 200 with a
            # non-JSON body (e.g. a proxy's HTML error page) must not escape as an
            # uncaught exception into the caller's own health-check try/except.
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
                if self.model_name in model_names or (
                    ":" not in self.model_name and f"{self.model_name}:latest" in model_names
                ):
                    return {"status": "available", "model": self.model_name, "all_models": model_names}
                docker_cmd = (
                    f"docker exec codebase-rag-ollama ollama pull {self.model_name}"
                    if self._is_ollama_containerized()
                    else f"ollama pull {self.model_name}"
                )
                return {
                    "status": "not_found",
                    "message": f"Model '{self.model_name}' not found",
                    "suggested_action": f"Run '{docker_cmd}'",
                    "available_models": model_names,
                }
            return {"status": "error", "message": f"Failed to get model list: {response.status_code}"}
        except (requests.exceptions.RequestException, ValueError) as e:
            return {"status": "error", "message": f"Error checking model availability: {e}"}

    def _is_ollama_containerized(self) -> bool:
        """Detect whether the Ollama this client talks to runs in the compose network.

        Based on `config.ollama_base_url`'s host rather than whether the app itself is
        containerized: the app can run on the host against a compose-networked Ollama, or
        against a native Ollama on localhost, so the remedy has to match where Ollama is.

        Matches positively against the compose service's DNS name (`docker/compose-dev.yml`
        sets `OLLAMA_BASE_URL=http://ollama:11434` for the app service) rather than excluding
        loopback-like hosts, so a LAN-hosted Ollama (`http://192.168.1.20:11434`) correctly
        gets the plain `ollama pull` remedy instead of a `docker exec codebase-rag-ollama`
        command naming a container that only exists in the compose network.
        """
        return urlparse(self.base_url).hostname == "ollama"
