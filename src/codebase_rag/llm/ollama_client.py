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
# The compose Ollama publishes here, off 11434, so it can't shadow a native install.
CONTAINER_OLLAMA_HOST_PORT = 11435
# Docker Desktop's documented host port for Model Runner once TCP is enabled.
MODEL_RUNNER_HOST_PORT = 12434
# The in-container hostname Model Runner answers on; never exercised by this project's own tests.
MODEL_RUNNER_INTERNAL_HOST = "model-runner.docker.internal"
# Ollama's own default port, matched on any host so a LAN-hosted instance is still recognized.
OLLAMA_DEFAULT_PORT = 11434
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


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
                if any(self._matches_configured_model(name) for name in model_names):
                    return {"status": "available", "model": self.model_name, "all_models": model_names}
                return {
                    "status": "not_found",
                    "message": f"Model '{self.model_name}' not found",
                    "suggested_action": self._suggest_remedy(model_names),
                    "available_models": model_names,
                }
            return {"status": "error", "message": f"Failed to get model list: {response.status_code}"}
        except (requests.exceptions.RequestException, ValueError) as e:
            return {"status": "error", "message": f"Error checking model availability: {e}"}

    def check_runtime_placement(self) -> dict[str, Any]:
        """Report whether the configured model is currently resident in GPU memory.

        Reads Ollama's running-models list, where `size_vram` counts the bytes the loaded
        model holds in VRAM. Placement is only knowable while the model is loaded, so a
        model nothing has touched yet reports `unknown` rather than `cpu`: claiming CPU
        inference on a cold start would be wrong on every GPU machine, every time.

        Returns a `placement` of `gpu`, `cpu`, or `unknown`, plus the endpoint that was
        asked, so a caller can report which backend the answer describes.
        """
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if response.status_code != 200:
                return {"placement": "unknown", "url": self.base_url}
            for running_model in response.json().get("models", []):
                if not self._matches_configured_model(running_model.get("name")):
                    continue
                size_vram = running_model.get("size_vram")
                if size_vram is None:
                    # An Ollama build that doesn't publish the field can't be read as CPU.
                    return {"placement": "unknown", "url": self.base_url}
                return {"placement": "gpu" if size_vram > 0 else "cpu", "url": self.base_url}
            return {"placement": "unknown", "url": self.base_url}
        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueError covers a 200 carrying a non-JSON body, as in check_connection.
            logger.debug("Could not determine runtime placement at %s: %s", self.base_url, e)
            return {"placement": "unknown", "url": self.base_url}

    def _suggest_remedy(self, model_names: list[str | None]) -> str:
        """Build the `not_found` remedy for the configured model, backend-appropriate.

        The compose and native Ollama cases keep their known-good commands. Model Runner
        gets `docker model pull`, verified to accept the same normalized name it reports
        back through `/api/tags`, so no separate reference form is needed. An unrecognized
        endpoint gets no guessed command at all, only the models the server actually has,
        matching the shape `OpenAICompatClient` uses for the same situation.
        """
        backend = self._resolve_backend()
        if backend == "compose_ollama":
            return f"Run 'docker exec codebase-rag-ollama ollama pull {self.model_name}'"
        if backend == "ollama":
            return f"Run 'ollama pull {self.model_name}'"
        if backend == "model_runner":
            return f"Run 'docker model pull {self.model_name}'"
        models_str = ", ".join(str(m) for m in model_names[:5] if m is not None)
        return f"Load '{self.model_name}' or set LLM_MODEL_NAME to: {models_str}"

    def _matches_configured_model(self, name: str | None) -> bool:
        """Match a model name reported by Ollama against the configured one.

        Ollama reports fully-qualified tags, so a configured bare name has to be compared
        against its `:latest` form as well.
        """
        if name is None:
            return False
        return name == self.model_name or (":" not in self.model_name and name == f"{self.model_name}:latest")

    def _resolve_backend(self) -> str:
        """Identify which backend `config.ollama_base_url` names.

        Based on `config.ollama_base_url`'s host rather than whether the app itself is
        containerized: the app can run on the host against a compose-networked Ollama, or
        against a native Ollama on localhost, so the remedy has to match where Ollama is.

        Returns one of "compose_ollama", "model_runner", "ollama", or "unrecognized".

        Compose Ollama has two independent ways to be reached, and neither subsumes the
        other: the compose service's DNS name, from another service on the compose network,
        or loopback on the container's published host port, from a process on the host. The
        container publishes 11435 so it cannot shadow a native Ollama on 11434, which makes
        the port enough to tell the two apart.

        Model Runner is matched the same way: loopback on its published host port, or the
        in-container hostname it answers on when reached from inside the compose network.

        Ollama proper is matched on port 11434 alone, on any host, rather than restricted to
        loopback. That is deliberate: it is what keeps a LAN-hosted Ollama
        (`http://192.168.1.20:11434`) on the plain `ollama pull` remedy instead of falling
        through to the unrecognized case, since no container claims that port.
        """
        parsed = urlparse(self.base_url)
        if parsed.hostname == "ollama":
            return "compose_ollama"
        if parsed.hostname in _LOOPBACK_HOSTS and parsed.port == CONTAINER_OLLAMA_HOST_PORT:
            return "compose_ollama"
        if parsed.hostname == MODEL_RUNNER_INTERNAL_HOST:
            return "model_runner"
        if parsed.hostname in _LOOPBACK_HOSTS and parsed.port == MODEL_RUNNER_HOST_PORT:
            return "model_runner"
        if parsed.port == OLLAMA_DEFAULT_PORT:
            return "ollama"
        return "unrecognized"
