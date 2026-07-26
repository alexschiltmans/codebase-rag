"""OpenAI-compatible LLM client using LangChain's ChatOpenAI."""

import logging
from collections.abc import Iterator
from typing import Any

import requests
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import SecretStr

from ..config import Config

logger = logging.getLogger(__name__)

GENERATION_MARGIN_TOKENS = 256
CHARS_PER_TOKEN = 4
MIN_PROMPT_BUDGET_CHARS = 2000


class OpenAICompatClient:
    """Client for OpenAI-compatible LLM APIs using ChatOpenAI.

    Supports LM Studio, llama.cpp server, vLLM, Jan, and other servers
    implementing the OpenAI chat-completions API.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        timeout: int = 120,
        context_window: int = 8192,
    ) -> None:
        config = Config.get_instance()

        self.model_name = model_name or config.llm_model_name
        self.base_url = (base_url or config.llm_base_url).rstrip("/")
        self.api_key = api_key or config.llm_api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

        if not self.base_url:
            raise ValueError("LLM_BASE_URL must be set for OpenAI-compatible backend")

        reservation_tokens = max_tokens + GENERATION_MARGIN_TOKENS
        self.prompt_budget_chars = (context_window - reservation_tokens) * CHARS_PER_TOKEN
        if self.prompt_budget_chars < MIN_PROMPT_BUDGET_CHARS:
            raise ValueError(
                f"OLLAMA_NUM_CTX={context_window} leaves a prompt budget of "
                f"{self.prompt_budget_chars} chars after reserving {reservation_tokens} tokens "
                f"({max_tokens} for generation + {GENERATION_MARGIN_TOKENS} margin). "
                f"Raise OLLAMA_NUM_CTX (governs this backend's context window too) or lower max_tokens."
            )

        # BaseChatOpenAI rather than ChatOpenAI: ChatOpenAI's _default_params renames
        # max_tokens to max_completion_tokens unconditionally, and plenty of OpenAI-compatible
        # servers (verified: Ollama's own /v1) only honor the older field, silently generating
        # past the configured cap instead of stopping at it. BaseChatOpenAI sends max_tokens
        # as given, which every documented target here (and Ollama's /v1) both accept.
        self._llm = BaseChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=SecretStr(self.api_key or "not-needed"),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        logger.info("Initialized OpenAICompatClient for model '%s' at %s", self.model_name, self.base_url)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The text prompt to send to the model.

        Returns:
            The generated text response.
        """
        logger.debug("Calling OpenAI-compatible backend with prompt length %d", len(prompt))
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
        logger.debug("Streaming from OpenAI-compatible backend with prompt length %d", len(prompt))
        for chunk in self._llm.stream(prompt, **kwargs):
            text = str(chunk.content)
            if text:
                yield text

    def _auth_headers(self) -> dict[str, str]:
        """Headers for the /models probes. ChatOpenAI already sends this key on generation
        requests; without it here, a server started with an API key requirement (e.g. vLLM's
        --api-key) 401s these checks while generation through ChatOpenAI works fine.
        """
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def check_connection(self) -> dict[str, Any]:
        """Check the connection to the OpenAI-compatible service."""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._auth_headers(), timeout=5)
            if response.status_code == 200:
                return {
                    "status": "connected",
                    "model": self.model_name,
                    "url": self.base_url,
                }
            return {
                "status": "error",
                "message": f"Server responded with status code {response.status_code}",
                "url": self.base_url,
            }
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": f"Cannot connect to {self.base_url}", "url": self.base_url}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Error checking connection: {e}", "url": self.base_url}

    def check_model_availability(self) -> dict[str, Any]:
        """Check if the configured model is available on the server.

        Queries /models directly rather than calling check_connection first,
        avoiding a duplicate GET.
        """
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._auth_headers(), timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models if isinstance(m, dict)]
                model_ids_str = [str(m) for m in model_ids if m is not None]
                if self.model_name in model_ids_str:
                    return {"status": "available", "model": self.model_name, "all_models": model_ids_str}
                models_str = ", ".join(model_ids_str[:5])
                return {
                    "status": "not_found",
                    "message": f"Model '{self.model_name}' not found on server",
                    "suggested_action": f"Load '{self.model_name}' or set LLM_MODEL_NAME to: {models_str}",
                    "available_models": model_ids_str,
                }
            return {"status": "error", "message": f"Failed to get model list: {response.status_code}"}
        except (requests.exceptions.RequestException, ValueError) as e:
            # ValueError covers response.json()'s json.JSONDecodeError.
            return {"status": "error", "message": f"Error checking model availability: {e}"}
