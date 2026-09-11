import logging
import os

import httpx

from .base import LLMProvider

logger = logging.getLogger(__name__)

# Right-sized generation options so Ollama does not run on untuned defaults.
# All three are overridable via .env - set to empty to leave unset.

# Cap total output tokens: stops the model rambling past a useful summary
# on CPU. Large enough for the ≤800-word structured JSON document summary.
_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
# Context window. Our prompts (document chunks, data profiles) fit well
# under this; oversized windows inflate KV-cache memory for no benefit.
_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
# Leave CPU thread count to Ollama unless OLLAMA_NUM_THREAD is explicitly
# set - hand-tuning can easily hurt more than it helps.
_NUM_THREAD = os.getenv("OLLAMA_NUM_THREAD")


def _tuned_options() -> dict:
    options: dict = {}
    if _NUM_PREDICT:
        options["num_predict"] = _NUM_PREDICT
    if _NUM_CTX:
        options["num_ctx"] = _NUM_CTX
    if _NUM_THREAD:
        try:
            options["num_thread"] = int(_NUM_THREAD)
        except ValueError:
            logger.warning("ignoring non-integer OLLAMA_NUM_THREAD=%r", _NUM_THREAD)
    return options


class OllamaProvider(LLMProvider):
    """
    Ollama - local development inference server.

    Talks to the Ollama REST API (/api/chat) directly over httpx with a
    bounded timeout, so a wedged/in-process-stuck server returns an error
    instead of hanging the HTTP request forever.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        host: str = "http://127.0.0.1:11434",
        timeout: float = 600.0,
    ):
        self.model = model
        self.host = (host or "http://127.0.0.1:11434").rstrip("/")
        self.timeout = timeout
        self.options = _tuned_options()
        logger.info("ollama generation options: %s", self.options)

    def chat(
        self,
        prompt: str,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": self.options,
        }
        if self.model.startswith("qwen3"):
            # qwen3 is a reasoning model: without this it emits a long
            # internal "think" block first, doubling latency and often
            # polluting the structured JSON output.
            payload["think"] = False
        if json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()

        return response.json()["message"]["content"]

    def list_models(self) -> list[str]:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{self.host}/api/tags")
            response.raise_for_status()
        return [
            m.get("name") or m.get("model")
            for m in response.json().get("models", [])
            if m.get("name") or m.get("model")
        ]