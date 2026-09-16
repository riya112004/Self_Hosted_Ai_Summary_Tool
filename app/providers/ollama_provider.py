import logging
import os

import httpx

from .base import LLMProvider

logger = logging.getLogger(__name__)

# Right-sized generation options so Ollama does not run on untuned defaults.
# num_predict caps the OUTPUT tokens (a summary is bounded to ~500 tokens;
# 200-400 words never needs more). num_ctx is the context window. Both are
# overridable via .env - set num_predict to 0 to disable the output cap.

def _env_int(name: str, default: int) -> int:
    """Read an int env var defensively; empty/invalid values fall back."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logger.warning("ignoring non-integer %s=%r", name, raw)
        return default


def _tuned_options() -> dict:
    options: dict = {}
    num_predict = _env_int("OLLAMA_NUM_PREDICT", 500)
    num_ctx = _env_int("OLLAMA_NUM_CTX", 2048)
    num_thread = os.getenv("OLLAMA_NUM_THREAD")
    if num_predict:
        options["num_predict"] = num_predict
    if num_ctx:
        options["num_ctx"] = num_ctx
    if num_thread:
        try:
            options["num_thread"] = int(num_thread)
        except ValueError:
            logger.warning("ignoring non-integer OLLAMA_NUM_THREAD=%r", num_thread)
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
        model: str = "qwen3:0.6b",
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
        options: dict | None = None,
    ) -> str:
        generation = {**self.options, **(options or {})}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": generation,
        }
        if self.model.startswith("qwen3"):
            # qwen3 is a reasoning model: without this it emits a long
            # internal "think" block first, doubling latency and often
            # polluting the structured JSON output.
            payload["think"] = False
        if json_mode:
            payload["format"] = "json"

        endpoint = f"{self.host}/api/chat"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(endpoint, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text.strip()
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"Ollama request failed ({response.status_code}) for model "
                    f"{self.model!r} at {endpoint}{suffix}"
                ) from exc

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