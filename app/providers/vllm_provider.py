import logging
import os

import httpx

from .base import LLMProvider

logger = logging.getLogger(__name__)

# Summary output is bounded (200-400 words) so cap generation tokens.
# Set VLLM_MAX_TOKENS=0 to disable the cap.


class VLLMProvider(LLMProvider):
    """
    vLLM - OpenAI-compatible production inference server.

    Supports structured/guided outputs: the JSON schema is sent via
    `guided_json` (vLLM guided decoding) so completions are schema-compliant,
    which pairs with the Pydantic validation layer. `max_tokens` caps the
    output so the model cannot over-generate past a full summary.
    """

    name = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        try:
            self.max_tokens = int(os.getenv("VLLM_MAX_TOKENS", "500"))
        except ValueError:
            logger.warning("ignoring non-integer VLLM_MAX_TOKENS")
            self.max_tokens = 500

    def _post(self, payload: dict) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def chat(
        self,
        prompt: str,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            if json_schema is not None:
                payload["guided_json"] = json_schema

        data = self._post(payload)
        return data["choices"][0]["message"]["content"]

    def list_models(self) -> list[str]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/models")
            response.raise_for_status()
        return [m["id"] for m in response.json()["data"] if m.get("id")]