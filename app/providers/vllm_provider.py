import httpx

from .base import LLMProvider


class VLLMProvider(LLMProvider):
    """
    vLLM - OpenAI-compatible production inference server.

    Supports structured/guided outputs: the JSON schema is sent via
    `guided_json` (vLLM guided decoding) so completions are schema-compliant,
    which pairs with the Pydantic validation layer.
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