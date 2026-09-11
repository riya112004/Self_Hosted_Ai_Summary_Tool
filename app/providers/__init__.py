import os

from dotenv import load_dotenv

from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .vllm_provider import VLLMProvider


def get_provider(name: str | None = None) -> LLMProvider:
    """
    Factory: pick the inference server from config.
    Default is Ollama (development). Set LLM_PROVIDER=vllm for production.
    """
    load_dotenv()
    name = name or os.getenv("LLM_PROVIDER", "ollama")

    if name == "ollama":
        return OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout=float(os.getenv("OLLAMA_TIMEOUT", "600")),
        )
    if name == "vllm":
        return VLLMProvider(
            base_url=os.getenv("VLLM_SERVER", "http://localhost:8000/v1"),
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER: {name!r}. Available: ollama, vllm"
    )