from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Common interface for inference servers.

    Development -> Ollama   (local, low-throughput)
    Production  -> vLLM     (high-throughput, structured/guided outputs)

    The rest of the app only talks to this interface, so switching servers
    is a config change, not a code change.
    """

    @abstractmethod
    def chat(
        self,
        prompt: str,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        """
        Send a single user prompt to the inference server.
        Returns the raw completion text.
        When json_mode is True and json_schema is provided, request a
        schema-constrained (guided) JSON response where the server supports it.
        """

    def list_models(self) -> list[str]:
        """Names of models served by the inference server."""
        raise NotImplementedError(f"{type(self).__name__} does not expose list_models")