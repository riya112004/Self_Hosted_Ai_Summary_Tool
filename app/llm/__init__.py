from .prompt_builder import build_prompt
from app.providers import get_provider, OllamaProvider
from app.schemas.summary import SummaryOutput, StructuredOutputError, parse_summary


class LLMService:
    """Orchestrates prompt building -> inference provider -> structured validation."""

    def __init__(self, provider=None, model: str | None = None):
        if model is not None:
            provider = OllamaProvider(model=model)
        self.provider = provider or get_provider()

    def generate_summary(
        self, data, mode: str = "system", strict: bool = True, auto_fallback: bool = False
    ) -> SummaryOutput:
        """
        Run the full pipeline.

        strict=False disables validation. auto_fallback=True retries in
        lenient mode once when strict validation fails, so a slightly
        malformed answer still produces a usable summary.
        """
        prompt = build_prompt(data, mode=mode)
        json_schema = SummaryOutput.model_json_schema()
        text = self.provider.chat(prompt, json_mode=True, json_schema=json_schema)
        try:
            return parse_summary(text, strict=True)
        except StructuredOutputError:
            if auto_fallback and strict:
                return parse_summary(text, strict=False)
            if strict:
                raise
            return parse_summary(text, strict=False)

    def generate_summary_from_profile(
        self,
        profile: dict,
        mode: str = "system",
        auto_fallback: bool = False,
    ) -> SummaryOutput:
        """
        Profile-first summarization: the only thing the LLM sees is the
        verified python-computed data profile - never the raw dataset.
        """
        from app.profile.prompt import build_profile_prompt

        prompt = build_profile_prompt(profile, mode=mode)
        json_schema = SummaryOutput.model_json_schema()
        text = self.provider.chat(prompt, json_mode=True, json_schema=json_schema)
        try:
            return parse_summary(text, strict=True)
        except StructuredOutputError:
            if auto_fallback:
                return parse_summary(text, strict=False)
            raise