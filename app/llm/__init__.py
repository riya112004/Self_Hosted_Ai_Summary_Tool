from app.providers import get_provider, OllamaProvider
from app.schemas.summary import SummaryOutput, StructuredOutputError, parse_summary


class LLMService:
    """Orchestrates prompt building -> inference provider -> structured validation."""

    def __init__(self, provider=None, model: str | None = None):
        if model is not None:
            provider = OllamaProvider(model=model)
        self.provider = provider or get_provider()

    def generate_summary_from_profile(
        self,
        profile: dict,
        auto_fallback: bool = False,
        sample: dict | None = None,
    ) -> SummaryOutput:
        """
        Profile-first summarization: the only thing the LLM sees is the
        verified python-computed data profile - never the raw dataset.
        `sample` is an optional compact first/random/last excerpt used as
        illustrative context alongside the verified statistics.
        """
        from app.llm.budget import build_budgeted_prompt
        from app.llm.guard import apply_fact_guard

        prompt, fitted_profile, fitted_sample = build_budgeted_prompt(
            profile, sample=sample
        )
        json_schema = SummaryOutput.model_json_schema()
        text = self.provider.chat(prompt, json_mode=True, json_schema=json_schema)
        try:
            out = parse_summary(text, strict=True)
        except StructuredOutputError:
            if auto_fallback:
                out = parse_summary(text, strict=False)
            else:
                raise
        return apply_fact_guard(out, fitted_profile, sample=fitted_sample)