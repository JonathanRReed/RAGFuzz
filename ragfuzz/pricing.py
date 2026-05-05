"""Provider pricing data for cost estimation."""

from __future__ import annotations

# Default local pricing per 1K tokens.
# Local providers do not bill through RAGFuzz, so the default estimate is zero.
DEFAULT_PRICING = {
    "prompt_per_1k": 0.0,
    "completion_per_1k": 0.0,
}

# Provider-specific pricing data
PROVIDER_PRICING = {
    "lmstudio": {
        "prompt_per_1k": 0.0,
        "completion_per_1k": 0.0,
    },
    "ollama": {
        "prompt_per_1k": 0.0,
        "completion_per_1k": 0.0,
    },
    "vllm": {
        "prompt_per_1k": 0.0,
        "completion_per_1k": 0.0,
    },
}


def get_pricing(provider_id: str, model_id: str | None = None) -> dict[str, float]:
    """Get pricing for a provider and model.

    Args:
        provider_id: The provider identifier.
        model_id: Optional model identifier for model-specific pricing.

    Returns:
        Dictionary with 'prompt_per_1k' and 'completion_per_1k' keys.
    """
    # Check for provider-specific pricing
    provider_data = PROVIDER_PRICING.get(provider_id, {})

    # If provider has model-specific pricing and model_id is provided
    if model_id and isinstance(provider_data, dict) and model_id in provider_data:
        result = provider_data[model_id]
        if isinstance(result, dict):
            return dict(result)  # type: ignore[return-value]
        return DEFAULT_PRICING

    # If provider has default pricing
    if isinstance(provider_data, dict) and "prompt_per_1k" in provider_data:
        return provider_data

    # Fall back to default pricing
    return DEFAULT_PRICING


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    provider_id: str,
    model_id: str | None = None,
) -> float:
    """Estimate cost for a request.

    Args:
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
        provider_id: The provider identifier.
        model_id: Optional model identifier.

    Returns:
        Estimated cost in USD.
    """
    pricing = get_pricing(provider_id, model_id)

    prompt_cost = (prompt_tokens / 1000) * pricing["prompt_per_1k"]
    completion_cost = (completion_tokens / 1000) * pricing["completion_per_1k"]

    return prompt_cost + completion_cost


def estimate_tokens(text: str, multiplier: float = 0.25) -> int:
    """Estimate number of tokens from text.

    Args:
        text: The text to estimate tokens for.
        multiplier: Multiplier to adjust estimate (default: 0.25 for chars).

    Returns:
        Estimated token count.
    """
    return int(len(text) * multiplier)
