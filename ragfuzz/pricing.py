"""Provider pricing data for cost estimation."""

from __future__ import annotations

# Default pricing per 1K tokens
# These are fallback values used when provider-specific pricing is not available
DEFAULT_PRICING = {
    "prompt_per_1k": 0.00001,
    "completion_per_1k": 0.00003,
}

# Provider-specific pricing data
# Prices are approximate and should be updated based on current rates
PROVIDER_PRICING = {
    # LM Studio - depends on the model being used
    "lmstudio": {
        "prompt_per_1k": 0.00001,
        "completion_per_1k": 0.00003,
    },
    # Ollama - typically free/local
    "ollama": {
        "prompt_per_1k": 0.0,
        "completion_per_1k": 0.0,
    },
    # OpenRouter - varies by model
    "openrouter": {
        "anthropic/claude-sonnet-4": {
            "prompt_per_1k": 0.003,
            "completion_per_1k": 0.015,
        },
        "anthropic/claude-3.5-sonnet": {
            "prompt_per_1k": 0.0015,
            "completion_per_1k": 0.0075,
        },
        "openai/gpt-4": {
            "prompt_per_1k": 0.03,
            "completion_per_1k": 0.06,
        },
        "openai/gpt-3.5-turbo": {
            "prompt_per_1k": 0.0005,
            "completion_per_1k": 0.0015,
        },
        "meta-llama/llama-3-70b-instruct": {
            "prompt_per_1k": 0.0007,
            "completion_per_1k": 0.0007,
        },
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
        return provider_data[model_id]

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
