"""Base provider interface."""

from __future__ import annotations

from typing import Any

from ragfuzz.models import Message, Response


class Provider:
    """Abstract base class for LLM providers."""

    def __init__(self, provider_id: str, base_url: str, api_key: str | None = None):
        """Initialize the provider.

        Args:
            provider_id: Unique identifier for this provider.
            base_url: Base URL for the provider API.
            api_key: Optional API key for authentication.
        """
        self.provider_id = provider_id
        self.base_url = base_url
        self.api_key = api_key

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> Response:
        """Send a chat completion request.

        Args:
            messages: List of chat messages.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Optional tool definitions.
            stream: Whether to stream responses.

        Returns:
            A Response object.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model identifiers.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check if the provider is accessible.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            await self.chat(
                messages=[Message(role="user", content="test")],
                model="model",
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    async def benchmark(self, num_requests: int = 5) -> dict[str, float]:
        """Run a micro-benchmark to estimate performance.

        Args:
            num_requests: Number of requests to run.

        Returns:
            Dictionary with performance metrics.
        """
        import time

        latencies = []
        tokens_per_sec = []

        for _ in range(num_requests):
            start = time.time()
            response = await self.chat(
                messages=[Message(role="user", content="Say hello")],
                model="model",
                max_tokens=10,
            )
            end = time.time()

            latency = end - start
            total_tokens = response.usage.get("total_tokens", 10)
            latencies.append(latency)
            tokens_per_sec.append(total_tokens / latency if latency > 0 else 0)

        return {
            "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency_s": min(latencies) if latencies else 0,
            "max_latency_s": max(latencies) if latencies else 0,
            "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0,
        }
