"""Provider health checking and benchmarking."""

from __future__ import annotations

import asyncio
from typing import Any

from ragfuzz.config import Config
from ragfuzz.providers.openai_compat import OpenAICompatProvider


class ProviderDoctor:
    """Provider health checker and benchmark."""

    def __init__(self, config: Config):
        """Initialize the provider doctor.

        Args:
            config: Application configuration.
        """
        self.config = config

    async def check_all(self, benchmark: bool = False) -> dict[str, dict[str, Any]]:
        """Check all configured providers.

        Args:
            benchmark: Whether to run performance benchmarks.

        Returns:
            Dictionary mapping provider IDs to health reports.
        """
        results = {}

        tasks = [
            self.check_provider(provider_id, benchmark) for provider_id in self.config.providers
        ]

        provider_results = await asyncio.gather(*tasks, return_exceptions=True)

        for provider_id, result in zip(self.config.providers, provider_results, strict=False):
            if isinstance(result, Exception):
                results[provider_id] = {
                    "status": "error",
                    "error": str(result),
                }
            else:
                results[provider_id] = result

        return results

    async def check_provider(self, provider_id: str, benchmark: bool = False) -> dict[str, Any]:
        """Check a single provider.

        Args:
            provider_id: The provider ID to check.
            benchmark: Whether to run performance benchmarks.

        Returns:
            Dictionary with health check results.
        """
        provider_config = self.config.get_provider(provider_id)
        if not provider_config:
            return {
                "status": "not_found",
                "error": f"Provider {provider_id} not found in configuration",
            }

        api_key = self.config.get_api_key(provider_id)

        # For some providers, API key is optional
        if not api_key and provider_config.type != "openai_compat":
            return {
                "status": "missing_api_key",
                "error": f"API key not found for {provider_id}",
            }

        provider = OpenAICompatProvider(
            provider_id=provider_config.id,
            base_url=provider_config.base_url,
            api_key=api_key,
        )

        report: dict[str, Any] = {
            "status": "unknown",
            "type": provider_config.type,
            "base_url": provider_config.base_url,
            "default_model": provider_config.default_model,
            "api_key_set": bool(api_key),
        }

        # Test basic connectivity
        try:
            healthy = await provider.health_check()
            if healthy:
                report["status"] = "healthy"
            else:
                report["status"] = "unhealthy"
                report["error"] = "Health check failed"
        except Exception as e:
            report["status"] = "error"
            report["error"] = str(e)
            return report

        # Probe capabilities
        try:
            models = await provider.list_models()
            report["models_available"] = len(models)
            report["models_sample"] = models[:5] if models else []

            report["supports_streaming"] = provider.supports_streaming()
            report["supports_tools"] = provider.supports_tools()
            report["max_context_estimate"] = provider.max_context_estimate()
        except Exception as e:
            report["capabilities_error"] = str(e)

        # Validate default model
        if provider_config.default_model:
            try:
                models = await provider.list_models()
                if provider_config.default_model not in models:
                    report["default_model_warning"] = (
                        f"Default model '{provider_config.default_model}' not found in available models"
                    )
            except Exception:
                pass

        # Run benchmarks if requested
        if benchmark:
            try:
                benchmark_results = await provider.benchmark(num_requests=3)
                report["benchmark"] = benchmark_results
            except Exception as e:
                report["benchmark_error"] = str(e)

        # Check for OpenRouter-specific requirements
        if "openrouter" in provider_id.lower():
            # Note: This is a simplified check. OpenRouter recommends HTTP-Referer and X-Title headers
            report["warning"] = (
                "OpenRouter recommends HTTP-Referer and X-Title headers for attribution"
            )

        if report.get("status") != "healthy":
            report["error"] = report.get("error", "Unknown error")

        return report

    def format_report(self, results: dict[str, dict[str, Any]]) -> str:
        """Format provider health check results as a human-readable string.

        Args:
            results: Dictionary of provider health check results.

        Returns:
            Formatted report string.
        """
        lines = ["Provider Health Report", "=" * 50, ""]

        for provider_id, report in results.items():
            status_emoji = {
                "healthy": "✅",
                "unhealthy": "❌",
                "error": "⚠️",
                "not_found": "❓",
                "missing_api_key": "🔑",
            }.get(report.get("status", "unknown"), "❓")

            lines.append(f"{status_emoji} {provider_id}")
            lines.append(f"   Status: {report.get('status', 'unknown')}")
            lines.append(f"   Type: {report.get('type', 'N/A')}")
            lines.append(f"   Base URL: {report.get('base_url', 'N/A')}")
            lines.append(f"   API Key: {'Set' if report.get('api_key_set') else 'Not set'}")

            if "models_available" in report:
                lines.append(f"   Models: {report['models_available']} available")
                if report.get("models_sample"):
                    lines.append(f"   Sample: {', '.join(report['models_sample'])}")

            if "supports_streaming" in report:
                lines.append(f"   Streaming: {'Yes' if report['supports_streaming'] else 'No'}")
            if "supports_tools" in report:
                lines.append(f"   Tools: {'Yes' if report['supports_tools'] else 'No'}")

            if "benchmark" in report:
                bench = report["benchmark"]
                lines.append(f"   Latency: {bench['avg_latency_s']:.3f}s avg")
                lines.append(f"   Throughput: {bench['avg_tokens_per_sec']:.1f} tok/s")

            if "error" in report:
                lines.append(f"   Error: {report['error']}")
            if "warning" in report:
                lines.append(f"   ⚠️  Warning: {report['warning']}")

            lines.append("")

        return "\n".join(lines)
