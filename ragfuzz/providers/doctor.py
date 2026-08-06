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
        results: dict[str, Any] = {}

        tasks = [
            self.check_provider(provider_id, benchmark) for provider_id in self.config.providers
        ]

        provider_results = await asyncio.gather(*tasks, return_exceptions=True)

        provider_ids = list(self.config.providers)
        for index, provider_id in enumerate(provider_ids):
            result = provider_results[index] if index < len(provider_results) else None
            if isinstance(result, BaseException):
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

        models: list[str] = []

        # Probe capabilities
        try:
            models = await provider.list_models()
            report["models_available"] = len(models)
            report["models_sample"] = models[:5] if models else []
            selected_model = self._select_chat_model(provider_id, provider_config.default_model, models)
            report["selected_model"] = selected_model

            report["supports_streaming"] = provider.supports_streaming()
            report["supports_tools"] = provider.supports_tools()
            report["max_context_estimate"] = provider.max_context_estimate()
        except Exception as e:
            report["capabilities_error"] = str(e)

        # Validate default model
        if provider_config.default_model and provider_config.default_model != "auto":
            try:
                if provider_config.default_model not in models:
                    report["default_model_warning"] = (
                        f"Default model '{provider_config.default_model}' not found in available models"
                    )
            except Exception:
                pass

        # Run benchmarks if requested
        if benchmark:
            try:
                benchmark_model = str(report.get("selected_model") or provider_config.default_model)
                benchmark_results = await provider.benchmark(
                    num_requests=3,
                    model=benchmark_model,
                )
                report["benchmark"] = benchmark_results
            except Exception as e:
                report["benchmark_error"] = str(e)

        if report.get("status") != "healthy":
            report["error"] = report.get("error", "Unknown error")

        return report

    def _select_chat_model(
        self,
        provider_id: str,
        configured_model: str,
        models: list[str],
    ) -> str | None:
        if not models:
            return None
        if configured_model != "auto" and configured_model in models:
            return configured_model

        lowered_provider = provider_id.lower()
        chat_models = [
            model
            for model in models
            if "embed" not in model.lower() and "rerank" not in model.lower()
        ]
        if lowered_provider == "ollama" and chat_models:
            return chat_models[0]
        return chat_models[0] if chat_models else models[0]

    def format_report(self, results: dict[str, dict[str, Any]]) -> str:
        """Format provider health check results as a human-readable string.

        Args:
            results: Dictionary of provider health check results.

        Returns:
            Formatted report string.
        """
        lines = ["Provider Health Report", "=" * 50, ""]

        for provider_id, report in results.items():
            status_label = {
                "healthy": "OK",
                "unhealthy": "FAIL",
                "error": "WARN",
                "not_found": "MISSING",
                "missing_api_key": "KEY",
            }.get(report.get("status", "unknown"), "UNKNOWN")

            lines.append(f"{status_label} {provider_id}")
            lines.append(f"   Status: {report.get('status', 'unknown')}")
            lines.append(f"   Type: {report.get('type', 'N/A')}")
            lines.append(f"   Base URL: {report.get('base_url', 'N/A')}")
            api_key_status = "Set" if report.get("api_key_set") else "Not required locally"
            lines.append(f"   API Key: {api_key_status}")
            if report.get("selected_model"):
                lines.append(f"   Selected Model: {report['selected_model']}")

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
                lines.append(f"   Warning: {report['warning']}")
            if "default_model_warning" in report:
                lines.append(f"   Warning: {report['default_model_warning']}")
            if "benchmark_error" in report:
                lines.append(f"   Benchmark Error: {report['benchmark_error']}")

            lines.append("")

        return "\n".join(lines)
