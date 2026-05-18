"""Async HTTP client wrapper for OpenAI-compatible endpoints."""

from __future__ import annotations

import os

import httpx

_client: httpx.AsyncClient | None = None
_client_settings: tuple[float, int, bool] | None = None

_TRUST_ENV_TRUE_VALUES = {"1", "true", "yes", "on"}


def should_trust_env(value: str | None = None) -> bool:
    """Return whether outbound HTTP should inherit proxy environment settings."""

    raw_value = os.environ.get("RAGFUZZ_HTTP_TRUST_ENV") if value is None else value
    return bool(raw_value and raw_value.strip().lower() in _TRUST_ENV_TRUE_VALUES)


def get_async_client(
    timeout: float = 300.0,
    max_connections: int = 100,
    *,
    trust_env: bool | None = None,
) -> httpx.AsyncClient:
    """Get or create a shared async HTTP client.

    Args:
        timeout: Request timeout in seconds.
        max_connections: Maximum number of concurrent connections.
        trust_env: Whether httpx should inherit proxy environment settings.
            Defaults to ``RAGFUZZ_HTTP_TRUST_ENV`` and otherwise stays local-first.

    Returns:
        A shared httpx.AsyncClient instance.
    """
    global _client
    global _client_settings

    resolved_trust_env = should_trust_env() if trust_env is None else trust_env
    settings = (timeout, max_connections, resolved_trust_env)
    if _client is None or _client_settings != settings or _client.is_closed:
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=20)
        _client = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=resolved_trust_env)
        _client_settings = settings

    return _client


async def close_client() -> None:
    """Close the shared async HTTP client."""
    global _client
    global _client_settings
    if _client is not None:
        await _client.aclose()
        _client = None
        _client_settings = None
