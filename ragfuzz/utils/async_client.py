"""Async HTTP client wrapper for OpenAI-compatible endpoints."""

from __future__ import annotations

import httpx

_client: httpx.AsyncClient | None = None


def get_async_client(timeout: float = 300.0, max_connections: int = 100) -> httpx.AsyncClient:
    """Get or create a shared async HTTP client.

    Args:
        timeout: Request timeout in seconds.
        max_connections: Maximum number of concurrent connections.

    Returns:
        A shared httpx.AsyncClient instance.
    """
    global _client

    if _client is None:
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=20)
        _client = httpx.AsyncClient(timeout=timeout, limits=limits)

    return _client


async def close_client() -> None:
    """Close the shared async HTTP client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
