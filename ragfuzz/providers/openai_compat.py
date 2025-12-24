"""OpenAI-compatible provider implementation."""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ragfuzz.models import Message, Response, ToolCall
from ragfuzz.providers.base import Provider
from ragfuzz.utils import get_async_client


class OpenAICompatProvider(Provider):
    """Provider for OpenAI-compatible endpoints (LM Studio, Ollama, OpenRouter)."""

    def __init__(
        self,
        provider_id: str,
        base_url: str,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        max_retries: int = 3,
    ):
        """Initialize the OpenAI-compatible provider.

        Args:
            provider_id: Unique identifier for this provider.
            base_url: Base URL for the OpenAI-compatible API.
            api_key: Optional API key for authentication.
            default_headers: Additional default headers to include.
            max_retries: Maximum number of retry attempts.
        """
        super().__init__(provider_id, base_url, api_key)
        self.default_headers = default_headers or {}
        self._max_retries = max_retries
        self._supports_streaming = True
        self._supports_tools = True
        self._max_context_estimate = 4096

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        timeout: float = 30.0,
    ) -> Response:
        """Send a chat completion request to an OpenAI-compatible endpoint.

        Args:
            messages: List of chat messages.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Optional tool definitions.
            stream: Whether to stream responses.
            timeout: Request timeout in seconds.

        Returns:
            A Response object.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        client = get_async_client()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.default_headers)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools

        payload["stream"] = stream

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        if stream:
            return await self._chat_stream(client, url, headers, payload)

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
            ),
        )
        async def _make_request():
            response = await client.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response

        response = await _make_request()
        data = response.json()
        return self._parse_response(data)

    async def _chat_stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Response:
        """Handle streaming chat completion.

        Args:
            client: HTTP client.
            url: Request URL.
            headers: Request headers.
            payload: Request payload.

        Returns:
            A Response object with accumulated content.
        """
        full_content = ""
        model = ""
        usage = {}

        async with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            full_content += delta.get("content", "")
                            if not model and "model" in data:
                                model = data["model"]
                        if "usage" in data:
                            usage = data["usage"]
                    except json.JSONDecodeError:
                        continue

        return Response(
            content=full_content,
            model=model,
            usage=usage,
        )

    def _parse_response(self, data: dict[str, Any]) -> Response:
        """Parse an OpenAI-style chat completion response.

        Args:
            data: JSON response data.

        Returns:
            A Response object.
        """
        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                )

        return Response(
            content=message.get("content", ""),
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            tool_calls=tool_calls,
        )

    async def list_models(self) -> list[str]:
        """List available models from the provider.

        Returns:
            List of model identifiers.
        """
        client = get_async_client()

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url.rstrip('/')}/models"

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
            ),
        )
        async def _make_request():
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response

        response = await _make_request()
        data = response.json()
        return [model["id"] for model in data.get("data", [])]

    def supports_streaming(self) -> bool:
        """Check if the provider supports streaming.

        Returns:
            True if streaming is supported.
        """
        return self._supports_streaming

    def supports_tools(self) -> bool:
        """Check if the provider supports tool calling.

        Returns:
            True if tool calling is supported.
        """
        return self._supports_tools

    def max_context_estimate(self) -> int:
        """Get the estimated max context length.

        Returns:
            Estimated max context length in tokens.
        """
        return self._max_context_estimate
