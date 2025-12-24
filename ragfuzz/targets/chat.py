"""Chat target - direct model calls via provider."""

from __future__ import annotations

from typing import Any

from ragfuzz.models import Message, Response
from ragfuzz.providers.base import Provider
from ragfuzz.targets.base import Target


class ChatTarget(Target):
    """Target that sends requests directly through a provider."""

    def __init__(self, target_id: str, provider: Provider, default_model: str | None = None):
        """Initialize the chat target.

        Args:
            target_id: Unique identifier for this target.
            provider: The provider to use.
            default_model: Default model to use.
        """
        super().__init__(
            target_id=target_id,
            endpoint=provider.base_url,
            capabilities=["chat"],
        )
        self.provider = provider
        self.default_model = default_model or provider.__dict__.get("default_model", "")

    async def execute(self, input_data: dict[str, Any]) -> Response:
        """Execute a test case by sending it through the provider.

        Args:
            input_data: Input data containing messages and optional parameters.

        Returns:
            A Response object.
        """
        messages = input_data.get("messages", [])
        model = input_data.get("model", self.default_model)
        temperature = input_data.get("temperature", 0.7)
        max_tokens = input_data.get("max_tokens")
        tools = input_data.get("tools")
        stream = input_data.get("stream", False)

        message_objects = []
        for msg in messages:
            if isinstance(msg, Message):
                message_objects.append(msg)
            elif isinstance(msg, dict):
                message_objects.append(Message(**msg))

        return await self.provider.chat(
            messages=message_objects,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=stream,
        )
