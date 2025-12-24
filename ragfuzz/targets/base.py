"""Base target interface."""

from __future__ import annotations

from typing import Any

from ragfuzz.models import Response


class Target:
    """Abstract base class for test targets."""

    def __init__(self, target_id: str, endpoint: str, capabilities: list[str] | None = None):
        """Initialize the target.

        Args:
            target_id: Unique identifier for this target.
            endpoint: Target endpoint URL.
            capabilities: List of capabilities supported by this target.
        """
        self.target_id = target_id
        self.endpoint = endpoint
        self.capabilities = capabilities or []

    async def execute(self, input_data: dict[str, Any]) -> Response:
        """Execute a test case against the target.

        Args:
            input_data: Input data for the test case.

        Returns:
            A Response object.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check if the target is accessible.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            await self.execute({"test": "health_check"})
            return True
        except Exception:
            return False

    def has_capability(self, capability: str) -> bool:
        """Check if the target has a specific capability.

        Args:
            capability: The capability to check.

        Returns:
            True if the capability is supported.
        """
        return capability in self.capabilities
