"""Generic HTTP target."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ragfuzz.models import Response
from ragfuzz.targets.base import Target
from ragfuzz.utils import get_async_client


class HTTPTarget(Target):
    """Target that sends requests to a generic HTTP endpoint."""

    def __init__(self, target_id: str, endpoint: str, headers: dict[str, str] | None = None):
        """Initialize HTTP target.

        Args:
            target_id: Unique identifier for this target.
            endpoint: Target endpoint URL.
            headers: Optional additional headers.
        """
        super().__init__(target_id=target_id, endpoint=endpoint, capabilities=["http"])
        self.headers = headers or {}

    async def execute(self, input_data: dict[str, Any]) -> Response:
        """Execute a test case by POSTing to a HTTP endpoint.

        Args:
            input_data: Input data to send in request body.

        Returns:
            A Response object.
        """
        client = get_async_client()

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
            ),
        )
        async def _make_request():
            response = await client.post(
                self.endpoint,
                json=input_data,
                headers=self.headers,
            )
            response.raise_for_status()
            return response

        response = await _make_request()
        data = response.json()
        output = data.get("output", "")
        metadata = {k: v for k, v in data.items() if k != "output"}

        return Response(
            content=output,
            model="http_target",
            metadata=metadata,
        )
