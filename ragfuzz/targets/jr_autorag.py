"""JR AutoRAG target with grey-box auditing capabilities."""

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


class JRAutoRAGTarget(Target):
    """Target for JR AutoRAG with grey-box audit endpoints."""

    def __init__(self, target_id: str, base_url: str, headers: dict[str, str] | None = None):
        """Initialize the JR AutoRAG target.

        Args:
            target_id: Unique identifier for this target.
            base_url: Base URL for the JR AutoRAG API.
            headers: Optional additional headers.
        """
        super().__init__(
            target_id=target_id,
            endpoint=base_url,
            capabilities=[
                "chat",
                "http",
                "retrieval_trace",
                "ingestion",
                "vector_ops",
            ],
        )
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}

    async def execute(self, input_data: dict[str, Any]) -> Response:
        """Execute a test case via. RAG Audit API.

        Args:
            input_data: Input data including query and metadata.

        Returns:
            A Response object with retrieval trace information.
        """
        client = get_async_client()

        url = f"{self.base_url}/rag/audit/query"

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
            ),
        )
        async def _make_request():
            response = await client.post(url, json=input_data, headers=self.headers)
            response.raise_for_status()
            return response

        response = await _make_request()
        data = response.json()

        return Response(
            content=data.get("answer", ""),
            model="jr_autorag",
            metadata={
                "trace_id": data.get("trace_id"),
                "retrieval": data.get("retrieval", {}),
                "prompt": data.get("prompt"),
                "timing": data.get("timing", {}),
            },
            trace_id=data.get("trace_id"),
        )

    async def ingest_documents(
        self, documents: list[dict[str, Any]], tags: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Ingest documents into. RAG system.

        Args:
            documents: List of documents to ingest.
            tags: Optional tags for documents.

        Returns:
            Ingestion result.
        """
        client = get_async_client()
        url = f"{self.base_url}/rag/audit/ingest"

        payload: dict[str, Any] = {"documents": documents}
        if tags:
            payload["tags"] = tags

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
            ),
        )
        async def _make_request():
            response = await client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response

        response = await _make_request()
        result: dict[str, Any] = response.json()
        return result

    async def upsert_chunks(self, chunks: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
        """Upsert chunks directly into. vector store.

        Args:
            chunks: List of chunks to upsert.
            run_id: Run ID for tagging.

        Returns:
            Upsert result.
        """
        client = get_async_client()
        url = f"{self.base_url}/rag/audit/upsert_chunks"

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
            ),
        )
        async def _make_request():
            response = await client.post(
                url, json={"chunks": chunks, "run_id": run_id}, headers=self.headers
            )
            response.raise_for_status()
            return response

        response = await _make_request()
        result: dict[str, Any] = response.json()
        return result

    async def delete_by_tag(self, tag_key: str, tag_value: str) -> dict[str, Any]:
        """Delete documents by tag.

        Args:
            tag_key: Tag key.
            tag_value: Tag value.

        Returns:
            Deletion result.
        """
        client = get_async_client()
        url = f"{self.base_url}/rag/audit/delete_by_tag"

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
            ),
        )
        async def _make_request():
            response = await client.post(
                url, json={"tag_key": tag_key, "tag_value": tag_value}, headers=self.headers
            )
            response.raise_for_status()
            return response

        response = await _make_request()
        result: dict[str, Any] = response.json()
        return result

    async def get_trace(self, trace_id: str) -> dict[str, Any]:
        """Get full trace for. a query.

        Args:
            trace_id: Trace ID.

        Returns:
            Full trace data.
        """
        client = get_async_client()
        url = f"{self.base_url}/rag/audit/trace/{trace_id}"

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
            ),
        )
        async def _make_request():
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            return response

        response = await _make_request()
        result: dict[str, Any] = response.json()
        return result
