"""Shared data models for ragfuzz."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A chat message."""

    role: str
    content: str


class ToolCall(BaseModel):
    """A tool call from the model."""

    name: str
    arguments: dict[str, Any]


class Response(BaseModel):
    """Response from a provider or target."""

    content: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class ScoreVector(BaseModel):
    """Vector of scores for a case."""

    leak_score: float = 0.0
    policy_violation_score: float = 0.0
    partial_success_score: float = 0.0
    refusal_latency_delta: float = 0.0
    tool_error_rate: float = 0.0
    retrieval_poison_influence: float = 0.0


class Case(BaseModel):
    """A single test case."""

    case_id: str
    run_id: str
    suite_id: str
    inputs: dict[str, Any]
    mutation_trace: list[str] = Field(default_factory=list)
    target_response: Response | None = None
    scores: ScoreVector = Field(default_factory=ScoreVector)
    trace_id: str | None = None
    rag_lens_url: str | None = None
    retrieval_snapshot: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    failure_signature: str | None = None
    mutation_graph_node_id: str | None = None
    mutation_path: list[dict[str, Any]] = Field(default_factory=list)


class FailureSignature(BaseModel):
    """Signature of a failure for deduplication."""

    suite_id: str
    target_id: str
    score_bin: str
    canary_hit_type: str | None
    tool_call_shape: str | None
    top_k_chunk_ids: list[str] = Field(default_factory=list)

    def to_hash(self) -> str:
        """Convert to a hash string for deduplication."""
        import hashlib
        import json

        data = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
