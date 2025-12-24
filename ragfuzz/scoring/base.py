"""Base scoring interface."""

from __future__ import annotations

from typing import Any

from ragfuzz.models import Response, ScoreVector


class Scorer:
    """Abstract base class for scorers."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the scorer.

        Args:
            config: Optional configuration for the scorer.
        """
        self.config = config or {}

    async def score(self, response: Response, context: dict[str, Any] | None = None) -> ScoreVector:
        """Score a response.

        Args:
            response: The response to score.
            context: Optional context for scoring.

        Returns:
            A ScoreVector.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
