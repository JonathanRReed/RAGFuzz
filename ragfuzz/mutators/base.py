"""Base mutator interface."""

from __future__ import annotations

from typing import Any


class Mutator:
    """Abstract base class for prompt mutators."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the mutator.

        Args:
            name: Unique identifier for this mutator.
            config: Optional configuration for the mutator.
        """
        self.name = name
        self.config = config or {}
        self._node_id_counter = 0

    async def mutate(self, input_text: str, context: dict[str, Any] | None = None) -> str:
        """Apply a mutation to the input text.

        Args:
            input_text: The input text to mutate.
            context: Optional context for the mutation.

        Returns:
            The mutated text.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def generate_node_id(self, parent_id: str, output_text: str) -> str:
        """Generate a unique node ID based on parent and output.

        Args:
            parent_id: The parent node ID.
            output_text: The output text of this mutation.

        Returns:
            A unique node ID.
        """
        import hashlib

        combined = f"{parent_id}:{self.name}:{output_text}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
