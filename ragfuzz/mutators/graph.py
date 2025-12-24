"""Mutation graph for tracking prompt evolution."""

from __future__ import annotations

import hashlib
from dataclasses import field
from typing import Any

from pydantic import BaseModel


class MutationNode(BaseModel):
    """A node in the mutation graph."""

    node_id: str
    parent_id: str | None
    mutator_name: str
    mutator_config: dict[str, Any] = field(default_factory=dict)
    output: str
    cached: bool = False
    timestamp: float = 0.0


class MutationGraph:
    """DAG representing prompt mutations."""

    def __init__(self) -> None:
        """Initialize an empty mutation graph."""
        self.nodes: dict[str, MutationNode] = {}
        self.adjacency: dict[str, list[str]] = {}

    def add_node(
        self,
        parent_id: str | None,
        mutator_name: str,
        output: str,
        mutator_config: dict[str, Any] | None = None,
        cached: bool = False,
    ) -> str:
        """Add a node to the mutation graph.

        Args:
            parent_id: The parent node ID, or None for root.
            mutator_name: Name of the mutator that created this node.
            output: The output text of this mutation.
            mutator_config: Configuration of the mutator.
            cached: Whether this node was cached.

        Returns:
            The node ID of the new node.
        """
        import time

        mutator_config = mutator_config or {}
        timestamp = time.time()

        node_id = self._generate_node_id(parent_id, mutator_name, output)

        node = MutationNode(
            node_id=node_id,
            parent_id=parent_id,
            mutator_name=mutator_name,
            mutator_config=mutator_config,
            output=output,
            cached=cached,
            timestamp=timestamp,
        )

        self.nodes[node_id] = node

        if parent_id is not None:
            if parent_id not in self.adjacency:
                self.adjacency[parent_id] = []
            self.adjacency[parent_id].append(node_id)

        return node_id

    def get_node(self, node_id: str) -> MutationNode | None:
        """Get a node by ID.

        Args:
            node_id: The node ID.

        Returns:
            The MutationNode or None if not found.
        """
        return self.nodes.get(node_id)

    def get_path(self, node_id: str) -> list[MutationNode]:
        """Get the path from root to the given node.

        Args:
            node_id: The target node ID.

        Returns:
            List of nodes from root to the target.
        """
        path = []
        current_id = node_id

        while current_id is not None:
            node = self.get_node(current_id)
            if node is None:
                break
            path.append(node)
            current_id = node.parent_id

        return list(reversed(path))

    def _generate_node_id(self, parent_id: str | None, mutator_name: str, output: str) -> str:
        """Generate a unique node ID.

        Args:
            parent_id: The parent node ID.
            mutator_name: Name of the mutator.
            output: Output text.

        Returns:
            A unique node ID.
        """
        combined = f"{parent_id}:{mutator_name}:{output}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph to a dictionary for serialization.

        Returns:
            Dictionary representation of the graph.
        """
        return {
            "nodes": {node_id: node.model_dump() for node_id, node in self.nodes.items()},
            "adjacency": self.adjacency,
        }
