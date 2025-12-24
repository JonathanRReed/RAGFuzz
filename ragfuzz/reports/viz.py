"""Visualization of mutation graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MutationGraphViz:
    """Visualizes mutation graphs as ASCII trees."""

    def __init__(self, format: str = "ascii"):
        """Initialize the visualizer.

        Args:
            format: Output format ('ascii' or 'mermaid').
        """
        self.format = format

    def visualize_case(self, case_path: str | Path) -> str:
        """Visualize the mutation graph for a case.

        Args:
            case_path: Path to the case JSON file.

        Returns:
            String representation of the mutation graph.
        """
        case_path = Path(case_path)

        if not case_path.exists():
            raise FileNotFoundError(f"Case file not found: {case_path}")

        case_data = json.loads(case_path.read_text())

        if self.format == "mermaid":
            return self._render_mermaid(case_data)
        else:
            return self._render_ascii(case_data)

    def _render_ascii(self, case_data: dict[str, Any]) -> str:
        """Render ASCII tree visualization.

        Args:
            case_data: Case data dictionary.

        Returns:
            ASCII tree string.
        """
        lines = []
        case_id = case_data.get("case_id", "unknown")
        mutation_path = case_data.get("mutation_path", [])

        lines.append(f"Case: {case_id}")
        lines.append("=" * 60)
        lines.append("")

        if not mutation_path:
            lines.append("(No mutation path - seed input)")
            return "\n".join(lines)

        for i, step in enumerate(mutation_path):
            node_id = step.get("node_id", "unknown")[:8]
            mutator = step.get("mutator", "unknown")
            output = step.get("output", "")

            lines.append(f"{'┌' if i > 0 else ''} {node_id} [{mutator}]")
            lines.append(f"│  Output: {output[:100]}{'...' if len(output) > 100 else ''}")
            if i < len(mutation_path) - 1:
                lines.append("│  └──")
            else:
                lines.append("└── Final")

        return "\n".join(lines)

    def _render_mermaid(self, case_data: dict[str, Any]) -> str:
        """Render Mermaid diagram.

        Args:
            case_data: Case data dictionary.

        Returns:
            Mermaid diagram string.
        """
        lines = []
        case_id = case_data.get("case_id", "unknown").replace("-", "_")
        mutation_path = case_data.get("mutation_path", [])

        lines.append("graph TD")
        lines.append(f"    Case_{case_id}[Case: {case_id}]")

        if not mutation_path:
            lines.append(f"    Case_{case_id}[Seed Input]")
            return "\n".join(lines)

        for i, step in enumerate(mutation_path):
            node_id = step.get("node_id", "unknown").replace("-", "_")[:16]
            mutator = step.get("mutator", "unknown")
            output = step.get("output", "")[:50]

            safe_node_id = f"Node_{i}_{node_id}"
            lines.append(f"    {safe_node_id}[{mutator}\\n{output}]")

            if i > 0:
                prev_node_id = (
                    f"Node_{i - 1}_{mutation_path[i - 1].get('node_id', '').replace('-', '_')[:16]}"
                )
                lines.append(f"    {prev_node_id} --> {safe_node_id}")

        return "\n".join(lines)

    def save_mermaid(self, case_path: str | Path, output_path: str | Path) -> None:
        """Save Mermaid diagram to file.

        Args:
            case_path: Path to the case JSON file.
            output_path: Path to save the Mermaid file.
        """
        mermaid_content = self.visualize_case(case_path)
        Path(output_path).write_text(mermaid_content)
