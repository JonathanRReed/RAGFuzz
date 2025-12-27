"""Run directory management."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragfuzz.config import Config


class RunDir:
    """Manages a run directory for storing test results."""

    def __init__(self, base_path: str | Path = "runs", run_id: str | None = None):
        """Initialize a run directory.

        Args:
            base_path: Base path for runs.
            run_id: Optional run ID. If not provided, generates one.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        if run_id:
            self.run_id = run_id
        else:
            self.run_id = self._generate_run_id()

        self.path = self.base_path / self.run_id
        self.path.mkdir(exist_ok=True)

        self.create_subdirs()

    def _generate_run_id(self) -> str:
        """Generate a unique run ID.

        Returns:
            A run ID string.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        import random

        suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
        return f"{timestamp}_{suffix}"

    def create_subdirs(self) -> None:
        """Create subdirectories for run."""
        subdirs = ["failures", "artifacts"]
        for subdir in subdirs:
            (self.path / subdir).mkdir(exist_ok=True)

    def write_run_config(
        self, config: Config, suite_config: Any, extra: dict[str, Any] | None = None
    ) -> None:
        """Write run configuration.

        Args:
            config: Application configuration.
            suite_config: Suite configuration.
            extra: Extra metadata.
        """
        extra = extra or {}

        run_config = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "default_provider": config.default_provider,
                "default_target": config.default_target,
                "budget": {
                    "max_runs": config.budget.max_runs,
                    "max_cost_usd": config.budget.max_cost_usd,
                },
            },
            "suite": suite_config.model_dump()
            if hasattr(suite_config, "model_dump")
            else suite_config,
            "extra": extra,
        }

        path = self.path / "run.json"
        path.write_text(json.dumps(run_config, indent=2, default=str))

    def generate_rag_lens_url(self, trace_id: str, base_url: str | None = None) -> str | None:
        """Generate a deep link to RAG Lens for a trace.

        Args:
            trace_id: The trace ID.
            base_url: Optional base URL for RAG Lens. Defaults to config value.

        Returns:
            The RAG Lens deep link URL or None.
        """
        if not trace_id or not trace_id.strip():
            return None

        rag_lens_base = base_url
        if not rag_lens_base or not rag_lens_base.strip():
            return None

        if not rag_lens_base.startswith(("http://", "https://")):
            return None

        return f"{rag_lens_base.rstrip('/')}/trace/{trace_id}"

    def write_case(self, case: dict[str, Any]) -> None:
        """Write a case to cases.jsonl file.

        Args:
            case: The case data to write.
        """
        try:
            import portalocker  # type: ignore[import-not-found]

            cases_file = self.path / "cases.jsonl"
            with cases_file.open("a") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                try:
                    f.write(json.dumps(case, default=str) + "\n")
                finally:
                    portalocker.unlock(f)
        except ImportError:
            cases_file = self.path / "cases.jsonl"
            with cases_file.open("a") as f:
                f.write(json.dumps(case, default=str) + "\n")

    def write_failure(self, case_id: str, case_data: dict[str, Any]) -> None:
        """Write a minimized failure case.

        Args:
            case_id: The case ID.
            case_data: The case data.
        """
        path = self.path / "failures" / f"{case_id}.json"
        path.write_text(json.dumps(case_data, indent=2, default=str))

    def get_path(self) -> Path:
        """Get run directory path.

        Returns:
            The Path object.
        """
        return self.path

    def get_cases_path(self) -> Path:
        """Get path to cases.jsonl.

        Returns:
            The Path object.
        """
        return self.path / "cases.jsonl"

    def get_failures_path(self) -> Path:
        """Get path to failures directory.

        Returns:
            The Path object.
        """
        return self.path / "failures"
