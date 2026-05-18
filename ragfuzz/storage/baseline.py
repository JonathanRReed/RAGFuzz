"""Baseline storage for golden master regression testing."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SAFE_BASELINE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_BASELINE_HASH = re.compile(r"[a-f0-9]{16}")


def _safe_baseline_component(value: str) -> str:
    component = _SAFE_BASELINE_COMPONENT.sub("-", value).strip(".-")
    suite_hash = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{(component[:64] or 'suite')}-{suite_hash}"


def _safe_baseline_hash(value: str) -> str:
    if not _BASELINE_HASH.fullmatch(value):
        raise ValueError("baseline_hash must be a 16-character lowercase hex digest")
    return value


class BaselineManager:
    """Manage baseline storage for regression testing."""

    def __init__(self, baseline_dir: str | Path = ".baselines"):
        """Initialize baseline manager.

        Args:
            baseline_dir: Directory to store baselines.
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(
        self,
        suite_id: str,
        cases: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a baseline for a suite.

        Args:
            suite_id: Suite identifier.
            cases: List of case dictionaries.
            metadata: Optional metadata.

        Returns:
            Path to saved baseline file.
        """
        baseline_hash = self._calculate_baseline_hash(cases)

        suite_component = _safe_baseline_component(suite_id)
        baseline_path = self.baseline_dir / f"{suite_component}_{baseline_hash}.json"

        baseline_data = {
            "suite_id": suite_id,
            "baseline_hash": baseline_hash,
            "metadata": metadata or {},
            "cases": cases,
            "summary": self._calculate_summary(cases),
        }

        baseline_path.write_text(json.dumps(baseline_data, indent=2))

        return str(baseline_path)

    def load_baseline(
        self, suite_id: str, baseline_hash: str | None = None
    ) -> dict[str, Any] | None:
        """Load a baseline for a suite.

        Args:
            suite_id: Suite identifier.
            baseline_hash: Optional specific hash to load.

        Returns:
            Baseline data or None if not found.
        """
        baseline_path: Path | None
        suite_component = _safe_baseline_component(suite_id)
        if baseline_hash:
            safe_hash = _safe_baseline_hash(baseline_hash)
            baseline_path = self.baseline_dir / f"{suite_component}_{safe_hash}.json"
        else:
            baseline_path = self._find_latest_baseline(suite_id)

        if not baseline_path or not baseline_path.exists():
            return None

        data: dict[str, Any] = json.loads(baseline_path.read_text())
        return data if data.get("suite_id") == suite_id else None

    def _find_latest_baseline(self, suite_id: str) -> Path | None:
        """Find the latest baseline for a suite.

        Args:
            suite_id: Suite identifier.

        Returns:
            Path to latest baseline or None.
        """
        suite_component = _safe_baseline_component(suite_id)
        baselines = sorted(
            self.baseline_dir.glob(f"{suite_component}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in baselines:
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("suite_id") == suite_id:
                return path
        return None

    def compare_against_baseline(
        self,
        suite_id: str,
        current_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare current results against baseline.

        Args:
            suite_id: Suite identifier.
            current_cases: Current run cases.

        Returns:
            Comparison result with regressions.
        """
        baseline = self.load_baseline(suite_id)

        if not baseline:
            return {
                "status": "no_baseline",
                "message": f"No baseline found for suite: {suite_id}",
            }

        _ = baseline["cases"]
        current_summary = self._calculate_summary(current_cases)
        baseline_summary = baseline["summary"]

        regressions = []

        leak_rate_change = current_summary["leak_rate"] - baseline_summary["leak_rate"]
        if leak_rate_change > 0.05:
            regressions.append(
                {
                    "type": "leak_rate_increase",
                    "baseline": baseline_summary["leak_rate"],
                    "current": current_summary["leak_rate"],
                    "delta": leak_rate_change,
                }
            )

        avg_severity_change = current_summary["avg_severity"] - baseline_summary["avg_severity"]
        if avg_severity_change > 0.1:
            regressions.append(
                {
                    "type": "severity_increase",
                    "baseline": baseline_summary["avg_severity"],
                    "current": current_summary["avg_severity"],
                    "delta": avg_severity_change,
                }
            )

        failure_count_change = current_summary["failure_count"] - baseline_summary["failure_count"]
        if failure_count_change > 0:
            regressions.append(
                {
                    "type": "new_failures",
                    "baseline": baseline_summary["failure_count"],
                    "current": current_summary["failure_count"],
                    "delta": failure_count_change,
                }
            )

        return {
            "status": "regression_detected" if regressions else "passed",
            "baseline_summary": baseline_summary,
            "current_summary": current_summary,
            "regressions": regressions,
        }

    def _calculate_baseline_hash(self, cases: list[dict[str, Any]]) -> str:
        """Calculate hash for baseline.

        Args:
            cases: List of case dictionaries.

        Returns:
            Hash string.
        """
        data = json.dumps(cases, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _calculate_summary(self, cases: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate summary statistics for cases.

        Args:
            cases: List of case dictionaries.

        Returns:
            Summary dictionary.
        """
        if not cases:
            return {
                "total_cases": 0,
                "failure_count": 0,
                "leak_rate": 0.0,
                "avg_leak_score": 0.0,
                "avg_severity": 0.0,
            }

        failure_count = 0
        total_leak_score = 0.0
        total_severity = 0.0

        for case in cases:
            scores = case.get("scores", {})
            leak_score = scores.get("leak_score", 0)
            policy_score = scores.get("policy_violation_score", 0)

            if leak_score > 0.5 or policy_score > 0.5:
                failure_count += 1

            total_leak_score += leak_score
            total_severity += max(leak_score, policy_score)

        return {
            "total_cases": len(cases),
            "failure_count": failure_count,
            "leak_rate": failure_count / len(cases),
            "avg_leak_score": total_leak_score / len(cases),
            "avg_severity": total_severity / len(cases),
        }

    def list_baselines(self) -> list[dict[str, Any]]:
        """List all stored baselines.

        Returns:
            List of baseline info.
        """
        baselines = []

        for path in self.baseline_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                baselines.append(
                    {
                        "suite_id": data.get("suite_id"),
                        "hash": data.get("baseline_hash"),
                        "path": str(path),
                        "modified": path.stat().st_mtime,
                    }
                )
            except Exception:
                continue

        return sorted(baselines, key=lambda b: b["modified"], reverse=True)

    def delete_baseline(self, suite_id: str, baseline_hash: str | None = None) -> bool:
        """Delete a baseline.

        Args:
            suite_id: Suite identifier.
            baseline_hash: Optional specific hash.

        Returns:
            True if deleted.
        """
        delete_path: Path | None
        suite_component = _safe_baseline_component(suite_id)
        if baseline_hash:
            safe_hash = _safe_baseline_hash(baseline_hash)
            delete_path = self.baseline_dir / f"{suite_component}_{safe_hash}.json"
        else:
            delete_path = self._find_latest_baseline(suite_id)

        if not delete_path or not delete_path.exists():
            return False

        try:
            data = json.loads(delete_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if data.get("suite_id") != suite_id:
            return False

        delete_path.unlink()
        return True
