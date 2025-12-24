"""Tests for storage module."""

import json
from pathlib import Path

from ragfuzz.config import BudgetConfig, Config, ProviderConfig
from ragfuzz.storage import RunDir


class TestRunDir:
    """Test run directory management."""

    def test_run_dir_creation(self, tmp_path: Path) -> None:
        """Test run directory creation."""
        run_dir = RunDir(base_path=tmp_path)

        assert run_dir.path.exists()
        assert (run_dir.path / "failures").exists()
        assert (run_dir.path / "artifacts").exists()

    def test_run_dir_with_custom_id(self, tmp_path: Path) -> None:
        """Test run directory with custom run ID."""
        run_dir = RunDir(base_path=tmp_path, run_id="custom_run_123")

        assert run_dir.run_id == "custom_run_123"
        assert (tmp_path / "custom_run_123").exists()

    def test_write_run_config(self, tmp_path: Path) -> None:
        """Test writing run configuration."""
        run_dir = RunDir(base_path=tmp_path)
        config = Config(
            providers={
                "test": ProviderConfig(
                    id="test",
                    type="openai_compat",
                    base_url="http://localhost/v1",
                    api_key_env="TEST_KEY",
                    default_model="model",
                )
            },
            budget=BudgetConfig(max_runs=100),
        )

        class MockSuite:
            def model_dump(self):
                return {"name": "test-suite"}

        run_dir.write_run_config(config, MockSuite())

        config_path = run_dir.path / "run.json"
        assert config_path.exists()

        data = json.loads(config_path.read_text())
        assert data["run_id"] == run_dir.run_id
        assert data["config"]["default_provider"] is None
        assert data["suite"]["name"] == "test-suite"

    def test_write_case(self, tmp_path: Path) -> None:
        """Test writing a case."""
        run_dir = RunDir(base_path=tmp_path)
        case = {
            "case_id": "case_000001",
            "run_id": run_dir.run_id,
            "suite_id": "test",
            "scores": {"leak_score": 1.0},
        }

        run_dir.write_case(case)

        cases_path = run_dir.path / "cases.jsonl"
        assert cases_path.exists()

        line = cases_path.read_text().strip()
        loaded = json.loads(line)
        assert loaded["case_id"] == "case_000001"

    def test_write_failure(self, tmp_path: Path) -> None:
        """Test writing a failure case."""
        run_dir = RunDir(base_path=tmp_path)
        case = {"case_id": "failed_case", "leaked": True}

        run_dir.write_failure("failed_case", case)

        failure_path = run_dir.path / "failures" / "failed_case.json"
        assert failure_path.exists()

        data = json.loads(failure_path.read_text())
        assert data["case_id"] == "failed_case"
