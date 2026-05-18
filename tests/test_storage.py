"""Tests for storage module."""

import json
from pathlib import Path

import pytest

from ragfuzz.config import BudgetConfig, Config, ProviderConfig
from ragfuzz.storage import RunDir
from ragfuzz.storage.baseline import BaselineManager


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


class TestBaselineManager:
    def test_baseline_suite_ids_cannot_escape_baseline_directory(self, tmp_path: Path) -> None:
        manager = BaselineManager(tmp_path / ".baselines")

        saved_path = Path(manager.save_baseline("../escaped", []))

        assert saved_path.parent == tmp_path / ".baselines"
        assert saved_path.name.startswith("escaped-")
        assert (tmp_path / "escaped_4f53cda18c2baa0c.json").exists() is False

    def test_baseline_hash_must_be_safe(self, tmp_path: Path) -> None:
        manager = BaselineManager(tmp_path / ".baselines")

        with pytest.raises(ValueError, match="baseline_hash"):
            manager.load_baseline("suite", "../not-safe")

    def test_delete_baseline_sanitizes_suite_ids(self, tmp_path: Path) -> None:
        manager = BaselineManager(tmp_path / ".baselines")
        saved_path = Path(manager.save_baseline("../escaped", []))

        assert manager.delete_baseline("../escaped", "4f53cda18c2baa0c") is True
        assert not saved_path.exists()

    def test_baseline_suite_id_collisions_do_not_cross_wire(self, tmp_path: Path) -> None:
        manager = BaselineManager(tmp_path / ".baselines")

        first_path = Path(
            manager.save_baseline(
                "suite/a",
                [{"case_id": "first", "scores": {"leak_score": 0.1}}],
            )
        )
        second_path = Path(
            manager.save_baseline(
                "suite:a",
                [{"case_id": "second", "scores": {"leak_score": 0.9}}],
            )
        )

        first = manager.load_baseline("suite/a")
        second = manager.load_baseline("suite:a")

        assert first_path != second_path
        assert first is not None
        assert second is not None
        assert first["suite_id"] == "suite/a"
        assert first["cases"][0]["case_id"] == "first"
        assert second["suite_id"] == "suite:a"
        assert second["cases"][0]["case_id"] == "second"

    def test_delete_baseline_does_not_remove_colliding_suite(self, tmp_path: Path) -> None:
        manager = BaselineManager(tmp_path / ".baselines")
        first_path = Path(manager.save_baseline("suite/a", []))
        second_path = Path(manager.save_baseline("suite:a", []))

        assert manager.delete_baseline("suite/a", "4f53cda18c2baa0c") is True

        assert not first_path.exists()
        assert second_path.exists()
        assert manager.load_baseline("suite:a") is not None
