"""Test configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create a temporary configuration file."""
    config_path = tmp_path / "ragfuzz.toml"
    config_content = """[providers.test]
type = "openai_compat"
base_url = "http://localhost:9999/v1"
api_key_env = "TEST_API_KEY"
default_model = "test-model"

[budget]
max_runs = 10
max_cost_usd = 1.0

default_provider = "test"
default_target = "chat"
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_suite(tmp_path: Path) -> Path:
    """Create a sample suite configuration."""
    suite_path = tmp_path / "test_suite.yaml"
    suite_content = """name: test-suite
requires:
  target: chat
inputs:
  - seed: "Test input"
canary:
  value: "TEST-CANARY"
mutations: []
scoring:
  heuristics: [canary_regex]
budget:
  runs: 5
"""
    suite_path.write_text(suite_content)
    return suite_path
