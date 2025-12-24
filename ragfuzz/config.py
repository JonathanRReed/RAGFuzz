"""Configuration management for ragfuzz using TOML."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    id: str
    type: str
    base_url: str
    api_key_env: str
    default_model: str
    supports_streaming: bool = True
    supports_tools: bool = True

    def __post_init__(self):
        if not self.base_url:
            raise ValueError("base_url cannot be empty")
        if not self.default_model:
            raise ValueError("default_model cannot be empty")


@dataclass
class BudgetConfig:
    """Budget constraints for runs."""

    max_runs: int | None = None
    max_cost_usd: float | None = None
    max_duration_seconds: int | None = None

    def __post_init__(self) -> None:
        """Validate budget constraints after initialization."""
        if self.max_runs is not None and self.max_runs <= 0:
            raise ValueError("max_runs must be positive or None")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be non-negative or None")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive or None")


@dataclass
class Config:
    """Main ragfuzz configuration."""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: str | None = None
    default_target: str | None = None
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    run_dir: str = "runs"
    cache_dir: str = ".cache"
    vram_threshold_mb: int = 1024

    @classmethod
    def load(cls, path: str | Path = "ragfuzz.toml") -> Config:
        """Load configuration from a TOML file.

        Args:
            path: Path to the configuration file.

        Returns:
            A Config instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the TOML is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if tomllib is None:
            raise ImportError(
                "tomli is required for Python < 3.11. Install with: pip install tomli"
            )

        try:
            data = tomllib.loads(path.read_text())
        except Exception as e:
            raise ValueError(f"Invalid TOML in {path}: {e}") from e

        config = cls()

        # Parse providers section
        providers_data = data.get("providers", {})
        for provider_id, provider_config in providers_data.items():
            config.providers[provider_id] = ProviderConfig(
                id=provider_id,
                type=provider_config.get("type", "openai_compat"),
                base_url=provider_config.get("base_url", ""),
                api_key_env=provider_config.get("api_key_env", ""),
                default_model=provider_config.get("default_model", ""),
                supports_streaming=provider_config.get("supports_streaming", True),
                supports_tools=provider_config.get("supports_tools", True),
            )

        # Parse budget section
        budget_data = data.get("budget", {})
        config.budget = BudgetConfig(
            max_runs=budget_data.get("max_runs"),
            max_cost_usd=budget_data.get("max_cost_usd"),
            max_duration_seconds=budget_data.get("max_duration_seconds"),
        )

        # Parse other settings
        config.run_dir = data.get("run_dir", "runs")
        config.cache_dir = data.get("cache_dir", ".cache")
        config.vram_threshold_mb = data.get("vram", {}).get("threshold_mb", 1024)

        # Parse default_provider and default_target from budget section or root
        config.default_provider = data.get("default_provider")
        config.default_target = data.get("default_target")

        if (
            not config.default_provider
            and "budget" in data
            and "default_provider" in data["budget"]
        ):
            config.default_provider = data["budget"]["default_provider"]

        if not config.default_target and "budget" in data and "default_target" in data["budget"]:
            config.default_target = data["budget"]["default_target"]

        return config

    @classmethod
    def create_default(cls, path: str | Path = "ragfuzz.toml") -> Config:
        """Create a default configuration file.

        Args:
            path: Path where to create the configuration file.

        Returns:
            A Config instance.
        """
        path = Path(path)
        default_config_content = """# ragfuzz configuration

[providers.lmstudio]
type = "openai_compat"
base_url = "http://localhost:1234/v1"
api_key_env = "LM_STUDIO_API_KEY"  # Optional for LM Studio
default_model = "model"

[providers.ollama]
type = "openai_compat"
base_url = "http://localhost:11434/v1"
api_key_env = "OLLAMA_API_KEY"  # Optional for Ollama
default_model = "llama2"

[providers.openrouter]
type = "openai_compat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
default_model = "anthropic/claude-sonnet-4"

[budget]
max_runs = 1000
max_cost_usd = 10.0
max_duration_seconds = 3600

default_provider = "lmstudio"
default_target = "chat"

[run]
run_dir = "runs"
cache_dir = ".cache"

[vram]
threshold_mb = 1024
"""
        path.write_text(default_config_content)
        return cls.load(path)

    def get_api_key(self, provider_id: str) -> str | None:
        """Get the API key for a provider from environment variables.

        Args:
            provider_id: The provider ID.

        Returns:
            The API key if set, None otherwise.
        """
        provider = self.providers.get(provider_id)
        if not provider:
            return None
        return os.environ.get(provider.api_key_env)

    def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """Get a provider configuration by ID.

        Args:
            provider_id: The provider ID.

        Returns:
            The ProviderConfig or None if not found.
        """
        return self.providers.get(provider_id)


class SuiteConfig(BaseModel):
    """Configuration for a test suite."""

    name: str
    requires: dict[str, Any] = Field(default_factory=dict)
    inputs: list[dict[str, Any]] = Field(default_factory=list, min_length=1)
    canary: dict[str, Any] = Field(default_factory=dict)
    mutations: list[dict[str, Any]] = Field(default_factory=list)
    scoring: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    cleanup: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v, _info):
        if not v or len(v) == 0:
            raise ValueError("At least one input seed is required")
        return v

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, v, _info):
        if "runs" in v and v["runs"] is not None and v["runs"] <= 0:
            raise ValueError("budget.runs must be positive")
        if "max_cost_usd" in v and v["max_cost_usd"] is not None and v["max_cost_usd"] < 0:
            raise ValueError("budget.max_cost_usd must be non-negative")
        return v

    @classmethod
    def load(cls, path: str | Path) -> SuiteConfig:
        """Load a suite configuration from a YAML file.

        Args:
            path: Path to the suite YAML file.

        Returns:
            A SuiteConfig instance.

        Raises:
            FileNotFoundError: If the suite file doesn't exist.
            ValueError: If the YAML is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Suite file not found: {path}")

        import yaml

        try:
            data = yaml.safe_load(path.read_text())
        except Exception as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        return cls(**data)
