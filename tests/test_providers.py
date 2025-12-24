"""Tests for providers module."""

from pathlib import Path

import pytest

from ragfuzz.config import Config, ProviderConfig
from ragfuzz.providers import OpenAICompatProvider


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_provider_config_creation(self) -> None:
        """Test creating a provider configuration."""
        config = ProviderConfig(
            id="test",
            type="openai_compat",
            base_url="http://localhost:1234/v1",
            api_key_env="TEST_KEY",
            default_model="model",
        )

        assert config.id == "test"
        assert config.type == "openai_compat"
        assert config.supports_streaming is True


class TestOpenAICompatProvider:
    """Test OpenAI-compatible provider."""

    @pytest.fixture
    def provider(self) -> OpenAICompatProvider:
        """Create a test provider instance."""
        return OpenAICompatProvider(
            provider_id="test",
            base_url="http://localhost:9999/v1",
            api_key="test-key",
        )

    def test_provider_initialization(self, provider: OpenAICompatProvider) -> None:
        """Test provider initialization."""
        assert provider.provider_id == "test"
        assert provider.base_url == "http://localhost:9999/v1"
        assert provider.api_key == "test-key"
        assert provider.supports_streaming() is True
        assert provider.supports_tools() is True

    def test_max_context_estimate(self, provider: OpenAICompatProvider) -> None:
        """Test max context estimate."""
        assert provider.max_context_estimate() == 4096

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider: OpenAICompatProvider) -> None:
        """Test health check with unreachable server."""
        healthy = await provider.health_check()
        assert healthy is False


class TestConfig:
    """Test Config class."""

    def test_create_default_config(self, tmp_path: Path) -> None:
        """Test creating default configuration."""
        config_path = tmp_path / "ragfuzz.toml"
        config = Config.create_default(config_path)

        assert config_path.exists()
        assert "lmstudio" in config.providers
        assert "ollama" in config.providers
        assert config.providers["lmstudio"].base_url == "http://localhost:1234/v1"
        assert config.providers["ollama"].base_url == "http://localhost:11434/v1"

    def test_load_config(self, temp_config: Path) -> None:
        """Test loading configuration from file."""
        config = Config.load(temp_config)

        assert "test" in config.providers
        assert config.providers["test"].type == "openai_compat"
        assert config.default_provider == "test"
        assert config.budget.max_runs == 10

    def test_get_provider(self, temp_config: Path) -> None:
        """Test getting a provider by ID."""
        config = Config.load(temp_config)
        provider = config.get_provider("test")

        assert provider is not None
        assert provider.id == "test"
        assert provider.base_url == "http://localhost:9999/v1"

    def test_get_nonexistent_provider(self, temp_config: Path) -> None:
        """Test getting a non-existent provider."""
        config = Config.load(temp_config)
        provider = config.get_provider("nonexistent")

        assert provider is None
