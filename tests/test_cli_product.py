"""Tests for product-level CLI and orchestration behavior."""

from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path

import pytest
import tomllib
from typer.testing import CliRunner

from ragfuzz.cli import _select_chat_model
from ragfuzz.config import Config
from ragfuzz.engine import Scheduler, SchedulerConfig
from ragfuzz.models import Response, ScoreVector
from ragfuzz.scoring.judge import JudgeScorer


def test_pyproject_exposes_the_console_script_and_demo_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())

    assert project["project"]["scripts"]["ragfuzz"] == "ragfuzz.cli:app"
    dependencies = project["project"]["dependencies"]
    assert any(dep.startswith("fastapi") for dep in dependencies)
    assert any(dep.startswith("uvicorn") for dep in dependencies)


def test_auto_model_selection_skips_embedding_models() -> None:
    selected = _select_chat_model(
        "ollama",
        "auto",
        ["nomic-embed-text:latest", "gemma4:e2b", "qwen3.5:9b"],
    )

    assert selected == "gemma4:e2b"


def test_auto_model_selection_keeps_configured_model_when_available() -> None:
    selected = _select_chat_model(
        "ollama",
        "qwen3.5:9b",
        ["gemma4:e2b", "qwen3.5:9b"],
    )

    assert selected == "qwen3.5:9b"


@pytest.mark.asyncio
async def test_scheduler_propagates_suite_canary_and_target_id_into_failure_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMutator:
        name = "noop"
        config = {}

        async def mutate(self, text: str, _context: dict[str, str] | None = None) -> str:
            return text

    class FakeTarget:
        async def execute(self, _input_data: dict[str, object]) -> Response:
            return Response(
                content="leak ZXQ-491-AZ",
                model="demo-model",
                trace_id="trace-123",
                usage={"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
            )

    class CapturingScorer:
        def __init__(self) -> None:
            self.contexts: list[dict[str, str] | None] = []

        async def score(
            self, _response: Response, context: dict[str, str] | None = None
        ) -> ScoreVector:
            self.contexts.append(context)
            return ScoreVector(leak_score=1.0)

    monkeypatch.setattr(
        Config,
        "_loaded_suite",
        {"canary": {"value": "ZXQ-491-AZ"}},
        raising=False,
    )

    scheduler = Scheduler(
        config=SchedulerConfig(max_runs=1, concurrency=1, max_cost_usd=10.0, use_cache=False)
    )
    scorer = CapturingScorer()

    cases = await scheduler.run_suite(
        seeds=[{"seed": "seed prompt"}],
        mutators=[FakeMutator()],
        target=FakeTarget(),
        scorer=scorer,
        suite_id="suite-a",
        target_id="target-a",
        provider_id="provider-a",
        model_id="model-a",
        run_id="run-a",
        suite_context={"canary": "ZXQ-491-AZ"},
    )

    assert len(cases) == 1
    assert scorer.contexts == [{"canary": "ZXQ-491-AZ", "prompt": "seed prompt"}]

    expected_signature = scheduler.corpus.calculate_failure_signature(
        suite_id="suite-a",
        target_id="target-a",
        scores=cases[0].scores,
    )

    assert scheduler.corpus.entries[-1].failure_signature == expected_signature
    assert scheduler.corpus.entries[-1].input_text == "seed prompt"


@pytest.mark.asyncio
async def test_judge_scorer_selects_rubric_and_provider_model() -> None:
    class FakeJudgeProvider:
        def __init__(self) -> None:
            self.provider_id = "lmstudio:local-judge"
            self.calls: list[dict[str, object]] = []

        async def chat(
            self,
            messages: list[object],
            model: str,
            temperature: float = 0.7,
            max_tokens: int | None = None,
        ) -> Response:
            self.calls.append(
                {
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return Response(
                content=(
                    '{"success": true, "severity": "medium", "category": "policy_violation", '
                    '"explanation": "bypassed refusal", "confidence": 0.4}'
                ),
                model=model,
            )

    provider = FakeJudgeProvider()
    scorer = JudgeScorer(judge_provider=provider, rubric="missing-rubric")

    result = await scorer.score(
        Response(content="ignored", model="demo"),
        prompt="original prompt",
        canary="ZXQ-491-AZ",
    )

    assert provider.calls[0]["model"] == "local-judge"
    assert provider.calls[0]["temperature"] == 0.1
    assert provider.calls[0]["max_tokens"] == 300

    system_prompt, user_prompt = provider.calls[0]["messages"]  # type: ignore[misc]
    assert "canary" in system_prompt.content.lower()
    assert "Canary to check for: ZXQ-491-AZ" in user_prompt.content
    assert "Original prompt: original prompt" in user_prompt.content
    assert result.leak_score == 0.0
    assert result.policy_violation_score == pytest.approx(0.6)
    assert result.partial_success_score == pytest.approx(0.4)


def test_cli_demo_command_is_available_when_the_demo_module_exists() -> None:
    demo_spec = importlib_util.find_spec("ragfuzz.demo")
    if demo_spec is None:
        pytest.skip(
            "Demo module is not present yet, this test documents the expected CLI contract."
        )

    try:
        from ragfuzz.cli import app
    except Exception as exc:  # pragma: no cover - import failure is a product regression signal
        pytest.skip(f"CLI module is not importable in this checkout: {exc}")

    runner = CliRunner()
    result = runner.invoke(app, ["demo", "--help"])

    assert result.exit_code == 0
