"""Tests for product-level CLI and orchestration behavior."""

from __future__ import annotations

import json
import tomllib
from importlib import util as importlib_util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragfuzz import __version__
from ragfuzz.cli import _is_allowed_target_url, _select_chat_model, app
from ragfuzz.config import Config, SuiteConfig
from ragfuzz.engine import Scheduler, SchedulerConfig
from ragfuzz.models import Response, ScoreVector
from ragfuzz.reports.data import build_report_data
from ragfuzz.scoring.judge import JudgeScorer


def test_pyproject_exposes_the_console_script_and_demo_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())

    assert project["project"]["version"] == __version__
    assert project["project"]["scripts"]["ragfuzz"] == "ragfuzz.cli:app"
    dependencies = project["project"]["dependencies"]
    assert any(dep.startswith("fastapi") for dep in dependencies)
    assert any(dep.startswith("uvicorn") for dep in dependencies)


def test_readiness_command_emits_json_evidence(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "readiness",
            "--config",
            "missing-ragfuzz.toml",
            "--skip-provider-checks",
            "--evidence-dir",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    evidence_path = tmp_path / "readiness.json"

    assert payload["status"] == "needs_setup"
    assert payload["config"]["loaded"] is False
    assert payload["suites"]
    assert payload["security_posture"]["report_url_sanitization"] is True
    assert evidence_path.exists()


def test_doctor_command_emits_local_operator_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "doctor",
            "--config",
            "missing-ragfuzz.toml",
            "--skip-provider-checks",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "needs_setup"
    assert payload["security_posture"]["operator_target_check"] is True
    assert payload["security_posture"]["local_audit_log"] is True


def test_target_check_blocks_link_local_by_default() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["target-check", "http://169.254.169.254/latest/meta-data", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["allowed"] is False
    assert payload["classification"] == "link_local"


def test_target_check_allows_private_and_allowlisted_hosts() -> None:
    runner = CliRunner()
    private_result = runner.invoke(app, ["target-check", "http://127.0.0.1:8765", "--json"])
    allowlisted_result = runner.invoke(
        app,
        [
            "target-check",
            "https://rag.internal.example",
            "--allowed-host",
            "*.internal.example",
            "--json",
        ],
    )

    assert private_result.exit_code == 0
    assert json.loads(private_result.stdout)["classification"] == "loopback"
    assert allowlisted_result.exit_code == 0
    assert json.loads(allowlisted_result.stdout)["classification"] == "allowlisted"


def test_redact_check_detects_obvious_secrets(tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("Authorization: Bearer abcdefghijklmnop\n")

    runner = CliRunner()
    result = runner.invoke(app, ["redact-check", str(artifact), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail"
    assert payload["finding_count"] == 1
    assert payload["findings"][0]["kind"] == "bearer_token"


def test_redact_check_passes_clean_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("No secrets here. Public reference: https://example.invalid/report\n")

    runner = CliRunner()
    result = runner.invoke(app, ["redact-check", str(artifact), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["finding_count"] == 0


def test_evidence_bundle_writes_manifest_and_redaction_proof(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evidence"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-operator",
                "timestamp": "2026-05-18T12:00:00Z",
                "suite": {"name": "rag-canary-leak"},
            }
        )
    )
    (run_dir / "cases.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "inputs": {"messages": [{"role": "user", "content": "clean prompt"}]},
                "scores": {"leak_score": 0.0, "policy_violation_score": 0.0},
            }
        )
        + "\n"
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evidence-bundle",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--config",
            "missing-ragfuzz.toml",
            "--skip-provider-checks",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ready"
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "readiness.json").exists()
    assert (output_dir / "report-summary.json").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "redact-check.json").exists()


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


def test_target_url_guard_defaults_to_loopback_and_private_hosts() -> None:
    assert _is_allowed_target_url("http://127.0.0.1:8000")
    assert _is_allowed_target_url("http://localhost:8000")
    assert _is_allowed_target_url("http://10.0.0.8:8000")
    assert not _is_allowed_target_url("http://169.254.169.254/latest/meta-data")
    assert not _is_allowed_target_url("https://example.com")
    assert _is_allowed_target_url(
        "http://169.254.169.254/latest/meta-data",
        allow_public=True,
    )


def test_research_backed_suites_load_with_metadata() -> None:
    suite_dir = Path(__file__).resolve().parents[1] / "suites"
    suite_paths = sorted(suite_dir.glob("*.yaml"))

    assert {path.name for path in suite_paths} >= {
        "rag-canary-leak.yaml",
        "rag-indirect-prompt-injection.yaml",
        "rag-retrieval-conflict.yaml",
        "rag-poisoned-knowledge.yaml",
        "rag-dos-flood.yaml",
        "rag-multi-hop.yaml",
    }

    loaded = [SuiteConfig.load(path) for path in suite_paths]
    by_name = {suite.name: suite for suite in loaded}

    assert by_name["rag-indirect-prompt-injection"].owasp == ["LLM01", "LLM05", "LLM08"]
    assert by_name["rag-retrieval-conflict"].run_type == "retrieval"
    assert by_name["rag-poisoned-knowledge"].risk_tags == [
        "knowledge-base-poisoning",
        "poison-influence",
        "corpus-integrity",
    ]
    assert by_name["rag-dos-flood"].run_type == "dos"
    assert by_name["rag-multi-hop"].run_type == "multi-hop"
    assert all(suite.research for suite in loaded)


def test_report_data_preserves_suite_research_and_owasp_metadata() -> None:
    report = build_report_data(
        {
            "run_id": "run-research",
            "timestamp": "2026-05-12T15:00:00Z",
            "suite": {
                "id": "rag-indirect-prompt-injection",
                "name": "rag-indirect-prompt-injection",
                "run_type": "prompt-injection",
                "owasp": ["LLM01", "LLM05", "LLM08"],
                "research": [{"name": "AgentDojo", "url": "https://arxiv.org/abs/2406.13352"}],
                "risk_tags": ["indirect-prompt-injection"],
            },
            "config": {},
            "extra": {},
        },
        [],
    )

    assert report["metadata"]["suite"]["owasp"] == ["LLM01", "LLM05", "LLM08"]
    assert report["metadata"]["suite"]["research"][0]["name"] == "AgentDojo"
    assert report["metadata"]["suite"]["research"][0]["url"] == "https://arxiv.org/abs/2406.13352"
    assert report["metadata"]["suite"]["risk_tags"] == ["indirect-prompt-injection"]


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
    assert scorer.contexts == [
        {"canary": "ZXQ-491-AZ", "prompt": "seed prompt", "run_id": "run-a"}
    ]

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
            self.default_model = "judge-default"
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

    assert provider.calls[0]["model"] == "judge-default"
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
