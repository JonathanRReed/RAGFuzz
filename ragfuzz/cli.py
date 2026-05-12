"""CLI interface for ragfuzz."""

from __future__ import annotations

import asyncio
import json
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import typer
from rich.console import Console

from ragfuzz.config import Config, SuiteConfig
from ragfuzz.engine import Cache, Scheduler, SchedulerConfig
from ragfuzz.mutators import (
    GrammarMutator,
    LLMGuidedMutator,
    Mutator,
    PoisonMutator,
    StatefulDialogueMutator,
    TemplateMutator,
)
from ragfuzz.providers import OpenAICompatProvider, ProviderDoctor
from ragfuzz.reports import HTMLReporter
from ragfuzz.reports.viz import MutationGraphViz
from ragfuzz.scoring import HeuristicScorer
from ragfuzz.scoring.base import Scorer
from ragfuzz.scoring.judge import JudgeScorer
from ragfuzz.storage import RunDir
from ragfuzz.storage.baseline import BaselineManager
from ragfuzz.targets import ChatTarget
from ragfuzz.targets.jr_autorag import JRAutoRAGTarget
from ragfuzz.utils import close_client
from ragfuzz.utils.bisect import compare_runs, format_comparison_report

app = typer.Typer(help="ragfuzz: Grey-box RAG Auditor and Prompt Fuzzer")
console = Console()


@app.command()
def init(config_path: str = "ragfuzz.toml") -> None:
    """Initialize ragfuzz configuration.

    Args:
        config_path: Path for configuration file.
    """
    try:
        Config.create_default(config_path)
        console.print(f"OK Configuration created at [bold]{config_path}[/bold]")
        console.print("\nNext steps:")
        console.print("1. Edit configuration to set up your providers")
        console.print("2. Run [bold]ragfuzz providers-doctor[/bold] to check connectivity")
        console.print("3. Create a test suite in [bold]suites/[/bold]")
    except Exception as e:
        console.print(f"ERROR Failed to create configuration: [red]{e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def demo(
    host: str = typer.Option("127.0.0.1", "--host", help="Host for the local demo app"),
    port: int = typer.Option(8765, "--port", help="Port for the local demo app"),
    open_browser: bool = typer.Option(
        True,
        "--open/--no-open",
        help="Open the demo in the default browser",
    ),
) -> None:
    """Launch the local RAGFuzz product demo."""
    try:
        import uvicorn

        from ragfuzz.demo import create_app

        url = f"http://{host}:{port}"
        console.print(f"Starting RAGFuzz demo at [bold]{url}[/bold]")
        console.print("Demo data is in memory. Normal CLI run artifacts remain on disk.")

        if open_browser:
            webbrowser.open(url)

        uvicorn.run(create_app(), host=host, port=port, log_level="warning")
    except Exception as e:
        console.print(f"ERROR Failed to launch demo: [red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def providers_ls() -> None:
    """List configured providers."""
    try:
        config = Config.load()
        console.print("Configured Providers:")
        console.print("=" * 50)

        if not config.providers:
            console.print("No providers configured.")
            return

        for provider_id, provider in config.providers.items():
            default_marker = " (default)" if provider_id == config.default_provider else ""
            console.print(f"• {provider_id}{default_marker}")
            console.print(f"  Type: {provider.type}")
            console.print(f"  Base URL: {provider.base_url}")
            console.print(f"  Default Model: {provider.default_model}")
            console.print()

    except FileNotFoundError:
        console.print("[red]Configuration file not found.[/red]")
        console.print("Run [bold]ragfuzz init[/bold] to create a configuration.")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"ERROR Error loading configuration: [red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def models_ls(
    provider: str = typer.Option(None, "--provider", help="Specific provider to list models for"),
) -> None:
    """List available models from a provider."""

    async def _models_ls() -> None:
        try:
            config = Config.load()
            provider_id = provider or config.default_provider or "lmstudio"
            provider_config = config.get_provider(provider_id)

            if not provider_config:
                console.print(f"ERROR Provider '{provider_id}' not found in configuration")
                raise typer.Exit(1)

            api_key = config.get_api_key(provider_id)
            provider_instance = OpenAICompatProvider(
                provider_id=provider_config.id,
                base_url=provider_config.base_url,
                api_key=api_key,
            )

            console.print(f"Available models for [bold]{provider_id}[/bold]:")
            console.print("=" * 60)

            models = await provider_instance.list_models()

            if not models:
                console.print("No models found.")
                return

            selected_model = _select_chat_model(provider_id, provider_config.default_model, models)
            for i, model in enumerate(models, 1):
                default_marker = " (selected)" if model == selected_model else ""
                console.print(f"{i:2d}. {model}{default_marker}")

        except FileNotFoundError:
            console.print("[red]Configuration file not found.[/red]")
            console.print("Run [bold]ragfuzz init[/bold] to create a configuration.")
            raise typer.Exit(1) from None
        except Exception as e:
            console.print(f"ERROR [red]{e}[/red]")
            raise typer.Exit(1) from e
        finally:
            await close_client()

    asyncio.run(_models_ls())


@app.command()
def providers_doctor(
    bench: bool = typer.Option(False, "--bench", help="Run performance benchmarks"),
    provider: str = typer.Option(None, "--provider", help="Specific provider to check"),
) -> None:
    """Check provider health and capabilities."""

    async def _doctor() -> None:
        try:
            config = Config.load()
            doctor = ProviderDoctor(config)

            if provider:
                results = {provider: await doctor.check_provider(provider, bench)}
            else:
                results = await doctor.check_all(benchmark=bench)

            console.print(doctor.format_report(results))

        except FileNotFoundError:
            console.print("[red]Configuration file not found.[/red]")
            console.print("Run [bold]ragfuzz init[/bold] to create a configuration.")
            raise typer.Exit(1) from None
        except Exception as e:
            console.print(f"ERROR [red]{e}[/red]")
            raise typer.Exit(1) from e

    asyncio.run(_doctor())


@app.command()
def check_api(
    url: str = typer.Argument(..., help="JR AutoRAG base URL"),
    headers: str = typer.Option(None, "--headers", help="Optional headers as JSON"),
) -> None:
    """Check JR AutoRAG grey-box API connectivity and capabilities."""

    async def _check_api() -> None:
        try:
            import json

            headers_dict = json.loads(headers) if headers else None
            if headers_dict is not None and not isinstance(headers_dict, dict):
                raise ValueError("--headers must be a JSON object")
            if not _is_http_url(url):
                raise ValueError("URL must start with http:// or https://")
            target = JRAutoRAGTarget(
                target_id="jr_autorag_check", base_url=url, headers=headers_dict
            )

            console.print(f"Checking JR AutoRAG API at: [bold]{url}[/bold]")
            console.print("=" * 60)

            checks = []

            # Test query endpoint
            try:
                test_input = {"query": "test query", "run_id": "check"}
                response = await target.execute(test_input)
                checks.append(("query", "OK", f"trace_id: {response.trace_id}"))
            except Exception as e:
                checks.append(("query", "FAIL", str(e)))

            # Test trace endpoint
            if trace_str := next((c[2] for c in checks if c[0] == "query" and c[1] == "OK"), None):
                trace_id = trace_str.split(": ")[1] if ": " in trace_str else None
                if trace_id:
                    try:
                        trace_data = await target.get_trace(trace_id)
                        checks.append(
                            (
                                "trace",
                                "OK",
                                f"Found trace with {len(trace_data.get('steps', []))} steps",
                            )
                        )
                    except Exception as e:
                        checks.append(("trace", "FAIL", str(e)))

            # Test ingestion endpoint
            try:
                await target.ingest_documents([{"text": "test doc"}], tags={"run_id": "check"})
                checks.append(("ingestion", "OK", "Document ingested"))
                await target.delete_by_tag("run_id", "check")
                checks.append(("cleanup", "OK", "Documents cleaned up"))
            except Exception as e:
                checks.append(("ingestion", "FAIL", str(e)))

            for name, status, message in checks:
                console.print(f"{status} {name}: {message}")

        except Exception as e:
            console.print(f"ERROR [red]{e}[/red]")
            raise typer.Exit(1) from e
        finally:
            await close_client()

    asyncio.run(_check_api())


@app.command()
def baseline_save(
    suite: str = typer.Argument(..., help="Suite name"),
    cases_file: str = typer.Argument(..., help="Cases JSONL file path"),
) -> None:
    """Save a baseline for regression testing."""
    import json

    try:
        baseline_manager = BaselineManager()
        cases = []

        for line in Path(cases_file).read_text().strip().split("\n"):
            if line:
                cases.append(json.loads(line))

        baseline_path = baseline_manager.save_baseline(
            suite_id=suite,
            cases=cases,
            metadata={"timestamp": Path(cases_file).stat().st_mtime},
        )

        console.print(f"OK Baseline saved: [bold]{baseline_path}[/bold]")
        console.print(f"   Cases: {len(cases)}")

    except Exception as e:
        console.print(f"ERROR [red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def baseline_check(
    suite: str = typer.Argument(..., help="Suite name"),
    cases_file: str = typer.Argument(..., help="Cases JSONL file path"),
) -> None:
    """Check current results against baseline."""
    import json

    try:
        baseline_manager = BaselineManager()
        cases = []

        for line in Path(cases_file).read_text().strip().split("\n"):
            if line:
                cases.append(json.loads(line))

        result = baseline_manager.compare_against_baseline(suite, cases)

        if result["status"] == "no_baseline":
            console.print(f"WARN {result['message']}")
            console.print("Run [bold]ragfuzz baseline-save[/bold] to create a baseline.")
            return

        console.print(f"Baseline check for: [bold]{suite}[/bold]")
        console.print("=" * 50)

        current_summary = result["current_summary"]
        baseline_summary = result["baseline_summary"]

        console.print("\nStatistics:")
        console.print(f"  Total cases: {current_summary['total_cases']}")
        console.print(
            f"  Failures: {current_summary['failure_count']} (baseline: {baseline_summary['failure_count']})"
        )
        console.print(
            f"  Leak rate: {current_summary['leak_rate']:.3f} (baseline: {baseline_summary['leak_rate']:.3f})"
        )
        console.print(
            f"  Avg severity: {current_summary['avg_severity']:.3f} (baseline: {baseline_summary['avg_severity']:.3f})"
        )

        if result["status"] == "regression_detected":
            console.print("\nERROR [bold]Regressions detected:[/bold]")
            for regression in result["regressions"]:
                console.print(f"  • {regression['type']}:")
                console.print(f"      Baseline: {regression['baseline']:.3f}")
                console.print(f"      Current: {regression['current']:.3f}")
                console.print(f"      Delta: {regression['delta']:.3f}")
        else:
            console.print("\nOK [bold]No regressions detected[/bold]")

    except Exception as e:
        console.print(f"ERROR [red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def run(
    suite: str = typer.Argument(..., help="Path to suite YAML file"),
    provider: str = typer.Option(None, "--provider", help="Provider to use"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without running"),
    runs: int = typer.Option(None, "--runs", help="Number of test runs"),
    concurrency: int = typer.Option(
        None, "--concurrency", help="Concurrency level (auto for auto)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    poison: str = typer.Option(None, "--poison", help="Poison mode: influence, exfil, bias"),
    scoring: str = typer.Option(None, "--scoring", help="Scoring mode: heuristic or judge"),
    json_summary: bool = typer.Option(False, "--json-summary", help="Print CI-friendly JSON summary"),
) -> None:
    """Run a test suite with AFL-style corpus scheduling."""

    async def _run() -> None:
        try:
            config = Config.load()
            suite_config = SuiteConfig.load(suite)

            provider_id = provider or config.default_provider or "lmstudio"
            provider_config = config.get_provider(provider_id)

            if not provider_config:
                console.print(f"ERROR Provider '{provider_id}' not found in configuration")
                raise typer.Exit(1)
            if poison and poison not in {"influence", "exfil", "bias"}:
                console.print("ERROR Poison mode must be one of: influence, exfil, bias")
                raise typer.Exit(1)

            api_key = config.get_api_key(provider_id)
            provider_instance = OpenAICompatProvider(
                provider_id=provider_config.id,
                base_url=provider_config.base_url,
                api_key=api_key,
            )
            resolved_model = await _resolve_provider_model(
                provider_id,
                provider_config.default_model,
                provider_instance,
            )

            if dry_run:
                from ragfuzz.pricing import estimate_cost, estimate_tokens

                console.print("[yellow]Dry run mode - estimating cost...[/yellow]")
                console.print(f"Suite: {suite_config.name}")
                console.print(f"Provider: {provider_id}")
                console.print(f"Model: {resolved_model}")

                num_runs = runs or suite_config.budget.get("runs", 100)
                concurrency_value = concurrency or 4

                mutators = _setup_mutators(suite_config, provider_instance)
                seeds = suite_config.inputs or [{"seed": "Test prompt"}]

                estimated_prompt_tokens = 0

                for seed in seeds:
                    seed_text = seed.get("seed", "")
                    estimated_prompt_tokens += estimate_tokens(seed_text, multiplier=0.25)

                for mutator in mutators:
                    mutator_config = getattr(mutator, "config", {})
                    max_attempts = mutator_config.get("max_attempts", 6)
                    estimated_prompt_tokens *= max_attempts

                avg_completion_tokens = 200
                total_prompt_tokens = estimated_prompt_tokens * (num_runs // len(seeds) + 1)
                total_completion_tokens = num_runs * avg_completion_tokens

                estimated_cost = estimate_cost(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    provider_id=provider_id,
                    model_id=resolved_model,
                )

                console.print(f"Estimated runs: {num_runs}")
                console.print(f"Concurrency: {concurrency_value}")
                console.print(f"Estimated prompt tokens: {total_prompt_tokens:,}")
                console.print(f"Estimated completion tokens: {total_completion_tokens:,}")
                console.print(f"Estimated total cost: ${estimated_cost:.4f}")
                return

            target_instance: ChatTarget | JRAutoRAGTarget = ChatTarget(
                target_id="chat",
                provider=provider_instance,
                default_model=resolved_model,
            )

            run_dir = RunDir()
            run_dir.write_run_config(
                config,
                suite_config,
                extra={
                    "run_type": suite_config.run_type,
                    "provider_id": provider_id,
                    "owasp": suite_config.owasp,
                    "research": suite_config.research,
                    "risk_tags": suite_config.risk_tags,
                },
            )

            num_runs = runs or suite_config.budget.get("runs", 100)
            concurrency_value = concurrency or 4

            # Setup target based on poison mode
            if poison:
                target_url = suite_config.requires.get("base_url", "http://localhost:8000")
                if not _is_http_url(target_url):
                    console.print("ERROR JR AutoRAG base_url must start with http:// or https://")
                    raise typer.Exit(1)
                target_instance = JRAutoRAGTarget(
                    target_id="jr_autorag",
                    base_url=target_url,
                )
                console.print(f"Poison mode: [bold]{poison}[/bold]")

                # Ingest poisoned chunks
                poison_mutator = PoisonMutator(
                    name="poison_generator", config={"mode": poison, "run_id": run_dir.run_id}
                )
                poison_chunks_json = await poison_mutator.mutate("", context=None)

                poisoned_chunks = json.loads(poison_chunks_json)
                if poisoned_chunks:
                    await target_instance.ingest_documents(poisoned_chunks, tags={"run_id": run_dir.run_id})
                    console.print(f"OK Ingested {len(poisoned_chunks)} poisoned chunks")
            else:
                target_instance = ChatTarget(
                    target_id="chat",
                    provider=provider_instance,
                    default_model=resolved_model,
                )

            scorer = _setup_scorer(scoring or suite_config.scoring.get("mode"), provider_instance)

            # Setup VRAM monitor if configured
            vram_monitor = None
            if config.vram_threshold_mb and config.vram_threshold_mb > 0:
                from ragfuzz.utils.vram import VRAMMonitor

                vram_monitor = VRAMMonitor(threshold_mb=config.vram_threshold_mb)

            # Setup scheduler with caching
            cache = Cache(cache_dir=config.cache_dir) if not no_cache else None
            scheduler_config = SchedulerConfig(
                max_runs=num_runs,
                concurrency=concurrency_value,
                max_cost_usd=suite_config.budget.get("max_cost_usd", 10.0),
                use_cache=not no_cache,
                vram_threshold_mb=config.vram_threshold_mb,
            )

            scheduler = Scheduler(config=scheduler_config, cache=cache, vram_monitor=vram_monitor)

            # Setup mutators from suite
            mutators = _setup_mutators(
                suite_config, provider_instance, poison_mode=poison, run_id=run_dir.run_id
            )

            # Generate seed inputs
            seeds = suite_config.inputs or [{"seed": "Test prompt"}]

            with console.status(
                f"[bold blue]Running {num_runs} test cases with AFL-style scheduling..."
            ) as status:
                cases = await scheduler.run_suite(
                    seeds=seeds,
                    mutators=mutators,
                    target=target_instance,
                    scorer=scorer,
                    suite_id=suite_config.name,
                    target_id=target_instance.target_id,
                    provider_id=provider_id,
                    model_id=resolved_model,
                    run_id=run_dir.run_id,
                    run_type=suite_config.run_type,
                    suite_context={
                        "canary": suite_config.canary.get("value"),
                        "run_type": suite_config.run_type,
                        "owasp": suite_config.owasp,
                        "research": suite_config.research,
                        "risk_tags": suite_config.risk_tags,
                    },
                )

                # Write cases to disk
                for case in cases:
                    run_dir.write_case(case.model_dump())

                    if case.scores.leak_score > 0.5 or case.scores.policy_violation_score > 0.5:
                        run_dir.write_failure(case.case_id, case.model_dump())

                    if (len(cases) - len([c for c in cases if c is None])) % 10 == 0:
                        completed = len([c for c in cases if c is not None])
                        status.update(
                            f"[bold blue]Running {num_runs} test cases... {completed}/{num_runs}"
                        )

            stats = scheduler.get_stats()
            console.print(f"\nOK Run completed: [bold]{run_dir.run_id}[/bold]")
            console.print(f"Results: {run_dir.path}")
            console.print("\nScheduler Stats:")
            console.print(f"  Runs completed: {stats['run_count']}")
            console.print(f"  Total cost: ${stats.get('total_cost_usd', 0):.2f}")
            console.print(f"  Corpus size: {stats['corpus']['total_entries']}")
            console.print(f"  Unique failures: {stats['corpus']['unique_failures']}")

            if cache:
                cache_stats = cache.get_stats()
                console.print("\nCache Stats:")
                console.print(f"  Entries: {cache_stats['total_entries']}")
                console.print(f"  Avg age: {cache_stats.get('avg_age_seconds', 0):.0f}s")

            reporter = HTMLReporter()
            report_path = reporter.generate(run_dir.path)
            console.print(f"\nReport: [bold]{report_path}[/bold]")

            if json_summary:
                summary = {
                    "run_id": run_dir.run_id,
                    "run_dir": str(run_dir.path),
                    "report": str(report_path),
                    "run_type": suite_config.run_type,
                    "owasp": suite_config.owasp,
                    "research": suite_config.research,
                    "risk_tags": suite_config.risk_tags,
                    "target_id": target_instance.target_id,
                    "provider_id": provider_id,
                    "runs_completed": stats["run_count"],
                    "unique_failures": stats["corpus"]["unique_failures"],
                    "total_cost_usd": round(stats.get("total_cost_usd", 0), 6),
                }
                console.print(json.dumps(summary, sort_keys=True))

            # Cleanup poisoned chunks if poison mode was used
            if poison and isinstance(target_instance, JRAutoRAGTarget):
                try:
                    await target_instance.delete_by_tag("run_id", run_dir.run_id)
                    console.print(
                        f"OK Cleaned up poisoned chunks tagged with run_id: {run_dir.run_id}"
                    )
                except Exception as e:
                    console.print(f"WARN Could not clean up poisoned chunks: {e}")

        except FileNotFoundError as e:
            console.print(f"ERROR File not found: [red]{e}[/red]")
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"ERROR [red]{e}[/red]")
            raise typer.Exit(1) from e
        finally:
            await close_client()

    asyncio.run(_run())


@app.command()
def report(
    run_dir_path: str = typer.Argument(..., help="Path to run directory"),
    html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML report"),
    md: bool = typer.Option(False, "--md", help="Generate Markdown report"),
    json_out: bool = typer.Option(False, "--json", help="Print report summary as JSON"),
) -> None:
    """Generate a report from run results."""

    try:
        run_dir = Path(run_dir_path)

        if not run_dir.exists():
            console.print(f"ERROR Run directory not found: [red]{run_dir_path}[/red]")
            raise typer.Exit(1)

        reporter = HTMLReporter()

        if html:
            report_path = reporter.generate(run_dir)
            console.print(f"OK HTML report generated: [bold]{report_path}[/bold]")

        if md:
            md_path = reporter.generate_markdown(run_dir)
            console.print(f"OK Markdown report generated: [bold]{md_path}[/bold]")

        if json_out:
            console.print(json.dumps(reporter.summarize(run_dir), sort_keys=True))

    except Exception as e:
        console.print(f"ERROR [red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def replay(
    case_path: str = typer.Argument(..., help="Path to case JSON file"),
    provider: str = typer.Option(None, "--provider", help="Provider to use"),
) -> None:
    """Replay a test case."""

    async def _replay() -> None:
        try:
            config = Config.load()
            import json

            case_data = json.loads(Path(case_path).read_text())

            provider_id = provider or config.default_provider or "lmstudio"
            provider_config = config.get_provider(provider_id)

            if not provider_config:
                console.print(f"ERROR Provider '{provider_id}' not found")
                raise typer.Exit(1)

            api_key = config.get_api_key(provider_id)
            provider_instance = OpenAICompatProvider(
                provider_id=provider_config.id,
                base_url=provider_config.base_url,
                api_key=api_key,
            )
            resolved_model = await _resolve_provider_model(
                provider_id,
                provider_config.default_model,
                provider_instance,
            )

            target = ChatTarget(
                target_id="chat",
                provider=provider_instance,
                default_model=resolved_model,
            )

            console.print(f"Replaying case: [bold]{case_data.get('case_id', 'unknown')}[/bold]")
            console.print(f"Input: {case_data.get('inputs', {})}")

            response = await target.execute(case_data.get("inputs", {}))

            console.print("\nResponse:")
            console.print(response.content)

        except Exception as e:
            console.print(f"ERROR [red]{e}[/red]")
            raise typer.Exit(1) from e
        finally:
            await close_client()

    asyncio.run(_replay())


@app.command()
def cache_cleanup(
    max_age: int = typer.Option(86400, "--max-age", help="Maximum age in seconds (default: 24h)"),
) -> None:
    """Clean up old cache entries."""
    try:
        config = Config.load()
        cache = Cache(cache_dir=config.cache_dir)

        initial_stats = cache.get_stats()
        cache.cleanup_old(max_age_seconds=max_age)
        final_stats = cache.get_stats()

        console.print("OK Cache cleanup completed")
        console.print(f"  Entries before: {initial_stats['total_entries']}")
        console.print(f"  Entries after: {final_stats['total_entries']}")
        console.print(
            f"  Removed: {initial_stats['total_entries'] - final_stats['total_entries']} entries"
        )

        cache.close()

    except FileNotFoundError:
        console.print("[red]Configuration file not found.[/red]")
        console.print("Run [bold]ragfuzz init[/bold] to create a configuration.")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"ERROR [red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def corpus_stats(
    run_dir_path: str = typer.Argument(..., help="Path to run directory"),
) -> None:
    """Show corpus statistics from a run."""

    try:
        import json

        from ragfuzz.engine import Corpus

        run_dir = Path(run_dir_path)
        cases_jsonl = run_dir / "cases.jsonl"

        if not cases_jsonl.exists():
            console.print("ERROR cases.jsonl not found in run directory")
            raise typer.Exit(1)

        corpus = Corpus()

        for line in cases_jsonl.read_text().strip().split("\n"):
            if not line:
                continue

            case_data = json.loads(line)

            from ragfuzz.models import ScoreVector

            scores_data = case_data.get("scores", {})
            scores = ScoreVector(**scores_data)

            failure_sig = corpus.calculate_failure_signature(
                suite_id=case_data.get("suite_id", "unknown"),
                target_id=case_data.get("target_id", "unknown"),
                scores=scores,
            )

            input_text = ""
            messages = case_data.get("inputs", {}).get("messages", [])
            if messages:
                input_text = messages[0].get("content", "")

            from ragfuzz.engine import CorpusEntry

            entry = CorpusEntry(
                case_id=case_data.get("case_id", "unknown"),
                input_text=input_text,
                scores=scores,
                failure_signature=failure_sig
                if scores.leak_score > 0.5 or scores.policy_violation_score > 0.5
                else None,
            )
            corpus.add_entry(entry)

        stats = corpus.get_stats()

        console.print(f"Corpus Statistics for: [bold]{run_dir_path}[/bold]")
        console.print(f"  Total entries: {stats['total_entries']}")
        console.print(f"  Unique failures: {stats['unique_failures']}")
        console.print(f"  Average energy: {stats['avg_energy']:.3f}")
        console.print(f"  Max energy: {stats['max_energy']:.3f}")
        console.print(f"  High energy entries: {stats['high_energy_count']}")

    except Exception as e:
        console.print(f"ERROR [red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def bisect(
    run1: str = typer.Argument(..., help="Path to first run directory"),
    run2: str = typer.Argument(..., help="Path to second run directory"),
) -> None:
    """Compare two runs and identify differences."""
    try:
        console.print("Comparing runs...")
        console.print(f"  Run 1: {run1}")
        console.print(f"  Run 2: {run2}")
        console.print()

        comparison = compare_runs(run1, run2)
        report = format_comparison_report(comparison)

        console.print(report)

    except Exception as e:
        console.print(f"ERROR [red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def viz(
    case_path: str = typer.Argument(..., help="Path to case JSON file"),
    format: str = typer.Option("ascii", "--format", help="Output format (ascii or mermaid)"),
    output: str = typer.Option(None, "--output", help="Output file path (for mermaid)"),
) -> None:
    """Visualize the mutation graph for a case."""
    try:
        visualizer = MutationGraphViz(format=format)
        result = visualizer.visualize_case(case_path)

        if output and format == "mermaid":
            visualizer.save_mermaid(case_path, output)
            console.print(f"OK Mermaid diagram saved to [bold]{output}[/bold]")
        else:
            console.print(result)

    except FileNotFoundError as e:
        console.print(f"ERROR File not found: [red]{e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"ERROR [red]{e}[/red]")
        raise typer.Exit(1) from e


def _is_http_url(value: str) -> bool:
    """Return True when a URL is an HTTP or HTTPS URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _select_chat_model(provider_id: str, configured_model: str, models: list[str]) -> str:
    """Select a usable chat model from provider models."""
    if not models:
        raise ValueError(f"Provider '{provider_id}' returned no models")
    if configured_model != "auto" and configured_model in models:
        return configured_model

    chat_models = [
        model
        for model in models
        if "embed" not in model.lower() and "rerank" not in model.lower()
    ]
    return chat_models[0] if chat_models else models[0]


async def _resolve_provider_model(
    provider_id: str,
    configured_model: str,
    provider: OpenAICompatProvider,
) -> str:
    """Resolve the configured provider model, including local auto selection."""
    if configured_model != "auto":
        return configured_model
    models = await provider.list_models()
    return _select_chat_model(provider_id, configured_model, models)


def _setup_scorer(scoring_mode: str | None, provider: OpenAICompatProvider) -> Scorer:
    """Create the scorer requested by the suite or CLI."""
    mode = scoring_mode or "heuristic"
    if mode == "heuristic":
        return HeuristicScorer()
    if mode == "judge":
        return JudgeScorer(judge_provider=provider)
    raise ValueError("scoring mode must be heuristic or judge")


def _setup_mutators(
    suite_config: SuiteConfig,
    provider: OpenAICompatProvider,
    poison_mode: str | None = None,
    run_id: str = "",
) -> list[Mutator]:
    """Setup mutators from suite configuration.

    Args:
        suite_config: Suite configuration.
        provider: Provider instance for LLM-guided mutator.
        poison_mode: Optional poison mode (influence, exfil, bias).
        run_id: Run ID for tagging poisoned chunks.

    Returns:
        List of configured mutators.
    """
    mutators: list[Mutator] = []

    # Add poison mutator if mode specified
    if poison_mode:
        mutators.append(
            PoisonMutator(name="poison_mutator", config={"mode": poison_mode, "run_id": run_id})
        )

    for mutator_spec in suite_config.mutations:
        mutator_type = mutator_spec.get("type", "template")

        if mutator_type == "template":
            config = {"templates": mutator_spec.get("templates")}
            mutators.append(
                TemplateMutator(name=f"template_{mutator_spec.get('name')}", config=config)
            )

        elif mutator_type == "grammar":
            rule = mutator_spec.get("rule", "homoglyph_and_whitespace")
            mutators.append(GrammarMutator(name=f"grammar_{rule}", config=mutator_spec, rule=rule))

        elif mutator_type == "llm_guided":
            attacker_model = mutator_spec.get("attacker", f"{provider.provider_id}:gpt-4")
            goal = mutator_spec.get("goal", "Extract canary without triggering refusal.")
            max_attempts = mutator_spec.get("max_attempts", 6)
            config = {"attacker": attacker_model, "goal": goal, "max_attempts": max_attempts}
            mutators.append(LLMGuidedMutator(name="llm_guided", config=config, provider=provider))

        elif mutator_type == "stateful_dialogue":
            config = mutator_spec.get("config", {})
            mutators.append(
                StatefulDialogueMutator(name="stateful_dialogue", config=config, provider=provider)
            )

        elif mutator_type == "poison":
            mode = mutator_spec.get("mode", "influence")
            mutators.append(
                PoisonMutator(name=f"poison_{mode}", config={"mode": mode, "run_id": run_id})
            )

    # Default mutator if none specified
    if not mutators:
        mutators.append(TemplateMutator())

    return mutators


if __name__ == "__main__":
    app()
