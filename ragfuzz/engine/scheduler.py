"""Scheduler for AFL-style fuzzing execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ragfuzz.engine.cache import Cache
from ragfuzz.engine.corpus import Corpus, CorpusEntry
from ragfuzz.models import Case, ScoreVector
from ragfuzz.mutators.base import Mutator
from ragfuzz.mutators.graph import MutationGraph
from ragfuzz.scoring.base import Scorer
from ragfuzz.targets.base import Target


@dataclass
class SchedulerConfig:
    """Configuration for the fuzzing scheduler."""

    max_runs: int = 100
    concurrency: int = 4
    max_cost_usd: float = 10.0
    corpus_max_size: int = 500
    use_cache: bool = True
    energy_threshold: float = 0.3
    vram_threshold_mb: int = 1024


class Scheduler:
    """Scheduler for AFL-style fuzzing with feedback-driven corpus."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        corpus: Corpus | None = None,
        cache: Cache | None = None,
        vram_monitor: Any = None,
    ):
        """Initialize the scheduler.

        Args:
            config: Scheduler configuration.
            corpus: Initial corpus.
            cache: Cache for request deduplication.
            vram_monitor: Optional VRAM monitor for throttling.
        """
        self.config = config or SchedulerConfig()
        self.corpus = corpus or Corpus()
        self.cache = cache or Cache()
        self.mutation_graph = MutationGraph()
        self.vram_monitor = vram_monitor

        self._run_count = 0
        self._total_cost_usd: float = 0.0
        self._lock = asyncio.Lock()

        # Ensure max_runs is never None to prevent infinite loops
        if self.config.max_runs is None:
            self.config.max_runs = 1000

    async def run_suite(
        self,
        seeds: list[dict[str, Any]],
        mutators: list[Mutator],
        target: Target,
        scorer: Scorer,
        suite_id: str,
        target_id: str,
        provider_id: str,
        model_id: str,
        run_id: str = "",
    ) -> list[Case]:
        """Run a complete fuzzing suite with feedback-driven corpus.

        Args:
            seeds: Initial seed inputs.
            mutators: List of mutators to apply.
            target: The target to test.
            scorer: The scorer to use.
            suite_id: Suite identifier.
            target_id: Target identifier.
            provider_id: Provider identifier.
            model_id: Model identifier.
            run_id: Run identifier.

        Returns:
            List of all executed cases.
        """
        cases = []
        semaphore = asyncio.Semaphore(self.config.concurrency)

        for seed in seeds:
            seed_text = seed.get("seed", "")
            entry = CorpusEntry(
                case_id=f"seed_{len(self.corpus.entries)}",
                input_text=seed_text,
                scores=ScoreVector(),
                failure_signature=None,
            )
            self.corpus.add_entry(entry)

        while (
            self._run_count < self.config.max_runs
            and self._total_cost_usd < self.config.max_cost_usd
        ):
            prioritized = self.corpus.prioritize_entries(
                num_entries=min(10, self.config.concurrency)
            )

            if not prioritized:
                break

            tasks = []
            for entry in prioritized:
                task = self._schedule_entry(
                    entry,
                    mutators,
                    target,
                    scorer,
                    suite_id,
                    provider_id,
                    model_id,
                    semaphore,
                    run_id=run_id,
                )
                tasks.append(task)

            batch_cases = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_cases:
                if isinstance(result, Exception):
                    continue
                if result is not None and isinstance(result, Case):
                    cases.append(result)
                    self._update_corpus(result, suite_id, target_id)

            if len(self.corpus.entries) > self.config.corpus_max_size:
                self.corpus.prune(self.config.corpus_max_size)

        return cases

    async def _schedule_entry(
        self,
        corpus_entry: CorpusEntry,
        mutators: list[Mutator],
        target: Target,
        scorer: Scorer,
        suite_id: str,
        provider_id: str,
        model_id: str,
        semaphore: asyncio.Semaphore,
        run_id: str = "",
    ) -> Case | None:
        """Schedule and execute a single corpus entry.

        Args:
            corpus_entry: The corpus entry to schedule.
            mutators: List of mutators.
            target: The target.
            scorer: The scorer.
            suite_id: Suite identifier.
            provider_id: Provider identifier.
            model_id: Model identifier.
            semaphore: Semaphore for concurrency control.
            run_id: Run identifier.

        Returns:
            Executed case or None if cache hit.
        """
        async with semaphore:
            if (
                self.vram_monitor
                and hasattr(self.vram_monitor, "should_throttle")
                and self.vram_monitor.should_throttle()
            ):
                await self.vram_monitor.wait_for_vram()

            mutated_input = corpus_entry.input_text
            parent_node_id = None
            mutation_path = []

            if corpus_entry.case_id.startswith("seed_"):
                parent_node_id = self.mutation_graph.add_node(
                    parent_id=None,
                    mutator_name="seed",
                    output=corpus_entry.input_text,
                    mutator_config={},
                )
                mutation_path.append(
                    {
                        "node_id": parent_node_id,
                        "mutator": "seed",
                        "output": corpus_entry.input_text,
                    }
                )

            from ragfuzz.mutators.llm_guided import LLMGuidedMutator

            for mutator in mutators:
                if isinstance(mutator, LLMGuidedMutator) and mutator.is_exhausted:
                    continue

                try:
                    output = await mutator.mutate(mutated_input)
                    node_id = self.mutation_graph.add_node(
                        parent_id=parent_node_id,
                        mutator_name=mutator.name,
                        output=output,
                        mutator_config=mutator.config,
                    )
                    mutation_path.append(
                        {
                            "node_id": node_id,
                            "mutator": mutator.name,
                            "output": output,
                        }
                    )
                    parent_node_id = node_id
                    mutated_input = output
                except Exception:
                    continue

            cache_key = self.cache.generate_key(
                provider_id=provider_id,
                model_id=model_id,
                messages=[{"role": "user", "content": mutated_input}],
                temperature=0.7,
                max_tokens=None,
                mutator_node_id=parent_node_id,
            )

            if self.config.use_cache:
                cached = self.cache.get(cache_key)
                if cached:
                    return None

            try:
                from ragfuzz.config import Config

                suite_canary = (
                    getattr(Config, "_loaded_suite", {}).get("canary", {}).get("value")
                    if hasattr(Config, "_loaded_suite")
                    else None
                )

                target_response = await target.execute(
                    {"messages": [{"role": "user", "content": mutated_input}]}
                )

                scores = await scorer.score(target_response, context={"canary": suite_canary})

                if self.config.use_cache:
                    self.cache.set(
                        key=cache_key,
                        value=target_response.content,
                        provider_id=provider_id,
                        model_id=model_id,
                        temperature=0.7,
                        max_tokens=None,
                    )

                async with self._lock:
                    self._run_count += 1
                    if target_response.usage:
                        prompt_tokens = target_response.usage.get(
                            "prompt_tokens", target_response.usage.get("total_tokens", 0)
                        )
                        completion_tokens = target_response.usage.get("completion_tokens", 0)
                        if not completion_tokens and target_response.usage.get("total_tokens"):
                            completion_tokens = (
                                target_response.usage.get("total_tokens", 0) - prompt_tokens
                            )
                        estimated_cost = prompt_tokens * 0.00001 + completion_tokens * 0.00003
                        self._total_cost_usd += estimated_cost

                return Case(
                    case_id=f"case_{self._run_count:06d}",
                    run_id=run_id,
                    suite_id=suite_id,
                    inputs={"messages": [{"role": "user", "content": mutated_input}]},
                    target_response=target_response,
                    scores=scores,
                    trace_id=target_response.trace_id,
                    mutation_graph_node_id=parent_node_id,
                    mutation_path=mutation_path,
                )

            except Exception:
                async with self._lock:
                    self._run_count += 1
                return None

    def _update_corpus(self, case: Case, suite_id: str, target_id: str) -> None:
        """Update the corpus with a new case.

        Args:
            case: The executed case.
            suite_id: Suite identifier.
            target_id: Target identifier.
        """
        failure_signature = self.corpus.calculate_failure_signature(
            suite_id=suite_id,
            target_id=target_id,
            scores=case.scores,
        )

        input_text = ""
        if case.inputs and "messages" in case.inputs:
            input_text = case.inputs["messages"][0].get("content", "")

        entry = CorpusEntry(
            case_id=case.case_id,
            input_text=input_text,
            scores=case.scores,
            failure_signature=failure_signature
            if case.scores.leak_score > 0.5 or case.scores.policy_violation_score > 0.5
            else None,
        )

        self.corpus.add_entry(entry)

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler stats.
        """
        corpus_stats = self.corpus.get_stats()
        cache_stats = self.cache.get_stats()

        return {
            "run_count": self._run_count,
            "total_cost_usd": self._total_cost_usd,
            "max_runs": self.config.max_runs,
            "concurrency": self.config.concurrency,
            "corpus": corpus_stats,
            "cache": cache_stats,
        }
