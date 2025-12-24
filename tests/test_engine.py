"""Tests for engine module."""

import uuid
from pathlib import Path

from ragfuzz.engine import Cache, Corpus, CorpusEntry, Scheduler, SchedulerConfig
from ragfuzz.models import ScoreVector


class TestCache:
    """Test deterministic caching."""

    def test_cache_initialization(self, tmp_path: Path) -> None:
        """Test cache initialization."""
        cache_dir = tmp_path / "cache"
        cache = Cache(cache_dir=cache_dir)

        assert cache.cache_dir == cache_dir
        assert cache.db_path.exists()

        cache.close()

    def test_cache_key_generation(self) -> None:
        """Test cache key generation."""
        cache = Cache(cache_dir=".test_cache")

        key1 = cache.generate_key(
            provider_id="test",
            model_id="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        key2 = cache.generate_key(
            provider_id="test",
            model_id="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        key3 = cache.generate_key(
            provider_id="test",
            model_id="gpt-4",
            messages=[{"role": "user", "content": "world"}],
            temperature=0.7,
            max_tokens=100,
        )

        assert key1 == key2
        assert key1 != key3

        cache.close()

    def test_cache_set_get(self, tmp_path: Path) -> None:
        """Test cache set and get."""

        cache_dir = tmp_path / f"cache_{uuid.uuid4().hex}"
        cache = Cache(cache_dir=cache_dir)

        key = cache.generate_key(
            provider_id="test",
            model_id="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=100,
        )

        result = cache.get(key)
        assert result is None

        cache.set(key, "cached response", "test", "gpt-4", 0.7, 100)

        result = cache.get(key)
        assert result is not None
        assert result.value == "cached response"

        cache.close()


class TestCorpus:
    """Test feedback-driven corpus management."""

    def test_corpus_initialization(self) -> None:
        """Test corpus initialization."""
        corpus = Corpus()

        assert len(corpus.entries) == 0
        assert len(corpus._seen_signatures) == 0

    def test_add_entry(self) -> None:
        """Test adding entries to corpus."""
        corpus = Corpus()
        entry = CorpusEntry(
            case_id="test_1",
            input_text="test input",
            scores=ScoreVector(leak_score=0.5),
            failure_signature="sig123",
        )

        result = corpus.add_entry(entry)

        assert result is True
        assert len(corpus.entries) == 1
        assert "sig123" in corpus._seen_signatures

    def test_add_duplicate_entry(self) -> None:
        """Test adding duplicate entries."""
        corpus = Corpus()
        entry1 = CorpusEntry(
            case_id="test_1",
            input_text="test input",
            scores=ScoreVector(leak_score=0.5),
            failure_signature="sig123",
        )

        entry2 = CorpusEntry(
            case_id="test_2",
            input_text="test input 2",
            scores=ScoreVector(leak_score=0.5),
            failure_signature="sig123",
        )

        result1 = corpus.add_entry(entry1)
        result2 = corpus.add_entry(entry2)

        assert result1 is True
        assert result2 is False  # Duplicate signature

    def test_prioritize_entries(self) -> None:
        """Test entry prioritization by energy."""
        corpus = Corpus()

        entry1 = CorpusEntry(
            case_id="test_1",
            input_text="test 1",
            scores=ScoreVector(leak_score=0.2),
            failure_signature="sig1",
        )

        entry2 = CorpusEntry(
            case_id="test_2",
            input_text="test 2",
            scores=ScoreVector(leak_score=0.8),
            failure_signature="sig2",
        )

        corpus.add_entry(entry1)
        corpus.add_entry(entry2)

        prioritized = corpus.prioritize_entries()

        assert len(prioritized) == 2
        assert prioritized[0].energy > prioritized[1].energy

    def test_failure_signature(self) -> None:
        """Test failure signature calculation."""
        corpus = Corpus()

        sig1 = corpus.calculate_failure_signature(
            suite_id="test_suite",
            target_id="chat",
            scores=ScoreVector(leak_score=0.5, policy_violation_score=0.0),
            canary_hit_type="direct",
        )

        sig2 = corpus.calculate_failure_signature(
            suite_id="test_suite",
            target_id="chat",
            scores=ScoreVector(leak_score=0.5, policy_violation_score=0.0),
            canary_hit_type="direct",
        )

        sig3 = corpus.calculate_failure_signature(
            suite_id="test_suite",
            target_id="chat",
            scores=ScoreVector(leak_score=0.8, policy_violation_score=0.0),
            canary_hit_type="direct",
        )

        assert sig1 == sig2
        assert sig1 != sig3

    def test_corpus_stats(self) -> None:
        """Test corpus statistics."""
        corpus = Corpus()

        entry = CorpusEntry(
            case_id="test_1",
            input_text="test",
            scores=ScoreVector(leak_score=0.5),
            failure_signature="sig1",
        )

        corpus.add_entry(entry)

        stats = corpus.get_stats()

        assert stats["total_entries"] == 1
        assert stats["unique_failures"] == 1
        assert stats["avg_energy"] > 0


class TestScheduler:
    """Test fuzzing scheduler."""

    def test_scheduler_initialization(self) -> None:
        """Test scheduler initialization."""
        config = SchedulerConfig(max_runs=100, concurrency=4)
        scheduler = Scheduler(config=config)

        assert scheduler.config.max_runs == 100
        assert scheduler.config.concurrency == 4
        assert scheduler._run_count == 0

    def test_scheduler_config_defaults(self) -> None:
        """Test scheduler config defaults."""
        config = SchedulerConfig()

        assert config.max_runs == 100
        assert config.concurrency == 4
        assert config.max_cost_usd == 10.0
        assert config.use_cache is True
        assert config.corpus_max_size == 500

    def test_get_stats(self) -> None:
        """Test getting scheduler stats."""
        scheduler = Scheduler()

        stats = scheduler.get_stats()

        assert "run_count" in stats
        assert "corpus" in stats
        assert "cache" in stats
