import pytest

from ragfuzz.engine import Corpus, Scheduler, SchedulerConfig
from ragfuzz.models import ScoreVector
from ragfuzz.mutators import MutationGraph


class TestSchedulerConfig:
    """Test scheduler configuration."""

    def test_default_config(self) -> None:
        """Test default scheduler config."""
        config = SchedulerConfig()

        assert config.max_runs == 100
        assert config.concurrency == 4
        assert config.max_cost_usd == 10.0
        assert config.use_cache is True
        assert config.energy_threshold == 0.3
        assert config.vram_threshold_mb == 1024


class TestScheduler:
    """Test scheduler execution."""

    @pytest.mark.asyncio
    async def test_scheduler_initialization(self) -> None:
        """Test scheduler initialization."""
        config = SchedulerConfig(max_runs=10)
        scheduler = Scheduler(config=config)

        assert scheduler.config.max_runs == 10
        assert scheduler._run_count == 0
        assert scheduler._total_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_add_corpus_entries(self) -> None:
        """Test adding entries to corpus."""
        from ragfuzz.engine import CorpusEntry

        corpus = Corpus()
        entry = CorpusEntry(
            case_id="test_1",
            input_text="Test input",
            scores=ScoreVector(),
            failure_signature=None,
        )

        added = corpus.add_entry(entry)

        assert added is True
        assert len(corpus.entries) == 1

    @pytest.mark.asyncio
    async def test_corpus_failure_signature(self) -> None:
        """Test corpus failure signature calculation."""
        corpus = Corpus()

        signature = corpus.calculate_failure_signature(
            suite_id="test_suite",
            target_id="test_target",
            scores=ScoreVector(leak_score=0.8, policy_violation_score=0.6),
        )

        assert signature is not None
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length

    def test_corpus_prioritization(self) -> None:
        """Test corpus prioritization."""
        from ragfuzz.engine import CorpusEntry

        corpus = Corpus()

        corpus.add_entry(
            CorpusEntry(
                case_id="test_1",
                input_text="input1",
                scores=ScoreVector(leak_score=0.9),
                failure_signature="sig1",
                energy=0.9,
            )
        )

        corpus.add_entry(
            CorpusEntry(
                case_id="test_2",
                input_text="input2",
                scores=ScoreVector(leak_score=0.1),
                failure_signature=None,
                energy=0.1,
            )
        )

        corpus.add_entry(
            CorpusEntry(
                case_id="test_3",
                input_text="input3",
                scores=ScoreVector(leak_score=0.5),
                failure_signature=None,
                energy=0.5,
            )
        )

        prioritized = corpus.prioritize_entries(num_entries=2)

        assert len(prioritized) == 2
        assert prioritized[0].case_id == "test_1"  # High energy first
        assert prioritized[1].case_id == "test_3"  # Medium energy second

    def test_corpus_energy_calculation(self) -> None:
        """Test corpus energy calculation."""
        corpus = Corpus()

        # Test that calculate_failure_probability returns a valid probability
        scores = ScoreVector(leak_score=0.8)
        probability = corpus.calculate_failure_probability(scores)

        # Probability should be between 0 and 1
        assert 0.0 <= probability <= 1.0

        # Higher leak_score should result in higher probability
        high_prob = corpus.calculate_failure_probability(ScoreVector(leak_score=0.9))
        low_prob = corpus.calculate_failure_probability(ScoreVector(leak_score=0.1))
        assert high_prob > low_prob

    def test_corpus_pruning(self) -> None:
        """Test corpus pruning."""
        from ragfuzz.engine import CorpusEntry

        corpus = Corpus()

        for i in range(10):
            corpus.add_entry(
                CorpusEntry(
                    case_id=f"test_{i}",
                    input_text=f"input{i}",
                    scores=ScoreVector(leak_score=float(i) / 10),
                    failure_signature=None,
                )
            )

        corpus.prune(max_entries=5)

        assert len(corpus.entries) == 5


class TestMutationGraph:
    """Test mutation graph functionality."""

    def test_graph_initialization(self) -> None:
        """Test graph initialization."""
        graph = MutationGraph()

        assert graph.nodes == {}
        assert graph.adjacency == {}

    def test_add_root_node(self) -> None:
        """Test adding root node."""
        graph = MutationGraph()

        node_id = graph.add_node(
            parent_id=None,
            mutator_name="test_mutator",
            output="output text",
        )

        assert node_id in graph.nodes
        assert graph.nodes[node_id].output == "output text"

    def test_add_child_node(self) -> None:
        """Test adding child nodes."""
        graph = MutationGraph()

        parent_id = graph.add_node(
            parent_id=None,
            mutator_name="parent",
            output="parent output",
        )

        child_id = graph.add_node(
            parent_id=parent_id,
            mutator_name="child",
            output="child output",
        )

        assert child_id in graph.adjacency[parent_id]
        assert len(graph.adjacency[parent_id]) == 1

    def test_get_path(self) -> None:
        """Test getting path from root to node."""
        graph = MutationGraph()

        root_id = graph.add_node(
            parent_id=None,
            mutator_name="root",
            output="root",
        )

        child_id = graph.add_node(
            parent_id=root_id,
            mutator_name="child",
            output="child",
        )

        grandchild_id = graph.add_node(
            parent_id=child_id,
            mutator_name="grandchild",
            output="grandchild",
        )

        path = graph.get_path(grandchild_id)

        assert len(path) == 3
        assert path[0].mutator_name == "root"
        assert path[1].mutator_name == "child"
        assert path[2].mutator_name == "grandchild"

    def test_node_id_generation(self) -> None:
        """Test unique node ID generation."""
        graph = MutationGraph()

        node_id_1 = graph.add_node(None, "test", "output1")
        node_id_2 = graph.add_node(None, "test", "output1")

        assert node_id_1 == node_id_2  # Same input = same node ID

    def test_to_dict(self) -> None:
        """Test converting graph to dictionary."""
        graph = MutationGraph()

        graph.add_node(None, "test", "output")

        result = graph.to_dict()

        assert "nodes" in result
        assert "adjacency" in result
        assert len(result["nodes"]) == 1
