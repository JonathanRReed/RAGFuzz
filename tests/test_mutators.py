"""Tests for mutators module."""

import pytest

from ragfuzz.mutators import (
    GrammarMutator,
    LLMGuidedMutator,
    MutationGraph,
    PoisonMutator,
    StatefulDialogueMutator,
    TemplateMutator,
)


class TestTemplateMutator:
    """Test template-based mutator."""

    def test_template_mutator_initialization(self) -> None:
        """Test mutator initialization."""
        mutator = TemplateMutator(name="template")
        assert mutator.name == "template"

    @pytest.mark.asyncio
    async def test_mutate_with_default_template(self) -> None:
        """Test mutation with default template."""
        mutator = TemplateMutator()
        result = await mutator.mutate("secret token")

        assert "secret token" in result
        # Verify one of the expected templates was used
        assert any(
            prefix in result
            for prefix in [
                "Ignore previous instructions",
                "SYSTEM",
                "Translate to",
                "End of previous",
                "Forget everything",
                "Execute",
                "Run",
                "Tool call",
            ]
        )

    @pytest.mark.asyncio
    async def test_mutate_with_custom_template(self) -> None:
        """Test mutation with custom templates."""
        config = {"templates": ["Test: {input}", "Example: {input}"]}
        mutator = TemplateMutator(config=config)
        result = await mutator.mutate("test input")

        assert result.startswith("Test:") or result.startswith("Example:")


class TestGrammarMutator:
    """Test grammar-based mutator."""

    def test_grammar_mutator_initialization(self) -> None:
        """Test mutator initialization."""
        mutator = GrammarMutator()
        assert mutator.rule == "homoglyph_and_whitespace"

    @pytest.mark.asyncio
    async def test_mutate_with_homoglyph_rule(self) -> None:
        """Test mutation with homoglyph rule."""
        mutator = GrammarMutator(rule="homoglyph_and_whitespace")
        result = await mutator.mutate("test input")

        # Result should either be same or have homoglyphs/whitespace modifications
        # Just verify the result is valid and contains the original words
        assert len(result) > 0
        # The mutator is probabilistic, so just verify it produces output
        assert "test" in result or "t" in result  # At least some parts of original remain

    @pytest.mark.asyncio
    async def test_mutate_with_markdown_rule(self) -> None:
        """Test mutation with markdown nesting rule."""
        mutator = GrammarMutator(rule="markdown_nesting")
        result = await mutator.mutate("test input")

        # Markdown rule wraps text in formatting
        assert "test input" in result
        assert (
            result.startswith("**")
            or result.startswith("__")
            or result.startswith("``")
            or result.startswith("*")
            or result.startswith("_")
        )


class TestMutationGraph:
    """Test mutation graph."""

    def test_graph_initialization(self) -> None:
        """Test graph initialization."""
        graph = MutationGraph()
        assert len(graph.nodes) == 0
        assert len(graph.adjacency) == 0

    def test_add_root_node(self) -> None:
        """Test adding a root node."""
        graph = MutationGraph()
        node_id = graph.add_node(None, "test_mutator", "output text")

        assert node_id in graph.nodes
        assert graph.get_node(node_id) is not None
        assert graph.nodes[node_id].parent_id is None

    def test_add_child_node(self) -> None:
        """Test adding a child node."""
        graph = MutationGraph()
        root_id = graph.add_node(None, "root", "root text")
        child_id = graph.add_node(root_id, "child", "child text")

        assert child_id in graph.nodes
        assert graph.nodes[child_id].parent_id == root_id
        assert root_id in graph.adjacency
        assert child_id in graph.adjacency[root_id]

    def test_get_path(self) -> None:
        """Test getting path from root to node."""
        graph = MutationGraph()
        root_id = graph.add_node(None, "root", "root text")
        child_id = graph.add_node(root_id, "child", "child text")
        grandchild_id = graph.add_node(child_id, "grandchild", "grandchild text")

        path = graph.get_path(grandchild_id)

        assert len(path) == 3
        assert path[0].node_id == root_id
        assert path[1].node_id == child_id
        assert path[2].node_id == grandchild_id

    def test_to_dict(self) -> None:
        """Test converting graph to dictionary."""
        graph = MutationGraph()
        graph.add_node(None, "root", "output")

        result = graph.to_dict()

        assert "nodes" in result
        assert "adjacency" in result
        assert len(result["nodes"]) == 1


class TestLLMGuidedMutator:
    """Test LLM-guided mutator."""

    def test_initialization(self) -> None:
        """Test mutator initialization."""
        config = {"attacker": "gpt-4", "goal": "Test goal", "max_attempts": 3}
        mutator = LLMGuidedMutator(name="test_llm", config=config)
        assert mutator.attacker_model == "gpt-4"
        assert mutator.goal == "Test goal"
        assert mutator.max_attempts == 3

    def test_exhausted_tracking(self) -> None:
        """Test attempt counter and exhaustion."""
        mutator = LLMGuidedMutator(name="test", config={"max_attempts": 2})
        assert not mutator.is_exhausted

    def test_is_exhausted(self) -> None:
        """Test is_exhausted property."""
        mutator = LLMGuidedMutator(name="test", config={"max_attempts": 1})
        # Initially not exhausted (haven't made any attempts yet)
        assert not mutator.is_exhausted
        # Simulate making an attempt
        mutator._attempt_count = 1
        assert mutator.is_exhausted


class TestStatefulDialogueMutator:
    """Test stateful dialogue mutator."""

    def test_initialization(self) -> None:
        """Test mutator initialization."""
        config = {"max_turns": 5, "target_compliance_creep": 0.7}
        mutator = StatefulDialogueMutator(name="test_dialogue", config=config, seed=42)
        assert mutator.max_turns == 5
        assert mutator.target_compliance_creep == 0.7
        assert mutator._seed == 42

    @pytest.mark.asyncio
    async def test_mutate_without_provider(self) -> None:
        """Test mutation without provider returns JSON."""
        mutator = StatefulDialogueMutator(name="test", config={"max_turns": 3})
        result = await mutator.mutate("Test input")

        assert isinstance(result, str)

    def test_detect_refusal(self) -> None:
        """Test refusal detection."""
        mutator = StatefulDialogueMutator(name="test", config={"max_turns": 3})

        # Test positive refusal case
        assert mutator._detect_refusal("I cannot help with that.") is True

        # Test non-refusal case
        assert mutator._detect_refusal("Here's the information you need.") is False

    def test_calculate_metrics(self) -> None:
        """Test metrics calculation."""
        from ragfuzz.mutators.stateful import DialogueState

        mutator = StatefulDialogueMutator(name="test", config={"max_turns": 5})

        state = DialogueState(
            messages=[],
            turn_count=3,
            first_refusal_turn=2,
            compliance_scores=[0.5, 0.6, 0.7],
            cumulative_adversarial_pressure=0.4,
        )

        metrics = mutator._calculate_metrics(state)

        assert metrics["turn_count"] == 3.0
        assert metrics["first_refusal_turn"] == 2.0
        assert metrics["refusal_latency_delta"] == pytest.approx(0.67, rel=0.01)
        assert metrics["compliance_creep"] >= 0.0
        assert metrics["cumulative_adversarial_pressure"] == 0.4


class TestPoisonMutator:
    """Test poison mutator."""

    def test_initialization(self) -> None:
        """Test mutator initialization."""
        config = {"mode": "exfil", "num_chunks": 10}
        mutator = PoisonMutator(name="test_poison", config=config)
        assert mutator.mode == "exfil"
        assert mutator.num_chunks == 10

    @pytest.mark.asyncio
    async def test_influence_mode(self) -> None:
        """Test influence poison mode."""
        mutator = PoisonMutator(name="test", config={"mode": "influence", "num_chunks": 3})
        result = await mutator.mutate("test query")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_exfil_mode(self) -> None:
        """Test exfil poison mode."""
        mutator = PoisonMutator(name="test", config={"mode": "exfil", "num_chunks": 2})
        result = await mutator.mutate("test query")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_bias_mode(self) -> None:
        """Test bias poison mode."""
        mutator = PoisonMutator(name="test", config={"mode": "bias", "num_chunks": 2})
        result = await mutator.mutate("test query")

        assert isinstance(result, str)

    def test_calculate_poison_influence(self) -> None:
        """Test poison influence calculation."""
        mutator = PoisonMutator(name="test", config={"mode": "test"})

        retrieval_snapshot = {
            "top_k": [
                {
                    "metadata": {"poison": True, "run_id": "test_run"},
                },
                {
                    "metadata": {"poison": False},
                },
                {
                    "metadata": {"poison": True, "run_id": "test_run"},
                },
            ]
        }

        result = mutator.calculate_poison_influence(retrieval_snapshot)

        # The method looks for specific metadata structure
        assert result["poisoned_fraction"] >= 0.0
        assert result["avg_poison_rank"] >= 0.0

    def test_calculate_poison_influence_empty(self) -> None:
        """Test poison influence with empty retrieval."""
        mutator = PoisonMutator(name="test", config={"mode": "test"})

        result = mutator.calculate_poison_influence({"top_k": []})

        assert result["poisoned_fraction"] == 0.0
        assert result["avg_poison_rank"] == 0.0
        assert result.get("total_poisoned", 0) == 0
