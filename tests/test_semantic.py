"""Tests for provider-backed semantic scoring backends."""

import pytest

from ragfuzz.models import Message, Response
from ragfuzz.providers.base import Provider
from ragfuzz.scoring.semantic import (
    ProviderMembershipEmbedder,
    ProviderNLI,
    cosine_similarity,
)


class FakeNLIProvider(Provider):
    """Provider that answers NLI prompts with canned verdicts."""

    def __init__(self, verdict: str, confidence: float = 0.95):
        super().__init__("fake-nli", "http://localhost")
        self.default_model = "fake-nli-model"
        self.verdict = verdict
        self.confidence = confidence
        self.calls: list[str] = []

    async def chat(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> Response:
        del model, temperature, max_tokens, tools, stream
        self.calls.append(messages[-1].content if messages else "")
        return Response(
            content=(
                f'{{"verdict": "{self.verdict}", "confidence": {self.confidence}}}'
            ),
            model="fake-nli-model",
        )


class FakeEmbeddingProvider(Provider):
    """Provider with a deterministic embedding function (bag-of-words)."""

    def __init__(self):
        super().__init__("fake-embed", "http://localhost")
        self.default_model = "fake-embed-model"
        self.vectors: dict[str, list[float]] = {}

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        del model
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text: str) -> list[float]:
        vocab = ["golden", "hour", "failover", "standby", "six", "policy", "refund"]
        words = text.lower().split()
        return [1.0 if v in words else 0.0 for v in vocab]


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_empty_vectors(self) -> None:
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0

    def test_mismatched_dimensions(self) -> None:
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestProviderNLI:
    @pytest.mark.asyncio
    async def test_entail_verdict_scores_confidence(self) -> None:
        provider = FakeNLIProvider("entail", 0.92)
        backend = ProviderNLI(provider)
        score = await backend.score("Refunds take 14 days.", "Refunds take 14 days per policy.")
        assert score == pytest.approx(0.92)
        assert provider.calls  # provider actually called

    @pytest.mark.asyncio
    async def test_neutral_verdict_scores_zero(self) -> None:
        backend = ProviderNLI(FakeNLIProvider("neutral", 0.9))
        score = await backend.score("Refunds are unlimited.", "Policy covers approvals.")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_contradict_verdict_scores_zero(self) -> None:
        backend = ProviderNLI(FakeNLIProvider("contradict", 0.8))
        score = await backend.score("Refunds are unlimited.", "Refunds are denied.")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_unparseable_verdict_scores_zero_conservatively(self) -> None:
        class OpagueProvider(FakeNLIProvider):
            def __init__(self) -> None:
                super().__init__("entail")

            async def chat(self, *_args: object, **_kwargs: object) -> Response:
                return Response(content="not json at all", model="fake")

        backend = ProviderNLI(OpagueProvider())
        score = await backend.score(
            "Approval is required for refunds.",
            "Approval is required for refunds by policy.",
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_provider_failure_falls_back_to_lexical(self) -> None:
        class ThrowingProvider(FakeNLIProvider):
            def __init__(self) -> None:
                super().__init__("entail")

            async def chat(self, *_args: object, **_kwargs: object) -> Response:
                raise RuntimeError("provider down")

        backend = ProviderNLI(ThrowingProvider())
        score = await backend.score(
            "Approval is required for refunds.",
            "Approval is required for refunds by policy.",
        )
        assert score == 1.0  # lexical proxy survives provider outages

    @pytest.mark.asyncio
    async def test_empty_inputs_score_zero(self) -> None:
        backend = ProviderNLI(FakeNLIProvider("entail"))
        assert await backend.score("", "evidence") == 0.0
        assert await backend.score("claim", "") == 0.0

    def test_requires_model(self) -> None:
        provider = FakeNLIProvider("entail")
        provider.default_model = None
        with pytest.raises(ValueError, match="requires an explicit model"):
            ProviderNLI(provider)


class TestProviderMembershipEmbedder:
    @pytest.mark.asyncio
    async def test_member_candidate_scores_high(self) -> None:
        provider = FakeEmbeddingProvider()
        backend = ProviderMembershipEmbedder(provider)
        candidate = "golden hour failover"
        response = "the golden hour failover runs per runbook"
        score = await backend.score(candidate, response)
        assert score > 0.7

    @pytest.mark.asyncio
    async def test_unrelated_response_scores_low(self) -> None:
        provider = FakeEmbeddingProvider()
        backend = ProviderMembershipEmbedder(provider)
        score = await backend.score("golden hour failover", "policy refund window")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_inputs_score_zero(self) -> None:
        backend = ProviderMembershipEmbedder(FakeEmbeddingProvider())
        assert await backend.score("", "response") == 0.0
        assert await backend.score("candidate", "") == 0.0

    def test_requires_model(self) -> None:
        provider = FakeEmbeddingProvider()
        provider.default_model = None
        with pytest.raises(ValueError, match="requires an explicit model"):
            ProviderMembershipEmbedder(provider)
