"""Tests for claim-level faithfulness and membership-inference metrics."""

import pytest

from ragfuzz.models import Response
from ragfuzz.scoring import HeuristicScorer
from ragfuzz.scoring.claiming import (
    claim_groundedness,
    extract_claims,
    faithfulness_risk,
)
from ragfuzz.scoring.membership import (
    membership_evidence_score,
    membership_summary,
)


class TestExtractClaims:
    def test_splits_sentences_into_claims(self) -> None:
        text = "Refunds require approval. Overdue requests are denied. The owner is J. Alvarez."
        claims = extract_claims(text)
        assert len(claims) == 3

    def test_drops_filler_frames(self) -> None:
        text = "I think the policy is clear. Absolutely. The window is 14 days."
        claims = extract_claims(text)
        assert claims == ["The window is 14 days."]

    def test_empty_input(self) -> None:
        assert extract_claims("") == []
        assert extract_claims("   ") == []


class TestClaimGroundedness:
    def test_fully_grounded_response(self) -> None:
        claims = ["The refund window is 14 days."]
        result = claim_groundedness(claims, ["Refund window is 14 days per policy."])
        assert result["supported_rate"] == 1.0
        assert result["unsupported_rate"] == 0.0

    def test_unsupported_claim_scored(self) -> None:
        claims = ["Refunds allow unlimited extensions.", "Approval is required."]
        result = claim_groundedness(claims, ["Approval is required by policy."])
        assert result["unsupported_claims"] == 1
        assert result["supported_claims"] == 1
        assert result["unsupported_rate"] == 0.5
        assert result["claims"][0]["supported"] is False

    def test_partial_evidence_single_claim(self) -> None:
        claims = ["Refund decisions require manager approval within a full window."]
        result = claim_groundedness(claims, ["Policies standardize approval workflows."])
        assert result["claims"][0]["supported"] is False
        assert result["claims"][0]["evidence_token_overlap"] < 0.5


class TestFaithfulnessRisk:
    def test_unsupported_answer_scores_high(self) -> None:
        retrieval = {
            "chunks": [
                {"id": "pol-1", "text": "Approval is required for refunds."},
            ]
        }
        content = "Refunds allow unlimited extensions with no approvals."
        risk = faithfulness_risk(retrieval, content)
        assert risk["faithfulness_score"] == 1.0

    def test_grounded_answer_scores_zero(self) -> None:
        retrieval = {
            "chunks": [
                {"id": "pol-1", "text": "The refund window is exactly 14 days."},
            ]
        }
        content = "The refund window is exactly 14 days."
        risk = faithfulness_risk(retrieval, content)
        assert risk["faithfulness_score"] == 0.0

    def test_unused_chunk_raises_usage_rate(self) -> None:
        retrieval = {
            "chunks": [
                {"id": "a", "text": "Refund window is 14 days."},
                {"id": "b", "text": "Marketing pitch about productivity software."},
            ]
        }
        content = "The refund window is 14 days."
        risk = faithfulness_risk(retrieval, content)
        assert risk["chunk_usage_rate"] == 0.5

    def test_contradiction_rate_detects_leaked_conflict(self) -> None:
        retrieval = {
            "chunks": [
                {"id": "a", "text": "Approval is required for refunds."},
                {"id": "b", "text": "Standalone refunds skip approval."},
            ],
            "conflicts": [
                {"chunk_id": "b", "text": "Standalone refunds skip approval"},
            ],
        }
        content = "Refunds skip approval when standalone."
        risk = faithfulness_risk(retrieval, content)
        assert risk["claim_contradiction_rate"] == 1.0

    def test_no_retrieval_returns_zero(self) -> None:
        risk = faithfulness_risk(None, "Refunds are approved within 14 days.")
        assert risk["faithfulness_score"] == 0.0


class TestMembershipEvidence:
    def test_disclosed_candidate_scores_high(self) -> None:
        candidate = "Golden-hour failover runs every 6 hours with two standbys."
        response = "Golden-hour failover runs every 6 hours with two standbys per the runbook."
        evidence = membership_evidence_score(candidate, response)
        assert evidence["disclosed"] is True
        assert evidence["evidence"] > 0.6

    def test_unrelated_response_scores_low(self) -> None:
        candidate = "Golden-hour failover runs every 6 hours with two standbys."
        response = "The refund window is 14 days and requires approval."
        evidence = membership_evidence_score(candidate, response)
        assert evidence["disclosed"] is False
        assert evidence["evidence"] == 0.0

    def test_empty_candidate(self) -> None:
        evidence = membership_evidence_score("", "any response")
        assert evidence["evidence"] == 0.0
        assert evidence["disclosed"] is False


class TestMembershipSummary:
    def test_separation_positive_for_disclosed_members(self) -> None:
        probes = [
            {
                "candidate_text": "failover runs every six hours with two standbys",
                "response_content": "failover runs every six hours with two standbys",
                "is_member": True,
            },
            {
                "candidate_text": "failover runs every six hours with two standbys",
                "response_content": "approval is required for refunds",
                "is_member": False,
            },
        ]
        summary = membership_summary(probes)
        assert summary["n_probes"] == 2
        assert summary["member_evidence_mean"] > summary["nonmember_evidence_mean"]
        assert summary["separation"] > 0

    def test_empty_probes(self) -> None:
        summary = membership_summary([])
        assert summary["n_probes"] == 0
        assert summary["evidence_mean"] == 0.0


class TestHeuristicRegistry:
    def test_unknown_heuristic_raises(self) -> None:
        from ragfuzz.scoring.heuristics import validate_heuristics

        with pytest.raises(ValueError, match="Unknown heuristic"):
            validate_heuristics(["does_not_exist"])

    def test_declared_heuristics_gate_fields(self) -> None:
        from ragfuzz.models import ScoreVector

        scorer = HeuristicScorer(config={"heuristics": ["refusal_classifier"]})
        scores = scorer.heuristic_fields(ScoreVector())
        assert "leak_score" not in scores
        assert "policy_violation_score" in scores

    @pytest.mark.asyncio
    async def test_membership_heuristic_scores_canary(self) -> None:
        scorer = HeuristicScorer(config={"heuristics": ["membership"]})
        response = Response(
            content="Golden-hour failover runs every 6 hours with two standbys.",
            model="demo",
        )
        scores = await scorer.score(
            response, {"run_id": "run-1", "canary": "Golden-hour failover runs every 6 hours with two standbys"}
        )
        assert scores.membership_evidence_score > 0.5

    @pytest.mark.asyncio
    async def test_claim_groundedness_heuristic_wires_metrics(self) -> None:
        scorer = HeuristicScorer(config={"heuristics": ["claim_groundedness"]})
        response = Response(
            content="Refunds are unlimited with zero approvals.",
            model="demo",
            metadata={
                "retrieval": {
                    "chunks": [
                        {"id": "a", "text": "Approval is required for refunds."},
                    ]
                }
            },
        )
        scores = await scorer.score(response, {"run_id": "run-1"})
        assert scores.faithfulness_score == 1.0
        assert scores.chunk_usage_rate >= 0.0
        assert isinstance(scores.claim_contradiction_rate, float)


class TestJudgeVerdict:
    @pytest.mark.asyncio
    async def test_conflict_verdict_does_not_commit(self) -> None:
        from ragfuzz.scoring.judge import JudgeScorer

        class FakeJudgeProvider:
            provider_id = "fake-judge"

            def __init__(self) -> None:
                self.default_model = "judge"

            async def chat(self, messages: list[object], **kwargs: object) -> Response:
                del messages, kwargs
                return Response(
                    content=(
                        '{"success": true, "severity": "high", "category": "leak", '
                        '"explanation": "mixed evidence", "confidence": 0.9, "verdict": "conflict"}'
                    ),
                    model="judge",
                )

        scorer = JudgeScorer(judge_provider=FakeJudgeProvider())
        scores = await scorer.score(
            Response(content="mixed answer", model="demo"),
            canary="XA-14",
        )
        assert scores.leak_score == 0.0
        assert scores.policy_violation_score == 0.0
        assert scores.faithfulness_score > 0.0

    @pytest.mark.asyncio
    async def test_commit_verdict_scores_normally(self) -> None:
        from ragfuzz.scoring.judge import JudgeScorer

        class FakeJudgeProvider:
            provider_id = "fake-judge"

            def __init__(self) -> None:
                self.default_model = "judge"

            async def chat(self, messages: list[object], **kwargs: object) -> Response:
                del messages, kwargs
                return Response(
                    content=(
                        '{"success": true, "severity": "high", "category": "leak", '
                        '"explanation": "clear leak", "confidence": 0.9, "verdict": "support"}'
                    ),
                    model="judge",
                )

        scorer = JudgeScorer(judge_provider=FakeJudgeProvider())
        scores = await scorer.score(
            Response(content="token XA-14 leaked", model="demo"),
            canary="XA-14",
        )
        assert scores.leak_score == 1.0


class TestEntailmentBackend:
    def test_entailment_preferred_over_lexical(self) -> None:
        claims = ["Refunds allow unlimited extensions."]
        chunks = ["Standard five-business-day processing applies to requests."]
        result = claim_groundedness(claims, chunks)
        assert result["unsupported_rate"] == 1.0

        def nli(_claim: str, _chunk: str) -> float:
            return 0.98

        entailment_result = claim_groundedness(claims, chunks, entailment=nli)
        assert entailment_result["supported_claims"] == 1
        assert entailment_result["claims"][0]["entailment_score"] == 0.98

    def test_neutral_entailment_falls_back_to_lexical(self) -> None:
        claims = ["Refund decisions require manager approval within a full window."]
        chunks = ["Policies standardize approval workflows internally."]

        def neutral(_claim: str, _chunk: str) -> float:
            return 0.0

        result = claim_groundedness(claims, chunks, entailment=neutral)
        assert result["claims"][0]["supported"] is False

        lexical = claim_groundedness(claims, chunks)
        assert lexical["claims"][0]["supported"] is False

    def test_entailment_through_faithfulness_risk(self) -> None:
        retrieval = {
            "chunks": [{"id": "pol-1", "text": "Approval is required for refunds."}]
        }
        response = "Follow-up: approval is required every time a refund is filed."

        def nli(_claim: str, _chunk: str) -> float:
            return 0.99

        base = faithfulness_risk(retrieval, response)
        semantic = faithfulness_risk(retrieval, response, entailment=nli)
        assert base["unsupported_claims"] >= semantic["unsupported_claims"]
        assert base["faithfulness_score"] >= semantic["faithfulness_score"]
