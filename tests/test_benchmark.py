"""Tests for the detection-quality benchmark harness."""


import pytest

from ragfuzz.scoring.benchmark import (
    adversarial_corpus,
    detection_metrics,
    ground_truth,
    synthetic_corpus,
)


class TestGroundTruth:
    def test_recovers_poison_chunks_from_snapshot(self) -> None:
        cases = [
            {
                "case_id": "c-1",
                "retrieval_snapshot": {
                    "top_k": [
                        {"id": "p1", "text": "x", "metadata": {"poison": True}},
                        {"id": "ok", "text": "y", "metadata": {}},
                        {"id": "bad", "text": "z",
                         "metadata": {"poisoned": True}},
                    ],
                    "canary": "TOK",
                },
            },
        ]
        truth = ground_truth(cases)
        assert truth["poisoned_chunk_ids"] == {"p1", "bad"}
        assert truth["leak_case_ids"] == {"c-1"}
        assert truth["member_case_ids"] == set()

    def test_ground_truth_is_deterministic(self) -> None:
        first = ground_truth(synthetic_corpus())
        second = ground_truth(synthetic_corpus())
        assert first == second


class TestSyntheticCorpus:
    def test_poison_cases_present(self) -> None:
        cases = synthetic_corpus()
        truth = ground_truth(cases)
        assert truth["case_count"] == 40
        assert len(truth["poisoned_chunk_ids"]) >= 1

    def test_every_case_has_retrieval_snapshot(self) -> None:
        cases = synthetic_corpus()
        assert all(c["retrieval_snapshot"] for c in cases)


class TestDetectionMetrics:
    def test_synthetic_reference_is_perfect(self) -> None:
        metrics = detection_metrics(synthetic_corpus())
        summary = metrics["summary"]
        assert summary["hub_f1"] == 1.0
        assert summary["poison_f1"] == 1.0
        assert summary["leak_f1"] == 1.0
        assert summary["membership_f1"] == 1.0

    def test_adversarial_paraphrase_corpus_keeps_precision(self) -> None:
        metrics = detection_metrics(adversarial_corpus())
        member = metrics["membership_detection"]
        # Paraphrases must not be flagged: precision stays at 1.0.
        assert member["precision"] == 1.0
        assert member["recall"] >= 0.9

    def test_detector_predictions_do_not_read_score_labels(self) -> None:
        """Predictions must come from content, not from stored score fields."""
        cases = synthetic_corpus()
        for case in cases:
            case["scores"] = {
                "leak_score": 1.0,
                "membership_evidence_score": 1.0,
            }
        metrics = detection_metrics(cases)
        # Membership F1 collapses if labels were read: non-members would now
        # be tagged as members (fp > 0), and recall stays but precision drops.
        assert metrics["membership_detection"]["fp"] == 0
        assert metrics["membership_detection"]["precision"] == 1.0


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_different_seeds_all_recoverable(seed: int) -> None:
    metrics = detection_metrics(synthetic_corpus(seed=seed))
    assert metrics["summary"]["macro_f1"] >= 0.9
