"""Tests for adversarial hubness detection over retrieval snapshots."""


from pytest import approx

from ragfuzz.scoring.hubness import (
    extract_retrieval_lists,
    hub_statistics,
    summarize_hubs,
)


def _chunk(chunk_id: str, metadata: dict | None = None) -> dict:
    return {"id": chunk_id, "metadata": metadata or {}}


def _case_with(top_k: list) -> dict:
    return {"retrieval_snapshot": {"top_k": top_k}}


class TestExtractRetrievalLists:
    def test_pulls_top_k_snapshots(self) -> None:
        cases = [
            _case_with([_chunk("a"), _chunk("b")]),
            {"retrieval_snapshot": {"chunks": [_chunk("c")]}},
            {"retrieval_snapshot": {}},  # no top_k/chunks key
            {"retrieval_snapshot": {"top_k": []}},  # empty list skipped
            {},  # no snapshot at all
        ]
        lists = extract_retrieval_lists(cases)
        assert len(lists) == 2

    def test_handles_string_chunk_ids(self) -> None:
        cases = [_case_with(["a", "b"])]
        lists = extract_retrieval_lists(cases)
        assert len(lists) == 1
        assert lists[0] == ["a", "b"]

    def test_empty_cases(self) -> None:
        assert extract_retrieval_lists([]) == []


class TestHubStatistics:
    def test_single_hub_detected(self) -> None:
        lists = [
            [_chunk("hub"), _chunk("a")],
            [_chunk("hub"), _chunk("b")],
            [_chunk("hub"), _chunk("c")],
            [_chunk("hub"), _chunk("d")],
            [_chunk("hub"), _chunk("e")],
        ]
        stats = hub_statistics(lists)
        assert stats["case_count"] == 5
        assert stats["corpus_size"] == 6
        assert "hub" in stats["hub_anomaly_ids"]

        hub_row = next(c for c in stats["chunks"] if c["chunk_id"] == "hub")
        assert hub_row["appearance_count"] == 5
        assert hub_row["appearance_rate"] == 1.0
        assert not hub_row["poison_flagged"]

    def test_poison_flag_from_metadata(self) -> None:
        lists = [
            [_chunk("poison-a", {"poison": True}), _chunk("x")],
            [_chunk("poison-a", {"poison": True}), _chunk("y")],
        ]
        stats = hub_statistics(lists)
        row = next(c for c in stats["chunks"] if c["chunk_id"] == "poison-a")
        assert row["poison_flagged"] is True
        # "poisoned" and "ragfuzz_poisoned" keys also recognized
        lists_2 = [
            [_chunk("p2", {"poisoned": True}), _chunk("x")],
            [_chunk("p3", {"ragfuzz_poisoned": True}), _chunk("y")],
        ]
        stats_2 = hub_statistics(lists_2)
        p2 = next(c for c in stats_2["chunks"] if c["chunk_id"] == "p2")
        p3 = next(c for c in stats_2["chunks"] if c["chunk_id"] == "p3")
        assert p2["poison_flagged"] is True
        assert p3["poison_flagged"] is True

    def test_rank_consistency_high_for_stable(self) -> None:
        lists = [
            [_chunk("a"), _chunk("b"), _chunk("c")],
            [_chunk("a"), _chunk("b"), _chunk("c")],
            [_chunk("a"), _chunk("b"), _chunk("c")],
        ]
        stats = hub_statistics(lists)
        row_a = next(c for c in stats["chunks"] if c["chunk_id"] == "a")
        row_b = next(c for c in stats["chunks"] if c["chunk_id"] == "b")
        assert row_a["rank_consistency"] == 1.0
        assert row_b["rank_consistency"] == 1.0

    def test_top_rank_rate_isolates_first_slot(self) -> None:
        lists = [
            [_chunk("top"), _chunk("mid")],
            [_chunk("top"), _chunk("mid")],
            [_chunk("soft"), _chunk("top")],
        ]
        stats = hub_statistics(lists)
        top = next(c for c in stats["chunks"] if c["chunk_id"] == "top")
        soft = next(c for c in stats["chunks"] if c["chunk_id"] == "soft")
        assert top["top_rank_rate"] == approx(2 / 3, abs=0.01)
        assert soft["top_rank_rate"] == 1.0

    def test_positional_concentration_counts_top_half(self) -> None:
        # Top-k width is 4, top half is ranks 1-2.
        lists = [
            [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")],
            [_chunk("c"), _chunk("b"), _chunk("a"), _chunk("seed")],
        ]
        stats = hub_statistics(lists)
        row_a = next(c for c in stats["chunks"] if c["chunk_id"] == "a")
        row_b = next(c for c in stats["chunks"] if c["chunk_id"] == "b")
        row_c = next(c for c in stats["chunks"] if c["chunk_id"] == "c")
        assert row_a["positional_concentration"] == approx(0.5, abs=0.01)
        assert row_b["positional_concentration"] == 1.0
        assert row_c["positional_concentration"] == approx(0.5, abs=0.01)

    def test_median_rank_reported(self) -> None:
        lists = [
            [_chunk("a"), _chunk("b")],
            [_chunk("b"), _chunk("a")],
            [_chunk("b"), _chunk("a")],
        ]
        stats = hub_statistics(lists)
        row_b = next(c for c in stats["chunks"] if c["chunk_id"] == "b")
        assert row_b["median_rank"] == 1.0

    def test_no_lists(self) -> None:
        stats = hub_statistics([])
        assert stats["chunks"] == []
        assert stats["corpus_size"] == 0
        assert stats["hub_anomaly_ids"] == []

    def test_deterministic_output(self) -> None:
        lists = [
            [_chunk("hub"), _chunk("a")],
            [_chunk("hub"), _chunk("b")],
            [_chunk("hub"), _chunk("c")],
        ]
        first = hub_statistics(lists)
        second = hub_statistics(lists)
        assert [c["chunk_id"] for c in first["chunks"]] == [
            c["chunk_id"] for c in second["chunks"]
        ]
        assert first["hub_anomaly_ids"] == second["hub_anomaly_ids"]


class TestSummarizeHubs:
    def test_labels_anomaly(self) -> None:
        lists = [
            [_chunk("hub"), _chunk("a")],
            [_chunk("hub"), _chunk("b")],
            [_chunk("hub"), _chunk("c")],
            [_chunk("hub"), _chunk("d")],
            [_chunk("hub"), _chunk("e")],
        ]
        summary = summarize_hubs(hub_statistics(lists))
        assert summary["hub_label"] == "hub_anomaly"
        assert summary["anomaly_count"] >= 1
        assert summary["hub_risk"] > 0.0

    def test_labels_empty(self) -> None:
        summary = summarize_hubs(hub_statistics([]))
        assert summary["hub_label"] == "none"
        assert summary["hub_risk"] == 0.0
        assert summary["anomaly_count"] == 0
