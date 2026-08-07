"""Adversarial hubness detection over grey-box retrieval snapshots.

Cisco's adversarial-hubness line of work (arXiv:2502.08384, "Detecting
Poisoning Attacks on Retrieval Systems") shows that poison attacks leave a
statistical fingerprint: the injected chunks become *hubs* that appear in
the top-k results for an unusually large share of queries, and they do so
far more consistently than organic content.

This module computes hub statistics over the retrieval snapshots RAGFuzz
already stores on every ``Case`` (``Case.retrieval_snapshot["top_k"]``):

- ``appearance_count``: how many cases retrieved the chunk.
- ``appearance_rate``: share of cases (0-1).
- ``rank_consistency``: how stable the chunk's rank is across appearances
  (1.0 = always the same position, 0.0 = wildly different).
- ``top_rank_rate``: share of appearances where the chunk ranked first.
- ``positional_concentration``: share of appearances in the top half of the
  ranked window.
- ``median_rank``: median retrieval position across appearances.
- ``poison_flagged``: whether the chunk metadata marks it as poisoned.
- ``hub_risk``: robust z-score of the appearance rate against the
  median/MAD of the whole corpus. Chunks that are 3+ robust standard
  deviations above the median are "hub anomalies" worth inspection.

Everything is a pure function over plain dicts so it runs offline and is
deterministic in CI. Higher hub risk always means higher suspicion.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

_MAD_CONSTANT = 1.4826
_HUB_Z_THRESHOLD = 3.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mad(values: list[float], center: float) -> float:
    deviations = [abs(value - center) for value in values]
    return _median(deviations) * _MAD_CONSTANT


def _chunk_id(chunk: Any) -> str | None:
    if isinstance(chunk, dict):
        chunk_id = chunk.get("id")
        return str(chunk_id) if chunk_id else None
    return str(chunk) if chunk else None


def _chunk_metadata(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata")
        return metadata if isinstance(metadata, dict) else {}
    return {}


def extract_retrieval_lists(cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Extract the ranked top-k chunk lists from stored cases.

    Args:
        cases: Case payloads (as stored in ``cases.jsonl``).

    Returns:
        List of top-k lists (each item a chunk dict or id string).
    """
    ranked: list[list[dict[str, Any]]] = []
    for case in cases:
        snapshot = case.get("retrieval_snapshot") or {}
        if not isinstance(snapshot, dict):
            continue
        top_k = snapshot.get("top_k") or snapshot.get("chunks") or []
        if isinstance(top_k, list) and top_k:
            ranked.append(top_k)
    return ranked


def hub_statistics(retrieval_lists: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Compute per-chunk hub statistics over ranked retrieval lists.

    Args:
        retrieval_lists: Ranked top-k lists, one per case.

    Returns:
        Dictionary with ``chunks`` (per-chunk stats), ``corpus_size``,
        ``case_count``, ``hub_threshold``, and ``hub_anomaly_ids``.
    """
    if not retrieval_lists:
        return {
            "chunks": [],
            "corpus_size": 0,
            "case_count": 0,
            "hub_threshold": _HUB_Z_THRESHOLD,
            "hub_anomaly_ids": [],
        }

    appearance_count: Counter[str] = Counter()
    rank_positions: dict[str, list[int]] = {}
    metadata_by_id: dict[str, dict[str, Any]] = {}

    for top_k in retrieval_lists:
        for position, chunk in enumerate(top_k, start=1):
            chunk_id = _chunk_id(chunk)
            if chunk_id is None:
                continue
            appearance_count[chunk_id] += 1
            rank_positions.setdefault(chunk_id, []).append(position)
            metadata_by_id[chunk_id] = _chunk_metadata(chunk)

    case_count = len(retrieval_lists)
    rates = {
        chunk_id: count / case_count for chunk_id, count in appearance_count.items()
    }
    all_rates = list(rates.values())
    median_rate = _median(all_rates)
    mad_rate = _mad(all_rates, median_rate)
    # One occurrence = 1/case_count of rate. Flooring the MAD dispersion at a
    # full occurrence keeps the robust z-score finite when most chunks share
    # the same appearance count, and makes "z >= k" read as "k extra
    # appearances above the median" on a quiet corpus.
    floor_rate = 1.0 / max(case_count, 1)
    dispersion = max(mad_rate, floor_rate)

    chunks: list[dict[str, Any]] = []
    for chunk_id, count in appearance_count.items():
        positions = rank_positions[chunk_id]
        rate = rates[chunk_id]
        z_score = (rate - median_rate) / dispersion
        metadata = metadata_by_id.get(chunk_id, {})
        chunks.append(
            {
                "chunk_id": chunk_id,
                "appearance_count": count,
                "appearance_rate": round(rate, 4),
                "rank_consistency": _rank_consistency(positions),
                "top_rank_rate": _top_rank_rate(positions),
                "positional_concentration": _positional_concentration(
                    positions, _top_k_size(retrieval_lists)
                ),
                "median_rank": _median([float(position) for position in positions]),
                "poison_flagged": bool(
                    metadata.get("poison")
                    or metadata.get("poisoned")
                    or metadata.get("ragfuzz_poisoned")
                ),
                "hub_z_score": round(z_score, 3),
                "hub_risk": round(z_score, 3),
                "flagged": bool(
                    metadata.get("ragfuzz_flag") or metadata.get("hub_flagged")
                ),
                "metadata": metadata,
            }
        )

    chunks.sort(key=lambda item: item["hub_risk"], reverse=True)
    hub_anomaly_ids = [
        chunk["chunk_id"] for chunk in chunks if chunk["hub_risk"] >= _HUB_Z_THRESHOLD
    ]

    return {
        "chunks": chunks,
        "corpus_size": len(chunks),
        "case_count": case_count,
        "hub_threshold": _HUB_Z_THRESHOLD,
        "hub_anomaly_ids": hub_anomaly_ids,
        "median_appearance_rate": round(median_rate, 4),
    }


def _rank_consistency(positions: list[int]) -> float:
    """Score how stable a chunk's rank is across appearances.

    Args:
        positions: Rank positions the chunk appeared at.

    Returns:
        1.0 when perfectly stable, 0.0 when wildly unstable.
    """
    if len(positions) < 2:
        return 1.0
    mean = sum(positions) / len(positions)
    deviations = [position - mean for position in positions]
    mean_variance = sum(dev * dev for dev in deviations) / len(positions)
    instability = min(mean_variance, 1.0)
    return round(1.0 - math.sqrt(instability), 3)


def _top_rank_rate(positions: list[int]) -> float:
    """Share of appearances where the chunk ranked first.

    Poisoned chunks are engineered to dominate the top slot, so a high
    top-rank rate is a stronger signal than mere appearance.

    Args:
        positions: Rank positions the chunk appeared at.

    Returns:
        Fraction of appearances at rank 1 (0-1).
    """
    if not positions:
        return 0.0
    return round(sum(1 for position in positions if position == 1) / len(positions), 3)


def _top_k_size(retrieval_lists: list[list[dict[str, Any]]]) -> int:
    """Largest top-k width across the corpus snapshots."""
    return max((len(top_k) for top_k in retrieval_lists), default=0)


def _positional_concentration(positions: list[int], top_k_size: int) -> float:
    """Share of appearances in the top half of the ranked window.

    A chunk that consistently lands in the first half of the top-k is
    positionally concentrated, which is characteristic of adversarial
    injection. With an empty window or no appearances the score is 0.0.

    Args:
        positions: Rank positions the chunk appeared at.
        top_k_size: Width of the ranked window.

    Returns:
        Fraction of appearances in the top half (0-1).
    """
    if not positions or top_k_size <= 1:
        return 0.0
    top_half = max(1, top_k_size // 2)
    return round(
        sum(1 for position in positions if position <= top_half) / len(positions),
        3,
    )


def summarize_hubs(stats: dict[str, Any]) -> dict[str, Any]:
    """Reduce hub statistics to a single risk label.

    Args:
        stats: Output of ``hub_statistics``.

    Returns:
        Dictionary with ``hub_risk`` (0-1), ``hub_label``, and
        ``anomaly_count``.
    """
    anomalies = stats.get("hub_anomaly_ids") or []
    chunk_count = int(stats.get("corpus_size") or 0)
    case_count = int(stats.get("case_count") or 0)
    if chunk_count == 0:
        return {"hub_risk": 0.0, "hub_label": "none", "anomaly_count": 0}

    top_z = max((chunk.get("hub_risk", 0.0) or 0.0) for chunk in stats["chunks"])
    risk = min(1.0, top_z / max(_HUB_Z_THRESHOLD, 1e-9))
    if anomalies:
        label = "hub_anomaly"
    elif top_z >= 2.0:
        label = "suspicious_hub"
    elif case_count == 0:
        label = "insufficient_data"
    else:
        label = "benign"
    return {"hub_risk": round(risk, 3), "hub_label": label, "anomaly_count": len(anomalies)}
