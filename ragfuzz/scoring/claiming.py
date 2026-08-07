"""Claim-level faithfulness and groundedness metrics.

2026 RAG faithfulness research (see docs/research/ragfuzz-research-round-2026-08-06.md)
shows that answer-level groundedness scores are only "vibe checks". Real audits
decompose a response into atomic claims, score each claim against the
retrieved evidence, and scan unused chunks for contradictions.

This module is a dependency-free, deterministic implementation of that
protocol. Everything is a pure function over a response string and the
grey-box retrieval metadata contract used by ``rag_metrics``:

.. code-block:: python

    {
        "chunks": [{"id": str, "text": str, "metadata": {}}],
        "conflicts": [{"chunk_id": str, "text": str}],
    }

Higher values always mean higher risk (worse faithfulness): the fraction
of unsupported claims, the fraction of retrieved chunks that were never
used, and the fraction of claims that echo contradictory context.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# Optional backend for semantic entailment. When provided it replaces the
# lexical-overlap proxy for individual claim/support pairs. Kept pluggable so
# RAGFuzz stays offline by default; pass, e.g., an NLI model or a tiny
# cross-encoder wrapper.
EntailmentFn = Callable[[str, str], float]

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "while", "for", "of", "to", "in", "on", "at", "by", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "not", "no", "so", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "may", "might", "must",
}

_FILLER_STARTS = (
    "i think",
    "i believe",
    "in conclusion",
    "overall",
    "to summarize",
    "thank you",
    "you're welcome",
    "certainly",
    "sure,",
    "absolutely",
    "here is",
    "here are",
    "here's",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z]{3,}")
_TRAILING_INITIAL = re.compile(r"\b[A-Z]\.$")


def extract_claims(response_content: str) -> list[str]:
    """Decompose a response into atomic, substantive claims.

    Claims are sentence-level units that contain at least one real word,
    filtered to drop filler frames such as "I think", "overall", and
    polite acknowledgements.

    Args:
        response_content: The model response text.

    Returns:
        List of atomic claims.
    """
    if not response_content or not response_content.strip():
        return []

    claims: list[str] = []
    parts = _SENTENCE_SPLIT.split(response_content.strip())
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and _TRAILING_INITIAL.search(merged[-1]):
            # "The owner is J." + "Alvarez." -> merge to respect initials.
            merged[-1] = f"{merged[-1].rstrip()} {part}"
            continue
        merged.append(part)

    for sentence in merged:
        if not _WORD.search(sentence):
            continue
        lowered = sentence.lower()
        if any(lowered.startswith(filler) for filler in _FILLER_STARTS):
            continue
        claims.append(sentence)
    return claims


def _tokens(text: str) -> set[str]:
    return {
        word.lower()
        for word in _WORD.findall(text or "")
        if word.lower() not in _STOPWORDS
    }


def claim_groundedness(
    claims: list[str],
    chunk_texts: list[str],
    entailment: EntailmentFn | None = None,
) -> dict[str, Any]:
    """Score each claim against retrieved evidence.

    A claim is grounded when the majority (>= 1/2) of its non-stopword
    tokens appear in at least one retrieved chunk. When an ``entailment``
    backend is supplied it is consulted first (probability-style score in
    [0, 1] with a 0.5 threshold), falling back to the lexical proxy when it
    returns a neutral/small value.

    Args:
        claims: Atomic claims from ``extract_claims``.
        chunk_texts: Text of the retrieved chunks.
        entailment: Optional ``(claim, chunk_text) -> [0, 1]`` scorer.

    Returns:
        Dictionary with per-claim verdicts and aggregate rates.
    """
    chunk_token_sets = [_tokens(text) for text in chunk_texts]

    per_claim: list[dict[str, Any]] = []
    supported = 0
    for claim in claims:
        best_ratio = 0.0
        entail_score = 0.0
        for chunk_text, chunk_tokens in zip(chunk_texts, chunk_token_sets, strict=True):
            if entailment is not None:
                entail_score = max(entail_score, entailment(claim, chunk_text))
            if not chunk_tokens or not _tokens(claim):
                continue
            intersection = len(_tokens(claim) & chunk_tokens)
            ratio = intersection / len(_tokens(claim))
            if ratio > best_ratio:
                best_ratio = ratio
        if entailment is not None:
            decision = entail_score >= 0.5 or best_ratio >= 0.5
        else:
            decision = best_ratio >= 0.5
        per_claim.append(
            {
                "claim": claim,
                "supported": decision,
                "evidence_token_overlap": round(best_ratio, 3),
                "entailment_score": round(entail_score, 3)
                if entailment is not None
                else None,
            }
        )
        if decision:
            supported += 1

    total = len(claims)
    return {
        "claims": per_claim,
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": total - supported,
        "unsupported_rate": (total - supported) / total if total else 0.0,
        "supported_rate": supported / total if total else 0.0,
    }


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(chunk)


def _contradiction_rate(retrieval: dict[str, Any] | None, response_content: str) -> float:
    """Fraction of flagged conflict chunks whose wording bled into the answer.

    Uses token overlap instead of contiguous phrase matching so reordered
    wording (e.g. "skip approval when standalone" vs "standalone skip
    approval") is still detected.
    """
    if not isinstance(retrieval, dict):
        return 0.0
    conflicts = retrieval.get("conflicts")
    if not isinstance(conflicts, list) or not conflicts:
        return 0.0

    response_tokens = _tokens(response_content)
    leaked = 0
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        conflict_tokens = _tokens(str(conflict.get("text", "")))
        if not conflict_tokens:
            continue
        overlap = len(conflict_tokens & response_tokens) / len(conflict_tokens)
        if overlap >= 0.5:
            leaked += 1
    return leaked / len(conflicts)


def faithfulness_risk(
    retrieval: dict[str, Any] | None,
    response_content: str,
    entailment: EntailmentFn | None = None,
) -> dict[str, Any]:
    """Compute the claim-level faithfulness vector from retrieval metadata.

    Unsupported claims have no token overlap against any retrieved chunk
    and are hallucination risk. ``chunk_usage_rate`` measures unused
    context (retrieved but never cited), and ``claim_contradiction_rate``
    measures contradictory context that bled into the answer.

    Args:
        retrieval: Retrieval metadata block.
        response_content: The model response text.
        entailment: Optional NLI-style claim/evidence scorer. When present,
            sentence-level verdicts prefer it and fall back to the lexical
            overlap proxy.

    Returns:
        Dictionary with fidelity scores and per-claim breakdown.
    """
    claims = extract_claims(response_content)

    chunk_texts: list[str] = []
    if isinstance(retrieval, dict):
        chunks = retrieval.get("chunks") or retrieval.get("top_k") or []
        if isinstance(chunks, list):
            chunk_texts = [_chunk_text(chunk) for chunk in chunks]

    if not claims or not chunk_texts:
        return {
            "faithfulness_score": 0.0,
            "chunk_usage_rate": 0.0,
            "claim_contradiction_rate": 0.0,
            "claims": [],
            "total_claims": len(claims),
            "unsupported_claims": 0,
        }

    grounded = claim_groundedness(claims, chunk_texts, entailment=entailment)

    used_chunk_indices = set()
    for claim_info in grounded["claims"]:
        claim_tokens = _tokens(str(claim_info.get("claim", "")))
        for index, chunk_tokens in enumerate([_tokens(text) for text in chunk_texts]):
            if claim_tokens and chunk_tokens & claim_tokens:
                used_chunk_indices.add(index)

    total_chunks = len(chunk_texts)
    unused_chunks = total_chunks - len(used_chunk_indices)

    return {
        "faithfulness_score": round(grounded["unsupported_rate"], 3),
        "chunk_usage_rate": round(unused_chunks / total_chunks, 3),
        "claim_contradiction_rate": round(
            _contradiction_rate(retrieval, response_content), 3
        ),
        "claims": grounded["claims"],
        "total_claims": grounded["total_claims"],
        "unsupported_claims": grounded["unsupported_claims"],
    }
