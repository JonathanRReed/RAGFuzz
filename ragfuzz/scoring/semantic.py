"""Provider-backed semantic scoring backends.

The pure lexical detectors in ``claiming.py`` and ``membership.py`` are the
offline, deterministic baseline. For real precision, SOTA faithfulness and
membership evidence want contextual judgment:

- ``ProviderEntailment`` turns a provider chat call into an NLI verdict
  (entail / neutral / contradict) for a claim/evidence pair, returning a
  probability-style score in ``[0, 1]``.
- ``ProviderMembershipEmbedder`` embeds a candidate against the response and
  scores cosine similarity as membership evidence.

Both backends take any ``Provider`` (a fake one in tests), so scoring stays
offline-capable while enabling model-backed precision when a provider is
reachable. Every backend is cancellation-safe and returns a plain float so it
drops cleanly into the existing ``EntailmentFn`` / membership call sites.
"""

from __future__ import annotations

import json
import math
import re

from ragfuzz.models import Message
from ragfuzz.providers.base import Provider

_NLI_PATTERN = re.compile(
    r"\{\s*\"verdict\"\s*:\s*\"(entail|neutral|contradict)\"[\s\S]*?"
    r"\"confidence\"\s*:\s*([0-9]*\.?[0-9]+)"
)

_NLI_SYSTEM = (
    "You are an NLI judge. Given a CLAIM and EVIDENCE, decide whether the "
    "evidence entails the claim (the claim follows from the evidence), "
    "contradicts it, or the evidence is neutral (unrelated or insufficient). "
    "Return ONLY JSON: {\"verdict\": \"entail\" | \"neutral\" | \"contradict\", "
    "\"confidence\": <float 0..1>}"
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in ``[0, 1]`` (0 for empty/zero vectors).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


class ProviderNLI:
    """Entailment backend backed by a provider chat call.

    Attributes:
        provider: The provider to call.
        model: Model identifier; falls back to ``provider.default_model``.
    """

    def __init__(self, provider: Provider, model: str | None = None):
        self.provider = provider
        self.model = model or provider.default_model
        if not self.model:
            raise ValueError("ProviderNLI requires an explicit model.")

    async def score(self, claim: str, evidence: str) -> float:
        """Return entailment probability of ``claim`` given ``evidence``.

        Returns:
            1.0-ish for ``entail`` scaled by confidence, 0.0 for ``neutral``
            and ``contradict``. Falls back to the lexical overlap proxy if
            the provider call fails or the verdict cannot be parsed.
        """
        if not claim.strip() or not evidence.strip():
            return 0.0
        payload = (
            f"CLAIM: {claim}\n\n"
            f"EVIDENCE: {evidence}\n\n"
            "Verdict and confidence in JSON only."
        )
        model = self.model
        assert model is not None
        try:
            response = await self.provider.chat(
                messages=[
                    Message(role="system", content=_NLI_SYSTEM),
                    Message(role="user", content=payload),
                ],
                model=model,
                temperature=0.0,
                max_tokens=64,
            )
            return self._parse_nli(response.content)
        except Exception:
            return _lexical_fallback(claim, evidence)

    def _parse_nli(self, content: str) -> float:
        """Parse an NLI verdict JSON blob into a confidence-weighted score."""
        match = _NLI_PATTERN.search(content)
        if match:
            verdict = match.group(1)
            confidence = min(max(float(match.group(2)), 0.0), 1.0)
            if verdict == "entail":
                return confidence
            return 0.0  # neutral and contradict both fail to support
        # Last-resort: scan raw JSON.
        try:
            data = json.loads(content)
            verdict = data.get("verdict")
            confidence = float(data.get("confidence", 1.0))
        except (json.JSONDecodeError, ValueError, AttributeError):
            return 0.0
        if verdict == "entail":
            return confidence
        return 0.0


def _lexical_fallback(claim: str, evidence: str) -> float:
    """Token-overlap entailment proxy (matches heuristics default)."""
    import re as _re

    words = set(_re.findall(r"[a-z]{3,}", claim.lower()))
    ev = set(_re.findall(r"[a-z]{3,}", evidence.lower()))
    if not words:
        return 0.0
    overlap = len(words & ev) / len(words)
    return 1.0 if overlap >= 0.5 else 0.0


class ProviderMembershipEmbedder:
    """Embedding-based membership scorer.

    Compares the candidate text and the response text in embedding space.
    Members (candidate present in the corpus) yield high cosine similarity to
    responses that reproduce them; paraphrased non-members yield lower scores.
    """

    def __init__(self, provider: Provider, model: str | None = None):
        self.provider = provider
        self.model = model or provider.default_model
        if not self.model:
            raise ValueError("ProviderMembershipEmbedder requires an explicit model.")

    async def score(self, candidate: str, response: str) -> float:
        """Return membership evidence in [0, 1] for ``candidate`` vs ``response``.

        Returns 0.0 on failure or empty inputs.
        """
        if not candidate.strip() or not response.strip():
            return 0.0
        model = self.model
        assert model is not None
        try:
            vectors = await self.provider.embed([candidate, response], model)
        except Exception:
            return 0.0
        if len(vectors) < 2:
            return 0.0
        sim = cosine_similarity(vectors[0], vectors[1])
        return round(sim, 3)
