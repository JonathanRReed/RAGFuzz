# RAGFuzz Research Round: Claim Belief, 2026-08

## Scope

Third research pass, focused on the two largest remaining RAG audit gaps identified
in the 2026-08 round: **answer-level faithfulness is only a vibe check**, and
**corpus membership inference is a first-class privacy attack surface**. It also
closes two process gaps surfaced by the metrics work: suite-declared heuristics now
actually gate what is measured, and runs are reproducible via `--seed`.

## Why This Round Matters

Retrieval-conditioned metrics already prove *what* was retrieved and from where;
they did not yet prove whether the *answer* faithfully restated it. 2026 RAG
faithfulness research is explicit that mean answer-level groundedness hides
per-claim hallucination. Separately, RAG-MIA work (MEntA, E-MIA) shows black-box
queries can determine whether a document is in a retrieval corpus — a privacy leak
RAGFuzz had no run type for. This round ships both.

## Key Sources And Capture

### Answer-level groundedness is a vibe check

- Thesis: decompose the answer into atomic claims, score each against retrieved
  evidence, and scan unused chunks for contradictions instead of averaging one
  score.
- Capture: `ragfuzz/scoring/claiming.py` implements a dependency-free, deterministic
  protocol: sentence-level atomic claims (filler-frame removal, initial-aware
  splitting), a lexical entailment proxy (majority non-stopword overlap against
  retrieved chunks), `chunk_usage_rate` (retrieved-but-never-cited context), and a
  conflict-chunk contradiction scan.
- Source: https://futureagi.com/ai-interview/2025/11/08/assuring-llm-output-faithfulness.html

### MEntA / E-MIA (RAG membership inference)

---

- MEntA is the practical 5-query surrogate-free corpus membership inference via
  entailment (~0.99 AUC); E-MIA's exam-styled questions pass guardrails nearly
  every time. LeakAngry shows query generation and instruction contributions
  combine.
- Capture: `ragfuzz/scoring/membership.py` scores distinctive-token overlap between
  a candidate document and a response, with member vs non-member separation
  aggregation. New run type `membership-inference` and
  `suites/rag-membership-inference.yaml`.
- Sources: MEntA (arXiv:2605.24312), E-MIA (arXiv:2605.00955), LeakDojo (2026).

### Cherry-pick override / judge honesty

- LLM judges commit directional verdicts even under mixed evidence (cherry-pick
  override).
- Capture: `JudgeResult.verdict` with `support/refute/conflict/no_commit`;
  `conflict` and `no_commit` never commit a directional score.
- Source: "Cherry-pick Override" (arXiv:2606.07834).

## What Shipped

- `ragfuzz/scoring/claiming.py`: `extract_claims`, `claim_groundedness`,
  `faithfulness_risk`.
- `ragfuzz/scoring/membership.py`: `membership_evidence_score`, `membership_summary`.
- New `ScoreVector` fields (`chunk_usage_rate`, `claim_contradiction_rate`,
  `membership_evidence_score`); `rag_risk_vector` and report keys extended.
- `HEURISTIC_FIELDS` registry in `HeuristicScorer`: suites declared under
  `scoring.heuristics` are validated (`validate_heuristics`) and actually gate
  which signals are computed; unknown names fail fast.
- `--seed` CLI option threaded into `_setup_mutators` and `SchedulerConfig`, and
  recorded in run metadata for deterministic replay.
- New suites: `rag-claim-grounded.yaml` (`faithfulness` run type) and
  `rag-membership-inference.yaml` (`membership-inference`). Run-type registry gains
  `membership-inference` and `prompt-leak`.
- Demo gains `faithfulness` and `membership-inference` scenarios.
- Release gate unchanged: `uv sync --locked --all-extras --dev`, `ruff`, `mypy`,
  `pytest` (153 passed).

## Open Lanes

- Local NLI (DeBERTa-style) as an optional entailment backend instead of the
  lexical proxy.
- Guardrail-drift suite: benign document injection shifting LLM-judge verdicts.
- Adversarial-hub detection for conspicuously central poison chunks.