# RAGFuzz Research Round, 2026-08

## Scope

Second research pass focused on retrieval-conditioned scoring and RAG red-teaming structure. The first round (2026-05-12) set the category direction. This round turns the recommended backlog items 2 and 3 into shipped behavior:

- retrieval-conditioned robustness metrics (RARE-Met style)
- SafeRAG white denial-of-service and soft-ad coverage
- poison provenance and source-trust first-class report fields

## Why This Round Matters

Most RAG security tooling still scores only the final answer. The failure modes that actually break RAG systems live upstream: what got retrieved, from which source, in what rank, and how the generator reacted to conflicting or flooded context. This round moves those signals into `ScoreVector`, the report bundle, the HTML/Markdown/JSON report renderers, the demo, and the starter suites, so an auditor can prove *why* an answer is unsafe, not just that it is.

## Key Sources And Capture

### RARE: Retrieval-Aware Robustness Evaluation, arXiv 2506.00789

- RARE-Met provides retrieval-conditioned robustness metrics under query perturbation, document perturbation, real retrieval changes, time-sensitive corpora, and multi-hop questions.
- Key metric capture in `ragfuzz/scoring/rag_metrics.py`:
  - poison provenance (injected document ids, top poison rank)
  - source trust fraction
  - top-k rank drift (baseline vs current retrieval)
  - conflict recovery (did untrusted conflict text bleed into the answer)
  - citation grounding (fabricated vs grounded citations)
  - multi-hop evidence loss (confident answer under missing evidence is a hallucination risk)
  - context-flood degradation (empty/truncated answer under padding)
- Source: https://arxiv.org/abs/2506.00789

### RAG Red Teaming

- Review of prompt artifact-driven RAG red-teaming flow: indirect prompt injection planted in the knowledge base, `pii`/RBAC checks, and vulnerability report output.
- Consequence captured: a threat vector family separated from the attack technique, matching the existing suite metadata (OWASP, research, risk tags) contract.
- Source: https://www.promptfoo.dev/blog/red-teaming-rag/

### SafeRAG, ACL 2025

- Task families: silver noise, inter-context conflict, soft ads, white denial-of-service.
- Capture in this round:
  - new run types `dos` (white DoS / context flood) and `soft-ad`
  - `suites/rag-dos-flood.yaml` with silver-noise and padding-resistance seeds
  - `HEAT` degradation signal in the `dos_degradation_score` column and demo scenario `dos`
- Source: https://aclanthology.org/2025.acl-long.230/

### PoisonedIR Cut, arXiv:2402.07867

- Treats the knowledge base as a practical attack surface: a few malicious texts can steer answers.
- Capture in this round: consistent `poison`, paging, `run_id` tags on chunks, energy refactors, poison provenance helper sharing a single scoping rule with the scheduler, and poisoned-fraction report fields.
- Source: https://arxiv.org/abs/2402.07867

## What Shipped

- `ragfuzz/scoring/rag_metrics.py`: pure-function metric library, fully unit-tested without a live provider.
- Six new `ScoreVector` fields (source trust, rank drift, conflict recovery, citation grounding, multi-hop, DoS degradation) plus a `rag_risk_vector` reducer used for report severity and primary risk labeling.
- `HeuristicScorer` now wires grey-box retrieval metadata from the target response into the metric fields; the scheduler carries the retrieval snapshot onto every `Case`.
- Run-type registry extended with `dos`, `soft-ad`, `multi-hop`.
- Thirty five new suites-level smoke gates; `tests/test_rag_metrics.py` adds 21 tests.
- Report bundle, HTML and Markdown renderers now surface `avg_rag_risk`, per-case RAG risk and primary risk label, and a primary-risk category fallback.
- Demo adds `dos` and `multi-hop` scenarios and synthetic RAG metric scores in the seeded runs.
- Judge, stateful-dialogue, and LLM-guided attacker workers now resolve a real default model from the provider instead of a fake `gpt-4`/provider-id string.
- The provider model resolution no longer hardcodes `local-model` defaults: `default_model = "auto"` means "ask the server".
- Project modernized: Python 3.11+ baseline, stdlib `tomllib`, pure-Python corpus math (numpy removed), CI matrix widened to 3.12/3.13 (+3.11 compatibility job).
- `Response.usage` widened to accept provider floats; provider and mutator typing hardened (47 files now verified by mypy).

## Open Lanes

- Agent/tool-aware RAG tests (AgentDojo-style excessive agency) remain a future lane.
- Packaged binary or container release artifacts with checksums and signed releases remain future distribution work.
- Soft-ad suite seeds and demo scenario are harmonized with the other suites but not separately CI-gated.

## Sources

- https://arxiv.org/abs/2506.00789 (RARE)
- https://aclanthology.org/2025.acl-long.230/ (SafeRAG)
- https://arxiv.org/abs/2402.07867 (PoisonedRAG)
- https://arxiv.org/abs/2406.13352 (AgentDojo)
- https://www.promptfoo.dev/blog/red-teaming-rag/ (prompt red-teaming guide)
- https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/ (OWASP LLM08)