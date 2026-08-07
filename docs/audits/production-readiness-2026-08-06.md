# RAGFuzz Production Readiness Audit, 2026-08-06

## Current Status

RAGFuzz 0.4.0 is an interview-ready and local-client-evaluation-ready RAG security workspace when run through the documented uv workflow. It is not positioned as a hosted multi-tenant SaaS.

The enterprise target is local IT and security operation: offline-capable setup, private provider endpoints, target authorization checks, redacted evidence bundles, repeatable CLI gates, and now retrieval-conditioned RAG metrics.

## Verified Strengths

- Local provider workflow for Ollama, LM Studio, and vLLM with `default_model = "auto"` server-driven selection.
- Python 3.11+ baseline, stdlib TOML parsing, no numpy dependency, CI matrix on 3.12/3.13 plus a 3.11 compatibility job.
- Current release gate adds 210 passing tests (`uv run pytest`) covering retrieval-conditioned metrics, claim-level faithfulness, membership-inference evidence, adversarial hubness, prompt-extraction detection, the heuristic registry, judge verdicts, a synthetic+adversarial detection benchmark, and provider-backed semantic scoring.
- Retrieval-conditioned robustness metrics in `ragfuzz/scoring/rag_metrics.py`: poison provenance, source trust, top-k rank drift, conflict recovery, citation grounding, multi-hop evidence loss, and context-flood degradation, wired into `ScoreVector`, `Case.retrieval_snapshot`, report bundles, HTML/Markdown/JSON outputs, and the demo.
- Claim-level faithfulness (`ragfuzz/scoring/claiming.py`) and corpus membership-inference evidence (`ragfuzz/scoring/membership.py`) with a pluggable NLI entailment backend falling back to lexical overlap. Adversarial hubness detection (`ragfuzz/scoring/hubness.py`, off `Case.retrieval_snapshot`) and system-prompt extraction scoring (`ragfuzz/scoring/prompt_leak.py`) are exposed as `corpus-hubs` CLI and the `prompt_leak` heuristic. `faithfulness`, `membership-inference`, `prompt-leak`, and `poisoning` run types. The heuristic registry (`HEURISTIC_FIELDS`) validates and gates suite-declared `scoring.heuristics`, and `--seed` makes runs reproducible.
- Evidence bundles now attach SHA-256 checksums to every artifact so downstream redaction checks and handoffs can verify integrity, not just presence.
- Measured detection quality, not just proxy scores: `ragfuzz/scoring/benchmark.py` ships a synthetic corpus and a paraphrase-hard adversarial corpus with ground-truth recovery and precision/recall/F1 per detector; `ragfuzz benchmark` reports macro F1 (currently 1.0 on both corpora) as an honest self-consistency result. A `--semantic` run flag upgrades heuristic scoring with provider-backed NLI entailment and embedding-based membership scoring (`ragfuzz/scoring/semantic.py`, `embed()` on providers), keeping the lexical proxy as default. Hub statistics now also expose `top_rank_rate`, `positional_concentration`, and `median_rank` beyond the MAD z-score.
- Research-backed starter suites for canary leakage, indirect prompt injection, retrieval conflict, poisoned knowledge, white-DoS context flood, multi-hop evidence chains, claim-level groundedness, corpus membership inference, system-prompt extraction, and adversarial hub detection.
- FastAPI dashboard with onboarding, provider checks, model selection, streaming demo runs, and report drilldowns.
- CLI artifacts under `runs/` with HTML, Markdown, and JSON report generation; secret redaction in reports and safer rendered report links.
- Baseline storage that prevents path traversal and sanitized-name collisions.
- Default local-first outbound HTTP behavior with enterprise proxy opt-in.
- Local operator commands for health checks, target policy validation, redaction scanning, and evidence bundle generation.
- Append-only local audit events for operator commands, runs, reports, target URLs, run ids, report paths, and poison cleanup status.
- Repo-level security policy, interview demo script, and client install handoff.

## Current Release Gate

Run these before showing the project:

```bash
uv sync --locked --all-extras --dev
uv run ruff check .
uv run mypy ragfuzz
uv run pytest
uv run ragfuzz doctor
uv run ragfuzz target-check http://127.0.0.1:8000
uv run ragfuzz readiness --evidence-dir evidence
uv run ragfuzz demo --host 127.0.0.1 --port 8765 --no-open
```

With a local model provider available, add:

```bash
uv run ragfuzz providers-doctor --provider ollama
uv run ragfuzz run suites/rag-canary-leak.yaml --provider ollama --runs 1 --concurrency 1 --json-summary
uv run ragfuzz corpus-hubs runs/<run_id>
uv run ragfuzz benchmark --json
uv run ragfuzz evidence-bundle --run-dir runs/<run_id> --output-dir evidence
uv run ragfuzz redact-check evidence
```

## Remaining Risks

- Demo stream evidence is partly synthetic after the provider sample, by design for fast walkthroughs. The UI and docs should keep that distinction clear.
- Soft-ad run type and suite seeds are defined and consistent, but there is no dedicated CI smoke for the soft-ad suite yet.
- Hub statistics depend on grey-box retrieval snapshots (`Case.retrieval_snapshot`); pure chat targets with no metadata produce empty hub reports by design.
- RAG metric scores depend on grey-box retrieval metadata from targets that expose it (JR AutoRAG style). Pure chat targets score 0 for retrieval metrics by design; the report already shows this.
- Public or link-local target testing requires explicit operator intent and should stay off by default.
- Packaged binary or container release artifacts, checksums, and signed releases remain future distribution work.
- The local audit log is suitable for workstation evidence, not centralized compliance retention.
- A hosted or multi-tenant SaaS deployment is out of scope for this local enterprise utility.

## Handoff Positioning

The best interview framing is that RAGFuzz is a focused local security evaluation product, not a generic chatbot wrapper. It shows concrete judgment in provider readiness, adversarial suite design, retrieval-conditioned scoring, redacted evidence, repeatable reports, and operator safety.
