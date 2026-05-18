# RAGFuzz Interview Demo Script

This script is the shortest credible walkthrough for an interviewer or reviewer.

## Setup Check

```bash
uv sync --locked --all-extras --dev
uv run ragfuzz doctor
uv run ragfuzz readiness --evidence-dir evidence
uv run ragfuzz providers-doctor --provider ollama
```

What to point out:

- RAGFuzz is local-first and does not require a hosted service.
- Provider readiness shows real local model state.
- The readiness report records suite inventory, security controls, operator docs, and recommended demo flow.
- The local audit log records operator actions without sending data to a hosted service.

## Real CLI Smoke

```bash
uv run ragfuzz run suites/rag-canary-leak.yaml --provider ollama --runs 1 --concurrency 1 --json-summary
uv run ragfuzz evidence-bundle --run-dir runs/<run_id> --output-dir evidence
uv run ragfuzz redact-check evidence
```

What to point out:

- The CLI writes durable artifacts under `runs/`.
- Reports include redacted evidence and suite metadata.
- The evidence bundle gives IT or app owners a manifest, HTML, Markdown, JSON summary, and redaction proof.
- The suite maps to RAG security risks instead of generic prompt testing.

## Dashboard Walkthrough

```bash
uv run ragfuzz demo --host 127.0.0.1 --port 8765 --no-open
```

Open `http://127.0.0.1:8765`.

Walk through:

- Provider cards, including endpoint status, model count, API-key state, and selected model.
- Scenario selector for canary leakage, indirect prompt injection, retrieval conflict, and poisoning.
- Streaming run events with case ids, risk labels, OWASP mapping, scores, and report links.
- Recent reports in JSON, HTML, Markdown preview, and raw Markdown.

## Close

The strongest positioning is:

RAGFuzz turns RAG security review into a repeatable local workflow. It checks model/provider readiness, runs adversarial suites, preserves replayable evidence, and produces redacted reports that can be reviewed by engineers, security reviewers, and technical stakeholders.
