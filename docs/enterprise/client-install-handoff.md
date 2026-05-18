# Client Install Handoff

RAGFuzz is ready for local client evaluation when the operator can produce a fresh readiness report, run one provider smoke test, and open at least one generated report.

## Install

```bash
uv sync --locked --all-extras --dev
uv run ragfuzz init
```

Edit `ragfuzz.toml` only for local provider endpoints, run and cache directories, and optional local API-key environment variable names.

## Enterprise Network Option

RAGFuzz defaults to proxy-free local HTTP so local Ollama, LM Studio, vLLM, and private target checks are not accidentally routed through shell proxy settings.

For corporate networks that require proxy environment variables:

```bash
RAGFUZZ_HTTP_TRUST_ENV=true uv run ragfuzz providers-doctor
```

## Handoff Evidence

Generate a local evidence bundle:

```bash
uv run ragfuzz doctor
uv run ragfuzz target-check http://127.0.0.1:8000
uv run ragfuzz readiness --evidence-dir evidence
uv run ragfuzz providers-doctor --provider ollama
uv run ragfuzz run suites/rag-canary-leak.yaml --provider ollama --runs 1 --concurrency 1 --json-summary
uv run ragfuzz evidence-bundle --run-dir runs/<run_id> --output-dir evidence
uv run ragfuzz redact-check evidence
```

The evidence directory should include:

- Config load status.
- uv lock presence.
- Suite catalog inventory.
- Security policy and handoff docs.
- Provider reachability unless intentionally skipped.
- Security posture controls.
- Recommended demo flow.
- HTML, Markdown, and JSON report artifacts when a run directory is supplied.
- `redact-check.json` and `manifest.json`.

## Safety Boundaries

- Keep the demo on `127.0.0.1` unless the network is trusted.
- Do not run poison-mode tests against systems without explicit permission.
- Use `--allow-public-target` only for intentional public or link-local targets.
- Use `--allowed-host` for approved internal enterprise targets that are not loopback or private IPs.
- Review reports before sharing externally.
- Store provider credentials in environment variables, not in suite files or reports.

## Local Audit Log

Operator actions append JSONL events to the configured cache directory, usually under `.cache/`. The audit log records:

- Local username when available.
- Command action and status.
- Target URL, suite, run id, and report output paths when applicable.
- Poison cleanup status when poison-mode tests are used.

The audit log is local evidence for IT review. It is not a hosted compliance system.

## Acceptance Gate

A client install is acceptable for evaluation when these pass:

```bash
uv run ruff check .
uv run mypy ragfuzz
uv run pytest
uv run ragfuzz doctor
uv run ragfuzz readiness --evidence-dir evidence
uv run ragfuzz redact-check evidence
```

For a live demo, also run the FastAPI dashboard and verify `/`, `/api/status`, `/api/runs/demo/stream`, `/api/runs/recent`, and one generated report.
