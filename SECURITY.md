# Security Policy

## Supported Use

RAGFuzz is a local-first RAG security evaluation workspace. The supported deployment model is a developer or evaluator running the CLI and demo on a trusted workstation or private network.

Do not expose the demo app directly to the public internet. It is designed for local walkthroughs, provider checks, and report review.

## Reporting Issues

Report security issues privately to the maintainer before public disclosure. Include:

- The affected command, endpoint, or report path.
- A minimal reproduction.
- Expected impact and any target preconditions.
- Whether secrets, private documents, or public targets were involved.

Do not include live credentials, API keys, cookies, or private customer data in reports.

## Security Guarantees

Current local-first controls include:

- JR AutoRAG target checks and poison-mode runs default to loopback or private hosts.
- Public or link-local targets require the explicit `--allow-public-target` option.
- `ragfuzz target-check` validates target URLs before testing and supports explicit enterprise host allowlists.
- HTTP proxy environment inheritance is disabled by default and can be enabled with `RAGFUZZ_HTTP_TRUST_ENV=true` for enterprise networks.
- Reports redact obvious secrets and drop unsafe report links.
- `ragfuzz redact-check` scans local artifacts for obvious unredacted secrets before handoff.
- `ragfuzz evidence-bundle` collects readiness evidence, report artifacts, manifest data, and redaction proof in a local directory.
- Operator actions append JSONL events to a local audit log under the configured cache directory.
- Baseline files are namespaced so path-like suite ids cannot escape the baseline directory or collide with similarly sanitized ids.
- Demo API responses include no-store cache headers and browser security headers.

## Operator Responsibilities

- Keep `uv.lock` committed and use `uv sync --locked --all-extras --dev` for reproducible setup.
- Run `uv run ragfuzz doctor` or `uv run ragfuzz readiness` before demos or handoff.
- Run `uv run ragfuzz target-check TARGET_URL` before testing a RAG service.
- Run `uv run ragfuzz evidence-bundle --run-dir runs/<run_id> --output-dir evidence` before handing results to another team.
- Run `uv run ragfuzz redact-check evidence` before moving artifacts outside the workstation.
- Review generated reports before sharing them outside the local environment.
- Use environment variables for provider credentials.
- Avoid running destructive poison-mode tests against systems you do not own or have explicit permission to test.
