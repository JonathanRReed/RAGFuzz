# RAGFuzz Product Context

## Register

product

## Product Purpose

RAGFuzz is a local-first RAG security evaluation workspace. It helps technical evaluators fuzz chat and retrieval systems, score failures, stream live evidence, and produce redacted reports that can be used in demos, client reviews, and regression workflows.

## Users

- AI engineers validating RAG behavior before release.
- Security reviewers checking prompt injection, leakage, poisoning, and retrieval robustness.
- Project managers or technical stakeholders who need a fast demo of risk, evidence, and next actions.
- Local-first teams that cannot send sensitive prompts or documents to a SaaS evaluator.

## Aha Moment

The user should see a local provider check, choose a risk scenario, stream a run, and open a redacted report in under five minutes. The product proves that RAG security can be evaluated as a repeatable workflow, not a manual prompt-testing exercise.

## Strategic Principles

- Local first: normal demo setup should work with Ollama, LM Studio, or vLLM without cloud credentials.
- Evidence over claims: every demo run should expose case ids, scores, risk labels, and report links.
- Fast onboarding: a reviewer should understand the workflow without reading the README first.
- Research-backed scope: roadmap claims should map to papers, benchmarks, OWASP risks, and peer tools.
- Safe by default: demo state is in memory, secrets are redacted, API responses are no-store, and local provider checks do not require persistent credentials.

## Voice

Direct, technical, and reviewer-friendly. Avoid hype. Make the workflow feel credible to engineers and legible to non-specialists.
