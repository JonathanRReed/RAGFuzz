# RAGFuzz Research and Peer Audit, 2026-05-12

## Executive Takeaways

RAGFuzz is already aimed at the right category: local-first RAG security evaluation with fuzzing, scoring, replayable artifacts, and redacted reports. The strongest upgrade path is not a generic chatbot UI. It is deeper RAG-specific coverage, stronger scenario catalogs, benchmark-style robustness metrics, and clearer handoff evidence.

## Research Signals

### SafeRAG, ACL 2025

SafeRAG frames RAG security around manipulated external knowledge. It classifies attack tasks into silver noise, inter-context conflict, soft ads, and white denial-of-service, then tests many RAG components and finds broad vulnerability. RAGFuzz should map suites to these task families and make conflict, noisy context, and service-quality degradation visible in reports.

Source: https://aclanthology.org/2025.acl-long.230/

### RARE, arXiv 2506.00789

RARE focuses on retrieval-aware robustness under query perturbations, document perturbations, real retrieval changes, time-sensitive corpora, and multi-hop questions. RAGFuzz should add retrieval-conditioned metrics, not only final answer scoring. Useful fields include query perturbation type, document perturbation type, top-k drift, conflict recovery, and multi-hop degradation.

Source: https://arxiv.org/abs/2506.00789

### PoisonedRAG, arXiv 2402.07867

PoisonedRAG treats the RAG knowledge base as a practical attack surface where a few malicious texts can steer answers to attacker-chosen targets. RAGFuzz already has poison scoring and a poison mutator, but should expose source-trust, retrieval rank, poison influence, and corpus integrity checks as first-class report fields.

Source: https://arxiv.org/abs/2402.07867

### AgentDojo, arXiv 2406.13352

AgentDojo is relevant because many RAG systems become agents when retrieved content influences tools or workflow actions. It emphasizes dynamic environments, untrusted tool data, adaptive attacks, and realistic tasks. RAGFuzz should keep a future lane for agent or tool-aware RAG tests, especially excessive agency, tool misuse, and indirect injection.

Source: https://arxiv.org/abs/2406.13352

### OWASP LLM Top 10 2025

OWASP LLM01 covers direct and indirect prompt injection, including RAG-modified documents. OWASP LLM08 is explicitly about vector and embedding weaknesses, including unauthorized access, cross-context leakage, embedding inversion, data poisoning, and behavior alteration. RAGFuzz should add OWASP mapping to suites, run summaries, and demo scenarios.

Sources:
- https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/

## Peer Open Source Signals

### promptfoo

promptfoo is the broadest peer for developer-first LLM evals and red teaming. It has CLI, local execution, CI/CD integration, vulnerability reports, and model comparison. RAGFuzz can differentiate by staying Python and local-provider first, but should match its crisp setup, report links, and config-driven workflows.

Source: https://github.com/promptfoo/promptfoo

### garak

garak is a mature vulnerability scanner with a probe vocabulary for hallucination, leakage, prompt injection, misinformation, toxicity, jailbreaks, and more. RAGFuzz should treat probe discoverability as a product feature: named suites, visible attack families, and concise result explanations.

Source: https://github.com/NVIDIA/garak

### PyRIT

PyRIT is a Microsoft red-teaming framework for proactively identifying generative AI risks. It is especially strong as an extensible security professional toolkit. RAGFuzz should borrow the idea of reusable risk workflows and keep advanced attack orchestration modular.

Source: https://github.com/microsoft/PyRIT

### Giskard

Giskard is moving toward modular, async-first evals for agentic systems, with RAG quality, groundedness, conformity, LLM-as-judge, and multi-turn testing. RAGFuzz should make groundedness, context relevance, and conformity more explicit in report summaries.

Source: https://github.com/Giskard-AI/giskard

## Recommended Upgrade Backlog

1. Suite metadata now supports OWASP ids, research references, and risk tags. Keep new suites on that contract.
2. Add retrieval-conditioned robustness metrics inspired by RARE: query perturbation type, document perturbation type, top-k drift, conflict recovery, and multi-hop degradation.
3. Expand SafeRAG coverage beyond the starter retrieval-conflict suite into soft ads and white denial-of-service scenarios.
4. Add poison provenance fields: source trust, retrieval rank, injected document id, and poison influence.
5. Add exportable readiness bundles: environment, provider status, suite map, reports, and redaction proof.
6. Keep demo onboarding aligned with real suite metadata so reviewer-facing claims stay grounded in runnable artifacts.
