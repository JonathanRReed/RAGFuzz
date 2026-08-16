# Research and Citation

RAGFuzz keeps its research claims, implementation notes, limitations, and citation metadata in the repository instead of treating the README as a benchmark paper.

## Start Here

- [Methodology, evidence, and citation](docs/research/METHODOLOGY.md)
- [Citation metadata](CITATION.cff)
- [Current production-readiness audit](docs/audits/production-readiness-2026-08-06.md)
- [August 2026 research-to-implementation record](docs/research/ragfuzz-research-round-2026-08-06.md)
- [Research and peer-tool audit](docs/research/ragfuzz-research-and-peer-audit-2026-05-12.md)
- [Security policy and operator boundary](SECURITY.md)

## Evidence Available From a Run

A normal CLI run can preserve the suite, seed, target policy, provider and model identifiers, replayable failure JSON, retrieval snapshots when the target exposes them, and HTML, Markdown, and JSON reports. `ragfuzz evidence-bundle` adds a manifest and SHA-256 checksums; `ragfuzz redact-check` scans the handoff directory for obvious unredacted secrets.

The exact contents depend on the target and suite. Retrieval-conditioned fields require a target-supplied retrieval snapshot. Missing grey-box metadata is a coverage boundary, not a passing result.

## Citation Rule

Use the repository's `CITATION.cff`. Cite a tagged release when it matches the source used. Include the full Git commit SHA for an untagged state.

The repository does not declare a DOI. Add one only after a real archival record exists.

## Claim Boundary

Built-in benchmark figures describe the synthetic and adversarial fixtures shipped with RAGFuzz. They are regression evidence for the detectors. They are not independent validation or a population estimate for production RAG systems.

The FastAPI demo may use seeded synthetic findings after a real provider sample. Durable evaluation evidence comes from the CLI workflow and saved run artifacts.
