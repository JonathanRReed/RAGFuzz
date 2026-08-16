# RAGFuzz Methodology, Evidence, and Citation

## Scope

RAGFuzz evaluates authorized chat and retrieval-augmented generation targets from a local workstation or private network. A result describes one concrete evaluation setup: the repository version, suite, target, provider, model, seed, command options, and available retrieval metadata.

The project is a security evaluation workspace. It does not claim that one score proves a system is safe, that its built-in corpora represent production traffic, or that results transfer unchanged across models and retrieval stacks.

## Reproducible Run Record

Keep these items with every run that may be reviewed or cited:

1. The tagged RAGFuzz version or full Git commit SHA.
2. The exact suite YAML, including any local changes.
3. The provider, model identifier, and endpoint type. Do not publish credentials.
4. The target authorization record and target-host policy used for the run.
5. The full command, including `--seed`, run count, concurrency, and whether semantic scoring was enabled.
6. The generated run directory, failure-case JSON files, and HTML, Markdown, or JSON reports.
7. The evidence-bundle `manifest.json`, artifact checksums, and redaction result before any handoff.

A minimal local workflow is:

```bash
uv sync --locked --all-extras --dev
uv run ragfuzz doctor
uv run ragfuzz target-check http://127.0.0.1:8000
uv run ragfuzz run suites/rag-canary-leak.yaml \
  --provider ollama \
  --runs 1 \
  --concurrency 1 \
  --seed 42 \
  --json-summary
uv run ragfuzz report runs/YOUR_RUN_ID --html --md --json
uv run ragfuzz evidence-bundle \
  --run-dir runs/YOUR_RUN_ID \
  --output-dir evidence
uv run ragfuzz redact-check evidence
```

## Evaluation Coverage

Suites define the run type, inputs, mutations, canary data, declared heuristics, and budget. Current coverage includes:

- prompt injection, jailbreak, leakage, and prompt extraction;
- retrieval conflict, poisoned knowledge, source-trust influence, and soft-ad behavior;
- faithfulness, claim grounding, citation grounding, and multi-hop evidence loss;
- context flooding and white denial-of-service behavior;
- corpus membership-inference evidence and adversarial retrieval hubness.

The repository includes starter suites under [`suites/`](../../suites/). They are starting points for a documented target, not universal test plans. Operators should adapt the corpus, seeds, expected behavior, and authorization boundary to the system being evaluated.

## Scoring and Interpretation

Suite-declared heuristic names are validated against the scoring registry. A case is scored only on the signals its suite declares.

Retrieval-conditioned metrics depend on a target returning a retrieval snapshot. When that metadata is absent, source trust, rank drift, conflict recovery, hubness, and related retrieval fields may be empty or zero by design. That absence should be reported as a coverage boundary, not interpreted as evidence that the target passed.

Lexical scoring is the local default. The optional `--semantic` path uses provider-backed entailment or embeddings where supported. Results can vary with the provider, model, corpus, target implementation, and nondeterministic generation behavior.

The built-in `ragfuzz benchmark` corpora measure detector self-consistency against synthetic and adversarial fixtures shipped with the repository. Those figures are useful regression evidence. They are not independent validation, a population estimate, or a claim about every production RAG system.

## Evidence Outputs

Normal CLI runs create durable artifacts under `runs/`. Depending on the command and suite, the evidence may include:

- run configuration and summary data;
- case identifiers, prompts, responses, mutations, traces, and retrieval snapshots;
- replayable failure-case JSON;
- HTML, Markdown, and JSON reports;
- evidence-bundle manifests and SHA-256 checksums;
- local operator audit events;
- redaction-scan output.

Review generated artifacts before sharing them. The redaction checks reduce obvious credential leakage, but they do not replace human review of private documents, customer data, or proprietary prompts.

## Demo Evidence Boundary

The FastAPI demo is an interview and walkthrough surface. Its provider readiness checks and provider sample can be real. Seeded findings and later stream content may be synthetic so the walkthrough can show different outcomes quickly.

Demo state is held in memory and clears when the process exits. Use the CLI workflow for durable evaluation records.

## Safety Boundary

Run RAGFuzz only against systems you own or have explicit permission to test. Loopback and private targets are the default. Public and link-local targets require explicit operator intent through `--allow-public-target`.

The local audit log records workstation evidence. It is not a centralized compliance-retention system. The demo is not designed for direct public-internet exposure, and hosted multi-tenant operation is outside the documented deployment model. See [`SECURITY.md`](../../SECURITY.md) for the maintained operator controls.

## Limitations to Report

A useful report should state at least these limitations when they apply:

- the model and provider version may change between runs;
- generation can remain nondeterministic even with a fixed RAGFuzz seed;
- retrieval metrics require target-supplied grey-box metadata;
- starter suites do not represent every production corpus or abuse pattern;
- semantic scoring depends on the selected provider and embedding or entailment behavior;
- synthetic demo evidence must remain distinct from provider-backed run evidence;
- public-target testing and destructive poisoning require explicit authorization;
- no single aggregate score establishes production safety.

The current production-readiness record is maintained in [`docs/audits/production-readiness-2026-08-06.md`](../audits/production-readiness-2026-08-06.md). The latest research-to-implementation map is [`ragfuzz-research-round-2026-08-06.md`](ragfuzz-research-round-2026-08-06.md).

## Citation

Repository citation metadata lives in [`CITATION.cff`](../../CITATION.cff). Cite a tagged release when one matches the source you used. For work based on an untagged repository state, include the full Git commit SHA.

A plain-text citation can use this shape:

```text
Reed, Jonathan R. RAGFuzz: A Local-First RAG Security Evaluation Workspace.
Version 0.4.0, GitHub repository, commit <full commit SHA>.
```

No DOI is declared in the citation metadata. Add an archival identifier only after a real archive record exists.

## Research Anchors

The implementation notes and starter suites draw on the sources linked in the repository research records, including RARE, SafeRAG, Ragas, AgentDojo, OWASP LLM guidance, and peer red-team tooling. Those sources motivate test and metric design. They do not endorse RAGFuzz or validate its results.
