# RAGFuzz Quickstart Guide

Get a real local RAGFuzz run working first, then use the demo dashboard for walkthroughs.

## Prerequisites

- Python 3.9+
- Ollama, LM Studio, or a local/private vLLM OpenAI-compatible server
- At least one local chat model installed

## Installation

```bash
git clone https://github.com/your-org/ragfuzz.git
cd ragfuzz
pip install -e ".[dev]"
```

## Quick Setup With Ollama

Start with Ollama because the generated config is local-first and defaults to Ollama.

```bash
ollama list
ragfuzz init
```

The generated `ragfuzz.toml` uses:

```toml
[providers.ollama]
base_url = "http://localhost:11434/v1"
default_model = "auto"

[budget]
default_provider = "ollama"
default_target = "chat"
```

`default_model = "auto"` selects the first non-embedding model returned by the provider. Set it to an exact model id when you want deterministic runs.

## Verify The Real Product Path

```bash
ragfuzz providers-doctor --provider ollama
ragfuzz models-ls --provider ollama
ragfuzz run suites/rag-canary-leak.yaml --provider ollama --runs 1 --concurrency 1 --json-summary
```

The run command writes durable artifacts under `runs/`. It is not demo-only.

Generate reports for the run directory printed by the command.

```bash
ragfuzz report runs/<run_id> --html --md --json
```

## Demo Dashboard

```bash
ragfuzz demo
```

Open `http://127.0.0.1:8765`.

The dashboard is for product explanation and recruiter or client walkthroughs. It still checks real local providers and lists real installed models, but its demo runs are in memory and clear when the app closes.

Use the dashboard to:

- Confirm which providers are reachable.
- Pick an installed model from a ready provider.
- Change scenario, case count, and injected findings.
- Stream a demo run with visible provider, mutation, scoring, and report stages.
- Open JSON, styled HTML, formatted Markdown preview, or raw Markdown report output.

## Other Local Providers

LM Studio:

```toml
[providers.lmstudio]
base_url = "http://localhost:1234/v1"
default_model = "auto"
```

Enable the local OpenAI-compatible server in LM Studio before running checks.

vLLM:

```bash
vllm serve MODEL_NAME --host 127.0.0.1 --port 8000
```

```toml
[providers.vllm]
base_url = "http://localhost:8000/v1"
default_model = "auto"
```

Local Ollama, LM Studio, and vLLM do not require API keys unless you start those servers with authentication enabled.

## Command Reference

| Command | Description |
| --- | --- |
| `ragfuzz init` | Create default local-first configuration |
| `ragfuzz demo` | Launch the local demo dashboard |
| `ragfuzz providers-ls` | List configured providers |
| `ragfuzz providers-doctor` | Check provider health |
| `ragfuzz models-ls` | List available provider models |
| `ragfuzz run <suite>` | Run a real test suite |
| `ragfuzz report <run>` | Generate reports |
| `ragfuzz replay <case>` | Replay a failure case |
| `ragfuzz baseline-save` | Save a regression baseline |
| `ragfuzz baseline-check` | Compare against a saved baseline |

## Troubleshooting

Provider will not connect:

```bash
ragfuzz providers-doctor --provider ollama
```

No models appear:

```bash
ragfuzz models-ls --provider ollama
```

Runs are slow:

- Use `--runs 1 --concurrency 1` for smoke tests.
- Use a smaller local model.
- Keep caching enabled unless debugging.

Out of memory:

- Reduce concurrency.
- Use a quantized model.
- Lower the VRAM threshold in `ragfuzz.toml`.

For the complete product guide, see [README.md](README.md).
