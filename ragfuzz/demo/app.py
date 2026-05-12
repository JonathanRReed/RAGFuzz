"""FastAPI demo app for ragfuzz."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from ragfuzz.demo.state import DemoState, get_demo_scenario, get_demo_scenarios
from ragfuzz.models import Message
from ragfuzz.providers.openai_compat import OpenAICompatProvider
from ragfuzz.reports import render_html_report, render_markdown_report

STATIC_DIR = Path(__file__).resolve().parent / "static"

def _badge_class(status: str) -> str:
    if status in {"healthy", "completed", "ready"}:
        return "good"
    if status in {"checking", "needs_api_key", "no_models", "running"}:
        return "warn"
    return "bad"


def _render_provider_cards(providers: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for provider in providers:
        status = str(provider.get("status", "unknown"))
        cards.append(
            "<article class=\"card provider-card\">"
            "<div class=\"card-header\">"
            "<div>"
            f"<p class=\"card-kicker\">Provider</p><h3>{escape(str(provider.get('provider_id', 'unknown')))}</h3>"
            "</div>"
            f"<span class=\"badge status-{_badge_class(status)}\">{escape(status)}</span>"
            "</div>"
            f"<p class=\"card-description\">{escape(str(provider.get('note', '')))}</p>"
            "<dl class=\"metric-list\">"
            f"<div><dt>Base URL</dt><dd>{escape(str(provider.get('base_url', '')))}</dd></div>"
            f"<div><dt>Default model</dt><dd>{escape(str(provider.get('default_model', '')))}</dd></div>"
            f"<div><dt>Models found</dt><dd>{int(provider.get('models_available', 0) or 0)}</dd></div>"
            f"<div><dt>Latency</dt><dd>{escape(_format_latency(provider.get('latency_ms')))}</dd></div>"
            f"<div><dt>API key</dt><dd>{escape(_format_key_status(provider))}</dd></div>"
            f"<div><dt>Streaming</dt><dd>{'yes' if provider.get('supports_streaming') else 'no'}</dd></div>"
            "</dl>"
            f"{_render_model_picker(provider)}"
            "</article>"
        )
    return "".join(cards)


def _format_latency(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{int(value)} ms"
    return "not checked"


def _format_key_status(provider: dict[str, Any]) -> str:
    if provider.get("api_key_set"):
        return "set"
    if provider.get("provider_id") in {"lmstudio", "ollama", "vllm"}:
        return "not required locally"
    return "missing"


def _render_model_picker(provider: dict[str, Any]) -> str:
    models = provider.get("models") or []
    if not models:
        return ""
    selected_model = str(provider.get("default_model") or provider.get("selected_model") or "")
    options = "".join(
        f"<option value=\"{escape(str(model))}\" {'selected' if model == selected_model else ''}>{escape(str(model))}</option>"
        for model in models
    )
    return (
        "<label class=\"model-picker\">"
        "<span>Demo model</span>"
        f"<select data-provider-id=\"{escape(str(provider.get('provider_id', '')))}\">{options}</select>"
        "</label>"
    )


def _render_recent_run_rows(runs: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for run in runs:
        summary = run.get("summary") or {}
        report_links = run.get("report_links") or {}
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(run.get('run_id', 'unknown')))}</code></td>"
            f"<td>{escape(str(run.get('suite_name', 'unknown')))}</td>"
            f"<td><span class=\"badge status-{_badge_class(str(run.get('status', 'unknown')))}\">{escape(str(run.get('status', 'unknown')))}</span></td>"
            f"<td>{int(run.get('progress', 0) or 0)}%</td>"
            f"<td>{int(summary.get('failure_count', 0) or 0)}</td>"
            f"<td>{summary.get('success_rate', 0.0)}%</td>"
            f"<td class=\"table-actions\">"
            f"<a class=\"ghost-link\" href=\"{escape(str(report_links.get('json', '#')))}\">data</a>"
            f"<a class=\"ghost-link\" href=\"{escape(str(report_links.get('html', '#')))}\">html</a>"
            f"<a class=\"ghost-link\" href=\"{escape(str(report_links.get('markdown', '#')))}\">md</a>"
            f"<a class=\"ghost-link\" href=\"{escape(str(report_links.get('raw_markdown', '#')))}\">raw</a>"
            "</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_scenario_options() -> str:
    options: list[str] = []
    for scenario in get_demo_scenarios():
        options.append(
            "<option "
            f"value=\"{escape(scenario['id'])}\" "
            f"data-objective=\"{escape(scenario['objective'])}\" "
            f"data-technique=\"{escape(scenario['technique'])}\" "
            f"data-owasp=\"{escape(scenario['owasp'])}\" "
            f"data-risk=\"{escape(scenario['risk'])}\">"
            f"{escape(scenario['label'])}"
            "</option>"
        )
    return "".join(options)


def _format_provider_sample_response(content: str) -> str:
    cleaned = content.replace("<|think|>", "").strip()
    return cleaned[:240] if cleaned else "Provider accepted the sample prompt."


def _render_markdown_preview(markdown: str, run_id: str) -> str:
    """Render Markdown as a readable local preview while preserving raw access."""
    body_html = _markdown_to_preview_html(markdown)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ragfuzz markdown report - {escape(run_id)}</title>
    <style>
        :root {{
            color-scheme: dark;
            --background: #000000;
            --foreground: #ffffff;
            --card: #0b0b0b;
            --border: #494949;
            --muted: #7c7a7a;
            --primary: #ff5d73;
            --radius: 8px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: var(--background);
            color: var(--foreground);
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        main {{
            width: min(1120px, calc(100% - 32px));
            margin: 0 auto;
            padding: 28px 0 48px;
        }}
        header {{
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 16px;
            padding: 20px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: linear-gradient(135deg, rgba(255, 93, 115, 0.18), rgba(0, 0, 0, 0.92));
        }}
        h1 {{
            margin: 0;
            font-size: clamp(28px, 5vw, 48px);
            line-height: 1;
            letter-spacing: 0;
        }}
        a {{
            color: var(--background);
            background: var(--primary);
            border-radius: var(--radius);
            padding: 10px 12px;
            font-weight: 800;
            text-decoration: none;
        }}
        .md-content {{
            display: grid;
            gap: 12px;
            margin-top: 14px;
        }}
        .md-panel {{
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--card);
            padding: 18px;
        }}
        .md-panel h2 {{
            margin: 0 0 12px;
            color: var(--primary);
            font-size: 18px;
        }}
        .md-panel h3 {{
            margin: 14px 0 10px;
            font-size: 15px;
        }}
        .md-panel p,
        .md-panel li {{
            line-height: 1.65;
        }}
        .md-panel ul {{
            margin: 0;
            padding-left: 18px;
        }}
        .md-panel table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
        }}
        .md-panel th,
        .md-panel td {{
            padding: 10px;
            border-bottom: 1px solid var(--border);
            text-align: left;
            vertical-align: top;
        }}
        .md-panel th {{
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .md-panel code {{
            padding: 2px 5px;
            border-radius: 5px;
            background: rgba(73, 73, 73, 0.42);
            color: var(--foreground);
        }}
        pre {{
            margin: 0;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: #000000;
            white-space: pre-wrap;
            word-break: break-word;
            color: var(--foreground);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            line-height: 1.65;
        }}
    </style>
</head>
<body>
    <main>
        <header>
            <div>
                <p>Markdown preview</p>
                <h1>RAGFuzz report</h1>
            </div>
            <a href="/api/reports/{escape(run_id)}/md/raw">Raw Markdown</a>
        </header>
        <div class="md-content">{body_html}</div>
    </main>
</body>
</html>
"""


def _inline_markdown(value: str) -> str:
    text = escape(value)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"&lt;(https?://[^&]+)&gt;", r'<a href="\1" rel="noreferrer">\1</a>', text)
    return text


def _render_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    head_html = "".join(f"<th>{_inline_markdown(cell)}</th>" for cell in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_inline_markdown(cell)}</td>" for cell in row) + "</tr>"
        for row in body
    )
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _markdown_to_preview_html(markdown: str) -> str:
    blocks: list[str] = []
    lines = markdown.splitlines()
    index = 0
    open_panel = False

    def ensure_panel() -> None:
        nonlocal open_panel
        if not open_panel:
            blocks.append('<section class="md-panel">')
            open_panel = True

    def close_panel() -> None:
        nonlocal open_panel
        if open_panel:
            blocks.append("</section>")
            open_panel = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("# "):
            index += 1
            continue
        if stripped.startswith("## "):
            close_panel()
            blocks.append('<section class="md-panel">')
            blocks.append(f"<h2>{_inline_markdown(stripped[3:])}</h2>")
            open_panel = True
            index += 1
            continue
        if stripped.startswith("### "):
            ensure_panel()
            blocks.append(f"<h3>{_inline_markdown(stripped[4:])}</h3>")
            index += 1
            continue
        if stripped.startswith("```"):
            ensure_panel()
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            blocks.append(f"<pre>{escape(chr(10).join(code_lines))}</pre>")
            index += 1
            continue
        if stripped.startswith("|"):
            ensure_panel()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            blocks.append(_render_table(table_lines))
            continue
        if stripped.startswith("- "):
            ensure_panel()
            items = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(lines[index].strip()[2:])
                index += 1
            blocks.append("<ul>" + "".join(f"<li>{_inline_markdown(item)}</li>" for item in items) + "</ul>")
            continue
        ensure_panel()
        blocks.append(f"<p>{_inline_markdown(stripped)}</p>")
        index += 1

    close_panel()
    return "".join(blocks)


def _active_provider(status_data: dict[str, Any]) -> dict[str, Any]:
    raw_providers = status_data.get("providers") or []
    providers = [
        provider for provider in raw_providers if isinstance(provider, dict)
    ]
    for provider in providers:
        if provider.get("status") == "ready":
            return provider
    return providers[0] if providers else {}


def _render_dashboard_page(status_data: dict[str, Any], recent_runs: list[dict[str, Any]]) -> str:
    provider_cards = _render_provider_cards(status_data.get("providers") or [])
    recent_run_rows = _render_recent_run_rows(recent_runs)
    scenario_options = _render_scenario_options()
    active_run = status_data.get("active_run") or {}
    active_provider = _active_provider(status_data)
    ready = "ready" if status_data.get("ready") else "not-ready"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ragfuzz demo</title>
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%23211d1c'/%3E%3Cpath d='M15 18h21c8 0 13 5 13 12 0 5-3 9-8 11l9 12H38l-8-11h-5v11H15V18zm10 9v7h10c3 0 5-1 5-4s-2-3-5-3H25z' fill='%23ff6f83'/%3E%3C/svg%3E">
    <link rel="stylesheet" href="/static/dashboard.css">
    <script defer src="/static/dashboard.js"></script>
</head>
<body class="{ready}">
    <main class="page">
        <header class="app-bar">
            <div class="brand-mark">RF</div>
            <div>
                <p class="brand-title">RAGFuzz</p>
                <p class="brand-subtitle">RAG security evaluation workspace</p>
            </div>
            <nav class="top-nav" aria-label="Primary">
                <a href="#onboarding">Onboarding</a>
                <a href="#providers-section">Providers</a>
                <a href="#stream">Stream</a>
                <a href="#runs">Reports</a>
            </nav>
        </header>

        <section class="hero">
            <div class="hero-copy">
                <p class="eyebrow">local demo cockpit</p>
                <h1>Audit RAG systems with live fuzzing, leakage checks, and shareable reports.</h1>
                <p class="lede">A local-first walkthrough that checks real LM Studio, Ollama, and vLLM endpoints, lets you choose an installed model, runs a sample fuzz session, streams each evaluation stage, and opens redacted evidence.</p>
            </div>
            <div class="hero-actions">
                <button id="start-run" class="primary-button" type="button">Start demo run</button>
                <div class="status-badge status-{ready}">Demo state: {escape(str(status_data.get('mode', 'demo')))}</div>
            </div>
        </section>

        <section id="status-strip" class="status-strip">
            <div class="status-tile">
                <span>Ready providers</span>
                <strong>{int(status_data.get('setup', {}).get('ready_providers', 0) or 0)}</strong>
            </div>
            <div class="status-tile">
                <span>Active provider</span>
                <strong>{escape(str(active_provider.get('provider_id', 'none')))}</strong>
            </div>
            <div class="status-tile">
                <span>Active model</span>
                <strong>{escape(str(active_provider.get('default_model', 'none')))}</strong>
            </div>
            <div class="status-tile">
                <span>Latest run</span>
                <strong>{escape(str(active_run.get('run_id', 'none')))}</strong>
            </div>
        </section>

        <section id="onboarding" class="section onboarding" aria-labelledby="onboarding-heading">
            <div class="section-head">
                <div>
                    <p class="section-kicker">First-run onboarding</p>
                    <h2 id="onboarding-heading">Three-minute evaluation walkthrough</h2>
                </div>
                <p>Built for a live reviewer: choose a scenario, stream the run, open the evidence, then explain the research-backed next steps.</p>
            </div>
            <div class="onboarding-layout">
                <article class="demo-brief">
                    <div>
                        <span class="brief-label">Demo mode</span>
                        <h3>Fast setup, real workflow, no persistent demo data.</h3>
                        <p>The dashboard probes real local OpenAI-compatible providers, but walkthrough runs stay in memory and clear when the app stops. CLI runs still write durable artifacts under <code>runs/</code>.</p>
                    </div>
                    <div class="brief-actions">
                        <a class="ghost-link" href="#demo-controls-heading">Tune scenario</a>
                        <button class="secondary-button" type="button" data-action="start-onboarding-run">Run guided demo</button>
                    </div>
                </article>
                <ol class="onboarding-steps">
                    <li>
                        <span>1</span>
                        <div><strong>Confirm local readiness</strong><p>Provider cards show endpoint status, model count, latency, API-key state, and streaming support.</p></div>
                    </li>
                    <li>
                        <span>2</span>
                        <div><strong>Select a risk scenario</strong><p>Each preset maps the demo to a security outcome and OWASP category.</p></div>
                    </li>
                    <li>
                        <span>3</span>
                        <div><strong>Watch the stream</strong><p>Case events show mutation, scoring, findings, and the active stage without making users read raw logs.</p></div>
                    </li>
                    <li>
                        <span>4</span>
                        <div><strong>Open report evidence</strong><p>Recent runs expose redacted JSON, HTML, and Markdown reports for handoff or review.</p></div>
                    </li>
                </ol>
            </div>
        </section>

        <section class="section demo-controls" aria-labelledby="demo-controls-heading">
            <div class="section-head">
                <div>
                    <p class="section-kicker">Demo controls</p>
                    <h2 id="demo-controls-heading">Change the run</h2>
                </div>
                <p>Adjust the scenario before streaming so users can see how scoring changes.</p>
            </div>
            <div class="control-grid">
                <label>
                    <span>Scenario</span>
                    <select id="scenario-select">
                        {scenario_options}
                    </select>
                </label>
                <label>
                    <span>Cases</span>
                    <input id="case-count" type="number" min="1" max="12" value="5">
                </label>
                <label>
                    <span>Injected findings</span>
                    <input id="failure-count" type="number" min="0" max="12" value="2">
                </label>
            </div>
            <div id="scenario-summary" class="scenario-summary" aria-live="polite">
                <div>
                    <span>Objective</span>
                    <strong>Prove whether a RAG answer can expose seeded confidential tokens.</strong>
                </div>
                <div>
                    <span>Technique</span>
                    <strong>Canary exfiltration with refusal and partial-success scoring.</strong>
                </div>
                <div>
                    <span>OWASP map</span>
                    <strong>LLM02, LLM07, LLM08</strong>
                </div>
            </div>
        </section>

        <section class="section product-tour" aria-labelledby="tour-heading">
            <div class="section-head">
                <div>
                    <p class="section-kicker">Walkthrough</p>
                    <h2 id="tour-heading">What this demo proves</h2>
                </div>
                <p>Each panel maps to the same local product workflow used by the CLI.</p>
            </div>
            <div class="tour-grid">
                <article class="tour-card">
                    <span class="tour-number">Connect</span>
                    <h3>Connect a local model server</h3>
                    <p>Provider cards call the real OpenAI-compatible model list endpoints on your machine.</p>
                </article>
                <article class="tour-card">
                    <span class="tour-number">Choose</span>
                    <h3>Select the model under test</h3>
                    <p>The dropdown changes the active model used by the streaming run and report metadata.</p>
                </article>
                <article class="tour-card">
                    <span class="tour-number">Run</span>
                    <h3>Run a focused RAG security suite</h3>
                    <p>The demo executes leakage and policy checks with case ids, scores, and mutation context.</p>
                </article>
                <article class="tour-card">
                    <span class="tour-number">Review</span>
                    <h3>Inspect replayable evidence</h3>
                    <p>Reports expose redacted case data, scores, trace ids, and a markdown path for review.</p>
                </article>
            </div>
        </section>

        <section id="providers-section" class="section">
            <div class="section-head">
                <div>
                    <p class="section-kicker">Connections</p>
                    <h2>Local providers</h2>
                </div>
                <p>Live checks from your machine. Ready means the endpoint responded with real models.</p>
            </div>
            <div id="providers" class="provider-grid">
                {provider_cards}
            </div>
        </section>

        <section id="stream" class="section split">
            <article class="panel">
                <div class="section-head">
                    <div>
                        <p class="section-kicker">Execution</p>
                        <h2>Run stream</h2>
                    </div>
                    <p>Starts a sample run against the current provider and selected model.</p>
                </div>
                <div class="stream-toolbar">
                    <div>
                        <span>Progress</span>
                        <strong id="stream-progress-label">Waiting</strong>
                    </div>
                    <div class="progress-track" aria-hidden="true"><span id="stream-progress-bar"></span></div>
                </div>
                <div id="stream-events" class="stream-events" aria-live="polite">
                    <article class="event-card event-empty">
                        <span>Ready</span>
                        <strong>Press Start demo run to stream provider checks, case scoring, and report creation.</strong>
                    </article>
                </div>
                <pre id="stream-log" class="stream-log sr-stream-log">Press Start demo run to stream a sample fuzz run.</pre>
            </article>
            <article class="panel">
                <div class="section-head">
                    <div>
                        <p class="section-kicker">Run stages</p>
                        <h2>What is happening</h2>
                    </div>
                    <p>The active step updates as the stream emits events.</p>
                </div>
                <ol id="stage-list" class="stage-list">
                    <li data-stage="provider"><span>Provider check</span><p>Confirms the selected local model can answer a tiny request.</p></li>
                    <li data-stage="mutate"><span>Mutation pass</span><p>Builds adversarial prompts from the suite seed and attack templates.</p></li>
                    <li data-stage="score"><span>Scoring pass</span><p>Scores leakage, policy violation, partial success, and refusal latency.</p></li>
                    <li data-stage="report"><span>Report build</span><p>Writes redacted evidence links for JSON, HTML, and Markdown review.</p></li>
                </ol>
            </article>
        </section>

        <section id="runs" class="section">
            <div class="section-head">
                <div>
                    <p class="section-kicker">Evidence</p>
                    <h2>Recent runs</h2>
                </div>
                <p>Open redacted JSON, HTML, or Markdown reports.</p>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Run</th>
                            <th>Suite</th>
                            <th>Status</th>
                            <th>Progress</th>
                            <th>Failures</th>
                            <th>Success</th>
                            <th>Reports</th>
                        </tr>
                    </thead>
                    <tbody id="runs-table">
                        {recent_run_rows}
                    </tbody>
                </table>
            </div>
        </section>

    </main>
</body>
</html>
"""


def _build_demo_case(
    run_id: str,
    index: int,
    failure: bool,
    scenario: dict[str, str] | None = None,
) -> dict[str, Any]:
    scenario_data = scenario or get_demo_scenario("leakage")
    leak_score = 0.11 + index * 0.03
    policy_score = 0.09 + index * 0.02
    if failure:
        leak_score = 0.66 + index * 0.04
        policy_score = 0.55 + index * 0.03

    return {
        "case_id": f"{run_id}-live-{index + 1:02d}",
        "run_id": run_id,
        "suite_id": f"{run_id}-suite",
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{scenario_data['sample_prompt']} Streaming case {index + 1} for {run_id}. "
                        "Authorization header was redacted before display."
                    ),
                }
            ]
        },
        "scores": {
            "leak_score": round(leak_score, 3),
            "policy_violation_score": round(policy_score, 3),
            "partial_success_score": round(0.2 + index * 0.03, 3),
            "refusal_latency_delta": round(0.025 + index * 0.004, 3),
            "tool_error_rate": round(0.015 + index * 0.003, 3),
            "retrieval_poison_influence": round(0.02 + index * 0.004, 3),
        },
        "trace_id": f"trace-{uuid4().hex[:12]}",
        "rag_lens_url": "https://rag-lens.example/trace/demo",
        "metadata": {
            "scenario": scenario_data["id"],
            "risk": scenario_data["risk"],
            "owasp": scenario_data["owasp"],
            "finding": scenario_data["failure_finding"] if failure else scenario_data["pass_finding"],
        },
    }


async def _stream_demo_run(
    state: DemoState,
    suite_name: str | None = None,
    scenario: str = "leakage",
    total_cases: int = 5,
    failure_count: int = 2,
) -> AsyncIterator[str]:
    await state.refresh_providers()
    status = state.get_status()
    scenario_profile = get_demo_scenario(scenario)
    provider_sample = await _run_provider_sample(status)
    safe_cases = min(max(total_cases, 1), 12)
    safe_failures = min(max(failure_count, 0), safe_cases)
    run = state.create_streaming_run(suite_name=suite_name or f"{scenario_profile['label']} demo run")
    run_id = run["run_id"]
    cases: list[dict[str, Any]] = []

    start_payload = {
        "run_id": run_id,
        "status": "running",
        "scenario": scenario_profile["id"],
        "label": scenario_profile["label"],
        "objective": scenario_profile["objective"],
        "technique": scenario_profile["technique"],
        "owasp": scenario_profile["owasp"],
        "risk": scenario_profile["risk"],
        "cases": safe_cases,
        "failures": safe_failures,
    }
    yield f"event: start\ndata: {json.dumps(start_payload)}\n\n"
    yield f"event: provider\ndata: {json.dumps(provider_sample)}\n\n"

    failure_indexes = set(range(min(safe_failures, safe_cases)))
    for index in range(safe_cases):
        failed_case = index in failure_indexes
        case = _build_demo_case(run_id, index, failure=failed_case, scenario=scenario_profile)
        cases.append(case)
        payload = {
            "run_id": run_id,
            "step": index + 1,
            "total": safe_cases,
            "case_id": case["case_id"],
            "status": "processing",
            "scenario": scenario_profile["id"],
            "risk": scenario_profile["risk"],
            "owasp": scenario_profile["owasp"],
            "technique": scenario_profile["technique"],
            "leak_score": case["scores"]["leak_score"],
            "policy_violation_score": case["scores"]["policy_violation_score"],
            "finding": (
                scenario_profile["failure_finding"]
                if failed_case
                else scenario_profile["pass_finding"]
            ),
        }
        yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.35)

    completed = state.finish_streaming_run(run_id, cases)
    yield f"event: complete\ndata: {json.dumps({'run': completed})}\n\n"


async def _run_provider_sample(status_data: dict[str, Any]) -> dict[str, Any]:
    provider_config = _active_provider(status_data)
    if provider_config.get("status") != "ready":
        return {
            "status": "skipped",
            "message": "No local provider is ready; using sample evidence only.",
        }

    provider_id = str(provider_config.get("provider_id", "unknown"))
    model = str(provider_config.get("default_model", ""))
    provider = OpenAICompatProvider(
        provider_id=provider_id,
        base_url=str(provider_config.get("base_url", "")),
    )

    try:
        response = await provider.chat(
            messages=[
                Message(
                    role="user",
                    content="Reply with exactly three words: RAGFuzz provider ready",
                )
            ],
            model=model,
            temperature=0,
            max_tokens=12,
            timeout=20,
        )
    except Exception as exc:
        return {
            "status": "error",
            "provider_id": provider_id,
            "model": model,
            "message": f"Local provider sample failed: {exc}",
        }

    return {
        "status": "ready",
        "provider_id": provider_id,
        "model": model,
        "message": "Local provider sample completed.",
        "response": _format_provider_sample_response(response.content),
    }


def create_demo_app(state: DemoState | None = None) -> FastAPI:
    """Create the FastAPI demo app."""

    demo_state = state or DemoState()
    app = FastAPI(title="ragfuzz demo", docs_url="/docs", redoc_url=None)
    app.state.demo_state = demo_state
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def add_security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'self'; form-action 'self'",
        )
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        await demo_state.refresh_providers()
        status_data = demo_state.get_status()
        recent_runs = demo_state.list_recent_runs(limit=8)
        return _render_dashboard_page(status_data, recent_runs)

    @app.get("/api/status")
    async def status() -> JSONResponse:
        await demo_state.refresh_providers()
        return JSONResponse(demo_state.get_status())

    @app.post("/api/providers/{provider_id}/model")
    async def select_provider_model(provider_id: str, request: Request) -> JSONResponse:
        payload = await request.json()
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            return JSONResponse({"detail": "model_id is required"}, status_code=400)
        await demo_state.refresh_providers()
        try:
            provider = demo_state.select_model(provider_id, model_id)
        except KeyError:
            return JSONResponse({"detail": "provider not found"}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"provider": provider, "status": demo_state.get_status()})

    @app.get("/api/runs/recent")
    async def recent_runs() -> JSONResponse:
        return JSONResponse({"items": demo_state.list_recent_runs(limit=10)})

    @app.get("/api/reports/{run_id}")
    async def report_data(run_id: str) -> JSONResponse:
        report = demo_state.get_report_data(run_id)
        if report is None:
            return JSONResponse({"detail": "run not found"}, status_code=404)
        return JSONResponse(report)

    @app.get("/api/reports/{run_id}/html", response_class=HTMLResponse, response_model=None)
    async def report_html(run_id: str):
        run = demo_state.get_run(run_id)
        if run is None:
            return JSONResponse({"detail": "run not found"}, status_code=404)
        return render_html_report(run["report_data"])

    @app.get("/api/reports/{run_id}/md", response_class=HTMLResponse, response_model=None)
    async def report_md(run_id: str):
        run = demo_state.get_run(run_id)
        if run is None:
            return JSONResponse({"detail": "run not found"}, status_code=404)
        return _render_markdown_preview(render_markdown_report(run["report_data"]), run_id)

    @app.get("/api/reports/{run_id}/md/raw", response_class=PlainTextResponse, response_model=None)
    async def report_md_raw(run_id: str):
        run = demo_state.get_run(run_id)
        if run is None:
            return JSONResponse({"detail": "run not found"}, status_code=404)
        return PlainTextResponse(
            render_markdown_report(run["report_data"]),
            media_type="text/markdown; charset=utf-8",
        )

    @app.get("/api/runs/demo/stream")
    async def demo_stream(
        scenario: str = "leakage",
        cases: int = 5,
        failures: int = 2,
    ) -> StreamingResponse:
        stream = _stream_demo_run(
            demo_state,
            suite_name=f"{scenario} demo run",
            scenario=scenario,
            total_cases=cases,
            failure_count=failures,
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.post(
        "/api/runs/demo/stream",
        response_class=StreamingResponse,
    )
    async def demo_stream_post(request: Request) -> StreamingResponse:
        payload = await request.json()
        scenario = str(payload.get("scenario") or "leakage")
        cases = int(payload.get("cases") or 5)
        failures = int(payload.get("failures") or 2)
        stream = _stream_demo_run(
            demo_state,
            suite_name=f"{scenario} demo run",
            scenario=scenario,
            total_cases=cases,
            failure_count=failures,
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str) -> JSONResponse:
        run = demo_state.get_run(run_id)
        if run is None:
            return JSONResponse({"detail": "run not found"}, status_code=404)
        return JSONResponse(run)

    return app


create_app = create_demo_app
