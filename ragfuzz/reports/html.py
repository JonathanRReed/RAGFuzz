"""HTML report generation."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .data import build_report_data, load_run_payload
from .markdown import MarkdownReporter


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _status_class(success_rate: float) -> str:
    if success_rate >= 90:
        return "good"
    if success_rate >= 75:
        return "warn"
    return "bad"


def _severity_class(severity: str) -> str:
    if severity == "high":
        return "high"
    if severity == "medium":
        return "medium"
    return "low"


def _render_metadata_section(metadata: dict[str, Any]) -> str:
    items: list[str] = []
    for group_name, group_value in metadata.items():
        if isinstance(group_value, dict) and group_value:
            nested_items = "".join(
                f"<li><span>{escape(str(key))}</span><strong>{escape(str(value))}</strong></li>"
                for key, value in group_value.items()
            )
            items.append(
                "<section class=\"panel\">"
                f"<h2>{escape(str(group_name))}</h2>"
                f"<ul class=\"kv-list\">{nested_items}</ul>"
                "</section>"
            )
        elif group_value:
            items.append(
                "<section class=\"panel\">"
                f"<h2>{escape(str(group_name))}</h2>"
                f"<p class=\"panel-text\">{escape(str(group_value))}</p>"
                "</section>"
            )

    return "".join(items)


def _render_case_rows(cases: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for case in cases:
        severity = str(case.get("severity", "low"))
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(case.get('case_id', 'unknown')))}</code></td>"
            f"<td><span class=\"badge severity-{_severity_class(severity)}\">{escape(severity)}</span></td>"
            f"<td>{escape(str(case.get('category', 'unknown')))}</td>"
            f"<td>{_format_number(case.get('leak_score'))}</td>"
            f"<td>{_format_number(case.get('policy_violation_score'))}</td>"
            f"<td>{escape(str(case.get('trace_id') or ''))}</td>"
            f"<td class=\"snippet\">{escape(str(case.get('input_text', '')))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_failure_cards(failures: list[dict[str, Any]]) -> str:
    if not failures:
        return (
            '<section class="panel panel-empty">'
            "<h2>Failures</h2>"
            "<p class=\"panel-text\">No failures were detected in this run.</p>"
            "</section>"
        )

    cards: list[str] = []
    for case in failures:
        links: list[str] = []
        rag_lens_url = case.get("rag_lens_url")
        if rag_lens_url:
            links.append(
                f'<a class="link-button" href="{escape(str(rag_lens_url))}" target="_blank" rel="noreferrer noopener">RAG Lens</a>'
            )

        links_html = "".join(links)
        cards.append(
            "<article class=\"failure-card\">"
            f"<div class=\"failure-head\"><code>{escape(str(case.get('case_id', 'unknown')))}</code>"
            f"<span class=\"badge severity-{_severity_class(str(case.get('severity', 'low')))}\">{escape(str(case.get('severity', 'low')))}</span>"
            "</div>"
            f"<div class=\"failure-meta\">Category: {escape(str(case.get('category', 'unknown')))}"
            f" · Leak: {_format_number(case.get('leak_score'))}"
            f" · Policy: {_format_number(case.get('policy_violation_score'))}</div>"
            f"<pre>{escape(str(case.get('input_text', '')))}</pre>"
            f"<div class=\"failure-links\">{links_html}</div>"
            "</article>"
        )
    return "".join(cards)


def render_html_report(report_data: dict[str, Any]) -> str:
    """Render a report bundle to polished, self-contained HTML."""

    summary = report_data.get("summary") or {}
    success_rate = float(summary.get("success_rate", 0.0) or 0.0)
    suite_name = escape(str(report_data.get("suite_name", "unknown")))
    run_id = escape(str(report_data.get("run_id", "unknown")))
    timestamp = escape(str(report_data.get("timestamp", "unknown")))

    summary_cards = [
        ("Run ID", run_id),
        ("Suite", suite_name),
        ("Total cases", str(summary.get("total_cases", 0))),
        ("Failures", str(summary.get("failure_count", 0))),
        ("Success rate", f"{success_rate:.1f}%"),
        ("Avg leak score", _format_number(summary.get("avg_leak_score"))),
        (
            "Avg policy score",
            _format_number(summary.get("avg_policy_violation_score")),
        ),
    ]

    cards_html = "".join(
        f"<div class=\"stat-card {('emphasis' if label == 'Success rate' else '')}\">"
        f"<span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
        for label, value in summary_cards
    )

    metadata_html = _render_metadata_section(report_data.get("metadata") or {})
    failures_html = _render_failure_cards(report_data.get("failures") or [])
    cases_rows = _render_case_rows(report_data.get("cases") or [])
    status_class = _status_class(success_rate)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ragfuzz report - {run_id}</title>
    <style>
        :root {{
            color-scheme: dark;
            --bg: #000000;
            --bg-elevated: #0a0a0a;
            --panel: #050505;
            --panel-border: #494949;
            --text: #ffffff;
            --muted: #7c7a7a;
            --good: #ffffff;
            --warn: #ff5d73;
            --bad: #ff5d73;
            --accent: #ff5d73;
            --radius: 8px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at 20% 0, rgba(255, 93, 115, 0.2), transparent 28rem),
                linear-gradient(180deg, rgba(73, 73, 73, 0.28), rgba(0, 0, 0, 0) 22rem),
                var(--bg);
            color: var(--text);
        }}
        .shell {{
            max-width: 1280px;
            margin: 0 auto;
            padding: 32px 20px 48px;
        }}
        .hero {{
            border: 1px solid var(--panel-border);
            border-radius: var(--radius);
            background: linear-gradient(135deg, rgba(255, 93, 115, 0.18), rgba(0, 0, 0, 0.96));
            padding: 24px;
            margin-bottom: 20px;
        }}
        .eyebrow {{
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        h1 {{
            margin: 0;
            font-size: clamp(28px, 3vw, 40px);
            line-height: 1.1;
        }}
        .subtitle {{
            margin-top: 10px;
            color: var(--muted);
        }}
        .subtle {{
            color: var(--muted);
            font-size: 14px;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 16px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255, 93, 115, 0.12);
            border: 1px solid rgba(255, 93, 115, 0.34);
            color: var(--text);
            font-size: 13px;
        }}
        .status-pill .dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 0 4px rgba(255, 93, 115, 0.14);
        }}
        .status-good .dot {{ background: var(--good); }}
        .status-warn .dot {{ background: var(--warn); }}
        .status-bad .dot {{ background: var(--bad); }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin-bottom: 18px;
        }}
        .stat-card {{
            border: 1px solid var(--panel-border);
            background: var(--bg-elevated);
            border-radius: var(--radius);
            padding: 16px;
            min-height: 92px;
        }}
        .stat-card span {{
            display: block;
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .stat-card strong {{
            font-size: 24px;
            line-height: 1.15;
            font-weight: 650;
            word-break: break-word;
        }}
        .stat-card.emphasis strong {{ color: var(--good); }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
        }}
        .panel {{
            border: 1px solid var(--panel-border);
            background: var(--panel);
            border-radius: var(--radius);
            padding: 18px;
        }}
        .panel h2 {{
            margin: 0 0 14px;
            font-size: 16px;
        }}
        .panel-text {{
            color: var(--muted);
            margin: 0;
        }}
        .kv-list {{
            list-style: none;
            margin: 0;
            padding: 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px;
        }}
        .kv-list li {{
            padding: 12px;
            border-radius: var(--radius);
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .kv-list span {{
            display: block;
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 6px;
        }}
        .kv-list strong {{
            font-size: 14px;
            word-break: break-word;
        }}
        .failure-grid {{
            display: grid;
            gap: 12px;
        }}
        .failure-card {{
            border: 1px solid var(--panel-border);
            background: rgba(255, 255, 255, 0.03);
            border-radius: var(--radius);
            padding: 16px;
        }}
        .failure-head {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}
        .failure-head code, code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        .failure-meta {{
            margin-top: 10px;
            color: var(--muted);
            font-size: 13px;
        }}
        .failure-card pre {{
            margin: 14px 0 0;
            padding: 14px;
            border-radius: var(--radius);
            background: #0a0f1d;
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: var(--text);
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .failure-links {{
            margin-top: 12px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .link-button {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 34px;
            padding: 0 12px;
            border-radius: 10px;
            border: 1px solid rgba(255, 93, 115, 0.44);
            color: #000000;
            background: var(--accent);
            text-decoration: none;
            font-weight: 800;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid transparent;
        }}
        .severity-high {{
            color: #ffd1d1;
            border-color: rgba(255, 123, 123, 0.25);
            background: rgba(255, 123, 123, 0.12);
        }}
        .severity-medium {{
            color: #ffe4b5;
            border-color: rgba(246, 198, 106, 0.25);
            background: rgba(246, 198, 106, 0.12);
        }}
        .severity-low {{
            color: #d7f8ec;
            border-color: rgba(77, 212, 167, 0.22);
            background: rgba(77, 212, 167, 0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding: 12px 10px;
            text-align: left;
            vertical-align: top;
            font-size: 13px;
        }}
        th {{
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 11px;
        }}
        .snippet {{
            max-width: 520px;
            color: var(--muted);
        }}
        .table-wrap {{
            overflow: auto;
        }}
        .section-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .section-title h2 {{
            margin: 0;
            font-size: 18px;
        }}
        .section-title .subtle {{
            margin: 0;
        }}
        @media (min-width: 960px) {{
            .grid {{
                grid-template-columns: 1.2fr 0.8fr;
            }}
            .report-table {{
                grid-column: 1 / -1;
            }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero status-{status_class}">
            <div class="eyebrow">ragfuzz report</div>
            <h1>{suite_name}</h1>
            <div class="subtitle">Run {run_id} completed at {timestamp}</div>
            <div class="status-pill status-{status_class}"><span class="dot"></span><span>Success rate {success_rate:.1f}%</span></div>
        </section>

        <section class="stat-grid">
            {cards_html}
        </section>

        <section class="grid">
            <section class="panel">
                <div class="section-title">
                    <h2>Metadata</h2>
                    <p class="subtle">Redacted before rendering</p>
                </div>
                {metadata_html or '<p class="panel-text">No metadata available.</p>'}
            </section>

            <section class="panel">
                <div class="section-title">
                    <h2>Failures</h2>
                    <p class="subtle">{len(report_data.get('failures') or [])} case(s)</p>
                </div>
                <div class="failure-grid">{failures_html}</div>
            </section>

            <section class="panel report-table">
                <div class="section-title">
                    <h2>Case Overview</h2>
                    <p class="subtle">{len(report_data.get('cases') or [])} case(s)</p>
                </div>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Case</th>
                                <th>Severity</th>
                                <th>Category</th>
                                <th>Leak</th>
                                <th>Policy</th>
                                <th>Trace</th>
                                <th>Input</th>
                            </tr>
                        </thead>
                        <tbody>
                            {cases_rows}
                        </tbody>
                    </table>
                </div>
            </section>
        </section>
    </main>
</body>
</html>
"""


class HTMLReporter:
    """Generates HTML reports for run results."""

    def __init__(self, template_dir: str | Path | None = None):
        self.template_dir = Path(template_dir) if template_dir else None

    def generate(
        self,
        run_dir_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate an HTML report from run data."""

        run_data, cases = load_run_payload(run_dir_path)
        report_data = build_report_data(run_data, cases)
        html = render_html_report(report_data)

        run_dir = Path(run_dir_path)
        output = Path(output_path) if output_path else run_dir / "report.html"
        output.write_text(html)
        return output

    def generate_markdown(
        self,
        run_dir_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate a Markdown report from run data."""

        return MarkdownReporter().generate(run_dir_path, output_path)

    def summarize(self, run_dir_path: str | Path) -> dict[str, Any]:
        """Return the redacted report data for machine-readable output."""

        run_data, cases = load_run_payload(run_dir_path)
        return build_report_data(run_data, cases)
