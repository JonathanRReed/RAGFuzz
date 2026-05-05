"""Markdown report generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .data import build_report_data, load_run_payload

_SCRIPT_PATTERN = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)


def _md_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _fmt_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _md_safe_text(value: Any) -> str:
    """Remove active-script text from Markdown code blocks."""

    return _SCRIPT_PATTERN.sub("[REDACTED SCRIPT]", str(value))


def render_markdown_report(report_data: dict[str, Any]) -> str:
    """Render a redacted report bundle to Markdown."""

    summary = report_data.get("summary") or {}
    failures = report_data.get("failures") or []
    cases = report_data.get("cases") or []
    metadata = report_data.get("metadata") or {}

    lines = [
        f"# RAGFuzz Report: {_md_escape(str(report_data.get('suite_name', 'unknown')))}",
        "",
        f"Run `{_md_escape(str(report_data.get('run_id', 'unknown')))}` completed at `{_md_escape(str(report_data.get('timestamp', 'unknown')))}`.",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total cases | {summary.get('total_cases', 0)} |",
        f"| Failures | {summary.get('failure_count', 0)} |",
        f"| Success rate | {summary.get('success_rate', 0.0)}% |",
        f"| Average leak score | {_fmt_number(summary.get('avg_leak_score'))} |",
        f"| Average policy violation score | {_fmt_number(summary.get('avg_policy_violation_score'))} |",
        "",
        "## Metadata",
        "",
    ]

    if metadata:
        for group_name, group_value in metadata.items():
            lines.append(f"### {_md_escape(str(group_name)).title()}")
            lines.append("")
            if isinstance(group_value, dict) and group_value:
                lines.append("| Key | Value |")
                lines.append("| --- | --- |")
                for key, value in group_value.items():
                    lines.append(f"| {_md_escape(str(key))} | `{_md_escape(str(value))}` |")
                lines.append("")
            else:
                lines.append(f"`{_md_escape(str(group_value))}`")
                lines.append("")
    else:
        lines.append("- No metadata available.")

    lines.extend(["", "## Failures"])

    if failures:
        for case in failures:
            lines.append("### `" + _md_escape(str(case.get("case_id", "unknown"))) + "`")
            lines.extend(
                [
                    f"- Severity: {_md_escape(str(case.get('severity', 'low')))}",
                    f"- Category: {_md_escape(str(case.get('category', 'unknown')))}",
                    f"- Leak score: {_fmt_number(case.get('leak_score'))}",
                    f"- Policy score: {_fmt_number(case.get('policy_violation_score'))}",
                ]
            )
            if case.get("trace_id"):
                lines.append("- Trace ID: `" + _md_escape(str(case.get("trace_id"))) + "`")
            if case.get("rag_lens_url"):
                lines.append(f"- RAG Lens: <{_md_escape(str(case.get('rag_lens_url')))}>")
            lines.extend(["", "```text", _md_safe_text(case.get("input_text", "")), "```", ""])
    else:
        lines.append("- No failures were detected in this run.")

    lines.extend(["## Case Overview", ""])

    if cases:
        lines.append("| Case | Severity | Category | Leak | Policy | Trace |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        for case in cases:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(str(case.get("case_id", "unknown"))),
                        _md_escape(str(case.get("severity", "low"))),
                        _md_escape(str(case.get("category", "unknown"))),
                        _fmt_number(case.get("leak_score")),
                        _fmt_number(case.get("policy_violation_score")),
                        _md_escape(str(case.get("trace_id") or "")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- No cases available.")

    return "\n".join(lines).rstrip() + "\n"


class MarkdownReporter:
    """Generates Markdown reports for run results."""

    def generate(
        self,
        run_dir_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate a Markdown report from run data."""

        run_data, cases = load_run_payload(run_dir_path)
        report_data = build_report_data(run_data, cases)
        markdown = render_markdown_report(report_data)

        run_dir = Path(run_dir_path)
        output = Path(output_path) if output_path else run_dir / "report.md"
        output.write_text(markdown)
        return output
