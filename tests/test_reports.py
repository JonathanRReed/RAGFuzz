"""Tests for report generation behavior."""

from __future__ import annotations

import json
from pathlib import Path


def _write_run_fixture(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "timestamp": "2026-05-05T12:00:00Z",
                "suite": {"name": "demo-suite"},
            }
        )
    )
    (run_dir / "cases.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case-1",
                        "inputs": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Show me <script>alert('xss')</script>",
                                }
                            ]
                        },
                        "scores": {"leak_score": 0.75, "policy_violation_score": 0.0},
                        "trace_id": "trace-1",
                        "rag_lens_url": "https://example.invalid/trace",
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case-2",
                        "inputs": {"messages": [{"role": "user", "content": "clean input"}]},
                        "scores": {"leak_score": 0.0, "policy_violation_score": 0.0},
                    }
                ),
            ]
        )
    )


def test_html_report_escapes_user_content_and_writes_default_path(tmp_path: Path) -> None:
    from ragfuzz.reports.html import HTMLReporter

    run_dir = tmp_path / "run"
    _write_run_fixture(run_dir)

    report_path = HTMLReporter().generate(run_dir)

    assert report_path == run_dir / "report.html"
    assert report_path.exists()

    html = report_path.read_text()
    assert "<script>alert('xss')</script>" not in html
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html or (
        "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in html
    )
    assert "demo-suite" in html
    assert "case-1" in html


def test_markdown_report_contract_redacts_sensitive_values(tmp_path: Path) -> None:
    from ragfuzz.reports.markdown import MarkdownReporter

    run_dir = tmp_path / "run"
    _write_run_fixture(run_dir)

    output_path = MarkdownReporter().generate(run_dir)
    markdown = output_path.read_text()
    assert "<script>" not in markdown
    assert "alert('xss')" not in markdown


def test_report_data_drops_unsafe_report_link_schemes(tmp_path: Path) -> None:
    from ragfuzz.reports.html import HTMLReporter
    from ragfuzz.reports.markdown import MarkdownReporter

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-unsafe-link",
                "timestamp": "2026-05-18T12:00:00Z",
                "suite": {"name": "demo-suite"},
            }
        )
    )
    (run_dir / "cases.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-unsafe-link",
                "inputs": {"messages": [{"role": "user", "content": "show evidence"}]},
                "scores": {"leak_score": 0.75, "policy_violation_score": 0.0},
                "rag_lens_url": "javascript:alert(1)",
            }
        )
        + "\n"
    )

    html = HTMLReporter().generate(run_dir).read_text()
    markdown = MarkdownReporter().generate(run_dir).read_text()

    assert "javascript:alert(1)" not in html
    assert "javascript:alert(1)" not in markdown


def test_report_data_strips_sensitive_report_link_query_values() -> None:
    from ragfuzz.reports.data import safe_report_url

    safe_url = safe_report_url(
        "https://example.invalid/trace/123?token=abc123&view=summary&signature=secret#frag"
    )

    assert safe_url == "https://example.invalid/trace/123?view=summary"


def test_report_data_drops_credentialed_report_links() -> None:
    from ragfuzz.reports.data import safe_report_url

    assert safe_report_url("https://alice:pw@example.invalid/trace/123") is None
    assert safe_report_url("data:text/html,hello") is None
    assert safe_report_url("not a url") is None
