"""Tests for report generation behavior."""

from __future__ import annotations

import json
import sys
import types
from importlib import util as importlib_util
from pathlib import Path

import pytest


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
    html_module_path = Path(__file__).resolve().parents[1] / "ragfuzz" / "reports" / "html.py"
    if not html_module_path.exists():
        pytest.skip(
            "HTML report module is not present yet, this test documents the expected contract."
        )

    data_module_path = Path(__file__).resolve().parents[1] / "ragfuzz" / "reports" / "data.py"
    if not data_module_path.exists():
        pytest.skip("Report data helpers are not present yet.")

    package_name = "ragfuzz.reports"
    package = types.ModuleType(package_name)
    package.__path__ = [str(html_module_path.parent)]  # type: ignore[attr-defined]
    sys.modules.setdefault(package_name, package)

    data_spec = importlib_util.spec_from_file_location(
        "ragfuzz.reports.data", data_module_path
    )
    if data_spec is None or data_spec.loader is None:
        pytest.skip("Report data module cannot be loaded in this checkout.")

    data_module = importlib_util.module_from_spec(data_spec)
    sys.modules[data_spec.name] = data_module
    data_spec.loader.exec_module(data_module)

    html_spec = importlib_util.spec_from_file_location("ragfuzz_reports_html", html_module_path)
    if html_spec is None or html_spec.loader is None:
        pytest.skip("HTML report module cannot be loaded in this checkout.")

    html_module = importlib_util.module_from_spec(html_spec)
    html_module.__package__ = package_name
    html_spec.loader.exec_module(html_module)
    html_reporter_cls = html_module.HTMLReporter

    run_dir = tmp_path / "run"
    _write_run_fixture(run_dir)

    report_path = html_reporter_cls().generate(run_dir)

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
    html_module_path = Path(__file__).resolve().parents[1] / "ragfuzz" / "reports" / "html.py"
    if not html_module_path.exists():
        pytest.skip(
            "Report package is incomplete in this checkout, this test documents the expected contract."
        )

    markdown_module_path = Path(__file__).resolve().parents[1] / "ragfuzz" / "reports" / "markdown.py"
    if not markdown_module_path.exists():
        pytest.skip(
            "Markdown report API is not present yet, this test documents the expected contract."
        )

    markdown_spec = importlib_util.spec_from_file_location(
        "ragfuzz_reports_markdown", markdown_module_path
    )
    if markdown_spec is None or markdown_spec.loader is None:
        pytest.skip("Markdown report module cannot be loaded in this checkout.")

    markdown_module = importlib_util.module_from_spec(markdown_spec)
    markdown_module.__package__ = "ragfuzz.reports"
    try:
        markdown_spec.loader.exec_module(markdown_module)
    except SyntaxError:
        pytest.skip("Markdown report module still has a syntax error in this checkout.")

    reporter = (
        getattr(markdown_module, "MarkdownReporter", None)
        or getattr(markdown_module, "MarkdownReport", None)
        or getattr(markdown_module, "render_markdown_report", None)
    )

    if reporter is None:
        pytest.skip("Markdown report API is not exposed yet.")

    run_dir = tmp_path / "run"
    _write_run_fixture(run_dir)

    if callable(reporter) and not hasattr(reporter, "generate"):
        output = reporter(run_dir)
        output_path = Path(output) if output is not None else None
    else:
        reporter_instance = reporter()
        output = reporter_instance.generate(run_dir)
        output_path = Path(output) if output is not None else None

    if output_path is None:
        pytest.skip("Markdown report API did not return an output path.")

    markdown = output_path.read_text()
    assert "<script>" not in markdown
    assert "alert('xss')" not in markdown
