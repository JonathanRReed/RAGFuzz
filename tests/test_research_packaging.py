from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_citation_metadata_matches_package_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert citation["cff-version"] == "1.2.0"
    assert str(citation["version"]) == project["version"]
    assert citation["repository-code"] == project["urls"]["Source"]
    assert citation["url"] == project["urls"]["Homepage"]
    assert citation["authors"] == [
        {"family-names": "Reed", "given-names": "Jonathan R."}
    ]
    assert project["authors"] == [{"name": "Jonathan R. Reed"}]


def test_research_index_only_links_to_existing_repository_files() -> None:
    research_index = ROOT / "RESEARCH.md"
    source = research_index.read_text(encoding="utf-8")
    repository_links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", source)

    assert repository_links
    for relative_path in repository_links:
        assert not relative_path.startswith(("http://", "https://"))
        assert (ROOT / relative_path).is_file(), relative_path


def test_methodology_keeps_required_evidence_boundaries_visible() -> None:
    methodology = (ROOT / "docs/research/METHODOLOGY.md").read_text(
        encoding="utf-8"
    )

    required_statements = (
        "full Git commit SHA",
        "coverage boundary",
        "not independent validation",
        "synthetic demo evidence",
        "no single aggregate score establishes production safety",
        "Run RAGFuzz only against systems you own or have explicit permission to test",
    )

    for statement in required_statements:
        assert statement in methodology
