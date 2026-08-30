"""
Tests for ArXiv Atom XML parsing and GitHub repo URL extraction
(src/scrapers/arxiv_papers.py). Parsed against a schema-accurate
fixture since export.arxiv.org is unreachable from the dev sandbox
(see module docstring in arxiv_papers.py) — the fixture's XML
structure matches arXiv's documented Atom 1.0 API response format
exactly (info.arxiv.org/help/api/user-manual.html).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scrapers.arxiv_papers import _parse_atom_response, extract_github_repo

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "arxiv_sample_response.xml"


def test_parses_expected_number_of_entries():
    xml = FIXTURE_PATH.read_text()
    papers = _parse_atom_response(xml)
    assert len(papers) == 2


def test_title_whitespace_is_collapsed():
    xml = FIXTURE_PATH.read_text()
    papers = _parse_atom_response(xml)
    assert "\n" not in papers[0]["title"]
    assert papers[0]["title"] == "Efficient Fallback Chains for Multi-Tier LLM Extraction Pipelines"


def test_authors_extracted():
    xml = FIXTURE_PATH.read_text()
    papers = _parse_atom_response(xml)
    assert papers[0]["authors"] == ["A. Researcher", "B. Scientist"]
    assert papers[1]["authors"] == ["C. Data Engineer"]


def test_arxiv_id_extracted():
    xml = FIXTURE_PATH.read_text()
    papers = _parse_atom_response(xml)
    assert papers[0]["arxiv_id"] == "2508.12345v1"


def test_categories_extracted():
    xml = FIXTURE_PATH.read_text()
    papers = _parse_atom_response(xml)
    assert "cs.AI" in papers[0]["categories"]
    assert "cs.CL" in papers[0]["categories"]


def test_github_repo_extracted_from_abstract():
    xml = FIXTURE_PATH.read_text()
    papers = _parse_atom_response(xml)
    repo = extract_github_repo(papers[0]["abstract"])
    assert repo == ("example-org", "llm-fallback-bench")


def test_no_github_repo_returns_none():
    xml = FIXTURE_PATH.read_text()
    papers = _parse_atom_response(xml)
    repo = extract_github_repo(papers[1]["abstract"])
    assert repo is None


def test_github_regex_handles_various_formats():
    cases = [
        ("Code at https://github.com/pytorch/pytorch for details.", ("pytorch", "pytorch")),
        ("See github.com/huggingface/transformers.", ("huggingface", "transformers")),
        ("No repo mentioned here at all.", None),
        ("Repo: github.com/openai/whisper/tree/main", ("openai", "whisper")),
    ]
    for text, expected in cases:
        assert extract_github_repo(text) == expected, f"Failed on: {text}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
