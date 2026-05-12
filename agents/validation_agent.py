from __future__ import annotations

from typing import TypedDict

from agents.summary_agent import _validate_summary
from utils.data_models import Paper, PipelineState


class ValidationResult(TypedDict):
    passed: bool
    checks: dict[str, bool]
    failures: list[str]


def run(papers: list[Paper], state: PipelineState) -> ValidationResult:
    checks = {
        "paper_count": _check_paper_count(papers),
        "summary_completeness": _check_summary_completeness(papers),
        "summary_format": _check_summary_format(papers),
        "categories": _check_categories(papers),
        "apa_refs": _check_apa_refs(papers),
        "no_doi_duplicates": _check_no_doi_duplicates(papers),
    }
    failures = [name for name, passed in checks.items() if not passed]

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[validation] {name}: {status}", flush=True)

    return ValidationResult(passed=len(failures) == 0, checks=checks, failures=failures)


def map_failures_to_stages(failures: list[str]) -> list[str]:
    stages = set()
    mapping = {
        "paper_count": "search",
        "no_doi_duplicates": "search",
        "summary_completeness": "summary",
        "summary_format": "summary",
        "categories": "category",
        "apa_refs": "category",
    }
    for f in failures:
        stage = mapping.get(f)
        if stage:
            stages.add(stage)
    return list(stages)


def _check_paper_count(papers: list[Paper]) -> bool:
    return len(papers) >= 1


def _check_summary_completeness(papers: list[Paper]) -> bool:
    for p in papers:
        if p.abstract is not None and not p.summary:
            return False
    return True


def _check_summary_format(papers: list[Paper]) -> bool:
    for p in papers:
        if p.summary and not _validate_summary(p.summary):
            return False
    return True


def _check_categories(papers: list[Paper]) -> bool:
    return all(len(p.categories) >= 1 for p in papers)


def _check_apa_refs(papers: list[Paper]) -> bool:
    for p in papers:
        if not p.apa_ref:
            return False
        year = str(p.year) if p.year else None
        if year and year not in p.apa_ref:
            return False
    return True


def _check_no_doi_duplicates(papers: list[Paper]) -> bool:
    seen: set[str] = set()
    for p in papers:
        if p.doi:
            if p.doi in seen:
                return False
            seen.add(p.doi)
    return True
