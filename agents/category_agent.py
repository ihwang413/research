from __future__ import annotations

import json
import re

import anthropic

from utils.data_models import Paper


def run(papers: list[Paper], categories: list[str], config: dict) -> list[Paper]:
    api_cfg = config.get("api", {})
    api_key = api_cfg.get("anthropic_key")
    model = config.get("summary", {}).get("model", "claude-sonnet-4-6")

    client = anthropic.Anthropic(api_key=api_key if api_key else None)

    if categories:
        papers = _assign_preset(papers, categories, client, model)
    else:
        papers = _auto_cluster(papers, client, model)

    for i, p in enumerate(papers):
        if not p.categories:
            papers[i] = p.model_copy(update={"categories": ["Uncategorized"]})

    papers = _generate_apa_refs(papers)

    papers.sort(key=lambda p: (p.categories[0] if p.categories else "zzz", -(p.year or 0)))

    print(f"[category] Done. Categories: {sorted({c for p in papers for c in p.categories})}", flush=True)
    return papers


def _assign_preset(papers: list[Paper], categories: list[str], client: anthropic.Anthropic, model: str) -> list[Paper]:
    paper_entries = []
    for i, p in enumerate(papers):
        text = f"{p.title}. {p.summary or ''}"
        paper_entries.append(f"{i}: {text[:300]}")

    categories_str = "\n".join(f"- {c}" for c in categories)
    papers_str = "\n".join(paper_entries)

    prompt = f"""You are categorizing academic papers into predefined research categories.

Categories:
{categories_str}

Papers (id: title + summary excerpt):
{papers_str}

For each paper, assign one or more categories from the list above. A paper may belong to multiple categories.
Return a JSON object where keys are paper IDs (as strings) and values are arrays of category names from the list.
Only use category names exactly as listed. Return ONLY the JSON object."""

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _strip_fences(resp.content[0].text.strip())
        mapping: dict[str, list[str]] = json.loads(raw)
        for i, p in enumerate(papers):
            assigned = mapping.get(str(i), [])
            valid = [c for c in assigned if c in categories]
            papers[i] = p.model_copy(update={"categories": valid})
    except (anthropic.APIError, json.JSONDecodeError, KeyError):
        for i, p in enumerate(papers):
            papers[i] = p.model_copy(update={"categories": [categories[0]] if categories else []})

    return papers


def _auto_cluster(papers: list[Paper], client: anthropic.Anthropic, model: str) -> list[Paper]:
    paper_entries = []
    for i, p in enumerate(papers):
        text = f"{p.title}. {p.summary or ''}"
        paper_entries.append(f"{i}: {text[:300]}")

    papers_str = "\n".join(paper_entries)

    prompt = f"""You are a research librarian organizing academic papers into coherent topic clusters.

Papers (id: title + summary excerpt):
{papers_str}

Identify 4-8 coherent topic clusters that best organize these papers. Name each cluster clearly and concisely.
Return a JSON object where keys are cluster names and values are arrays of paper IDs (as integers).
Each paper should appear in at least one cluster. Return ONLY the JSON object."""

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _strip_fences(resp.content[0].text.strip())
        clusters: dict[str, list[int]] = json.loads(raw)

        id_to_categories: dict[int, list[str]] = {}
        for cluster_name, ids in clusters.items():
            for pid in ids:
                id_to_categories.setdefault(int(pid), []).append(cluster_name)

        for i, p in enumerate(papers):
            cats = id_to_categories.get(i, ["General"])
            papers[i] = p.model_copy(update={"categories": cats})
    except (anthropic.APIError, json.JSONDecodeError, KeyError, ValueError):
        for i, p in enumerate(papers):
            papers[i] = p.model_copy(update={"categories": ["General"]})

    return papers


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    return text


def _generate_apa_refs(papers: list[Paper]) -> list[Paper]:
    updated = []
    for p in papers:
        updated.append(p.model_copy(update={"apa_ref": _generate_apa(p)}))
    return updated


def _generate_apa(paper: Paper) -> str:
    authors = paper.authors or []
    if not authors:
        author_str = "Unknown Author"
    elif len(authors) == 1:
        author_str = _format_author(authors[0])
    elif len(authors) <= 6:
        formatted = [_format_author(a) for a in authors]
        author_str = ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    else:
        formatted = [_format_author(a) for a in authors[:6]]
        author_str = ", ".join(formatted) + ", et al."

    year = f"({paper.year})." if paper.year else "(n.d.)."
    title = paper.title or "Untitled"
    venue = f" {paper.venue}." if paper.venue else ""
    doi_str = f" https://doi.org/{paper.doi}" if paper.doi else ""

    return f"{author_str} {year} {title}.{venue}{doi_str}".strip()


def _format_author(name: str) -> str:
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0]
    last = parts[-1]
    initials = " ".join(p[0].upper() + "." for p in parts[:-1] if p)
    return f"{last}, {initials}"
