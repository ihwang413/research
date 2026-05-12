from __future__ import annotations

import httpx

from utils.api_client import OpenAlexClient, SemanticScholarClient
from utils.data_models import Paper


# index_filter → (OpenAlex extra filter, min h_index, max h_index)
_INDEX_CONFIG: dict[str, tuple[str, int, int]] = {
    "all":      ("",                    0,  9999),
    "scopus":   ("indexed_in:scopus",   0,  9999),
    "esci":     ("indexed_in:scopus",   5,  39),
    "ssci_sci": ("indexed_in:scopus",   40, 9999),
}


async def run(
    config: dict,
    keyword: str,
    year_from: int,
    year_to: int,
    max_papers: int,
    language: str = "en",
    index_filter: str = "all",
) -> list[Paper]:
    api_cfg = config.get("api", {})
    search_cfg = config.get("search", {})
    per_page = search_cfg.get("per_page", 25)
    warn_threshold = search_cfg.get("abstract_missing_warn_threshold", 0.30)

    oa_extra, hindex_min, hindex_max = _INDEX_CONFIG.get(index_filter, _INDEX_CONFIG["all"])

    label = {"all": "전체", "scopus": "Scopus", "esci": "ESCI", "ssci_sci": "SSCI/SCI"}.get(index_filter, index_filter)
    print(f"[search] 인덱스 필터: {label}", flush=True)

    async with httpx.AsyncClient(timeout=30.0) as http:
        oa_client = OpenAlexClient(http, api_cfg.get("openalex_email"))
        ss_client = SemanticScholarClient(http, api_cfg.get("semantic_scholar_key") or None)

        raw_works: list[dict] = []
        cursor = "*"
        print(f"[search] OpenAlex 검색 중...", flush=True)
        while len(raw_works) < max_papers * 3:  # 필터 후 줄어들 수 있으므로 넉넉히
            works, next_cursor = await oa_client.search(
                keyword, year_from, year_to, language, per_page, cursor,
                extra_filter=oa_extra,
            )
            if not works:
                break
            raw_works.extend(works)
            if next_cursor is None:
                break
            cursor = next_cursor
            print(f"[search] OpenAlex: {len(raw_works)}편 수집", end="\r", flush=True)
            if len(raw_works) >= max_papers * 3:
                break

        print(f"\n[search] OpenAlex raw: {len(raw_works)}편", flush=True)

        # h_index 필터 (SSCI/SCI, ESCI)
        if index_filter in ("ssci_sci", "esci") and raw_works:
            source_ids = list({
                (w.get("primary_location") or {}).get("source", {}).get("id")
                for w in raw_works
                if (w.get("primary_location") or {}).get("source", {}).get("id")
            })
            print(f"[search] {len(source_ids)}개 학술지 h_index 조회 중...", flush=True)
            hindex_map = await oa_client.batch_source_hindex(source_ids)

            before = len(raw_works)
            raw_works = [
                w for w in raw_works
                if hindex_min
                <= hindex_map.get(
                    (w.get("primary_location") or {}).get("source", {}).get("id", ""), 0
                )
                <= hindex_max
            ]
            print(f"[search] h_index 필터({hindex_min}~{hindex_max}): {before}→{len(raw_works)}편", flush=True)

        papers = [_normalize_openalex(w) for w in raw_works]

        # Semantic Scholar 보충 (Scopus/index 필터 없을 때만)
        if index_filter == "all" and len(papers) < max_papers * 0.7:
            remaining = max_papers - len(papers)
            print(f"[search] Semantic Scholar 보충 검색 ({remaining}편)...", flush=True)
            ss_papers, _ = await ss_client.search(keyword, year_from, year_to, remaining)
            for p in ss_papers:
                papers.append(_normalize_ss(p))

    papers = _deduplicate(papers)[:max_papers]

    missing = sum(1 for p in papers if p.abstract is None)
    if papers and missing / len(papers) > warn_threshold:
        print(
            f"WARNING: {missing}/{len(papers)} ({missing/len(papers):.0%}) 논문 초록 없음",
            flush=True,
        )

    print(f"[search] 완료. {len(papers)}편 (인덱스: {label})", flush=True)
    return papers


def _normalize_openalex(raw: dict) -> Paper:
    doi = raw.get("doi")
    if doi:
        doi = doi.replace("https://doi.org/", "").lower()

    primary = raw.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name")

    authors = []
    for a in raw.get("authorships", []):
        name = (a.get("author") or {}).get("display_name")
        if name:
            authors.append(name)

    url = (primary.get("landing_page_url") or
           (raw.get("open_access") or {}).get("oa_url"))

    abstract = OpenAlexClient.reconstruct_abstract(raw.get("abstract_inverted_index"))

    oa_id = raw.get("id", "")
    short_id = oa_id.replace("https://openalex.org/", "") if oa_id else oa_id

    return Paper(
        id=short_id or oa_id,
        doi=doi or None,
        title=raw.get("title") or "",
        abstract=abstract,
        year=raw.get("publication_year"),
        venue=venue,
        authors=authors,
        url=url,
        source="openalex",
    )


def _normalize_ss(raw: dict) -> Paper:
    ext_ids = raw.get("externalIds") or {}
    doi = ext_ids.get("DOI")
    if doi:
        doi = doi.lower()

    authors = [a.get("name", "") for a in raw.get("authors", []) if a.get("name")]

    pub_venue = raw.get("publicationVenue") or {}
    venue = pub_venue.get("name") or raw.get("venue")

    return Paper(
        id=raw.get("paperId") or raw.get("corpusId", ""),
        doi=doi or None,
        title=raw.get("title") or "",
        abstract=raw.get("abstract"),
        year=raw.get("year"),
        venue=venue,
        authors=authors,
        url=raw.get("url"),
        source="semantic_scholar",
    )


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    seen_dois: dict[str, Paper] = {}
    result: list[Paper] = []
    no_doi: list[Paper] = []

    for p in papers:
        if p.doi:
            if p.doi not in seen_dois:
                seen_dois[p.doi] = p
                result.append(p)
        else:
            no_doi.append(p)

    seen_titles = {p.title.lower() for p in result}
    for p in no_doi:
        if p.title.lower() not in seen_titles:
            result.append(p)
            seen_titles.add(p.title.lower())

    return result
