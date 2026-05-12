from __future__ import annotations

import asyncio

import httpx


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    headers: dict | None = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> dict:
    for attempt in range(max_attempts):
        try:
            resp = await client.get(url, params=params, headers=headers or {})
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", base_delay * (2 ** attempt)))
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if attempt < max_attempts - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))
            else:
                raise RuntimeError(f"HTTP {e.response.status_code} after {max_attempts} attempts: {url}") from e
        except httpx.RequestError as e:
            if attempt < max_attempts - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))
            else:
                raise RuntimeError(f"Request failed after {max_attempts} attempts: {url}") from e
    raise RuntimeError(f"Max retry attempts exceeded: {url}")


class OpenAlexClient:
    BASE = "https://api.openalex.org"

    def __init__(self, client: httpx.AsyncClient, email: str | None = None):
        self._client = client
        self._email = email

    def _headers(self) -> dict:
        if self._email:
            return {"User-Agent": f"research_agent/1.0 (mailto:{self._email})"}
        return {}

    async def search(
        self,
        query: str,
        year_from: int,
        year_to: int,
        language: str,
        per_page: int,
        cursor: str = "*",
        extra_filter: str = "",
    ) -> tuple[list[dict], str | None]:
        filters = (
            f"publication_year:>{year_from - 1},"
            f"publication_year:<{year_to + 1},"
            f"language:{language}"
        )
        if extra_filter:
            filters += f",{extra_filter}"
        params = {
            "search": query,
            "filter": filters,
            "per-page": per_page,
            "cursor": cursor,
            "select": "id,doi,title,abstract_inverted_index,publication_year,primary_location,authorships,open_access",
        }
        data = await get_with_retry(self._client, f"{self.BASE}/works", params, self._headers())
        works = data.get("results", [])
        meta = data.get("meta", {})
        next_cursor = meta.get("next_cursor")
        return works, next_cursor

    async def batch_source_hindex(self, source_ids: list[str]) -> dict[str, int]:
        """OpenAlex source IDs → h_index 매핑. 100개씩 배치 처리."""
        if not source_ids:
            return {}
        short = [sid.replace("https://openalex.org/", "") for sid in source_ids]
        result: dict[str, int] = {}
        for i in range(0, len(short), 100):
            batch = short[i : i + 100]
            params = {
                "filter": "ids.openalex:" + "|".join(batch),
                "per-page": len(batch),
                "select": "id,h_index",
            }
            data = await get_with_retry(self._client, f"{self.BASE}/sources", params, self._headers())
            for src in data.get("results", []):
                result[src["id"]] = src.get("h_index") or 0
        return result

    @staticmethod
    def reconstruct_abstract(inverted_index: dict | None) -> str | None:
        if not inverted_index:
            return None
        positions: list[tuple[int, str]] = []
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions.append((pos, word))
        if not positions:
            return None
        positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in positions)


class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,year,authors,venue,externalIds,url,publicationVenue"

    def __init__(self, client: httpx.AsyncClient, api_key: str | None = None):
        self._client = client
        self._api_key = api_key
        self._unauthenticated = api_key is None

    def _headers(self) -> dict:
        if self._api_key:
            return {"x-api-key": self._api_key}
        return {}

    async def search(
        self,
        query: str,
        year_from: int,
        year_to: int,
        limit: int,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        if self._unauthenticated:
            await asyncio.sleep(1.1)
        params = {
            "query": query,
            "fields": self.FIELDS,
            "year": f"{year_from}-{year_to}",
            "limit": min(limit, 100),
            "offset": offset,
        }
        data = await get_with_retry(self._client, f"{self.BASE}/paper/search", params, self._headers())
        papers = data.get("data", [])
        total = data.get("total", 0)
        return papers, total
