from __future__ import annotations

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    id: str
    doi: str | None = None
    title: str
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] = Field(default_factory=list)
    url: str | None = None
    source: Literal["openalex", "semantic_scholar"]
    summary: str | None = None
    categories: list[str] = Field(default_factory=list)
    apa_ref: str | None = None


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class PipelineState(BaseModel):
    run_id: str
    keyword: str
    stage: Literal["search", "summary", "category", "validation", "complete"]
    stage_status: StageStatus
    attempt: dict[str, int] = Field(default_factory=dict)
    paper_count: int = 0
    papers_processed: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: str
    updated_at: str


def papers_from_json(path: Path) -> list[Paper]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Paper.model_validate(p) for p in data]


def papers_to_json(papers: list[Paper], path: Path) -> None:
    _atomic_write(path, json.dumps([p.model_dump() for p in papers], ensure_ascii=False, indent=2))


def load_state(path: Path) -> PipelineState:
    return PipelineState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_state(state: PipelineState, path: Path) -> None:
    _atomic_write(path, state.model_dump_json(indent=2))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise
