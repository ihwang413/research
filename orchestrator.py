from __future__ import annotations

import argparse
import asyncio
import atexit
import http.server
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from agents import category_agent, search_agent, summary_agent, validation_agent
from utils.data_models import (
    Paper,
    PipelineState,
    StageStatus,
    papers_to_json,
    save_state,
)

MAX_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic paper research pipeline")
    parser.add_argument("--keyword", required=True, help="Search keyword or phrase")
    parser.add_argument("--years", default="2020-2024", help="Year range, e.g. 2020-2024")
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--categories", default="", help="Comma-separated category names")
    parser.add_argument("--index-filter", default="all",
                        choices=["all", "scopus", "esci", "ssci_sci"],
                        help="Journal index filter: all / scopus / esci / ssci_sci")
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    def expand_env(match: re.Match) -> str:
        var = match.group(1)
        return os.environ.get(var, "")

    raw = re.sub(r"\$\{([^}]+)\}", expand_env, raw)
    return yaml.safe_load(raw)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_state(args: argparse.Namespace) -> PipelineState:
    return PipelineState(
        run_id=str(uuid.uuid4()),
        keyword=args.keyword,
        stage="search",
        stage_status=StageStatus.PENDING,
        attempt={"search": 0, "summary": 0, "category": 0, "validation": 0},
        paper_count=0,
        papers_processed=0,
        errors=[],
        started_at=_now(),
        updated_at=_now(),
    )


async def _run_stage(
    name: str,
    state: PipelineState,
    output_dir: Path,
    fn: Callable,
    save_fn: Callable[[Any], None],
) -> Any:
    while state.attempt[name] < MAX_ATTEMPTS:
        state.stage = name  # type: ignore[assignment]
        state.stage_status = StageStatus.RUNNING
        state.updated_at = _now()
        save_state(state, output_dir / "pipeline_state.json")
        print(f"[{name}] Starting (attempt {state.attempt[name] + 1}/{MAX_ATTEMPTS})", flush=True)

        try:
            result = await fn()
            state.attempt[name] += 1
            state.stage_status = StageStatus.DONE
            state.updated_at = _now()
            save_state(state, output_dir / "pipeline_state.json")
            save_fn(result)
            return result
        except Exception as e:
            state.attempt[name] += 1
            state.errors.append(f"{name}: {e}")
            state.stage_status = StageStatus.FAILED
            state.updated_at = _now()
            save_state(state, output_dir / "pipeline_state.json")
            print(f"[{name}] FAILED: {e}", flush=True)
            if state.attempt[name] >= MAX_ATTEMPTS:
                raise SystemExit(f"Stage '{name}' failed after {MAX_ATTEMPTS} attempts") from e

    raise SystemExit(f"Stage '{name}' exceeded max attempts")


def _launch_dashboard(output_dir: Path) -> None:
    port = 8765

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_dir.parent), **kwargs)

        def log_message(self, format, *args):
            pass

    def serve():
        with http.server.HTTPServer(("", port), Handler) as httpd:
            httpd.serve_forever()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    print(f"Dashboard: http://localhost:{port}/{output_dir.name}/dashboard.html", flush=True)


async def run_pipeline(args: argparse.Namespace, config: dict) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dashboard:
        import shutil
        src = Path(__file__).parent / "dashboard.html"
        dst = output_dir / "dashboard.html"
        if src.exists():
            shutil.copy2(src, dst)
        _launch_dashboard(output_dir)

    year_from, year_to = _parse_years(args.years)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    state = _init_state(args)
    save_state(state, output_dir / "pipeline_state.json")

    index_filter = getattr(args, "index_filter", "all")
    papers: list[Paper] = await _run_stage(
        "search", state, output_dir,
        fn=lambda: search_agent.run(
            config, args.keyword, year_from, year_to, args.max_papers,
            index_filter=index_filter,
        ),
        save_fn=lambda p: papers_to_json(p, output_dir / "raw_papers.json"),
    )
    state.paper_count = len(papers)

    papers = await _run_stage(
        "summary", state, output_dir,
        fn=lambda: summary_agent.run(papers, config),
        save_fn=lambda p: papers_to_json(p, output_dir / "summarized_papers.json"),
    )

    papers = await _run_stage(
        "category", state, output_dir,
        fn=lambda: asyncio.get_event_loop().run_in_executor(
            None, category_agent.run, papers, categories, config
        ),
        save_fn=lambda p: papers_to_json(p, output_dir / "categorized_papers.json"),
    )

    validation_attempts = 0
    while validation_attempts < MAX_ATTEMPTS:
        state.stage = "validation"  # type: ignore[assignment]
        state.stage_status = StageStatus.RUNNING
        state.updated_at = _now()
        save_state(state, output_dir / "pipeline_state.json")

        result = validation_agent.run(papers, state)
        state.attempt["validation"] += 1

        if result["passed"]:
            break

        failed_stages = validation_agent.map_failures_to_stages(result["failures"])
        print(f"[validation] Failed checks: {result['failures']}. Re-running: {failed_stages}", flush=True)
        state.errors.append(f"validation: {result['failures']}")
        validation_attempts += 1

        if validation_attempts >= MAX_ATTEMPTS:
            print("[validation] Max validation retries reached. Saving partial results.", flush=True)
            break

        for stage_name in failed_stages:
            if stage_name == "search":
                papers = await _run_stage(
                    "search", state, output_dir,
                    fn=lambda: search_agent.run(config, args.keyword, year_from, year_to, args.max_papers),
                    save_fn=lambda p: papers_to_json(p, output_dir / "raw_papers.json"),
                )
            elif stage_name == "summary":
                papers = await _run_stage(
                    "summary", state, output_dir,
                    fn=lambda: summary_agent.run(papers, config),
                    save_fn=lambda p: papers_to_json(p, output_dir / "summarized_papers.json"),
                )
            elif stage_name == "category":
                papers = await _run_stage(
                    "category", state, output_dir,
                    fn=lambda: asyncio.get_event_loop().run_in_executor(
                        None, category_agent.run, papers, categories, config
                    ),
                    save_fn=lambda p: papers_to_json(p, output_dir / "categorized_papers.json"),
                )

    papers_to_json(papers, output_dir / "final_results.json")
    state.stage = "complete"  # type: ignore[assignment]
    state.stage_status = StageStatus.DONE
    state.papers_processed = len(papers)
    state.updated_at = _now()
    save_state(state, output_dir / "pipeline_state.json")

    print(f"\n[done] {len(papers)} papers saved to {output_dir}/final_results.json", flush=True)
    _print_summary(papers)


def _parse_years(years_str: str) -> tuple[int, int]:
    parts = years_str.split("-")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    year = int(parts[0])
    return year, year


def _print_summary(papers: list[Paper]) -> None:
    from collections import Counter
    cats = Counter(c for p in papers for c in p.categories)
    print("\n--- Category Summary ---")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count} papers")


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {args.config}")
    except KeyError as e:
        sys.exit(f"Missing required config key: {e}")

    asyncio.run(run_pipeline(args, config))


if __name__ == "__main__":
    main()
