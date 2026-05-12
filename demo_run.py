"""
데모 실행 스크립트 - Claude API 없이 전체 파이프라인 흐름 시연
검색은 실제 OpenAlex API, 요약은 mock 데이터 사용
"""
from __future__ import annotations

import asyncio
import json
import sys
sys.path.insert(0, '.')

from pathlib import Path
from agents import search_agent, validation_agent
from agents.category_agent import run as cat_run, _generate_apa
from agents.summary_agent import _validate_summary
from utils.data_models import Paper, PipelineState, StageStatus, papers_to_json, save_state
import uuid
from datetime import datetime, timezone

CONFIG = {
    'api': {'openalex_email': '', 'semantic_scholar_key': None, 'anthropic_key': None},
    'search': {'per_page': 10, 'abstract_missing_warn_threshold': 0.30},
    'summary': {'model': 'claude-sonnet-4-6', 'max_tokens': 512, 'batch_size': 10},
}

MOCK_SUMMARIES = [
    "Purpose: To review how AI chatbots are applied in higher education and their effectiveness.\nMethod: Systematic literature review of 57 studies published between 2017 and 2022.\nResults: AI chatbots improved student engagement and provided 24/7 support, but lacked emotional intelligence.",
    "Purpose: To map AI technology applications in STEM education over a decade.\nMethod: Systematic review of 100 papers from 2011 to 2021 using PRISMA guidelines.\nResults: AI-STEM integration improved learning outcomes, especially in personalized learning and intelligent tutoring.",
    "Purpose: To examine how conversational AI collaborates with human teachers in language education.\nMethod: Systematic review of 34 empirical studies from 2010 to 2022 using thematic analysis.\nResults: Human-AI collaboration enhanced speaking practice opportunities but required teacher scaffolding for effectiveness.",
    "Purpose: To investigate FATE principles (Fairness, Accountability, Transparency, Ethics) in AI applied to higher education.\nMethod: Systematic review of 138 articles using bibliometric and content analysis.\nResults: Most AI systems lacked transparency mechanisms; ethical frameworks for higher education AI remain underdeveloped.",
    "Purpose: To synthesize empirical evidence on AI use in online learning and distance education contexts.\nMethod: Systematic review of 52 empirical studies conducted between 2010 and 2022.\nResults: AI tools significantly enhanced learner engagement and adaptive feedback in distance education settings.",
]


async def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    state = PipelineState(
        run_id=run_id,
        keyword="AI in education systematic review",
        stage="search",
        stage_status=StageStatus.PENDING,
        attempt={"search": 0, "summary": 0, "category": 0, "validation": 0},
        paper_count=0, papers_processed=0, errors=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    print("=" * 60)
    print("  학술논문 검색·요약 멀티에이전트 시스템 (데모)")
    print(f"  키워드: {state.keyword}")
    print("=" * 60)

    # ── Stage 1: Search ──────────────────────────────────────────
    print("\n[1/4] 검색 에이전트 실행 중...")
    state.stage = "search"
    state.stage_status = StageStatus.RUNNING
    state.attempt["search"] = 1
    save_state(state, output_dir / "pipeline_state.json")

    papers = await search_agent.run(CONFIG, state.keyword, 2022, 2024, 5)
    papers_to_json(papers, output_dir / "raw_papers.json")
    state.paper_count = len(papers)
    state.stage_status = StageStatus.DONE
    save_state(state, output_dir / "pipeline_state.json")

    print(f"  → {len(papers)}개 논문 검색 완료\n")
    for i, p in enumerate(papers):
        print(f"  [{i+1}] {p.title[:70]}")
        print(f"       {p.year} | {p.venue or 'N/A'} | DOI: {p.doi or 'N/A'}")

    # ── Stage 2: Summary (mock) ──────────────────────────────────
    print("\n[2/4] 요약 에이전트 실행 중... (데모: mock 요약 사용)")
    state.stage = "summary"
    state.stage_status = StageStatus.RUNNING
    state.attempt["summary"] = 1
    save_state(state, output_dir / "pipeline_state.json")

    for i, p in enumerate(papers):
        summary = MOCK_SUMMARIES[i % len(MOCK_SUMMARIES)]
        papers[i] = p.model_copy(update={"summary": summary})

    papers_to_json(papers, output_dir / "summarized_papers.json")
    state.stage_status = StageStatus.DONE
    save_state(state, output_dir / "pipeline_state.json")

    print(f"  → {len(papers)}개 논문 요약 완료")
    print(f"\n  예시 요약 (논문 1):")
    for line in papers[0].summary.split("\n"):
        print(f"    {line}")

    # ── Stage 3: Category (mock - no Claude needed) ─────────────
    print("\n[3/4] 범주화 에이전트 실행 중... (mock 범주 할당)")
    state.stage = "category"
    state.stage_status = StageStatus.RUNNING
    state.attempt["category"] = 1
    save_state(state, output_dir / "pipeline_state.json")

    category_map = ["기술개발", "기술개발", "효과성연구", "윤리및정책", "효과성연구"]
    for i, p in enumerate(papers):
        cat = category_map[i % len(category_map)]
        apa = _generate_apa(p)
        papers[i] = p.model_copy(update={"categories": [cat], "apa_ref": apa})

    papers.sort(key=lambda p: (p.categories[0], -(p.year or 0)))
    papers_to_json(papers, output_dir / "categorized_papers.json")
    state.stage_status = StageStatus.DONE
    save_state(state, output_dir / "pipeline_state.json")

    print(f"  → {len(papers)}개 논문 범주화 완료")
    from collections import Counter
    cats = Counter(c for p in papers for c in p.categories)
    for cat, cnt in sorted(cats.items()):
        print(f"    {cat}: {cnt}편")

    # ── Stage 4: Validation ──────────────────────────────────────
    print("\n[4/4] 검증 에이전트 실행 중...")
    state.stage = "validation"
    state.stage_status = StageStatus.RUNNING
    state.attempt["validation"] = 1
    save_state(state, output_dir / "pipeline_state.json")

    result = validation_agent.run(papers, state)

    papers_to_json(papers, output_dir / "final_results.json")
    state.stage = "complete"
    state.stage_status = StageStatus.DONE
    state.papers_processed = len(papers)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    save_state(state, output_dir / "pipeline_state.json")

    # ── Final Output ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  최종 결과")
    print("=" * 60)

    current_cat = None
    for p in papers:
        cat = p.categories[0] if p.categories else "Uncategorized"
        if cat != current_cat:
            current_cat = cat
            print(f"\n  ▶ {cat}")
            print(f"  {'─' * 50}")

        print(f"\n  [{p.year}] {p.title}")
        print(f"  저널: {p.venue or 'N/A'}")
        if p.summary:
            for line in p.summary.split("\n"):
                print(f"    {line}")
        print(f"  APA: {p.apa_ref}")

    print(f"\n{'=' * 60}")
    print(f"  검증 결과: {'PASS' if result['passed'] else 'FAIL'}")
    for check, ok in result['checks'].items():
        print(f"    {'✓' if ok else '✗'} {check}")
    print(f"\n  저장 위치: {output_dir}/final_results.json")

    report_path = output_dir / "report.html"
    _generate_report(papers, result, state, report_path)
    print(f"  보고서: {report_path.resolve()}")
    print("=" * 60)

    import subprocess
    subprocess.Popen(["open", str(report_path.resolve())])


def _generate_report(papers, result, state, path: Path) -> None:
    import json as _json
    from collections import Counter

    cats = sorted({c for p in papers for c in p.categories})
    by_cat: dict[str, list] = {c: [] for c in cats}
    for p in papers:
        for c in p.categories:
            by_cat[c].append(p)

    checks_html = "".join(
        f'<li class="{"ok" if ok else "fail"}">{"✓" if ok else "✗"} {name}</li>'
        for name, ok in result["checks"].items()
    )

    sections_html = ""
    for cat, ps in sorted(by_cat.items()):
        rows = ""
        for p in ps:
            summary_lines = (p.summary or "").replace("\n", "<br>")
            url = p.url or (f"https://doi.org/{p.doi}" if p.doi else "#")
            rows += f"""
            <tr>
              <td><a href="{url}" target="_blank">{_esc(p.title)}</a></td>
              <td>{p.year or "—"}</td>
              <td>{_esc(p.venue or "—")}</td>
              <td class="summary">{summary_lines}</td>
              <td class="apa">{_esc(p.apa_ref or "")}</td>
            </tr>"""
        sections_html += f"""
        <section>
          <h2>{_esc(cat)} <span class="count">{len(ps)}편</span></h2>
          <table>
            <thead><tr><th>제목</th><th>연도</th><th>저널</th><th>요약</th><th>APA</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Report — {_esc(state.keyword)}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f7;color:#1d1d1f}}
header{{background:#1d1d1f;color:#fff;padding:20px 32px}}
header h1{{font-size:20px;font-weight:700}}
header p{{font-size:13px;color:#aeaeb2;margin-top:4px}}
main{{max-width:1100px;margin:0 auto;padding:28px 20px}}
.meta{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
.chip{{background:#fff;border-radius:20px;padding:6px 14px;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.chip b{{color:#1d4ed8}}
.checks{{background:#fff;border-radius:12px;padding:16px 20px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.checks h3{{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#6e6e73;margin-bottom:10px}}
.checks ul{{list-style:none;display:flex;flex-wrap:wrap;gap:8px}}
.checks li{{font-size:13px;padding:3px 10px;border-radius:20px}}
.checks li.ok{{background:#d1fae5;color:#065f46}}
.checks li.fail{{background:#fee2e2;color:#991b1b}}
section{{background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
section h2{{font-size:16px;font-weight:700;margin-bottom:14px;color:#1d1d1f}}
section h2 .count{{font-size:13px;font-weight:400;color:#6e6e73;margin-left:6px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:8px 10px;border-bottom:2px solid #f2f2f7;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#6e6e73}}
td{{padding:10px;border-bottom:1px solid #f9f9f9;vertical-align:top}}
tr:last-child td{{border-bottom:none}}
td a{{color:#1d4ed8;text-decoration:none;font-weight:500}}
td a:hover{{text-decoration:underline}}
.summary{{font-size:12px;color:#374151;line-height:1.6}}
.apa{{font-size:11px;color:#9ca3af}}
</style>
</head>
<body>
<header>
  <h1>Research Report</h1>
  <p>키워드: {_esc(state.keyword)} &nbsp;|&nbsp; 생성: {state.updated_at[:10]}</p>
</header>
<main>
  <div class="meta">
    <div class="chip">논문 수 <b>{len(papers)}</b></div>
    <div class="chip">카테고리 <b>{len(cats)}</b></div>
    <div class="chip">검증 <b>{"PASS ✓" if result["passed"] else "FAIL ✗"}</b></div>
    <div class="chip">Run ID <b>{state.run_id[:8]}</b></div>
  </div>
  <div class="checks">
    <h3>품질 검증 결과</h3>
    <ul>{checks_html}</ul>
  </div>
  {sections_html}
</main>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


asyncio.run(main())
