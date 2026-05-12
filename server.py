"""
python server.py 실행 후 http://localhost:8765 접속
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"

_lock = threading.Lock()
_state: dict = {"running": False, "logs": [], "done": False, "error": None, "proc": None}


def _add_log(line: str) -> None:
    with _lock:
        _state["logs"].append(line)


def _run(params: dict) -> None:
    DATA.mkdir(exist_ok=True)
    (DATA / "pipeline.log").write_text("", encoding="utf-8")

    categories = params.get("categories", "").strip()
    cmd = [
        sys.executable, str(BASE / "orchestrator.py"),
        "--keyword", params["keyword"],
        "--years", f"{params['year_from']}-{params['year_to']}",
        "--max-papers", str(params["max_papers"]),
        "--output-dir", str(DATA),
        "--index-filter", params.get("index_filter", "all"),
    ]
    if categories:
        cmd += ["--categories", categories]

    env = os.environ.copy()

    with _lock:
        _state["logs"] = [f"[server] 파이프라인 시작: {' '.join(cmd[2:])}"]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(BASE),
        )
        with _lock:
            _state["proc"] = proc

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _add_log(line)

        proc.wait()

        if proc.returncode == 0:
            _generate_report()
            with _lock:
                _state["done"] = True
        else:
            with _lock:
                _state["error"] = f"파이프라인 종료 코드: {proc.returncode}"
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
    finally:
        with _lock:
            _state["running"] = False
            _state["proc"] = None


def _generate_report() -> None:
    results_path = DATA / "final_results.json"
    state_path = DATA / "pipeline_state.json"
    if not results_path.exists():
        return

    papers_raw = json.loads(results_path.read_text(encoding="utf-8"))
    pipeline_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    keyword = pipeline_state.get("keyword", "")
    updated = pipeline_state.get("updated_at", "")[:10]

    by_cat: dict[str, list] = {}
    for p in papers_raw:
        for c in (p.get("categories") or ["미분류"]):
            by_cat.setdefault(c, []).append(p)

    def esc(s: str) -> str:
        return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    sections = ""
    for cat in sorted(by_cat):
        ps = by_cat[cat]
        rows = ""
        for p in ps:
            url = p.get("url") or (f"https://doi.org/{p['doi']}" if p.get("doi") else "#")
            summary_html = (
                esc(p.get("summary") or "")
                .replace("\\n", "<br>")
                .replace("Purpose:", "<b>Purpose:</b>")
                .replace("Method:", "<b>Method:</b>")
                .replace("Results:", "<b>Results:</b>")
            )
            rows += (
                f"<tr>"
                f"<td><a href='{esc(url)}' target='_blank'>{esc(p.get('title',''))}</a></td>"
                f"<td style='white-space:nowrap'>{p.get('year','—')}</td>"
                f"<td style='font-size:12px;color:#555'>{esc(p.get('venue','—'))}</td>"
                f"<td class='summary'>{summary_html}</td>"
                f"<td class='apa'>{esc(p.get('apa_ref',''))}</td>"
                f"</tr>"
            )
        sections += (
            f"<section>"
            f"<h2>{esc(cat)} <span class='count'>{len(ps)}편</span></h2>"
            f"<table><thead><tr><th>제목</th><th>연도</th><th>저널</th><th>요약</th><th>APA</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            f"</section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Report — {esc(keyword)}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f7;color:#1d1d1f;font-size:14px}}
header{{background:#1d1d1f;color:#fff;padding:20px 32px}}
header h1{{font-size:20px;font-weight:700}}
header p{{font-size:13px;color:#aeaeb2;margin-top:4px}}
main{{max-width:1200px;margin:0 auto;padding:24px 20px}}
.meta{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}}
.chip{{background:#fff;border-radius:20px;padding:6px 14px;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.chip b{{color:#1d4ed8}}
section{{background:#fff;border-radius:12px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
section h2{{font-size:16px;font-weight:700;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #f2f2f7}}
.count{{font-size:13px;font-weight:400;color:#6e6e73;margin-left:6px}}
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;padding:8px 10px;border-bottom:2px solid #f2f2f7;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#6e6e73;white-space:nowrap}}
td{{padding:10px;border-bottom:1px solid #f9f9f9;vertical-align:top}}
tr:last-child td{{border-bottom:none}}
td a{{color:#1d4ed8;text-decoration:none;font-weight:500}}
td a:hover{{text-decoration:underline}}
.summary{{font-size:13px;color:#374151;line-height:1.7;min-width:280px}}
.apa{{font-size:11px;color:#9ca3af;min-width:180px}}
</style></head><body>
<header>
  <h1>Research Report</h1>
  <p>키워드: {esc(keyword)} &nbsp;|&nbsp; 총 {len(papers_raw)}편 &nbsp;|&nbsp; 생성: {updated}</p>
</header>
<main>
  <div class="meta">
    <div class="chip">총 논문 <b>{len(papers_raw)}</b>편</div>
    <div class="chip">카테고리 <b>{len(by_cat)}</b>개</div>
  </div>
  {sections}
</main></body></html>"""
    (DATA / "report.html").write_text(html, encoding="utf-8")
    _add_log("[server] 보고서 생성 완료 → data/report.html")


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")

    def _json(self, data: object, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _html_file(self, path: Path) -> None:
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        p = self.path.split("?")[0]

        if p in ("/", "/app.html"):
            self._html_file(BASE / "app.html")

        elif p == "/api/status":
            sp = DATA / "pipeline_state.json"
            data = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"stage": "idle"}
            with _lock:
                data["server_running"] = _state["running"]
                data["server_done"] = _state["done"]
                data["server_error"] = _state["error"]
            self._json(data)

        elif p == "/api/logs":
            with _lock:
                self._json({"logs": list(_state["logs"]), "running": _state["running"], "done": _state["done"], "error": _state["error"]})

        elif p == "/api/results":
            rp = DATA / "final_results.json"
            self._json(json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else [])

        elif p == "/api/report":
            rp = DATA / "report.html"
            if rp.exists():
                self._html_file(rp)
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(length))

            if not params.get("keyword", "").strip():
                self._json({"error": "키워드를 입력해주세요."}, 400)
                return

            if not os.environ.get("ANTHROPIC_API_KEY"):
                self._json({"error": "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다."}, 400)
                return

            with _lock:
                if _state["running"]:
                    self._json({"error": "이미 파이프라인이 실행 중입니다."}, 409)
                    return
                _state["running"] = True
                _state["done"] = False
                _state["error"] = None

            threading.Thread(target=_run, args=(params,), daemon=True).start()
            self._json({"status": "started"})

        elif self.path == "/api/stop":
            with _lock:
                proc = _state.get("proc")
            if proc:
                proc.terminate()
                self._json({"status": "stopped"})
            else:
                self._json({"status": "not_running"})

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    port = 8765
    DATA.mkdir(exist_ok=True)
    httpd = http.server.HTTPServer(("", port), Handler)
    url = f"http://localhost:{port}"
    print(f"서버 시작: {url}")
    print("브라우저에서 위 주소를 열거나 아래 명령으로 열어주세요:")
    print(f"  open {url}")
    print("종료: Ctrl+C")
    try:
        import subprocess as _sp
        _sp.Popen(["open", url])
    except Exception:
        pass
    httpd.serve_forever()
