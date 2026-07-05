"""Generate the mobile test dashboard into ~/agent-share/cffg-tests.html.

Runs the suite (junitxml), groups tests by M-epic, diffs against the previous
run's snapshot (NEW badges), and renders a self-contained dark/light HTML page
served by the agent-share file browser (port 8080 over Tailscale).

Run after any test-affecting commit:  uv run python tools/dashboard.py
When a new test file appears, add it to EPIC_FILES.
"""

from __future__ import annotations

import datetime
import html
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHARE = Path.home() / "agent-share"
OUT = SHARE / "cffg-tests.html"
STATE = SHARE / ".cffg-dashboard-state.json"

EPIC_FILES = {
    "M0 · Foundations": ["test_hygiene.py"],
    "M1 · Core IR": ["test_ir.py", "test_serialize.py", "test_edits.py", "test_compose.py"],
    "M2 · BP Engine": ["test_engine_tree.py", "test_engine_hmm.py"],
    "M3 · VMP + GFE": ["test_vmp.py", "test_gfe.py"],
}
FILE_TO_EPIC = {f: e for e, fs in EPIC_FILES.items() for f in fs}
EPIC_ORDER = list(EPIC_FILES) + ["Unassigned"]


def run_suite() -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        xml_path = tmp.name
    subprocess.run(
        ["uv", "run", "pytest", "-q", f"--junitxml={xml_path}"],
        cwd=REPO, capture_output=True, text=True,
    )
    tests = []
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        file = (case.get("classname") or "").split(".")[-1] + ".py"
        failure = case.find("failure") is not None or case.find("error") is not None
        tests.append(
            {
                "id": f"{file}::{case.get('name')}",
                "file": file,
                "name": case.get("name"),
                "ms": round(float(case.get("time", 0)) * 1000, 1),
                "ok": not failure,
                "epic": FILE_TO_EPIC.get(file, "Unassigned"),
            }
        )
    return tests


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() or "?"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"previous_ids": [], "history": []}


def trend_svg(history: list[dict]) -> str:
    pts = [h["total"] for h in history][-40:]
    if len(pts) < 2:
        return ""
    w, h_px, pad = 640, 80, 8
    lo, hi = 0, max(pts)
    xs = [pad + i * (w - 2 * pad) / (len(pts) - 1) for i in range(len(pts))]
    ys = [h_px - pad - (v - lo) / (hi - lo or 1) * (h_px - 2 * pad) for v in pts]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--line)">'
        f"<title>{html.escape(hh['when'])} · {hh['total']} tests · {hh['commit']}</title></circle>"
        for x, y, hh in zip(xs, ys, history[-40:])
    )
    return (
        f'<svg viewBox="0 0 {w} {h_px}" preserveAspectRatio="none" role="img" '
        f'aria-label="Total tests per run">'
        f'<path d="{path}" fill="none" stroke="var(--line)" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>{dots}</svg>'
    )


def render(tests: list[dict], new_ids: set[str], history: list[dict], commit: str) -> str:
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    failed = [t for t in tests if not t["ok"]]
    by_epic: dict[str, list[dict]] = {}
    for t in tests:
        by_epic.setdefault(t["epic"], []).append(t)
    max_count = max(len(v) for v in by_epic.values())

    def tile(label, value, accent=""):
        return (
            f'<div class="tile{" " + accent if accent else ""}">'
            f'<div class="tile-v">{value}</div><div class="tile-l">{label}</div></div>'
        )

    status = (
        '<span class="pill ok">✓ all passing</span>'
        if not failed
        else f'<span class="pill bad">✗ {len(failed)} failing</span>'
    )
    sections = []
    for epic in EPIC_ORDER:
        ts = by_epic.get(epic)
        if not ts:
            continue
        n_new = sum(1 for t in ts if t["id"] in new_ids)
        bar_pct = 100.0 * len(ts) / max_count
        rows = "".join(
            '<div class="row">'
            + (
                '<span class="st ok">✓</span>'
                if t["ok"]
                else '<span class="st bad">✗ FAIL</span>'
            )
            + f'<span class="tn">{html.escape(t["name"])}</span>'
            + (f'<span class="new">NEW</span>' if t["id"] in new_ids else "")
            + f'<span class="ms">{t["ms"]:g} ms</span></div>'
            for t in sorted(ts, key=lambda t: (t["file"], t["name"]))
        )
        newtxt = f' · <span class="accent">+{n_new} new</span>' if n_new else ""
        sections.append(
            f'<section><h2>{html.escape(epic)} <span class="count">{len(ts)} tests{newtxt}</span></h2>'
            f'<div class="bar"><div class="fill" style="width:{bar_pct:.0f}%"></div></div>{rows}</section>'
        )

    trend = trend_svg(history)
    trend_block = f'<section><h2>Total tests per run</h2>{trend}</section>' if trend else ""
    n_new_total = sum(1 for t in tests if t["id"] in new_ids)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CFFG tests</title>
<style>
:root {{
  --surface:#fcfcfb; --card:#f2f2ef; --ink:#0b0b0b; --ink2:#52514e; --mut:#8b8a85;
  --line:#2a78d6; --good:#0ca30c; --crit:#d03b3b; --chip:#2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --surface:#1a1a19; --card:#242423; --ink:#fff; --ink2:#c3c2b7; --mut:#8b8a85;
           --line:#3987e5; --good:#0ca30c; --crit:#d03b3b; --chip:#3987e5; }}
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--surface); color:var(--ink);
  font:15px/1.45 -apple-system,system-ui,sans-serif; padding:16px; max-width:760px; margin:0 auto; }}
header {{ margin-bottom:16px; }}
h1 {{ font-size:20px; }} .meta {{ color:var(--ink2); font-size:13px; margin-top:2px; }}
.pill {{ font-size:13px; font-weight:600; padding:2px 10px; border-radius:999px; }}
.pill.ok {{ color:var(--good); border:1.5px solid var(--good); }}
.pill.bad {{ color:var(--crit); border:1.5px solid var(--crit); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px; margin:14px 0; }}
.tile {{ background:var(--card); border-radius:10px; padding:10px 12px; }}
.tile-v {{ font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }}
.tile-l {{ color:var(--ink2); font-size:12px; }}
.tile.bad .tile-v {{ color:var(--crit); }} .tile.new .tile-v {{ color:var(--chip); }}
section {{ background:var(--card); border-radius:10px; padding:12px 14px; margin:10px 0; }}
h2 {{ font-size:15px; margin-bottom:8px; }}
.count {{ color:var(--ink2); font-weight:400; font-size:13px; }} .accent {{ color:var(--chip); }}
.bar {{ height:10px; margin:2px 0 10px; }}
.fill {{ height:10px; background:var(--line); border-radius:0 4px 4px 0; min-width:10px; }}
.row {{ display:flex; align-items:baseline; gap:8px; padding:3px 0;
  font:12.5px ui-monospace,Menlo,monospace; border-top:1px solid var(--surface); }}
.st.ok {{ color:var(--good); }} .st.bad {{ color:var(--crit); font-weight:700; }}
.tn {{ flex:1; word-break:break-word; color:var(--ink); }}
.ms {{ color:var(--mut); font-variant-numeric:tabular-nums; white-space:nowrap; }}
.new {{ color:var(--chip); border:1px solid var(--chip); border-radius:4px;
  font-size:10px; font-weight:700; padding:0 4px; }}
svg {{ width:100%; height:80px; display:block; }}
footer {{ color:var(--mut); font-size:12px; margin:18px 0 8px; }}
</style></head><body>
<header>
  <h1>CFFG test suite {status}</h1>
  <div class="meta">{when} · commit {commit} · {len(tests)} tests</div>
</header>
<div class="tiles">
  {tile("total tests", len(tests))}
  {tile("new this run", n_new_total, "new")}
  {tile("failing", len(failed), "bad" if failed else "")}
  {tile("suite time", f"{sum(t['ms'] for t in tests) / 1000:.1f}s")}
</div>
{trend_block}
{"".join(sections)}
<footer>generated by tools/dashboard.py · regen: uv run python tools/dashboard.py</footer>
</body></html>"""


def main() -> None:
    tests = run_suite()
    if not tests:
        sys.exit("no tests collected — junitxml empty?")
    state = load_state()
    prev = set(state["previous_ids"])
    new_ids = {t["id"] for t in tests} - prev if prev else set()
    commit = git_head()
    state["history"].append(
        {
            "when": datetime.datetime.now().strftime("%m-%d %H:%M"),
            "total": len(tests),
            "failed": sum(1 for t in tests if not t["ok"]),
            "commit": commit,
        }
    )
    state["history"] = state["history"][-120:]
    OUT.write_text(render(tests, new_ids, state["history"], commit))
    state["previous_ids"] = sorted(t["id"] for t in tests)
    STATE.write_text(json.dumps(state))
    failed = sum(1 for t in tests if not t["ok"])
    print(f"{OUT}: {len(tests)} tests, {failed} failing, {len(new_ids)} new")


if __name__ == "__main__":
    main()
