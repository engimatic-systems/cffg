"""Generate the mobile test dashboard and tracked suite-runtime history.

Consumes a completed default-suite JUnit report, groups tests by M-epic, diffs
against the previous run's snapshot (NEW badges), and renders a self-contained
dark/light HTML page served by the agent-share file browser (port 8080 over
Tailscale). ``--run-suite`` remains the one-command standalone path; it runs
pytest exactly once with live output, renders that report into
``~/agent-share/cffg-tests.html``, and records measured wall time in
``tools/test-suite-runtimes.json`` so the next agent can budget its wait.

Tests that render diagrams record SVGs into tests/.artifacts/ (gitignored)
via the `viz_gallery` fixture; those rows become twirl-down galleries here,
with images copied to ~/agent-share/cffg-viz/ so the page works remotely.

After an existing default run:
  uv run python tools/dashboard.py --junitxml /tmp/cffg-tests.xml
Standalone short run + refresh:
  uv run python tools/dashboard.py --run-suite
When a new test file appears, add it to EPIC_FILES.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHARE = Path.home() / "agent-share"
OUT = SHARE / "cffg-tests.html"
STATE = SHARE / ".cffg-dashboard-state.json"
RUNTIMES = REPO / "tools" / "test-suite-runtimes.json"
ARTIFACTS = REPO / "tests" / ".artifacts"
GALLERY = SHARE / "cffg-viz"
RUNTIME_HISTORY_LIMIT = 3
SUITE_SELECTION = "not slow"

EPIC_FILES = {
    "M0 · Foundations": ["test_hygiene.py", "test_core_imports.py", "test_moment_matching_schema.py"],
    "M1 · Core IR": ["test_ir.py", "test_serialize.py", "test_edits.py", "test_compose.py", "test_inits.py", "test_compiler_contracts.py", "test_compiled.py", "test_topology.py", "test_constraint_details.py", "test_specialized.py", "test_registered.py", "test_resolution.py", "test_assembly.py"],
    "M2 · BP Engine": ["test_engine_tree.py", "test_engine_hmm.py", "test_schedule.py", "test_discrete_kernels.py"],
    "M3 · VMP + GFE": [
        "test_vmp.py",
        "test_gfe.py",
        "test_dirichlet_param_edges.py",
        "test_mu3_derivation.py",
        "test_mu3.py",
        "test_goal_hyperprior.py",
        "test_shared_goals.py",
        "test_factored_rollout.py",
        "test_outer_gfe_formalism.py",
    ],
    "M4 · Epistemic behavior": [
        "test_tmaze.py",
        "test_tmaze_learning.py",
        "test_behavioral_epistemics.py",
    ],
    "M5 · Runtime": [
        "test_engine_contracts.py",
        "test_exact_colony.py",
        "test_failure_facts.py",
        "test_failure_transport.py",
        "test_failure_diagnostics.py",
        "test_outer_failures.py",
        "test_equality_cavities.py",
        "test_mixture_collapse.py",
        "test_mixture_rank.py",
        "test_mixture_evidence.py",
        "test_hybrid_planning.py",
        "test_cstr_planning.py",
        "test_cstr_envelope.py",
        "test_gaussian_readiness.py",
        "test_adaptive_startup.py",
        "test_outer_coordinates.py",
        "test_execution.py",
        "test_execution_specialized.py",
        "test_schedule_program.py",
        "test_lifecycle.py",
        "test_iteration.py",
        "test_run_strategies.py",
        "test_ledger.py",
        "test_readout_owners.py",
        "test_extension_ownership.py",
        "test_colony_failures.py",
        "test_constraint_data.py",
        "test_constraint_reconstruction.py",
        "test_agent.py",
        "test_agent_mu3.py",
        "test_agent_spec.py",
        "test_constraint_flip.py",
        "test_colony.py",
        "test_public_api.py",
        "test_runtime_status.py",
    ],
    "M6 · Tooling": [
        "test_bundle.py",
        "test_bundle_learning.py",
        "test_bundle_resume.py",
        "test_dashboard.py",
        "test_estimate.py",
        "test_edit_codec.py",
        "test_ticket_dashboard.py",
        "test_ticket_rewrite_scheduler.py",
    ],
    "M8 · Plotting": [
        "test_viz_diagram.py",
        "test_viz_compress.py",
        "test_viz_layout.py",
        "test_viz_svg.py",
    ],
    "M9 · Gaussian": [
        "test_gauss_rules.py", "test_kalman.py", "test_gaussian_kernels.py",
        "test_first_order_precision.py",
        "test_nonlinear_rules.py", "test_nonlinear_schema.py", "test_nonlinear_engine.py",
        "test_neural.py",
        "test_fixed_point_gradients.py",
        "test_neural_training.py",
        "test_neural_gfe.py",
        "test_amortization.py",
        "test_gaussian_kernel_schedules.py",
    ],
    "L1 · B-learning": ["test_b_learning.py", "test_learnables.py", "test_learning_schema.py"],
    "L2 · Dynamics curiosity": ["test_b_epistemics.py"],
    "M13 · Exp. families": [
        "test_ef.py",
        "test_ef_properties.py",
        "test_ef_rules.py",
        "test_ef_engine.py",
        "test_kernels.py",
    ],
    "M15 · Learning (Gaussian)": [
        "test_conj.py",
        "test_gauss_vmp.py",
        "test_params.py",
        "test_gaussian_learnables.py",
        "test_hybrid_learnables.py",
    ],
    "M16 · Plates": ["test_plates.py"],
    "M17 · Constraints": ["test_csp.py"],
    "M14 · Hybrid": [
        "test_mixg.py",
        "test_swkf_reference.py",
        "test_hybrid_engine.py",
        "test_hybrid_readouts.py",
        "test_switching_kf.py",
    ],
    "R1 · Hybrid epistemics": [
        "test_hybrid_epistemics.py",
        "test_hybrid_gfe.py",
        "test_recurrence_learning_numerics.py",
        "test_recurrence_checkpoint.py",
        "test_recurrence_link.py",
        "test_structured_recurrence.py",
        "test_regime_pair_derivation.py",
        "test_hybrid_separator_derivation.py",
        "test_structured_recurrence_engine.py",
        "test_structured_recurrence_learning.py",
        "test_rslds_fit.py",
    ],
    "SG · Glucose": ["test_glucose.py"],
}
FILE_TO_EPIC = {f: e for e, fs in EPIC_FILES.items() for f in fs}
EPIC_ORDER = list(EPIC_FILES) + ["Unassigned"]


class DashboardError(ValueError):
    """A JUnit report cannot truthfully certify the current default suite."""


def run_suite() -> tuple[Path, int, float]:
    """Run the default short suite; return its report, status, and wall time."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        xml_path = Path(tmp.name)
    started = time.monotonic()
    result = subprocess.run(
        ["uv", "run", "pytest", "-q", f"--junitxml={xml_path}"], cwd=REPO
    )
    return xml_path, result.returncode, time.monotonic() - started


def repository_test_files(repo: Path = REPO) -> set[str]:
    return {path.name for path in (repo / "tests").glob("test_*.py")}


def newest_test_input(
    repo: Path = REPO, *, exclude: Path | None = None
) -> tuple[float, Path]:
    """Newest mtime among files whose contents can affect the default result."""
    candidates = [repo / "pyproject.toml", repo / "uv.lock"]
    for directory in (repo / "src", repo / "tests", repo / "tools"):
        candidates.extend(directory.rglob("*.py"))
    existing = [
        path for path in candidates
        if path.is_file() and (exclude is None or path.resolve() != exclude.resolve())
    ]
    if not existing:
        raise DashboardError(f"no test inputs found under {repo}")
    newest = max(existing, key=lambda path: path.stat().st_mtime)
    return newest.stat().st_mtime, newest


def load_junit(
    xml_path: Path,
    *,
    expected_files: set[str] | None = None,
    newest_input: tuple[float, Path] | None = None,
) -> list[dict]:
    """Parse and validate a default-suite JUnit report.

    Completeness is checked without spawning pytest: every repository test
    module must appear, declared suite counts must match the testcase records,
    and the report must be newer than the code/test/tool inputs it certifies.
    """
    if not xml_path.is_file():
        raise DashboardError(f"JUnit report does not exist: {xml_path}")
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as error:
        raise DashboardError(f"malformed JUnit report {xml_path}: {error}") from error

    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise DashboardError(f"JUnit report has no testsuite: {xml_path}")
    cases = list(root.iter("testcase"))
    if not cases:
        raise DashboardError(f"JUnit report has no testcases: {xml_path}")
    try:
        declared = sum(int(suite.get("tests", "0")) for suite in suites)
        declared_failed = sum(
            int(suite.get("failures", "0")) + int(suite.get("errors", "0"))
            for suite in suites
        )
    except ValueError as error:
        raise DashboardError(f"JUnit report has a non-integer count: {xml_path}") from error
    if declared != len(cases):
        raise DashboardError(
            f"JUnit count mismatch: suites declare {declared}, report contains {len(cases)} cases"
        )

    tests = []
    for case in cases:
        # classname is dotted and may end in a test class (tests.test_x.TestY);
        # the module is the last segment that looks like a test file.
        parts = [p for p in (case.get("classname") or "").split(".") if p.startswith("test_")]
        file = (parts[-1] if parts else "?") + ".py"
        failure = case.find("failure") is not None or case.find("error") is not None
        try:
            milliseconds = round(float(case.get("time", 0)) * 1000, 1)
        except ValueError as error:
            raise DashboardError(
                f"JUnit testcase {case.get('name')!r} has invalid time {case.get('time')!r}"
            ) from error
        tests.append(
            {
                "id": f"{file}::{case.get('name')}",
                "file": file,
                "name": case.get("name"),
                "ms": milliseconds,
                "ok": not failure,
                "epic": FILE_TO_EPIC.get(file, "Unassigned"),
            }
        )

    actual_failed = sum(not test["ok"] for test in tests)
    if declared_failed != actual_failed:
        raise DashboardError(
            f"JUnit failure-count mismatch: suites declare {declared_failed}, "
            f"report contains {actual_failed} failed cases"
        )

    if expected_files is not None:
        present = {test["file"] for test in tests}
        missing = sorted(expected_files - present)
        if missing:
            raise DashboardError(
                "JUnit report is partial; missing test modules: " + ", ".join(missing)
            )
    if newest_input is not None:
        newest_mtime, newest_path = newest_input
        if xml_path.stat().st_mtime < newest_mtime:
            raise DashboardError(
                f"JUnit report is stale; {newest_path} is newer than {xml_path}"
            )
    return tests


def collect_gallery() -> dict[str, list[dict]]:
    """Copy test-recorded SVGs into the share and map test id -> images."""
    shutil.rmtree(GALLERY, ignore_errors=True)
    gallery: dict[str, list[dict]] = {}
    for sidecar in sorted(ARTIFACTS.glob("*.json")) if ARTIFACTS.is_dir() else []:
        svg = sidecar.with_suffix(".svg")
        if not svg.exists():
            continue
        meta = json.loads(sidecar.read_text())
        GALLERY.mkdir(parents=True, exist_ok=True)
        shutil.copy2(svg, GALLERY / svg.name)
        gallery.setdefault(meta["id"], []).append(
            {"file": svg.name, "tag": meta.get("tag", "")}
        )
    return gallery


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() or "?"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"previous_ids": [], "history": []}


def record_suite_runtime(
    tests: list[dict],
    *,
    wall_seconds: float,
    pytest_code: int,
) -> dict:
    """Persist measured wall-clock history for future agent wait budgets.

    Summed JUnit testcase time omits collection, imports, JAX compilation
    outside testcase calls, and pytest teardown.  Agents therefore wait from
    ``wall_seconds``; ``case_seconds`` remains useful for diagnosing the gap.
    Reused JUnit reports never call this function because their wall time is
    unknowable after the fact. ``previous-3`` includes the latest completed run.
    """
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise DashboardError(
            f"suite wall time must be positive and finite, got {wall_seconds!r}"
        )
    if RUNTIMES.exists():
        try:
            runtime_state = json.loads(RUNTIMES.read_text())
        except json.JSONDecodeError as error:
            raise DashboardError(
                f"malformed runtime history {RUNTIMES}: {error}"
            ) from error
        version = (
            runtime_state.get("schema_version")
            if isinstance(runtime_state, dict) else None
        )
        history_key = "recent" if version == 1 else "previous-3"
        if version not in (1, 2) or not isinstance(runtime_state.get(history_key), list):
            raise DashboardError(
                f"invalid runtime history schema in {RUNTIMES}"
            )
        previous = list(runtime_state[history_key])
    else:
        previous = []

    failed = sum(1 for test in tests if not test["ok"])
    record = {
        "completed_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "commit": git_head(),
        "test_count": len(tests),
        "failed": failed,
        "pytest_exit_code": pytest_code,
        "result": "passed" if pytest_code == 0 and failed == 0 else "failed",
        "selection": SUITE_SELECTION,
        "wall_seconds": round(float(wall_seconds), 3),
        "case_seconds": round(sum(test["ms"] for test in tests) / 1000, 3),
    }
    previous.append(record)
    previous = previous[-RUNTIME_HISTORY_LIMIT:]
    # Compare like-for-like suite selections so an opt-in slow run cannot
    # poison the default suite's wait budget.
    comparable = [
        entry for entry in previous
        if entry.get("selection") == SUITE_SELECTION
    ]
    observed = [entry["wall_seconds"] for entry in comparable]
    # Use the slowest comparable measurement, add 25% headroom, and round
    # upward to a human-friendly 30-second polling boundary.
    padded = max(observed) * 1.25
    suggested_wait = max(60, math.ceil(padded / 30) * 30)
    runtime_state = {
        "schema_version": 2,
        "latest": record,
        "suggested_wait_seconds": suggested_wait,
        "previous-3": previous,
    }
    RUNTIMES.parent.mkdir(parents=True, exist_ok=True)
    temporary = RUNTIMES.with_name(f".{RUNTIMES.name}.tmp")
    temporary.write_text(json.dumps(runtime_state, indent=2, sort_keys=True) + "\n")
    temporary.replace(RUNTIMES)
    return runtime_state

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


def test_row(t: dict, new_ids: set[str], gallery: dict[str, list[dict]]) -> str:
    inner = (
        (
            '<span class="st ok">✓</span>'
            if t["ok"]
            else '<span class="st bad">✗ FAIL</span>'
        )
        + f'<span class="tn">{html.escape(t["name"])}</span>'
        + ('<span class="new">NEW</span>' if t["id"] in new_ids else "")
    )
    images = gallery.get(t["id"], [])
    if not images:
        return f'<div class="row">{inner}<span class="ms">{t["ms"]:g} ms</span></div>'
    shots = "".join(
        f'<figure><a href="cffg-viz/{html.escape(img["file"])}" target="_blank">'
        f'<img src="cffg-viz/{html.escape(img["file"])}" loading="lazy" '
        f'alt="{html.escape(img["tag"] or t["name"])}"></a>'
        + (f'<figcaption>{html.escape(img["tag"])}</figcaption>' if img["tag"] else "")
        + "</figure>"
        for img in images
    )
    return (
        '<details class="viz"><summary class="row">'
        + inner
        + f'<span class="shots-n">◫ {len(images)}</span>'
        + f'<span class="ms">{t["ms"]:g} ms</span></summary>'
        + f'<div class="shots">{shots}</div></details>'
    )


def render(
    tests: list[dict],
    new_ids: set[str],
    history: list[dict],
    commit: str,
    gallery: dict[str, list[dict]],
) -> str:
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
            test_row(t, new_ids, gallery)
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
details.viz {{ border-left:3px solid rgba(42,120,214,.5); border-radius:4px;
  background:rgba(42,120,214,.07); margin:2px 0; }}
details.viz > summary {{ list-style:none; cursor:pointer; padding:3px 6px; }}
details.viz > summary::-webkit-details-marker {{ display:none; }}
.shots-n {{ color:var(--chip); font-size:11px; white-space:nowrap; }}
details.viz[open] .shots-n {{ opacity:.5; }}
.shots {{ padding:4px 6px 8px; display:grid; gap:8px; }}
.shots figure {{ margin:0; }}
.shots img {{ display:block; width:100%; background:#fff; border-radius:6px;
  border:1px solid rgba(0,0,0,.15); }}
.shots figcaption {{ color:var(--ink2); font-size:11px; margin-top:2px; }}
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
<footer>generated by tools/dashboard.py · regen: uv run python tools/dashboard.py --run-suite</footer>
</body></html>"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--junitxml", type=Path, metavar="PATH",
        help="reuse an existing completed default-suite JUnit report",
    )
    mode.add_argument(
        "--run-suite", action="store_true",
        help="run the default short suite once, then render its report",
    )
    return parser.parse_args(argv)


def publish(tests: list[dict]) -> int:
    """Render and persist the dashboard; return the number of failed tests."""
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
    gallery = collect_gallery()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(tests, new_ids, state["history"], commit, gallery))
    state["previous_ids"] = sorted(t["id"] for t in tests)
    STATE.write_text(json.dumps(state))
    failed = sum(1 for t in tests if not t["ok"])
    n_shots = sum(len(v) for v in gallery.values())
    print(
        f"{OUT}: {len(tests)} tests, {failed} failing, {len(new_ids)} new, "
        f"{n_shots} diagrams in {len(gallery)} galleries"
    )
    return failed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generated_report = None
    pytest_code = 0
    wall_seconds = None
    if args.run_suite:
        generated_report, pytest_code, wall_seconds = run_suite()
        xml_path = generated_report
    else:
        xml_path = args.junitxml

    try:
        tests = load_junit(
            xml_path,
            expected_files=repository_test_files(),
            newest_input=newest_test_input(exclude=xml_path),
        )
        failed = publish(tests)
        if wall_seconds is not None:
            record_suite_runtime(
                tests,
                wall_seconds=wall_seconds,
                pytest_code=pytest_code,
            )
    except DashboardError as error:
        sys.exit(str(error))
    finally:
        if generated_report is not None:
            generated_report.unlink(missing_ok=True)

    if pytest_code or failed:
        sys.exit(
            f"dashboard reflects a failing suite: pytest exit {pytest_code}, "
            f"{failed} failing test(s)"
        )


if __name__ == "__main__":
    main()
