"""Truthful one-pass generation of the mobile test dashboard (cff-kut1)."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "dashboard.py"


def load_dashboard():
    spec = importlib.util.spec_from_file_location("cffg_test_dashboard", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dashboard():
    return load_dashboard()


def write_report(path: Path, cases: list[tuple[str, str, float, bool]]) -> Path:
    root = ET.Element("testsuites", name="pytest tests")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="pytest",
        tests=str(len(cases)),
        failures=str(sum(failed for *_, failed in cases)),
        errors="0",
    )
    for module, name, seconds, failed in cases:
        case = ET.SubElement(
            suite,
            "testcase",
            classname=f"tests.{module}",
            name=name,
            time=str(seconds),
        )
        if failed:
            ET.SubElement(case, "failure", message="intentional")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def valid_report(tmp_path: Path) -> Path:
    return write_report(
        tmp_path / "report.xml",
        [
            ("test_alpha", "test_one", 0.125, False),
            ("test_beta", "test_two", 0.25, False),
        ],
    )


def test_load_junit_accepts_complete_fresh_report(dashboard, tmp_path):
    report = valid_report(tmp_path)
    tests = dashboard.load_junit(
        report,
        expected_files={"test_alpha.py", "test_beta.py"},
        newest_input=(report.stat().st_mtime - 1.0, tmp_path / "older.py"),
    )
    assert [test["id"] for test in tests] == [
        "test_alpha.py::test_one",
        "test_beta.py::test_two",
    ]
    assert [test["ms"] for test in tests] == [125.0, 250.0]


def test_load_junit_rejects_missing_malformed_empty_and_mismatched(dashboard, tmp_path):
    with pytest.raises(dashboard.DashboardError, match="does not exist"):
        dashboard.load_junit(tmp_path / "missing.xml")

    malformed = tmp_path / "malformed.xml"
    malformed.write_text("<testsuites>")
    with pytest.raises(dashboard.DashboardError, match="malformed"):
        dashboard.load_junit(malformed)

    empty = tmp_path / "empty.xml"
    empty.write_text('<testsuite tests="0"/>')
    with pytest.raises(dashboard.DashboardError, match="no testcases"):
        dashboard.load_junit(empty)

    mismatch = write_report(
        tmp_path / "mismatch.xml", [("test_alpha", "test_one", 0.1, False)]
    )
    tree = ET.parse(mismatch)
    tree.getroot().find("testsuite").set("tests", "2")
    tree.write(mismatch)
    with pytest.raises(dashboard.DashboardError, match="count mismatch"):
        dashboard.load_junit(mismatch)

    failure_mismatch = valid_report(tmp_path)
    tree = ET.parse(failure_mismatch)
    tree.getroot().find("testsuite").set("failures", "1")
    tree.write(failure_mismatch)
    with pytest.raises(dashboard.DashboardError, match="failure-count mismatch"):
        dashboard.load_junit(failure_mismatch)


def test_load_junit_rejects_partial_and_stale_reports(dashboard, tmp_path):
    report = valid_report(tmp_path)
    with pytest.raises(dashboard.DashboardError, match="partial.*test_gamma.py"):
        dashboard.load_junit(
            report,
            expected_files={"test_alpha.py", "test_beta.py", "test_gamma.py"},
        )

    newer = tmp_path / "newer.py"
    newer.write_text("# changed after the run")
    newer_time = report.stat().st_mtime + 5.0
    os.utime(newer, (newer_time, newer_time))
    with pytest.raises(dashboard.DashboardError, match="stale.*newer.py"):
        dashboard.load_junit(report, newest_input=(newer_time, newer))


def configure_publish(dashboard, monkeypatch, tmp_path, expected_files):
    monkeypatch.setattr(dashboard, "OUT", tmp_path / "cffg-tests.html")
    monkeypatch.setattr(dashboard, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(dashboard, "RUNTIMES", tmp_path / "runtimes.json")
    monkeypatch.setattr(dashboard, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(dashboard, "GALLERY", tmp_path / "gallery")
    monkeypatch.setattr(dashboard, "git_head", lambda: "abc123")
    monkeypatch.setattr(dashboard, "repository_test_files", lambda: expected_files)
    monkeypatch.setattr(
        dashboard,
        "newest_test_input",
        lambda **kwargs: (0.0, tmp_path / "old.py"),
    )


def test_reuse_mode_never_runs_pytest(dashboard, monkeypatch, tmp_path):
    report = valid_report(tmp_path)
    expected = {"test_alpha.py", "test_beta.py"}
    configure_publish(dashboard, monkeypatch, tmp_path, expected)

    def forbidden():
        raise AssertionError("reuse mode launched pytest")

    monkeypatch.setattr(dashboard, "run_suite", forbidden)
    dashboard.main(["--junitxml", str(report)])
    page = dashboard.OUT.read_text()
    assert "2 tests" in page
    assert "all passing" in page


def test_run_suite_mode_runs_once_and_removes_its_temporary_report(
    dashboard, monkeypatch, tmp_path
):
    report = valid_report(tmp_path)
    expected = {"test_alpha.py", "test_beta.py"}
    configure_publish(dashboard, monkeypatch, tmp_path, expected)
    calls = []

    def run_once():
        calls.append(True)
        return report, 0, 123.25

    monkeypatch.setattr(dashboard, "run_suite", run_once)
    dashboard.main(["--run-suite"])
    assert calls == [True]
    assert not report.exists()
    timing = dashboard.RUNTIMES.read_text()
    assert '"wall_seconds": 123.25' in timing
    assert '"selection": "not slow"' in timing
    assert '"suggested_wait_seconds": 180' in timing


@pytest.mark.parametrize("code,elapsed,result", [
    (124, 1200.01, "timed_out"), (0, 1200.01, "over_budget"), (130, 12.0, "incomplete"),
])
def test_interrupted_or_over_budget_run_preserves_last_dashboard(
    dashboard, monkeypatch, tmp_path, code, elapsed, result
):
    report = valid_report(tmp_path)
    configure_publish(dashboard, monkeypatch, tmp_path, {"test_alpha.py", "test_beta.py"})
    dashboard.OUT.write_text("previous dashboard")
    monkeypatch.setattr(dashboard, "run_suite", lambda: (report, code, elapsed))
    with pytest.raises(SystemExit, match="no passing result published"):
        dashboard.main(["--run-suite"])
    assert dashboard.OUT.read_text() == "previous dashboard"
    assert report.exists()
    state = json.loads(dashboard.RUNTIMES.read_text())
    assert state["latest"]["result"] == result
    assert state["latest"]["test_count"] is None
    assert state["suggested_wait_seconds"] <= 1200


def test_nonzero_pytest_exit_cannot_publish_green_report(dashboard, monkeypatch, tmp_path):
    report = valid_report(tmp_path)
    configure_publish(dashboard, monkeypatch, tmp_path, {"test_alpha.py", "test_beta.py"})
    dashboard.OUT.write_text("previous dashboard")
    monkeypatch.setattr(dashboard, "run_suite", lambda: (report, 1, 2.0))
    with pytest.raises(SystemExit, match="pytest exited 1"):
        dashboard.main(["--run-suite"])
    assert dashboard.OUT.read_text() == "previous dashboard"
    assert report.exists()
    assert json.loads(dashboard.RUNTIMES.read_text())["latest"]["result"] == "failed"


def test_deadline_stops_pytest_and_its_child_process(dashboard, monkeypatch, tmp_path):
    child_file = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        f"Path({str(child_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    original = subprocess.Popen

    def start(_command, **kwargs):
        return original([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(dashboard.subprocess, "Popen", start)
    monkeypatch.setattr(dashboard, "SHORT_SUITE_SECONDS", 1.0)
    report, code, elapsed = dashboard.run_suite()
    monkeypatch.setattr(dashboard.subprocess, "Popen", original)
    try:
        assert code == 124
        assert elapsed < 5.0
        child = int(child_file.read_text())
        state = subprocess.run(["ps", "-p", str(child), "-o", "stat="], capture_output=True, text=True)
        assert state.returncode != 0 or state.stdout.strip().startswith("Z")
    finally:
        report.unlink(missing_ok=True)
        if child_file.exists():
            try:
                os.kill(int(child_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("version,history_key", [(1, "recent"), (2, "previous-3")])
def test_runtime_history_is_bounded_and_reuse_does_not_overwrite_it(
    dashboard, monkeypatch, tmp_path, version, history_key
):
    report = valid_report(tmp_path)
    expected = {"test_alpha.py", "test_beta.py"}
    configure_publish(dashboard, monkeypatch, tmp_path, expected)
    dashboard.RUNTIMES.write_text(
        json.dumps(
            {
                "schema_version": version,
                history_key: [
                    {
                        "wall_seconds": float(index + 1),
                        "result": "passed",
                    }
                    for index in range(dashboard.RUNTIME_HISTORY_LIMIT)
                ],
            }
        )
    )
    tests = dashboard.load_junit(report)
    state = dashboard.record_suite_runtime(tests, wall_seconds=100.0, pytest_code=1)
    assert state["schema_version"] == 2
    assert "recent" not in state
    assert len(state["previous-3"]) == dashboard.RUNTIME_HISTORY_LIMIT == 3
    assert state["previous-3"][0]["wall_seconds"] == 2.0
    assert state["previous-3"][-1] == state["latest"]
    assert state["latest"]["result"] == "failed"
    assert state["latest"]["case_seconds"] == 0.375
    before = dashboard.RUNTIMES.read_text()

    dashboard.main(["--junitxml", str(report)])
    assert dashboard.RUNTIMES.read_text() == before


def test_runtime_wait_uses_only_comparable_retained_runs(
    dashboard, monkeypatch, tmp_path
):
    report = valid_report(tmp_path)
    configure_publish(
        dashboard, monkeypatch, tmp_path, {"test_alpha.py", "test_beta.py"}
    )
    dashboard.RUNTIMES.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent": [
                    {"selection": "not slow", "wall_seconds": 8000.0},
                    {"selection": "not slow", "wall_seconds": 180.0},
                    {"selection": "slow", "wall_seconds": 9000.0},
                ],
            }
        )
    )
    tests = dashboard.load_junit(report)
    state = dashboard.record_suite_runtime(tests, wall_seconds=100.0, pytest_code=0)
    assert [r["wall_seconds"] for r in state["previous-3"]] == [180.0, 9000.0, 100.0]
    assert state["suggested_wait_seconds"] == 240
    state = dashboard.record_suite_runtime(tests, wall_seconds=90.0, pytest_code=0)
    assert [r["wall_seconds"] for r in state["previous-3"]] == [9000.0, 100.0, 90.0]
    assert state["suggested_wait_seconds"] == 150
    state = dashboard.record_suite_runtime(tests, wall_seconds=80.0, pytest_code=0)
    assert [r["wall_seconds"] for r in state["previous-3"]] == [100.0, 90.0, 80.0]
    assert state["suggested_wait_seconds"] == 150


@pytest.mark.parametrize(
    "state",
    [
        [],
        {"schema_version": 2, "recent": []},
        {"schema_version": 2, "previous-3": {}},
        {"schema_version": 3, "previous-3": []},
    ],
)
def test_runtime_history_rejects_invalid_schema(
    dashboard, monkeypatch, tmp_path, state
):
    configure_publish(dashboard, monkeypatch, tmp_path, set())
    original = json.dumps(state)
    dashboard.RUNTIMES.write_text(original)
    with pytest.raises(
        dashboard.DashboardError, match="invalid runtime history schema"
    ):
        dashboard.record_suite_runtime([], wall_seconds=10.0, pytest_code=0)
    assert dashboard.RUNTIMES.read_text() == original


def test_failing_reused_report_renders_red_and_exits_nonzero(
    dashboard, monkeypatch, tmp_path
):
    report = write_report(
        tmp_path / "failed.xml", [("test_alpha", "test_broken", 0.1, True)]
    )
    configure_publish(dashboard, monkeypatch, tmp_path, {"test_alpha.py"})
    with pytest.raises(SystemExit, match="1 failing test"):
        dashboard.main(["--junitxml", str(report)])
    page = dashboard.OUT.read_text()
    assert "1 failing" in page
    assert "FAIL" in page
