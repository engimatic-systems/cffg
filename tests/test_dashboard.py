"""Truthful one-pass generation of the mobile test dashboard (cff-kut1)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from suite_selection import nodeids_digest

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


def write_report(
    path: Path, cases: list[tuple[str, str, float, str]], *, workers: int = 1
) -> Path:
    nodeids = [f"tests/{module}.py::{name}" for module, name, _, _ in cases]
    failures = sum(outcome == "failure" for *_, outcome in cases)
    root = ET.Element("testsuites", name="pytest tests")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="pytest",
        tests=str(len(cases)),
        failures=str(failures),
        skipped=str(sum(outcome == "skipped" for *_, outcome in cases)),
        errors="0",
    )
    receipt = {
        "selection": "not slow",
        "scope": "repository",
        "selected_count": len(cases),
        "nodeids_sha256": nodeids_digest(nodeids),
        "workers": workers,
        "exitstatus": int(bool(failures)),
    }
    suite_properties = ET.SubElement(suite, "properties")
    ET.SubElement(suite_properties, "property", name="cffg.suite", value=json.dumps(receipt))
    for (module, name, seconds, outcome), nodeid in zip(cases, nodeids, strict=True):
        case = ET.SubElement(
            suite,
            "testcase",
            classname=f"tests.{module}",
            name=name,
            time=str(seconds),
        )
        case_properties = ET.SubElement(case, "properties")
        ET.SubElement(case_properties, "property", name="cffg.nodeid", value=nodeid)
        if outcome != "passed":
            ET.SubElement(case, outcome, message="intentional")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def valid_report(tmp_path: Path, *, workers: int = 1) -> Path:
    return write_report(
        tmp_path / "report.xml",
        [
            ("test_alpha", "test_one", 0.125, "passed"),
            ("test_beta", "test_two", 0.25, "passed"),
        ],
        workers=workers,
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
        tmp_path / "mismatch.xml", [("test_alpha", "test_one", 0.1, "passed")]
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

    def forbidden(*args, **kwargs):
        raise AssertionError("reuse mode launched pytest")

    monkeypatch.setattr(dashboard, "run_suite", forbidden)
    monkeypatch.setattr(dashboard.subprocess, "run", forbidden)
    dashboard.main(["--junitxml", str(report)])
    page = dashboard.OUT.read_text()
    assert "2 tests" in page
    assert "all passing" in page
    assert report.exists()
    assert not dashboard.RUNTIMES.exists()


@pytest.mark.parametrize(
    "system,architecture,cores,pages,expected",
    [
        ("darwin", "arm64", 12, 2**24, 4),
        ("darwin", "arm64", 11, 2**24, 1),
        ("darwin", "arm64", 12, 2**24 - 1, 1),
        ("darwin", "arm64", None, 2**24, 1),
        ("darwin", "x86_64", 12, 2**24, 1),
        ("linux", "aarch64", 32, 2**25, 1),
    ],
)
def test_default_workers_respect_host_limits(
    dashboard, monkeypatch, system, architecture, cores, pages, expected
):
    monkeypatch.setattr(dashboard.sys, "platform", system)
    monkeypatch.setattr(dashboard.platform, "machine", lambda: architecture)
    monkeypatch.setattr(dashboard.os, "cpu_count", lambda: cores)
    sizes = {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(dashboard.os, "sysconf", sizes.__getitem__, raising=False)
    assert dashboard.default_worker_count() == expected


@pytest.mark.parametrize("workers,expected", [(None, 4), (1, 1), (3, 3)])
def test_runner_starts_one_default_selection(dashboard, monkeypatch, workers, expected):
    monkeypatch.setattr(dashboard, "default_worker_count", lambda: 4)
    launches = []

    def run(command, *, cwd, **kwargs):
        launches.append((command, cwd))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(dashboard.subprocess, "run", run)
    report, code, elapsed = dashboard.run_suite(workers)
    try:
        command = ["uv", "run", "pytest", "-q", "-m", "not slow"]
        if expected > 1:
            command.extend(["-n", str(expected), "--dist", "loadfile", "--max-worker-restart", "0"])
        assert launches == [(command + [f"--junitxml={report}"], dashboard.REPO)]
        assert code == 1 and elapsed >= 0
    finally:
        report.unlink(missing_ok=True)


@pytest.mark.parametrize("args", [
    ["--run-suite", "--workers", "0"],
    ["--run-suite", "--workers", "-1"],
    ["--junitxml", "saved.xml", "--workers", "1"],
])
def test_cli_rejects_invalid_worker_options(dashboard, args):
    with pytest.raises(SystemExit) as rejected:
        dashboard.parse_args(args)
    assert rejected.value.code == 2


@pytest.mark.parametrize("worker_args,workers", [([], 4), (["--workers", "3"], 3)])
def test_run_suite_mode_runs_once_and_removes_its_temporary_report(
    dashboard, monkeypatch, tmp_path, worker_args, workers
):
    report = valid_report(tmp_path, workers=workers)
    expected = {"test_alpha.py", "test_beta.py"}
    configure_publish(dashboard, monkeypatch, tmp_path, expected)
    calls = []

    def run_once(workers):
        calls.append(workers)
        return report, 0, 123.25

    monkeypatch.setattr(dashboard, "run_suite", run_once)
    monkeypatch.setattr(dashboard, "default_worker_count", lambda: 4)
    dashboard.main(["--run-suite", *worker_args])
    assert calls == [workers]
    assert not report.exists()
    timing = dashboard.RUNTIMES.read_text()
    assert '"wall_seconds": 123.25' in timing
    assert '"selection": "not slow"' in timing
    assert json.loads(timing)["latest"]["workers"] == workers
    assert '"suggested_wait_seconds": 180' in timing


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


@pytest.mark.parametrize("workers,expected_wait", [(1, 750), (4, 150)])
def test_wait_uses_matching_worker_count_including_legacy_serial_records(
    dashboard, monkeypatch, tmp_path, workers, expected_wait
):
    configure_publish(dashboard, monkeypatch, tmp_path, set())
    history = [
        {"selection": "not slow", "wall_seconds": 600.0},
        {"selection": "not slow", "workers": 4, "wall_seconds": 120.0},
    ]
    dashboard.RUNTIMES.write_text(json.dumps({"schema_version": 2, "previous-3": history}))
    state = dashboard.record_suite_runtime([], wall_seconds=90.0, pytest_code=0, workers=workers)
    assert state["latest"]["workers"] == workers
    assert state["suggested_wait_seconds"] == expected_wait


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


@pytest.mark.parametrize("run_suite", [False, True], ids=["reuse", "run"])
def test_complete_failing_suite_renders_red_and_exits_nonzero(
    dashboard, monkeypatch, tmp_path, run_suite
):
    report = write_report(
        tmp_path / "failed.xml", [("test_alpha", "test_broken", 0.1, "failure")]
    )
    configure_publish(dashboard, monkeypatch, tmp_path, {"test_alpha.py"})
    args = ["--junitxml", str(report)]
    if run_suite:
        monkeypatch.setattr(dashboard, "run_suite", lambda workers: (report, 1, 4.0))
        args = ["--run-suite", "--workers", "1"]
    with pytest.raises(SystemExit, match="1 failing test"):
        dashboard.main(args)
    page = dashboard.OUT.read_text()
    assert "1 failing" in page
    assert "FAIL" in page
    if run_suite:
        assert not report.exists()
        assert json.loads(dashboard.RUNTIMES.read_text())["latest"]["result"] == "failed"
    else:
        assert report.exists()
        assert not dashboard.RUNTIMES.exists()


@pytest.mark.parametrize("code,recorded_workers", [(2, 4), (0, 1)])
def test_runner_rejects_a_report_inconsistent_with_the_launch(
    dashboard, monkeypatch, tmp_path, code, recorded_workers
):
    report = valid_report(tmp_path, workers=recorded_workers)
    configure_publish(dashboard, monkeypatch, tmp_path, {"test_alpha.py", "test_beta.py"})
    dashboard.OUT.write_text("previous dashboard")
    monkeypatch.setattr(dashboard, "run_suite", lambda workers: (report, code, 12.0))
    with pytest.raises(SystemExit):
        dashboard.main(["--run-suite", "--workers", "4"])
    assert dashboard.OUT.read_text() == "previous dashboard"
    assert not dashboard.RUNTIMES.exists()
    assert not report.exists()


def test_reuse_requires_the_suite_receipt_before_publishing(dashboard, monkeypatch, tmp_path):
    report = valid_report(tmp_path)
    tree = ET.parse(report)
    suite = tree.getroot().find("testsuite")
    suite.remove(suite.find("properties"))
    tree.write(report)
    configure_publish(dashboard, monkeypatch, tmp_path, {"test_alpha.py", "test_beta.py"})
    with pytest.raises(SystemExit):
        dashboard.main(["--junitxml", str(report)])
    assert report.exists()
    assert not any(path.exists() for path in (dashboard.OUT, dashboard.STATE, dashboard.RUNTIMES))


def test_reuse_displays_skipped_cases_separately(dashboard, monkeypatch, tmp_path):
    report = write_report(
        tmp_path / "skipped.xml",
        [("test_alpha", "test_ok", 0.1, "passed"), ("test_alpha", "test_skip", 0.0, "skipped")],
    )
    configure_publish(dashboard, monkeypatch, tmp_path, {"test_alpha.py"})
    dashboard.main(["--junitxml", str(report)])
    page = dashboard.OUT.read_text()
    assert "1 passed, 1 skipped" in page
    assert "all passing" not in page
    assert "skipped</span><span class=\"tn\">test_skip</span>" in page
