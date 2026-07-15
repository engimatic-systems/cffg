"""Truthful one-pass generation of the mobile test dashboard (cff-kut1)."""

from __future__ import annotations

import importlib.util
import os
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
        return report, 0

    monkeypatch.setattr(dashboard, "run_suite", run_once)
    dashboard.main(["--run-suite"])
    assert calls == [True]
    assert not report.exists()


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
