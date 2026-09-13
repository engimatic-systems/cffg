"""Real pytest invocations must not label narrowed coverage as a PR gate."""

import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


@pytest.mark.parametrize(
    "arguments, expected_selection, expected_scope, expected_count",
    [
        (["-m", "not slow"], "not slow", "repository", 4),
        ([], "not slow and not pr_head", "partial", 3),
        (["-m", "not slow", "-k", "fast"], "not slow", "partial", 3),
        (["tests/test_one.py", "tests/test_two.py", "-m", "not slow"], "not slow", "partial", 4),
        (["-m", "not slow", "--deselect", "tests/test_one.py::test_expanded"], "not slow", "partial", 3),
        (["-m", "not slow", "--lf"], "not slow", "partial", 2),
        (["-m", "not slow", "-o", "python_functions=test_fast*"], "not slow", "partial", 3),
        (["-m", "not slow", "-c", "alternate.ini"], "not slow", "partial", 3),
    ],
    ids=(
        "pr-gate", "developer", "keyword", "explicit-files", "deselected", "last-failed",
        "discovery-override", "alternate-config",
    ),
)
def test_real_collection_provenance(tmp_path, arguments, expected_selection, expected_scope, expected_count):
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = -m 'not slow and not pr_head'\nmarkers =\n"
        "    slow: opt-in validation\n    pr_head: PR validation\n"
    )
    (tmp_path / "alternate.ini").write_text(
        (tmp_path / "pytest.ini").read_text() + "python_functions = test_fast*\n"
    )
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["suite_selection"]\n')
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_one.py").write_text(
        "import pytest\n"
        "def test_fast_a(): pass\n"
        "@pytest.mark.pr_head\ndef test_expanded(): pass\n"
        "@pytest.mark.slow\ndef test_long(): pass\n"
    )
    (tests / "test_two.py").write_text("def test_fast_b(): pass\ndef test_fast_c(): pass\n")
    cache = tmp_path / ".pytest_cache" / "v" / "cache"
    cache.mkdir(parents=True)
    (cache / "lastfailed").write_text(json.dumps({
        "tests/test_one.py::test_fast_a": True,
        "tests/test_two.py::test_fast_b": True,
    }))
    report = tmp_path / "report.xml"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}", *arguments],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    root = ET.parse(report).getroot()
    rows = root.findall(".//property[@name='cffg.suite']")
    assert len(rows) == 1
    assert json.loads(rows[0].get("value")) == {
        "selection": expected_selection,
        "scope": expected_scope,
        "selected_count": expected_count,
    }
    assert len(root.findall(".//testcase")) == expected_count
