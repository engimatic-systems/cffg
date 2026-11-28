"""One collection per process and one JUnit outcome per selected pytest node."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from suite_selection import nodeids_digest, validate_junit


@pytest.fixture
def project(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = \"-m 'not slow'\"\nmarkers = ['slow']\n"
    )
    # Load only the real selection plugin; the synthetic project needs no JAX.
    (tests / "conftest.py").write_text(
        "from pathlib import Path\nfrom suite_selection import register\n"
        "def pytest_configure(config):\n    register(config, Path(__file__).resolve().parents[1])\n"
        "def pytest_collection_finish(session):\n"
        "    worker = getattr(session.config, 'workerinput', {}).get('workerid', 'serial')\n"
        "    with open(worker + '.collected', 'a') as f: f.write('1')\n"
    )
    (tests / "test_a.py").write_text(
        "import pytest\nclass TestClass:\n"
        "    @pytest.mark.parametrize('value', ['colon::value', 'brackets[1]'])\n"
        "    def test_param(self, value): assert value\n"
        "@pytest.mark.xfail(strict=True)\ndef test_expected_failure(): assert False\n"
        "@pytest.mark.slow\ndef test_slow(): assert False\n"
    )
    return tmp_path


def run(project, *args, expected=0):
    report = project / "result.xml"
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "tools"))
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "xdist.plugin", *args,
         f"--junitxml={report}"], cwd=project, env=env, capture_output=True, text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return report


@pytest.mark.parametrize("workers", [1, 2])
def test_serial_and_xdist_certify_the_same_unique_selected_nodes(project, workers):
    args = ["-n", str(workers), "--dist=loadfile"] if workers > 1 else []
    report = run(project, *args)
    ids = ["tests/test_a.py::TestClass::test_param[colon::value]",
           "tests/test_a.py::TestClass::test_param[brackets[1]]",
           "tests/test_a.py::test_expected_failure"]
    assert validate_junit(report) == {
        "scope": "repository", "selection": "not slow", "workers": workers,
        "selected_count": 3, "exitstatus": 0, "nodeids_sha256": nodeids_digest(ids),
    }
    # Reusing an archive has no collection side effects.
    validate_junit(report)
    collections = list(project.glob("*.collected"))
    assert len(collections) == workers
    assert all(path.read_text() == "1" for path in collections)


@pytest.mark.parametrize("args", [("-k", "test"), ("tests/test_a.py",),
                                 ("--ignore", "unused.py"), ("--lf",), ("-m", "not slow and not absent")])
def test_partial_invocation_cannot_certify_even_when_the_same_cases_run(project, args):
    with pytest.raises(ValueError, match="repository"):
        validate_junit(run(project, *args))


def test_one_failure_is_reportable_but_never_certified_as_passing(project):
    (project / "tests/test_fail.py").write_text("def test_fail(): assert False\n")
    report = run(project, "-n2", expected=1)
    assert validate_junit(report, require_pass=False)["exitstatus"] == 1
    with pytest.raises(ValueError, match="failed"):
        validate_junit(report)


def test_lost_worker_cannot_certify_a_complete_run(project):
    (project / "tests/test_crash.py").write_text("import os\ndef test_crash(): os._exit(7)\n")
    report = run(project, "-n2", "--max-worker-restart=0", expected=1)
    with pytest.raises(ValueError):
        validate_junit(report, require_pass=False)


def test_reuse_rejects_partial_duplicate_substituted_and_inconsistent_reports(project):
    report = run(project)
    source = report.read_bytes()

    def check(mutate):
        root = ET.fromstring(source)
        suite = root.find("testsuite")
        cases = suite.findall("testcase")
        mutate(suite, cases)
        ET.ElementTree(root).write(report)
        with pytest.raises(ValueError):
            validate_junit(report, require_pass=False)

    def substitute(suite, cases):
        cases[0].set("name", "test_substitute")
        cases[0].find("properties/property").set("value", "tests/test_a.py::TestClass::test_substitute")

    def interrupt(suite, cases):
        prop = suite.find("properties/property")
        info = json.loads(prop.get("value"))
        prop.set("value", json.dumps(dict(info, exitstatus=2)))

    check(lambda suite, cases: (suite.remove(cases[0]), suite.set("tests", "2")))
    check(lambda suite, cases: (suite.remove(cases[0]), suite.append(cases[1])))
    check(lambda suite, cases: cases[0].set("name", "test_substitute"))
    check(substitute)
    check(interrupt)
    check(lambda suite, cases: suite.set("skipped", "0"))
