"""Validate the selected pytest node IDs in a reusable short-suite JUnit report."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def nodeids_digest(nodeids):
    """Hash the whole selection independent of worker completion order."""
    data = json.dumps(sorted(nodeids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def _property(element, name):
    matches = element.findall(f"./properties/property[@name='{name}']")
    if len(matches) != 1 or not matches[0].get("value"):
        raise ValueError(f"JUnit report lacks a unique {name} property")
    return matches[0].attrib["value"]


def _case_identity(case):
    nodeid = _property(case, "cffg.nodeid")
    address, bracket, params = nodeid.partition("[")
    path, *names = address.split("::")
    if not names or not path.endswith(".py"):
        raise ValueError(f"invalid pytest node ID {nodeid!r}")
    classname = ".".join([path.removesuffix(".py").replace("/", "."), *names[:-1]])
    if (case.get("classname"), case.get("name")) != (classname, names[-1] + bracket + params):
        raise ValueError(f"JUnit testcase name disagrees with node ID {nodeid!r}")
    return nodeid


def validate_junit(path, *, require_pass=True):
    """Return collection provenance or raise ValueError, without importing pytest.

    With require_pass=False a complete tests-failed report is also accepted.
    An interrupt, worker loss or collection-only run cannot certify coverage.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"cannot read JUnit report: {exc}") from exc
    suites = list(root.iter("testsuite"))
    if root.tag not in {"testsuites", "testsuite"} or len(suites) != 1:
        raise ValueError("JUnit report must contain exactly one testsuite")
    suite = suites[0]
    info = json.loads(_property(suite, "cffg.suite"))
    if not isinstance(info, dict) or info.get("scope") != "repository" or info.get("selection") != "not slow":
        raise ValueError("JUnit report does not certify the repository 'not slow' selection")

    cases = suite.findall("testcase")
    if not cases or len(cases) != len(list(root.iter("testcase"))):
        raise ValueError("missing or misplaced JUnit testcases")
    ids = [_case_identity(case) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pytest node ID in JUnit report")
    if (type(info.get("selected_count")) is not int or info["selected_count"] != len(ids)
            or info.get("nodeids_sha256") != nodeids_digest(ids)):
        raise ValueError("JUnit cases do not match the collected test selection")

    counts = Counter()
    for case in cases:
        outcomes = [child.tag for child in case if child.tag in {"failure", "error", "skipped"}]
        if len(outcomes) > 1:
            raise ValueError("multiple JUnit outcomes for one selected test")
        counts.update(outcomes)
    expected = {"tests": len(ids), "failures": counts["failure"],
                "errors": counts["error"], "skipped": counts["skipped"]}
    if any(suite.get(name) != str(value) for name, value in expected.items()):
        raise ValueError("JUnit outcome counts disagree with the testcases")

    if type(info.get("workers")) is not int or info["workers"] < 1:
        raise ValueError("invalid JUnit worker count")
    failed = bool(counts["failure"] or counts["error"])
    if type(info.get("exitstatus")) is not int or info["exitstatus"] != int(failed):
        raise ValueError("pytest exit status disagrees with recorded outcomes")
    if require_pass and failed:
        raise ValueError("JUnit report records a failed test suite")
    return info


def _repository_scope(config, repository):
    paths = [(Path(config.invocation_params.dir) / arg).resolve() for arg in config.args]
    narrowing = ("keyword", "lf", "ignore", "ignore_glob", "deselect", "override_ini",
                 "stepwise", "collectonly", "setuponly", "setupplan", "pyargs")
    return (
        config.rootpath == repository
        and config.inipath == repository / "pyproject.toml"
        and paths in ([repository], [repository / "tests"])
        and config.option.markexpr == "not slow"
        and not any(config.getoption(name, default=False) for name in narrowing)
    )


def register(config, repository):
    """Attach the collection witness only for JUnit runs, on controller and workers."""
    if not config.getoption("xmlpath", default=None) and not hasattr(config, "workerinput"):
        return

    import pytest
    from _pytest.junitxml import xml_key

    class Selection:
        def __init__(self):
            self.expected = []
            self.collection = None
            self.worker_collections = []
            self.worker_count = 0

        def pytest_itemcollected(self, item):
            if item.get_closest_marker("slow") is None:
                self.expected.append(item.nodeid)

        def pytest_collection_finish(self, session):
            ids = [item.nodeid for item in session.items]
            complete = (_repository_scope(config, repository) and bool(ids)
                        and len(ids) == len(set(ids)) and sorted(ids) == sorted(self.expected))
            self.collection = dict(
                selection=config.option.markexpr, scope="repository" if complete else "partial",
                selected_count=len(ids), nodeids_sha256=nodeids_digest(ids),
            )
            for item in session.items:
                item.user_properties.append(("cffg.nodeid", item.nodeid))
            if hasattr(config, "workeroutput"):
                config.workeroutput["cffg.suite"] = self.collection

        @pytest.hookimpl(optionalhook=True)
        def pytest_xdist_setupnodes(self, specs):
            self.worker_count = len(specs)

        @pytest.hookimpl(optionalhook=True)
        def pytest_testnodedown(self, node, error):
            output = getattr(node, "workeroutput", {})
            valid = error is None and output.get("exitstatus") in (0, 1)
            self.worker_collections.append(output.get("cffg.suite") if valid else None)

        @pytest.hookimpl(tryfirst=True)
        def pytest_sessionfinish(self, exitstatus):
            # Only the controller owns the JUnit writer. Its session.items is
            # empty under xdist; every worker must attest the same collection.
            writer = config.stash.get(xml_key, None)
            if writer is None:
                return
            receipt = self.collection
            if self.worker_count:
                matches = (len(self.worker_collections) == self.worker_count
                           and all(r == self.worker_collections[0] for r in self.worker_collections))
                receipt = self.worker_collections[0] if matches else None
            if receipt is None:
                receipt = {"scope": "partial", "selection": config.option.markexpr}
            receipt = dict(receipt, exitstatus=int(exitstatus), workers=self.worker_count or 1)
            writer.add_global_property("cffg.suite", json.dumps(receipt, sort_keys=True))

    config.pluginmanager.register(Selection(), "cffg-suite-selection")
