"""Record whether a JUnit report covers the complete repository PR selection."""

import json
from pathlib import Path

import pytest


_COLLECTED = pytest.StashKey[list]()
_PROVENANCE = pytest.StashKey[dict]()


def pytest_itemcollected(item):
    item.config.stash.setdefault(_COLLECTED, []).append(item)


def pytest_collection_finish(session):
    config = session.config
    collected = config.stash.get(_COLLECTED, [])
    expected = {item.nodeid for item in collected if item.get_closest_marker("slow") is None}
    selected = [item.nodeid for item in session.items]
    root = config.rootpath.resolve()
    arguments = config.args
    repository = (
        len(arguments) == 1
        and "::" not in arguments[0]
        and (Path(config.invocation_params.dir) / arguments[0]).resolve() in {root, root / "tests"}
    )
    narrowed = any(config.getoption(option, default=False) for option in (
        "keyword", "lf", "deselect", "ignore", "ignore_glob", "override_ini", "inifilename",
    ))
    complete = repository and not narrowed and set(selected) == expected and len(selected) == len(expected)
    config.stash[_PROVENANCE] = {
        "selection": config.option.markexpr,
        "scope": "repository" if complete else "partial",
        "selected_count": len(selected),
    }
    if _COLLECTED in config.stash:
        del config.stash[_COLLECTED]


@pytest.fixture(scope="session", autouse=True)
def _record_suite_selection(request, record_testsuite_property):
    record_testsuite_property("cffg.suite", json.dumps(request.config.stash[_PROVENANCE], sort_keys=True))
