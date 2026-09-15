"""Prospective moderate inventory and source identity; no numerical imports."""

import json
from pathlib import Path
import subprocess

from d12_compile_scaling import digest

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).with_name("d12_moderate_spec.json")
SCHEMA = "cffg.d12-moderate-report.v1"
MODES = ("raw-zero", "feasible", "perturbed")
BUDGETS = dict(outer=200, trials=24, linear=128, backtrack=0.5, armijo=0.0001)


def solve_cases():
    return [
        dict(id=f"t{t}-xi{xi}-{mode}", T=t, K=4, d=2, xi=xi, messages=mode)
        for t in (32, 100)
        for xi in (0, 1)
        for mode in MODES
    ]


def storage_cases():
    variants = (
        ("solve_from_coordinate", 200, 128),
        ("solve_from_coordinate", 400, 128),
        ("solve_from_coordinate", 200, 256),
        ("iteration", 200, 128),
        ("iteration", 200, 256),
    )
    return [
        dict(
            id=f"t{t}-{program}-o{outer}-l{linear}",
            T=t,
            K=4,
            d=2,
            program=program,
            budgets={**BUDGETS, "outer": outer, "linear": linear},
        )
        for t in (32, 100)
        for program, outer, linear in variants
    ]


def load_spec():
    spec = json.loads(SPEC.read_text())
    if spec["schema"] != "cffg.d12-moderate-plan.v1" or spec["cases"] != solve_cases():
        raise ValueError("expected exactly the twelve frozen moderate starts")
    if spec["storage"]["cases"] != storage_cases() or spec["budgets"] != BUDGETS:
        raise ValueError(
            "prospective budgets or complete-solve storage variants changed"
        )
    if spec["storage"]["logical_shapes"] != [
        dict(T=t, K=k, d=d) for t, k, d in ((32, 4, 2), (100, 4, 2), (100, 20, 10))
    ]:
        raise ValueError("logical shape inventory changed")
    inherited = spec["inherited_spec"]
    fixture = spec["fixture"]
    for path, expected in (
        (inherited["path"], inherited["sha256"]),
        (fixture["formulas_source"], fixture["formulas_sha256"]),
        (
            spec["storage"]["logical_contract"]["path"],
            spec["storage"]["logical_contract"]["sha256"],
        ),
    ):
        if digest(ROOT / path) != expected:
            raise ValueError(f"frozen specification changed: {path}")
    tiny = json.loads((ROOT / inherited["path"]).read_text())
    for key in ("coordinate", "acceptance", "strategy"):
        if spec[key] != tiny[key]:
            raise ValueError(f"qualified Y15D {key} changed")
    return spec


def source_paths():
    paths = sorted((ROOT / "src/cffg").glob("hybrid_*.py"))
    paths += sorted((ROOT / "tools").glob("d12_moderate*.*"))
    paths += [
        ROOT / name
        for name in (
            "tools/d12_compile_scaling.py",
            "tools/d12_compile_fixture.py",
            "tools/d12_compile_spec.json",
            "uv.lock",
            "pyproject.toml",
            "src/cffg/__init__.py",
            "src/cffg/experiments/__init__.py",
            "src/cffg/experiments/hybrid_separator.py",
            "src/cffg/experiments/regime_pair.py",
            "tests/fixtures/hybrid-joint-candidate-spec.json",
        )
    ]
    return sorted(str(p.relative_to(ROOT)) for p in paths if p.is_file())


def identities():
    git = lambda *args: subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()
    return dict(
        git_revision=git("rev-parse", "HEAD"),
        git_status=git("status", "--porcelain"),
        spec_sha256=digest(SPEC),
        sources={name: digest(ROOT / name) for name in source_paths()},
    )
