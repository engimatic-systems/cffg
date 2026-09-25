"""Standalone examples are executable documentation of public workflows."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
EXPECTED_OUTPUT = {
    "discrete_agent.py": "discrete agent:",
    "discrete_rollout.py": "discrete rollout: action posterior =",
    "linear_gaussian.py": "linear Gaussian: x1 mean =",
    "exponential_family.py": "exponential family: Gamma(shape, rate) =",
    "hard_constraint.py": "hard constraint: marginal modes =",
    "nonlinear_gaussian.py": "nonlinear Gaussian: x mean =",
    "learning_bundle.py": "learning bundle: posterior counts =",
    "hybrid_rslds.py": "hybrid rSLDS: z1 =",
}
PUBLIC_MODULES = {"cffg", "cffg.core", "cffg.models", "cffg.learn"}


@pytest.mark.parametrize("filename", sorted(EXPECTED_OUTPUT))
def test_example_has_a_documented_standalone_public_entrypoint(filename):
    source = (EXAMPLES / filename).read_text()
    syntax = ast.parse(source)

    assert ast.get_docstring(syntax)
    assert any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
        for node in syntax.body
    )
    assert 'if __name__ == "__main__":' in source
    assert ".require_converged()" in source
    for node in ast.walk(syntax):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("cffg"):
            assert node.module in PUBLIC_MODULES


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        pytest.param(*item, marks=pytest.mark.slow)
        if item[0] in {"discrete_rollout.py", "linear_gaussian.py", "nonlinear_gaussian.py", "hybrid_rslds.py"}
        else item
        for item in EXPECTED_OUTPUT.items()
        if item[0] != "discrete_agent.py"
    ],
)
def test_example_runs_as_a_standalone_program(tmp_path, filename, expected):
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / filename)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert expected in completed.stdout
