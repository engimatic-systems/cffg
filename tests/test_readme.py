"""Keep the newcomer guide synchronized with executable public surfaces."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import cffg
import cffg.core
import cffg.models
import cffg.viz


REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
EXAMPLES = {
    "examples/discrete_agent.py",
    "examples/discrete_rollout.py",
    "examples/exponential_family.py",
    "examples/hard_constraint.py",
    "examples/hybrid_rslds.py",
    "examples/learning_bundle.py",
    "examples/linear_gaussian.py",
    "examples/nonlinear_gaussian.py",
}
CAPABILITIES = (
    "Build the graph IR, compose, plate, and edit",
    "Build a corridor agent",
    "Build a general discrete rollout",
    "Build a factored rollout",
    "Compile, run, and swap parameters",
    "Use GFE and parameter epistemics",
    "Run a receding-horizon agent",
    "Run the costly-cue behavioral oracle",
    "Learn declared parameters",
    "Persist and continue",
    "Run linear-Gaussian inference",
    "Run scalar exponential-family inference",
    "Run hybrid mixture and recurrence inference",
    "Build or fit an rSLDS",
    "Use nonlinear Gaussian factors",
    "Use neural means and fixed-point gradients",
    "Train or amortize neural inference",
    "Use hard constraints or MAP",
    "Batch matching structures",
    "Register an ordinary custom factor",
    "Render a model",
    "Estimate resources",
)


def _marked(text: str, name: str) -> str:
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    assert text.count(start) == text.count(end) == 1
    return text.split(start, 1)[1].split(end, 1)[0]


def _symbols(text: str, name: str) -> list[str]:
    return re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", _marked(text, name))


def _anchors(path: Path) -> set[str]:
    anchors = set()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", path.read_text(), re.MULTILINE):
        slug = re.sub(r"[^a-z0-9 _-]", "", heading.lower()).replace(" ", "-")
        anchors.add(re.sub(r"-+", "-", slug))
    return anchors


def test_documented_facades_match_the_runtime_exports():
    text = README.read_text()
    assert _symbols(text, "facade-cffg") == cffg.__all__
    assert _symbols(text, "facade-core") == cffg.core.__all__
    assert _symbols(text, "facade-models") == cffg.models.__all__
    assert _symbols(text, "facade-viz") == cffg.viz.__all__


def test_marked_quickstart_is_valid_facade_only_python():
    marked = _marked(README.read_text(), "executable-quickstart")
    assert marked.count("```python") == 1
    assert marked.count("```") == 2
    source = marked.split("```python", 1)[1].split("```", 1)[0]
    syntax = ast.parse(source)
    imports = [node for node in ast.walk(syntax) if isinstance(node, ast.ImportFrom)]
    assert imports and all(node.module == "cffg" for node in imports)
    assert all(alias.name in cffg.__all__ for node in imports for alias in node.names)
    assert ".require_converged()" in source
    assert "objective_contract" in source


def test_capability_map_and_builtin_vocabularies_are_complete():
    text = README.read_text()
    table = text.split("## Capability map", 1)[1].split("## The mental model", 1)[0]
    labels = re.findall(r"^\| \[([^]]+)\]\(#[^)]+\)", table, re.MULTILINE)
    assert tuple(labels) == CAPABILITIES
    assert all(line.count("|") == 8 for line in table.splitlines() if line.startswith("|"))
    for factor in cffg.core.FACTOR_TYPES:
        assert f"`{factor}`" in text
    for constraint in cffg.core.ConstraintKind:
        assert f"`{constraint.value}`" in text


def test_local_links_and_example_catalogue_resolve():
    text = README.read_text()
    destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    linked_examples = {
        destination.split("#", 1)[0]
        for destination in destinations
        if destination.startswith("examples/")
    }
    assert linked_examples == EXAMPLES

    for destination in destinations:
        target, _, fragment = destination.partition("#")
        path = README if not target else REPO / target
        assert path.exists(), destination
        if fragment and path.suffix == ".md":
            assert fragment in _anchors(path), destination


def test_installation_uses_a_complete_uv_add_command():
    text = README.read_text()
    assert "uv init --python 3.12" in text
    assert "uv add --editable ../cffg" in text
    assert "uv run python -c 'import cffg;" in text
    assert "\n[tool.uv.sources]\n" not in text
