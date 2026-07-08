"""Shared fixtures: a canonical HMM time-slice template.

One slice of the state-space model from Part II Fig. 7 (minus controls):
transition into z, an equality node exposing z to the observation branch and
to the next slice, observation factor emitting x. Glue ports head/tail chain
slices; the data port awaits an observation clamp.

    head(z_in) --[trans]-- z_mid --[eq]-- z_out : tail
                            |
                          z_obs --[obs]-- x : data
"""

import json
import re
import shutil
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)  # FE parity tests need f64

from cffg.core import (
    BlobRef,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)

# Rendered-diagram gallery: tests record SVGs here (gitignored); the dashboard
# (tools/dashboard.py) picks them up and shows them under the matching test row.
VIZ_ARTIFACTS = Path(__file__).parent / ".artifacts"


def pytest_sessionstart(session):
    shutil.rmtree(VIZ_ARTIFACTS, ignore_errors=True)


@pytest.fixture
def viz_gallery(request):
    """Record diagrams this test produces: `viz_gallery(template, level=4,
    tag="...")`. Accepts a Template (rendered via cffg.viz) or a ready SVG
    string. Each image gets a JSON sidecar naming the test, so the dashboard
    can attach it to the right row without any registry to maintain."""

    from cffg.viz import render_svg

    file = Path(request.node.location[0]).name
    test_id = f"{file}::{request.node.name}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{Path(file).stem}-{request.node.name}")
    count = iter(range(100))

    def record(template_or_svg, *, level=4, tag=""):
        svg = (
            template_or_svg
            if isinstance(template_or_svg, str)
            else render_svg(template_or_svg, level=level)
        )
        suffix = f"-{tag}" if tag else ""
        stem = f"{slug}-{next(count):02d}{re.sub(r'[^A-Za-z0-9_.-]+', '-', suffix)}"
        VIZ_ARTIFACTS.mkdir(exist_ok=True)
        (VIZ_ARTIFACTS / f"{stem}.svg").write_text(svg)
        (VIZ_ARTIFACTS / f"{stem}.json").write_text(
            json.dumps({"id": test_id, "tag": tag})
        )

    return record

K = 3  # latent cardinality
M = 2  # observation cardinality


def make_hmm_slice(name: str = "hmm_slice") -> Template:
    return Template(
        name=name,
        variables={
            "z_in": VariableSpec("z_in", K),
            "z_mid": VariableSpec("z_mid", K),
            "z_obs": VariableSpec("z_obs", K),
            "z_out": VariableSpec("z_out", K),
            "x": VariableSpec("x", M),
        },
        factors={
            "trans": FactorSpec(
                "trans", "matrix_cat", edges={"out": "z_mid", "in": "z_in"}, param_blobs={"matrix": "B"}
            ),
            "eq": FactorSpec("eq", "equality", edges={"a": "z_mid", "b": "z_obs", "c": "z_out"}),
            "obs": FactorSpec(
                "obs", "matrix_cat", edges={"out": "x", "in": "z_obs"}, param_blobs={"matrix": "A"}
            ),
        },
        ports={
            "head": PortSpec("head", "z_in", PortRole.GLUE),
            "tail": PortSpec("tail", "z_out", PortRole.GLUE),
            "sense": PortSpec("sense", "x", PortRole.DATA),
        },
        blobs={
            "A": BlobRef("A", "blobs/A.safetensors"),
            "B": BlobRef("B", "blobs/B.safetensors"),
        },
    )


@pytest.fixture
def hmm_slice() -> Template:
    return make_hmm_slice()
