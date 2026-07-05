"""Shared fixtures: a canonical HMM time-slice template.

One slice of the state-space model from Part II Fig. 7 (minus controls):
transition into z, an equality node exposing z to the observation branch and
to the next slice, observation factor emitting x. Glue ports head/tail chain
slices; the data port awaits an observation clamp.

    head(z_in) --[trans]-- z_mid --[eq]-- z_out : tail
                            |
                          z_obs --[obs]-- x : data
"""

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
