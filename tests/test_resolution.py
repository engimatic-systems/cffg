"""Data-resolution contracts, including the final validation checkpoints."""

from dataclasses import replace

import numpy as np
import pytest
from test_ef_engine import gamma_poisson

from cffg import kernels
from cffg.compiler import CompileError, compile_template
from cffg.core import (
    BlobRef,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)


def invalid_unused_learnable(template):
    template.blobs["unused"] = BlobRef("unused", "unused.npy")
    template.learnables["bad"] = LearnableSpec("bad", "dirichlet", ("unused",))


@pytest.mark.parametrize(
    "cardinality,error,match",
    [
        (2**63, OverflowError, "too large"),
        (2, CompileError, r"learnable 'bad': no factor reads blobs \['unused'\]"),
    ],
)
def test_prefix_conversion_precedes_learning_validation(cardinality, error, match):
    names = ("a", "b", "c")
    template = Template(
        "prefix-before-learning",
        variables={name: VariableSpec(name, cardinality) for name in names},
        factors={"eq": FactorSpec("eq", "equality", {name: name for name in names})},
        ports={name: PortSpec(name, name, PortRole.GLUE) for name in names},
    )
    invalid_unused_learnable(template)
    with pytest.raises(error, match=match):
        compile_template(template)


@pytest.mark.parametrize("invalid_learning", [False, True])
def test_numeric_publication_follows_learning_validation(invalid_learning, monkeypatch):
    template, arrays = gamma_poisson(1)
    kernel = kernels.kernel_for("gamma")

    def pack(*args):
        return {"eta": np.array([[object(), object()]], dtype=object)}

    monkeypatch.setitem(
        kernels.KERNELS,
        "gamma",
        replace(kernel, binding=replace(kernel.binding, pack_params=pack)),
    )
    if invalid_learning:
        invalid_unused_learnable(template)
        with pytest.raises(CompileError, match="no factor reads blobs"):
            compile_template(template, arrays=arrays)
    else:
        with pytest.raises(
            TypeError, match="published arrays cannot contain Python objects"
        ):
            compile_template(template, arrays=arrays)
