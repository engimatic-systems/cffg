"""Portable scalar Gaussian sensor models used across nonlinear integration checks."""

import numpy as np

from cffg.core import (
    BlobRef, FactorSpec, FunctionSpec, MomentProjection, PortRole, PortSpec,
    Template, VariableSpec, moment_matching,
)


def model(method="linearization", *, affine=False, observed=False, dim=1):
    arrays = {
        "m": np.full(dim, .25), "P": np.eye(dim) * .5,
        "e": np.full(dim, .8), "R": np.eye(dim) * .09,
        "V": np.eye(dim) * .16,
        "theta": np.concatenate((np.eye(dim).ravel(), np.zeros(dim))) if affine else np.array([.3]),
    }
    factors = {
        "prior": FactorSpec("prior", "gaussian", {"out": "x"},
                            param_blobs={"mean": "m", "cov": "P"}),
        "sensor": FactorSpec("sensor", "gaussian_nonlinear", {"in": "x", "out": "y"},
                             param_blobs={"parameters": "theta", "cov": "V"},
                             function=FunctionSpec("cffg.mean.affine" if affine else "cffg.mean.cubic",
                                                   "1.0.0", method=method)),
    }
    if not observed:
        factors["evidence"] = FactorSpec("evidence", "gaussian", {"out": "y"},
                                       param_blobs={"mean": "e", "cov": "R"})
    template = Template(
        "nonlinear", variables={v: VariableSpec(v, dim=dim) for v in ("x", "y")},
        factors=factors,
        ports={"y": PortSpec("y", "y", PortRole.DATA)} if observed else {},
        blobs={name: BlobRef(name, name + ".npy") for name in arrays},
        constraints=[moment_matching("sensor", "in", MomentProjection.GAUSSIAN)],
    )
    return template, arrays


def adaptive_pair(method="linearization", *, evidence=.3, variance=.2):
    arrays = {
        "m": np.array([0.]), "P": np.array([[1.]]),
        "e1": np.array([evidence]), "e2": np.array([-evidence]),
        "R": np.array([[variance]]), "V": np.array([[variance]]),
        "theta": np.array([.3]),
    }
    factors = {
        "prior": FactorSpec("prior", "gaussian", {"out": "x0"}, param_blobs={"mean": "m", "cov": "P"}),
        "copy": FactorSpec("copy", "equality", {"a": "x0", "b": "x1", "c": "x2"}),
    }
    for i in (1, 2):
        name = f"sensor{i}"
        factors[name] = FactorSpec(name, "gaussian_nonlinear", {"in": f"x{i}", "out": f"y{i}"},
                                   param_blobs={"parameters": "theta", "cov": "V"},
                                   function=FunctionSpec("cffg.mean.cubic", "1.0.0", method=method))
        factors[f"evidence{i}"] = FactorSpec(f"evidence{i}", "gaussian", {"out": f"y{i}"},
                                            param_blobs={"mean": f"e{i}", "cov": "R"})
    return Template(
        "adaptive-pair", variables={name: VariableSpec(name, dim=1) for name in ("x0", "x1", "x2", "y1", "y2")},
        factors=factors, blobs={name: BlobRef(name, name + ".npy") for name in arrays},
        constraints=[moment_matching(f"sensor{i}", "in", MomentProjection.GAUSSIAN) for i in (1, 2)],
    ), arrays
