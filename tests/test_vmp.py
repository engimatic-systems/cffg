"""VMP mean-field bead: geometric messages, exact in the conjugate
single-latent case, provably different from BP when it should be."""

import numpy as np
import pytest

from cffg.compiler import CompileError, compile_template
from cffg.core import (
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.ir import BlobRef
from cffg.engine import Engine

K, M, KU = 3, 4, 2
rng = np.random.default_rng(23)


def col_stochastic(rows, cols):
    m = rng.uniform(0.1, 1.0, size=(rows, cols))
    return m / m.sum(axis=0, keepdims=True)


D = rng.dirichlet(np.ones(K))
A = col_stochastic(M, K)


def make_conjugate(mean_field: bool) -> Template:
    t = Template(
        name="conjugate",
        variables={"z": VariableSpec("z", K), "x": VariableSpec("x", M)},
        factors={
            "prior": FactorSpec("prior", "categorical", edges={"out": "z"}, param_blobs={"statistic": "d"}),
            "lik": FactorSpec("lik", "matrix_cat", edges={"out": "x", "in": "z"}, param_blobs={"matrix": "A"}),
        },
        ports={"px": PortSpec("px", "x", PortRole.DATA)},
        blobs={"d": BlobRef("d", "d.npy"), "A": BlobRef("A", "A.npy")},
    )
    if mean_field:
        t.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "lik"))
    return t


def posterior(x_obs):
    p = D * A[x_obs]
    return p / p.sum()


def test_vmp_exact_in_conjugate_case():
    x_obs = 2
    engine = Engine(compile_template(make_conjugate(True), arrays={"d": D, "A": A}))
    state = engine.bind_data(engine.init_state(), {"px": x_obs})
    state, deltas = engine.run(state, 20, tol=1e-14)
    assert deltas[-1] < 1e-12
    np.testing.assert_allclose(engine.marginals(state)["z"], posterior(x_obs), atol=1e-10)
    # For a single latent, mean-field is exact: BFE still equals -log evidence.
    np.testing.assert_allclose(engine.bfe(state), -np.log((D * A[x_obs]).sum()), atol=1e-9)


def test_vmp_matches_bp_here():
    x_obs = 1
    results = []
    for mf in (False, True):
        engine = Engine(compile_template(make_conjugate(mf), arrays={"d": D, "A": A}))
        state = engine.bind_data(engine.init_state(), {"px": x_obs})
        state, _ = engine.run(state, 20)
        results.append(engine.marginals(state)["z"])
    np.testing.assert_allclose(results[0], results[1], atol=1e-10)


def make_tm(mean_field: bool) -> Template:
    t = Template(
        name="tm_mf",
        variables={
            "o": VariableSpec("o", K),
            "i": VariableSpec("i", K),
            "u": VariableSpec("u", KU),
        },
        factors={
            "tm": FactorSpec(
                "tm", "transition_mixture",
                edges={"out": "o", "in": "i", "selector": "u"},
                param_blobs={"matrices": "Bs"},
            )
        },
        ports={
            "po": PortSpec("po", "o", PortRole.DATA),
            "pi": PortSpec("pi", "i", PortRole.DATA),
            "pu": PortSpec("pu", "u", PortRole.DATA),
        },
        blobs={"Bs": BlobRef("Bs", "Bs.npy")},
    )
    if mean_field:
        t.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "tm"))
    return t


BS = np.stack([col_stochastic(K, K) for _ in range(KU)])


def test_vmp_tm_with_clamped_parents_matches_bp():
    """With all parents at point masses the geometric and arithmetic
    expectations coincide."""
    obs = {"pi": 0, "pu": 1, "po": None}
    outs = []
    for mf in (False, True):
        engine = Engine(compile_template(make_tm(mf), arrays={"Bs": BS}))
        state = engine.bind_data(engine.init_state(), obs)
        state, _ = engine.run(state, 10)
        outs.append(engine.marginals(state)["o"])
    np.testing.assert_allclose(outs[0], BS[1][:, 0], atol=1e-12)
    np.testing.assert_allclose(outs[1], BS[1][:, 0], atol=1e-12)


def test_vmp_differs_from_bp_with_two_free_edges():
    """Geometric vs arithmetic marginalization must actually differ when more
    than one neighbor is uncertain — otherwise the bead is a no-op."""
    obs = {"po": 0, "pi": None, "pu": None}
    outs = []
    for mf in (False, True):
        engine = Engine(compile_template(make_tm(mf), arrays={"Bs": BS}))
        state = engine.bind_data(engine.init_state(), obs)
        state, _ = engine.run(state, 40)
        outs.append(engine.marginals(state)["u"])
    assert not np.allclose(outs[0], outs[1], atol=1e-6)


def test_mean_field_on_equality_rejected():
    t = make_conjugate(False)
    t.variables["y"] = VariableSpec("y", K)
    t.variables["w"] = VariableSpec("w", K)
    t.factors["eq"] = FactorSpec("eq", "equality", edges={"a": "z", "b": "y", "c": "w"})
    # rewire to keep degrees legal: y,w dangle as data ports
    t.ports["py"] = PortSpec("py", "y", PortRole.DATA)
    t.ports["pw"] = PortSpec("pw", "w", PortRole.DATA)
    del t.factors["prior"]  # z: lik + eq = degree 2
    t.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "eq"))
    with pytest.raises(CompileError, match="ill-defined"):
        compile_template(t, arrays={"d": D, "A": A})
