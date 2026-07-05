"""CSP tier: on tree graphs, BP marginals match exhaustive enumeration
exactly and the BFE equals -log Z. The 'true BFE' invariant, tier one."""

import numpy as np
import pytest

from cffg.compiler import CompileError, compile_template
from cffg.core import FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.ir import BlobRef
from cffg.engine import Engine

K, MA, MB = 3, 2, 4
rng = np.random.default_rng(7)


def col_stochastic(rows, cols):
    m = rng.uniform(0.1, 1.0, size=(rows, cols))
    return m / m.sum(axis=0, keepdims=True)


D = rng.dirichlet(np.ones(K))
A = col_stochastic(MA, K)
BM = col_stochastic(MB, K)


def make_tree() -> Template:
    """prior -> z0 = (za, zb); za emits xa, zb emits xb."""
    return Template(
        name="tree_csp",
        variables={
            "z0": VariableSpec("z0", K),
            "za": VariableSpec("za", K),
            "zb": VariableSpec("zb", K),
            "xa": VariableSpec("xa", MA),
            "xb": VariableSpec("xb", MB),
        },
        factors={
            "prior": FactorSpec("prior", "categorical", edges={"out": "z0"}, param_blobs={"statistic": "d"}),
            "eq": FactorSpec("eq", "equality", edges={"a": "z0", "b": "za", "c": "zb"}),
            "obsA": FactorSpec("obsA", "matrix_cat", edges={"out": "xa", "in": "za"}, param_blobs={"matrix": "A"}),
            "obsB": FactorSpec("obsB", "matrix_cat", edges={"out": "xb", "in": "zb"}, param_blobs={"matrix": "Bm"}),
        },
        ports={
            "pa": PortSpec("pa", "xa", PortRole.DATA),
            "pb": PortSpec("pb", "xb", PortRole.DATA),
        },
        blobs={
            "d": BlobRef("d", "blobs/d.npy"),
            "A": BlobRef("A", "blobs/A.npy"),
            "Bm": BlobRef("Bm", "blobs/Bm.npy"),
        },
    )


ARRAYS = {"d": D, "A": A, "Bm": BM}


def run_tree(observations):
    engine = Engine(compile_template(make_tree(), arrays=ARRAYS))
    state = engine.bind_data(engine.init_state(), observations)
    state, deltas = engine.run(state, 12, tol=1e-14)
    assert deltas[-1] < 1e-12  # tree: converged, hard
    return engine, state


def joint():
    # joint[z, xa, xb] = d[z] * A[xa, z] * Bm[xb, z]
    return np.einsum("z,az,bz->zab", D, A, BM)


def test_fully_observed_marginals_and_bfe():
    xa, xb = 1, 2
    engine, state = run_tree({"pa": xa, "pb": xb})
    q = engine.marginals(state)

    p_z = joint()[:, xa, xb]
    z_exact = p_z / p_z.sum()
    for var in ("z0", "za", "zb"):
        np.testing.assert_allclose(q[var], z_exact, atol=1e-10)
    np.testing.assert_allclose(q["xa"], np.eye(MA)[xa], atol=1e-10)

    log_evidence = np.log(p_z.sum())
    np.testing.assert_allclose(engine.bfe(state), -log_evidence, atol=1e-9)


def test_missing_observation_marginalized():
    xa = 0
    engine, state = run_tree({"pa": xa, "pb": None})
    q = engine.marginals(state)

    p_z_xb = joint()[:, xa, :]  # [K, MB]
    z_exact = p_z_xb.sum(axis=1)
    z_exact /= z_exact.sum()
    np.testing.assert_allclose(q["z0"], z_exact, atol=1e-10)

    xb_pred = p_z_xb.sum(axis=0)
    np.testing.assert_allclose(q["xb"], xb_pred / xb_pred.sum(), atol=1e-10)

    np.testing.assert_allclose(engine.bfe(state), -np.log(p_z_xb.sum()), atol=1e-9)


def test_no_observations_bfe_is_zero():
    """With nothing clamped, Z = 1 for a normalized generative model."""
    engine, state = run_tree({})
    np.testing.assert_allclose(engine.bfe(state), 0.0, atol=1e-9)
    np.testing.assert_allclose(engine.marginals(state)["z0"], D, atol=1e-10)


def test_transition_mixture_against_einsum():
    KU = 2
    bs = np.stack([col_stochastic(K, K) for _ in range(KU)])
    t = Template(
        name="tm_unit",
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
        blobs={"Bs": BlobRef("Bs", "blobs/Bs.npy")},
    )
    engine = Engine(compile_template(t, arrays={"Bs": bs}))

    # in and selector clamped: q(o) is the selected column.
    state = engine.bind_data(engine.init_state(), {"pi": 2, "pu": 1, "po": None})
    state, _ = engine.run(state, 3)
    np.testing.assert_allclose(engine.marginals(state)["o"], bs[1][:, 2], atol=1e-12)

    # out clamped, others free: q(u) from the einsum reference.
    state = engine.bind_data(engine.init_state(), {"po": 0, "pi": None, "pu": None})
    state, _ = engine.run(state, 3)
    ref = np.einsum("uoi,o->u", bs, np.eye(K)[0])  # uniform in-message cancels
    np.testing.assert_allclose(engine.marginals(state)["u"], ref / ref.sum(), atol=1e-12)


def test_compile_rejects_unterminated_prior_port():
    t = make_tree()
    t.ports["pz"] = PortSpec("pz", "z0", PortRole.PRIOR)
    del t.factors["prior"]
    with pytest.raises(CompileError, match="unterminated"):
        compile_template(t, arrays=ARRAYS)


def test_compile_rejects_non_stochastic_params():
    bad = dict(ARRAYS)
    bad["A"] = np.ones((MA, K))
    with pytest.raises(CompileError, match="column-stochastic"):
        compile_template(make_tree(), arrays=bad)
