"""EF domains through the compiler and engine (cff-8s6a): the third arena.

Model builders follow the FFG discipline: an edge joins at most two factors,
so N observations of one rate/probability fan out through a chain of
equality nodes — exactly the shape the Gaussian tests use.
"""

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


def gamma_poisson(n_obs: int = 3, a0: float = 2.0, b0: float = 1.5) -> tuple[Template, dict]:
    """Gamma(a0, b0) prior on a rate, n_obs Poisson counts of it."""
    t = Template(name=f"gamma_poisson_{n_obs}")
    t.variables["lam0"] = VariableSpec("lam0", ef="gamma")
    t.factors["prior"] = FactorSpec(
        "prior", "gamma", edges={"out": "lam0"}, param_blobs={"shape": "a0", "rate": "b0"}
    )
    from cffg.core import BlobRef

    t.blobs["a0"] = BlobRef("a0", "blobs/a0.npy")
    t.blobs["b0"] = BlobRef("b0", "blobs/b0.npy")
    tail = "lam0"
    for i in range(n_obs):
        if i < n_obs - 1:  # branch through an equality node
            br, nxt = f"lam_b{i}", f"lam{i + 1}"
            t.variables[br] = VariableSpec(br, ef="gamma")
            t.variables[nxt] = VariableSpec(nxt, ef="gamma")
            t.factors[f"eq{i}"] = FactorSpec(
                f"eq{i}", "equality", edges={"a": tail, "b": br, "c": nxt}
            )
            rate, tail = br, nxt
        else:
            rate = tail
        t.variables[f"x{i}"] = VariableSpec(f"x{i}", ef="poisson")
        t.factors[f"obs{i}"] = FactorSpec(
            f"obs{i}", "poisson_obs", edges={"out": f"x{i}", "rate": rate}
        )
        t.ports[f"count{i}"] = PortSpec(f"count{i}", f"x{i}", PortRole.DATA)
    arrays = {"a0": np.array(a0), "b0": np.array(b0)}
    return t, arrays


def beta_bernoulli(n_obs: int = 4, a0: float = 2.0, b0: float = 5.0) -> tuple[Template, dict]:
    """Beta(a0, b0) prior on a probability, n_obs Bernoulli flips of it."""
    t = Template(name=f"beta_bernoulli_{n_obs}")
    t.variables["p0"] = VariableSpec("p0", ef="beta")
    t.factors["prior"] = FactorSpec(
        "prior", "beta", edges={"out": "p0"}, param_blobs={"a": "a0", "b": "b0"}
    )
    from cffg.core import BlobRef

    t.blobs["a0"] = BlobRef("a0", "blobs/a0.npy")
    t.blobs["b0"] = BlobRef("b0", "blobs/b0.npy")
    tail = "p0"
    for i in range(n_obs):
        if i < n_obs - 1:
            br, nxt = f"p_b{i}", f"p{i + 1}"
            t.variables[br] = VariableSpec(br, ef="beta")
            t.variables[nxt] = VariableSpec(nxt, ef="beta")
            t.factors[f"eq{i}"] = FactorSpec(
                f"eq{i}", "equality", edges={"a": tail, "b": br, "c": nxt}
            )
            prob, tail = br, nxt
        else:
            prob = tail
        t.variables[f"y{i}"] = VariableSpec(f"y{i}", ef="bernoulli")
        t.factors[f"obs{i}"] = FactorSpec(
            f"obs{i}", "bernoulli_obs", edges={"out": f"y{i}", "prob": prob}
        )
        t.ports[f"flip{i}"] = PortSpec(f"flip{i}", f"y{i}", PortRole.DATA)
    arrays = {"a0": np.array(a0), "b0": np.array(b0)}
    return t, arrays


# --- compile plans -------------------------------------------------------------


def test_ef_arena_shape():
    t, arrays = gamma_poisson(3)
    a = compile_template(t, arrays=arrays)
    # edges: lam0, lam_b0, lam1, lam_b1, lam2, x0, x1, x2 -> 16 ef slots
    assert a.n_efslots == 16
    assert a.n_slots == 0 and a.n_gslots == 0  # nothing leaks into other arenas
    types = sorted(g.type for g in a.ef_groups)
    assert types == ["equality", "gamma", "poisson_obs"]
    (prior,) = [g for g in a.ef_groups if g.type == "gamma"]
    np.testing.assert_allclose(a.log_params[prior.name + ":eta"], [[1.0, -1.5]])
    (obs,) = [g for g in a.ef_groups if g.type == "poisson_obs"]
    assert obs.family == "gamma" and len(obs.members) == 3 and a.n_ef_obs == 3
    assert sorted(a.ef_data) == ["count0", "count1", "count2"]
    # interior edges: the lam chain (5 edges), not the exterior counts
    (fam, pairs) = a.ef_interior[0]
    assert fam == "gamma" and pairs.shape == (5, 2)


def test_ef_slots_do_not_collide_with_discrete():
    """EF slot numbering is its own arena: a model mixing discrete and EF
    edges must not write EF rows into the discrete message array."""
    t, arrays = gamma_poisson(1)
    t.variables["z"] = VariableSpec("z", 3)
    t.factors["cat"] = FactorSpec(
        "cat", "categorical", edges={"out": "z"}, param_blobs={"statistic": "c"}
    )
    from cffg.core import BlobRef

    t.blobs["c"] = BlobRef("c", "blobs/c.npy")
    t.ports["zp"] = PortSpec("zp", "z", PortRole.DATA)
    arrays["c"] = np.array([0.5, 0.25, 0.25])
    a = compile_template(t, arrays=arrays)
    assert a.n_slots == 2 and a.n_efslots == 4
    assert a.interior_slot_pairs.shape == (0, 2)  # z is exterior, lam is EF


def test_unobserved_obs_factor_rejected():
    t, arrays = gamma_poisson(1)
    del t.ports["count0"]
    t.ports["glue"] = PortSpec("glue", "x0", PortRole.GLUE)
    with pytest.raises(CompileError, match="must be a DATA port"):
        compile_template(t, arrays=arrays)


def test_data_port_on_message_edge_rejected():
    """A DATA port directly on a gamma edge is a clamp, not conditioning."""
    t = Template(name="bad")
    from cffg.core import BlobRef

    t.variables["lam"] = VariableSpec("lam", ef="gamma")
    t.factors["prior"] = FactorSpec(
        "prior", "gamma", edges={"out": "lam"}, param_blobs={"shape": "a0", "rate": "b0"}
    )
    t.blobs["a0"] = BlobRef("a0", "x.npy")
    t.blobs["b0"] = BlobRef("b0", "x.npy")
    t.ports["bad"] = PortSpec("bad", "lam", PortRole.DATA)
    with pytest.raises(CompileError, match="EF DATA ports must clamp"):
        compile_template(t, arrays={"a0": np.array(1.0), "b0": np.array(1.0)})


def test_improper_prior_rejected():
    t, arrays = gamma_poisson(1)
    arrays["b0"] = np.array(-2.0)
    with pytest.raises(CompileError, match="improper gamma prior"):
        compile_template(t, arrays=arrays)


def test_delta_on_ef_edge_rejected():
    t, arrays = gamma_poisson(2)
    t.constraints.append(ConstraintSpec(ConstraintKind.DELTA, "lam0"))
    with pytest.raises(CompileError, match="delta constraint on EF edge"):
        compile_template(t, arrays=arrays)


def test_bead_on_ef_node_rejected():
    t, arrays = gamma_poisson(1)
    t.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "prior"))
    with pytest.raises(CompileError, match="beads on EF nodes"):
        compile_template(t, arrays=arrays)
