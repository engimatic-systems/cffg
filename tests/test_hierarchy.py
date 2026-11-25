"""Direct-path oracles for the two-clock categorical CFFG elimination schedule."""

import copy
from itertools import product

import jax
import numpy as np
import pytest

from cffg.core import BlobRef, FactorSpec
from cffg.runtime_types import RunStatus
from cffg.compiler import compile_template
from cffg.hierarchy import Hierarchy
from cffg.hierarchy_model import HierarchySpec, canonical_model


def tables(dtype=np.float64):
    return tuple(np.asarray(x, dtype=dtype) for x in (
        [.46, .54], [[.84, .27], [.16, .73]], [.66, .34],
        [[[.86, .38], [.14, .62]], [[.25, .87], [.75, .13]]],
        [[.88, .28], [.12, .72]]))


def paths(start, b, a, sequence):
    """Enumerate every fast path and every unclamped emission. No message recursion."""
    nr, nx = start.shape
    qo = np.zeros((nr, nx))
    prior = start.sum(axis=1)
    evidence = np.zeros(nr)
    costs = np.zeros(3)
    weights = []
    for r, x0 in product(range(nr), range(nx)):
        for states in product(range(nx), repeat=len(sequence)):
            for obs in product(*[range(len(a)) if o is None else [o] for o in sequence]):
                previous = (x0, *states[:-1])
                tr = np.prod([b[r,j,i] for j,i in zip(states, previous, strict=True)])
                em = np.prod([a[y,i] for y,i in zip(obs, states, strict=True)])
                mass = start[r,x0] * tr * em
                if mass:
                    initial = start[r,x0] / prior[r]
                    costs -= mass * np.log([initial,tr,em])
                    evidence[r] += initial * tr * em
                    qo[r,states[-1]] += mass
                    weights.append(mass)
    z = qo.sum()
    weights = np.asarray(weights)/z
    return qo/z, evidence, costs/z, np.sum(weights * np.log(weights)), -np.log(z)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_joint_every_term_entropies_and_correlation_across_periods(dtype):
    p,t,d,b,a = tables(dtype)
    h = Hierarchy.from_tables(p,t,d,b,a,thickness=3)
    start = np.outer(p,d).astype(float)
    b,a,t = (v.astype(float) for v in (b,a,t))
    tol = 96*np.finfo(dtype).eps
    for cycle in ([1, None, 0], [1, 1, 0]):
        for i, obs in enumerate(cycle):
            q,c,terms,neg_entropy,fe = paths(start,b,a,cycle[:i+1])
            r = h.preview(obs).require_converged()
            committed = h.step(obs).require_converged()
            np.testing.assert_array_equal(committed.joint,r.joint)
            np.testing.assert_allclose(r.joint,q,atol=tol,rtol=0)
            np.testing.assert_allclose(r.parent,q.sum(axis=1),atol=tol,rtol=0)
            np.testing.assert_allclose(r.child,q.sum(axis=0),atol=tol,rtol=0)
            np.testing.assert_allclose(r.log_cavity,np.log(c),atol=tol,rtol=0)
            np.testing.assert_allclose([r.ledger.child_prior,r.ledger.child_transitions,r.ledger.child_emissions],
                                       terms,atol=tol,rtol=0)
            np.testing.assert_allclose(r.ledger.child_conditional_negative_entropy+r.ledger.child_endpoint_negative_entropy,
                                       neg_entropy,atol=tol,rtol=0)
            np.testing.assert_allclose(r.ledger.regime_separator_entropy,-r.parent @ np.log(r.parent),atol=tol,rtol=0)
            np.testing.assert_allclose(r.free_energy,fe,atol=tol,rtol=0)
            assert r.period_complete is (i == 2)
        start = t @ q
        np.testing.assert_allclose(h.current_joint,start,atol=tol,rtol=0)
        assert np.abs(start-np.outer(start.sum(axis=1),start.sum(axis=0))).max() > .001
    assert h.child_ticks == 6 and h.parent_periods == 2
    np.testing.assert_allclose(h.child_statistics.sum(),6,atol=tol)
    np.testing.assert_allclose(h.parent_statistics.sum(),2,atol=tol)


def test_one_clock_native_flooding_and_scheduled_ledger():
    from cffg.engine import Engine
    p,t,d,b,a = tables()
    spec,parent,pa,child,ca = canonical_model(p,t,d,b,a,1)
    h = Hierarchy(spec,parent,pa,child,ca)
    unrolled = copy.deepcopy(child)
    del unrolled.ports[spec.child_port]
    unrolled.factors["slow prior"] = FactorSpec("slow prior","categorical",{"out":"regime"},param_blobs={"statistic":"p"})
    unrolled.blobs["p"] = BlobRef("p","p.npy")
    arenas = compile_template(unrolled,arrays={**ca,"p":p})
    result = h.step(1)
    for schedule in ("flood","two_pass"):
        engine = Engine(arenas,schedule=schedule)
        state = engine.bind_data(engine.init_state(),{"reading":1})
        out = engine.run(state,80,tol=1e-11,harden_last=0)
        marginals = engine.marginals(out.require_converged())
        np.testing.assert_allclose(marginals["regime"],result.parent,atol=1e-10)
        np.testing.assert_allclose(marginals["new"],result.child,atol=1e-10)
        np.testing.assert_allclose(engine.bfe(out.state),result.free_energy,atol=1e-10)


def test_support_budgets_vacuity_and_reference_execution():
    p = [1.,0.]
    # Regime 0 sets state 0, regime 1 sets state 1. Third outcome is impossible.
    t = np.eye(2)
    b = np.array([[[1,1],[0,0]], [[0,0],[1,1]]],float)
    a = np.array([[1,0],[0,1],[0,0]],float)
    h = Hierarchy.from_tables(p,t,[1,0],b,a,thickness=2)
    for options, child_status, parent_status in [
        ({"child_budget":0},RunStatus.BUDGET_EXHAUSTED,None),
        ({"parent_budget":0},RunStatus.CONVERGED,RunStatus.BUDGET_EXHAUSTED),
        ({"coupled_budget":0},None,None),
    ]:
        r = h.step(0,**options)
        assert (r.status,r.child_status,r.parent_status) == (RunStatus.BUDGET_EXHAUSTED,child_status,parent_status)
    impossible = h.step(2)
    assert impossible.child_status is RunStatus.INFEASIBLE and impossible.parent_status is None
    disjoint = h.step(1)
    assert disjoint.child_status is RunStatus.CONVERGED and disjoint.parent_status is RunStatus.INFEASIBLE
    assert np.isneginf(disjoint.log_cavity[0]) and disjoint.log_cavity[1] == 0
    assert h.child_ticks == 0 and h.parent_statistics.sum() == h.child_statistics.sum() == 0
    with jax.disable_jit():
        direct = h.preview(0)
    same = h.step(0)
    assert same.free_energy == direct.free_energy == 0
    np.testing.assert_array_equal(same.parent,p)
    assert np.isneginf(same.log_cavity[1])
    vacuous = Hierarchy.from_tables(None,t,[1,0],b,a,thickness=1).step(None)
    np.testing.assert_array_equal(vacuous.parent,[.5,.5])
    assert abs(vacuous.free_energy) < 1e-12


def test_long_float32_period_remains_finite_beyond_linear_space_underflow():
    p,t,d,b,a = tables(np.float32)
    a = np.array([[1.,1.], [1e-12,2e-12]],np.float32)
    h = Hierarchy.from_tables(p,t,d,b,a,thickness=64)
    for tick in range(64):
        r = h.step(1).require_converged()
        # A linear-space forward likelihood underflows inside this period.
        assert np.isfinite(r.log_cavity).all() and np.isfinite(r.free_energy)
        if tick == 55:
            resumed = Hierarchy.from_tables(p,t,d,b,a,thickness=64)
            resumed.restore(h.checkpoint())
            h = resumed
    assert h.parent_periods == 1 and min(-r.log_cavity) > 1000


def test_declaration_rejects_unsupported_semantics_and_invalid_inputs():
    args = canonical_model(*tables(),2)
    spec,parent,pa,child,ca = args
    for k in (True,0,-1,1.5):
        with pytest.raises(ValueError,match="thickness"):
            Hierarchy.from_tables(*tables(),thickness=k)
    changed = copy.deepcopy(child)
    changed.variables["regime"] = type(changed.variables["regime"])("regime",3)
    with pytest.raises(ValueError,match="cardinalit"):
        Hierarchy(spec,parent,pa,changed,ca)
    with pytest.raises(ValueError,match="nonnegative"):
        Hierarchy(spec,parent,{**pa,"d":[-.1,1.1]},child,ca)
    with pytest.raises(ValueError,match="stochastic"):
        Hierarchy(spec,parent,{**pa,"d":[0.,0.]},child,ca)
    assert HierarchySpec.from_json(spec.to_json()) == spec
    with pytest.raises(ValueError,match="unversioned"):
        HierarchySpec.from_json("{}")
    assert Hierarchy(*args).structure_hash != Hierarchy.from_tables(*tables(),thickness=1).structure_hash


def test_child_failure_gates_parent_and_nonfinite_joint_never_commits(monkeypatch):
    import cffg.hierarchy as module

    h = Hierarchy.from_tables(*tables(), thickness=1)
    original_child, original_parent = module.child_message, module.parent_message

    def invalid_child(*args):
        return (*original_child(*args)[:-1], False)

    def forbidden(*args):
        pytest.fail("parent consumed a divergent child result")

    monkeypatch.setattr(module, "child_message", invalid_child)
    monkeypatch.setattr(module, "parent_message", forbidden)
    r = h.step(0)
    assert (r.status, r.child_status, r.parent_status) == (RunStatus.DIVERGED, RunStatus.DIVERGED, None)

    def invalid_parent(*args):
        z, log_joint, terms, stat = original_parent(*args)
        return z, log_joint, terms.at[0].set(np.nan), stat

    monkeypatch.setattr(module, "child_message", original_child)
    monkeypatch.setattr(module, "parent_message", invalid_parent)
    r = h.step(0)
    assert (r.status, r.child_status, r.parent_status) == (RunStatus.DIVERGED, RunStatus.CONVERGED, RunStatus.DIVERGED)
    assert h.child_ticks == h.parent_periods == 0


def test_vmap_child_eliminates_independent_observations():
    import jax.numpy as jnp
    from cffg.hierarchy_kernel import child_message, initial_cost

    p, _, d, b, a = tables()
    frontier = jnp.log(jnp.broadcast_to(d, (len(p), len(d))))
    args = (frontier, jnp.zeros(len(p)), initial_cost(frontier), jnp.log(b), jnp.log(a))
    observations = jnp.array([-1, 0, 1])
    batched = jax.vmap(child_message, in_axes=(None, None, None, None, None, 0))(*args, observations)
    for i, observation in enumerate(observations):
        with jax.disable_jit():
            uncompiled = child_message(*args, observation)
        for actual, expected in zip(batched, uncompiled, strict=True):
            np.testing.assert_allclose(actual[i], expected, atol=32*np.finfo(float).eps)
