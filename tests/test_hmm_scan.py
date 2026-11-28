"""All directed scan outputs, support, energy and derivatives of the HMM lowering."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.benchmarks.scheduling import hmm_chain
from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.compiler import compile_template
from cffg.core.serialize import from_json, to_json
from cffg.engine import Engine, RunStatus, _replace
from cffg.execution import scans
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.schedule import build_schedule
from cffg.schedule_program import make_propagate


def fixture(n=18, k=3):
    template = hmm_chain(n, k)
    rng = np.random.default_rng(77)
    arrays = {name: rng.uniform(0.1, 2, size) for name, size in
              (("d", k), ("B", (k, k)), ("A", (k, k)))}
    return template, {name: x / x.sum(axis=0) for name, x in arrays.items()}


def args(state):
    return (state.messages, state.g_eta, state.g_lam, state.ef_eta,
            state.g_obs, state.g_mask, state.ef_obs, state.ef_mask, state.params)


def generic(arena, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(scans, "make_hmm_scan", lambda *_: None)
        return Engine(arena)


def test_lowering_depends_on_roles_ports_and_semiring():
    template, arrays = fixture()
    def lowering(template, semiring="sum", **kwargs):
        arena = compile_template(template, arrays=arrays, semiring=semiring)
        schedule = build_schedule(arena)
        return schedule, make_propagate(arena, schedule.stages, **kwargs).hmm_fused

    s, selected = lowering(template)
    assert selected == (True, True)
    rebuilt, reselected = lowering(from_json(to_json(template)))
    assert reselected == selected
    assert (rebuilt.describe(), rebuilt.fingerprint) == (s.describe(), s.fingerprint)
    assert lowering(template, fuse_hmm=False)[1] == (False, False)
    assert lowering(template, "max")[1] == (False, False)
    assert lowering(template, executors=replace(SCHEDULED_EXECUTORS))[1] == (False, False)
    rectangular = from_json(to_json(template))
    for i in range(1, 19):
        name = f"x{i}"
        rectangular.variables[name] = replace(rectangular.variables[name], cardinality=4)
    rectangular_arena = compile_template(rectangular, arrays={**arrays, "A": np.full((4, 3), .25)})
    rectangular_stages = build_schedule(rectangular_arena).stages
    assert make_propagate(rectangular_arena, rectangular_stages).hmm_fused == (False, False)
    # Valid ternary equalities, but now the outgoing chain is in the other role.
    for f in template.factors.values():
        if f.type == "equality":
            f.edges["b"], f.edges["c"] = f.edges["c"], f.edges["b"]
    assert lowering(template)[1] == (False, False)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_slots_marginals_certificate_energy_and_derivatives(dtype, monkeypatch):
    n = 18
    template, arrays = fixture(n)
    arena = compile_template(template, arrays=arrays)
    fused, plain, flood = Engine(arena), generic(arena, monkeypatch), Engine(arena, schedule="flood")
    state = jax.tree.map(lambda x: x.astype(dtype) if x.dtype.kind == "f" else x,
                         fused.bind_data(fused.init_state(), {"y1": 1, "y7": 0, "y18": 2}))
    eps = np.finfo(dtype).eps
    ff = fused.run(state, 1)
    pf = plain.run(state, 1)
    fl = flood.run(state, 3*n + 10, tol=8*eps)
    assert ff.status is pf.status is fl.status is RunStatus.CONVERGED
    assert ff.iterations == pf.iterations == 1
    assert ff.state.messages.dtype == dtype
    np.testing.assert_allclose(ff.state.messages, pf.state.messages, rtol=128*eps, atol=128*eps)
    for result in (pf, fl):
        expected = fused.marginals(result.state)
        for name, row in fused.marginals(ff.state).items():
            np.testing.assert_allclose(row, expected[name], rtol=128*eps, atol=128*eps)
        np.testing.assert_allclose(ff.free_energies, result.free_energies[-1:],
                                   rtol=256*eps*n, atol=256*eps*n)
    certified = fused.step(ff.state)
    assert np.max(np.abs(certified.messages - ff.state.messages) /
                  (1 + np.abs(ff.state.messages))) <= 64*eps

    def gradient(engine):
        # Differentiate probabilities, including the log conversion. Every per-slice
        # row is independent here, a stronger check than summing tied-parameter gradients.
        def energy(probabilities):
            params = jax.tree.map(jnp.log, probabilities)
            m, ge, gl, ef, _ = engine._propagate(*args(state)[:-1], params)
            return engine._bfe(m, ge, gl, state.g_obs, state.g_mask,
                               ef, state.ef_obs, state.ef_mask, params)
        return jax.jit(jax.grad(energy))(jax.tree.map(jnp.exp, state.params))

    g, pg = gradient(fused), gradient(plain)
    for key in g:
        np.testing.assert_allclose(g[key], pg[key], atol=512*eps*n, rtol=512*eps)


def test_nonstationary_inputs_retain_every_role_and_failure_owner():
    template, arrays = fixture()
    arena = compile_template(template, arrays=arrays)
    engine = Engine(arena)
    state = engine.init_state()
    rng = np.random.default_rng(321)
    state = _replace(state, messages=jnp.asarray(rng.normal(size=state.messages.shape)),
                     params={key: values + rng.normal(0, .2, values.shape)
                             for key, values in state.params.items()})
    fused = jax.jit(make_propagate(arena, engine.schedule.stages))
    plain = jax.jit(make_propagate(arena, engine.schedule.stages, fuse_hmm=False))
    # Arbitrary non-normalized cavities expose stale or silently omitted role outputs.
    for s in (state, _replace(state, messages=state.messages.at[
            arena.message_indices(arena.data_slots["y5"], 3)].set(jnp.nan))):
        lhs, rhs = fused(*args(s)), plain(*args(s))
        np.testing.assert_allclose(lhs[0], rhs[0], rtol=3e-14, atol=3e-14, equal_nan=True)
        np.testing.assert_array_equal(lhs[-1], rhs[-1])
        if s is not state:
            assert int(lhs[-1].reason) != 0
            assert engine.run(s, 1).status is RunStatus.DIVERGED


def test_literal_support_and_empty_support_per_colony_member():
    template = hmm_chain(20, 2)
    arrays = {"d": np.array([.5, .5]), "B": np.eye(2), "A": np.eye(2)}
    colony = Colony()
    ids = [colony.spawn(template, arrays, observations=obs)
           for obs in ({"y20": 1}, {"y1": 0, "y20": 1}, {})]
    result = colony.run(1)
    assert [result[i].status for i in ids] == [RunStatus.CONVERGED, RunStatus.INFEASIBLE,
                                               RunStatus.CONVERGED]
    engine = next(iter(colony.engines.values()))
    for row in engine.marginals(result[ids[0]].state).values():
        np.testing.assert_array_equal(row, [0, 1])
    state = jax.tree.map(lambda x: x.astype(jnp.float32) if x.dtype.kind == "f" else x,
                         engine.bind_data(engine.init_state(), {"y20": 1}))
    solved = engine.run(state, 1)
    assert solved.status is RunStatus.CONVERGED
    assert solved.state.messages.dtype == jnp.float32
    assert jnp.any(solved.state.messages == jnp.float32(NEG))
    s = _replace(state, messages=state.messages.at[
        engine.arena.message_indices(engine.arena.data_slots["y20"], 2)].set(NEG))
    assert engine.run(s, 1).status is RunStatus.INFEASIBLE


@pytest.mark.slow
def test_1024_slice_literal_identity_chain():
    template = hmm_chain(1024, 2)
    arena = compile_template(template, arrays={"d": np.array([.5, .5]), "B": np.eye(2), "A": np.eye(2)})
    engine = Engine(arena)
    assert all(make_propagate(arena, engine.schedule.stages).hmm_fused)
    result = engine.run(engine.bind_data(engine.init_state(), {"y1024": 0}), 1)
    assert result.status is RunStatus.CONVERGED
    assert result.iterations == 1
    for marginal in engine.marginals(result.state).values():
        np.testing.assert_array_equal(marginal, [1., 0.])
