"""Mixed fixed-support phase grouping, trust carry, and public certification."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import Engine, compile_template
from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.models.tmaze import build_tmaze
from cffg import outer
from cffg.outer_candidate import Candidate
from cffg.outer_batch import ChartCall, ChartKernel, run_phases
from cffg.outer_fixed import _anderson
from cffg.runtime_types import _replace
from test_outer_failures import scalar_arrays, scalar_chart, plan  # noqa: F401


def test_charts_regroup_at_phase_boundaries_without_reordering_results():
    calls = []
    def kernel(name):
        def many(x):
            calls.append((name, tuple(x)))
            return x + 1
        return ChartKernel(lambda x: many(x[None])[0], many)
    a, b = kernel("soft"), kernel("hard")

    def member(i, route):
        value = np.asarray(i)
        for stage in route:
            value = yield ChartCall(stage, (value,))
        return int(value)

    results = run_phases([member(8, [a, b]), member(0, []), member(3, [b, b])])
    assert results == [10, 0, 5]
    assert calls == [("soft", (8,)), ("hard", (3,)), ("hard", (9, 4))]


def test_standalone_map_rejects_speculative_candidate_at_false_boundary(scalar_chart):
    visited = []

    def mapping(arrays):
        value = float(arrays.messages[0])
        visited.append(value)
        return scalar_arrays(2. if value == 0 else 1.), jnp.asarray(0.), jnp.int32(value == 3.)

    def candidate(*args):
        proposed = scalar_arrays(3.)
        return Candidate(proposed, proposed, 0., 1, 1, jnp.array([3., 0.]), True)

    phases = outer.solve_outer(
        scalar_arrays(0.), SimpleNamespace(edges={}), mapping, lambda _: 0., lambda _: False,
        n_iters=4, tol=1e-6, candidate_limit=2,
        kernel_factory=lambda _: plan(), stage_kernel=lambda *_: (ChartKernel(candidate, None), None),
    )
    result = run_phases([phases])[0]
    assert 3. in visited
    assert result.reason.value == "root_converged" and result.iterations == 1
    np.testing.assert_array_equal(result.arrays.messages, [1.])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_anderson_history_uses_only_occupied_singular_vectors(dtype):
    rng = np.random.default_rng(12453)
    ys, rs = rng.normal(size=(2, 6, 16)).astype(dtype)
    rs[:, 1::2] = rs[:, ::2]
    scale = np.linspace(.2, 2., 16).astype(dtype)
    solve = jax.jit(_anderson)
    for n in (2, 4, 6):
        dy, dr = np.diff(ys[-n:], axis=0).T, np.diff(rs[-n:], axis=0).T
        matrix = dr / scale[:, None]
        coeff = np.linalg.lstsq(matrix, rs[-1] / scale, rcond=np.finfo(dtype).eps * max(matrix.shape))[0]
        expect = rs[-1] - (dy + dr) @ coeff
        np.testing.assert_allclose(solve(ys, rs, n, scale), expect,
                                   atol=35*np.finfo(dtype).eps, rtol=35*np.finfo(dtype).eps)
    np.testing.assert_array_equal(solve(ys, np.ones_like(rs), 6, scale), np.ones(16))


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_public_colony_mixed_trust_freezes_members_and_preserves_support(monkeypatch, dtype):
    observed = []
    build = ChartKernel.build

    def record(operation):
        kernel = build(operation)
        def capture(fn, *args):
            result = fn(*args)
            if hasattr(result, "accepted"):
                observed.extend(np.atleast_1d(result.accepted).tolist())
            return result
        return ChartKernel(lambda *args: capture(kernel.one, *args),
                           lambda *args: capture(kernel.many, *args))
    monkeypatch.setattr(ChartKernel, "build", record)
    with jax.enable_x64(dtype is np.float64):
        template, arrays = build_tmaze()
        colony = Colony()
        ids = [colony.spawn(template, arrays) for _ in range(5)]
        engine = next(iter(colony.engines.values()))
        prior = engine.arena.member_of["prior"].parameter("statistic")
        for i, mass in enumerate((.5, .7, 1.)):
            state = colony.instances[ids[i]].state
            logp = np.full(8, NEG, dtype=dtype)
            logp[0] = np.log(mass)
            if mass < 1:
                logp[1] = np.log1p(-mass)
            params = {**state.params, prior.key: state.params[prior.key].at[prior.index].set(logp)}
            colony.instances[ids[i]].state = _replace(state, params=params)
        state = colony.instances[ids[3]].state
        colony.instances[ids[3]].state = _replace(state, messages=state.messages.at[0].set(jnp.nan))
        reference = Engine(compile_template(template, arrays=arrays))
        colony.instances[ids[4]].state = reference.run(
            colony.instances[ids[0]].state, 200, tol=1e-10,
        ).require_converged()
        expect = [reference.run(colony.instances[i].state, 10, tol=1e-10, harden_last=10) for i in ids]
        assert not reference._outer_kernels.stages
        found = colony.run(10, tol=1e-10, outer_solver="compiled_trust_region")
        assert any(count > 0 for count in observed)
        assert found[ids[4]].iterations < found[ids[0]].iterations
        eps = np.finfo(dtype).eps
        for i, other in zip(ids, expect, strict=True):
            result = found[i]
            assert (result.status, result.termination_reason, result.failure) == (
                other.status, other.termination_reason, other.failure)
            if i == ids[3]:
                assert result.outer_map_evaluations == 0
                continue
            left, right = np.asarray(result.state.messages), np.asarray(other.state.messages)
            np.testing.assert_array_equal(left == NEG, right == NEG)
            # Compare normalized beliefs; chart offsets are a gauge of the messages.
            for name, marginal in engine.marginals(result.state).items():
                np.testing.assert_allclose(marginal, reference.marginals(other.state)[name],
                                           atol=max(100*eps, 5e-9))
            np.testing.assert_allclose(engine.bfe(result.state), reference.bfe(other.state),
                                       atol=max(100*eps, 5e-9))
