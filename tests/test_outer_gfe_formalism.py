"""Executable anchors for the mixed-GFE outer-solver formalism."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.agent import RollingAgent
from cffg.compiler import NEG, compile_template
from cffg.engine import Engine, RunStatus, _replace
from cffg.models import build_factored_corridor
from cffg.models.rollout import build_rollout
from cffg.models.tmaze import build_tmaze
from cffg.outer import (
    OuterArrays,
    _make_trust_kernel,
    build_outer_chart,
    outer_chart_signature,
)


def _categorical_chart(engine: Engine, state) -> np.ndarray:
    """Pack one log-ratio chart per supported categorical message."""
    coordinates = []
    messages = np.asarray(state.messages)
    for edge in sorted(engine.arena.edges.values(), key=lambda edge: edge.name):
        if edge.dim or edge.ef or edge.param_shape:
            continue
        for slot in (edge.slot0, edge.slot1):
            indices = engine.arena.message_indices(slot, edge.card)
            block = messages[indices]
            support = np.flatnonzero(block != NEG)
            if len(support) > 1:
                coordinates.extend(block[support[1:]] - block[support[0]])
    return np.asarray(coordinates)


def _gauge_shift(engine: Engine, state):
    messages = np.asarray(state.messages).copy()
    shifts = np.linspace(-0.25, 0.25, engine.arena.n_slots)
    for edge in engine.arena.edges.values():
        if edge.dim or edge.ef or edge.param_shape:
            continue
        for slot in (edge.slot0, edge.slot1):
            indices = engine.arena.message_indices(slot, edge.card)
            supported = messages[indices] != NEG
            messages[indices[supported]] += shifts[slot]
    return _replace(state, messages=jnp.asarray(messages))


def _undamped_outer_step(engine: Engine, state):
    assert not engine.arena.mu3_groups
    messages, g_eta, g_lam, ef_eta, inner_residual, inner_code, *_ = engine._iter_step(
        state.messages,
        state.g_eta,
        state.g_lam,
        state.g_obs,
        state.g_mask,
        state.ef_eta,
        state.ef_obs,
        state.ef_mask,
        state.params,
        0.0,
        True,
    )
    assert int(inner_code) == 0
    assert np.isfinite(float(inner_residual))
    return _replace(
        state,
        messages=messages,
        g_eta=g_eta,
        g_lam=g_lam,
        ef_eta=ef_eta,
    )


def _outer_residual(engine: Engine, state) -> np.ndarray:
    return (
        _categorical_chart(engine, _undamped_outer_step(engine, state))
        - _categorical_chart(engine, state)
    )


def _checked_d1_fixed_point() -> tuple[Engine, object]:
    k, u = 4, 2
    sensor = np.full((k, k), 0.1 / (k - 1))
    np.fill_diagonal(sensor, 0.9)
    transition_counts = np.stack(
        [
            np.full((k, k), 60.0 / k),
            np.ones((k, k)),
        ]
    )
    template, arrays = build_rollout(
        "outer-residual-d1",
        steps=2,
        d=np.full(k, 1.0 / k),
        A=sensor,
        c=np.full(k, 1.0 / k),
        e=np.full(u, 1.0 / u),
        Bs=np.full((u, k, k), 1.0 / k),
        gfe=False,
        epistemic_b=True,
    )
    arrays["Bc"] = transition_counts
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.run(
        engine.init_state(), 80, tol=1e-12
    ).require_converged()
    return engine, state


def _checked_tmaze_fixed_point() -> tuple[Engine, object]:
    template, arrays = build_tmaze()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.run(
        engine.init_state(), 400, tol=1e-13
    ).require_converged()
    return engine, state


@pytest.mark.slow
def test_packed_outer_residual_is_gauge_invariant_at_checked_fixed_points():
    for engine, state in (_checked_tmaze_fixed_point(), _checked_d1_fixed_point()):
        shifted = _gauge_shift(engine, state)
        np.testing.assert_allclose(
            _categorical_chart(engine, shifted),
            _categorical_chart(engine, state),
            atol=2e-14,
        )
        residual = _outer_residual(engine, state)
        shifted_residual = _outer_residual(engine, shifted)
        np.testing.assert_allclose(shifted_residual, residual, atol=2e-10)
        assert np.max(np.abs(residual), initial=0.0) < 2e-10


def test_matrix_free_trust_kernel_has_single_and_vmap_parity():
    target = jnp.asarray([0.25, -0.5, 0.75])
    kernel = _make_trust_kernel(lambda value: jnp.tanh(value) - target, 3)
    coordinates = jnp.asarray(
        [[1.0, -1.0, 0.2], [-0.4, 0.3, 1.5]]
    )
    scales = jnp.asarray([[1.0, 0.5, 2.0], [0.8, 1.2, 0.7]])
    radii = jnp.asarray([0.35, 1.1])
    accelerated = jnp.zeros_like(coordinates)
    flags = jnp.zeros(2, dtype=bool)

    batched = jax.vmap(kernel)(
        coordinates, scales, radii, accelerated, flags
    )
    sequential = [
        kernel(coordinates[i], scales[i], radii[i], accelerated[i], flags[i])
        for i in range(2)
    ]
    for batched_value, *single_values in zip(batched, *sequential):
        np.testing.assert_allclose(
            batched_value,
            np.stack(single_values),
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_array_less(
        np.linalg.norm(np.asarray(batched[0]), axis=1),
        np.asarray(radii) + 1e-12,
    )


def test_benign_candidate_path_keeps_the_full_root_certificate():
    template, arrays = build_tmaze()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()

    cold = engine.run(state, 400, tol=1e-13)
    warm = engine.run(state, 400, tol=1e-13)
    legacy = engine.run(
        state,
        400,
        tol=1e-13,
        damp=0.5,
        outer_solver="picard",
    )

    assert cold.status is warm.status is legacy.status is RunStatus.CONVERGED
    assert cold.termination_reason.value == "root_converged"
    assert cold.outer_residual < 1e-13
    assert np.isfinite(cold.outer_gradient_norm)
    assert cold.trial_evaluations == cold.linear_iterations == 0
    assert cold.solver_compilations == 1
    assert warm.solver_compilations == 0
    assert warm.outer_map_evaluations == cold.outer_map_evaluations
    for name, marginal in engine.marginals(legacy.state).items():
        np.testing.assert_allclose(
            engine.marginals(cold.state)[name], marginal, atol=2e-12
        )
    np.testing.assert_allclose(
        engine.bfe(cold.state), engine.bfe(legacy.state), atol=2e-12
    )

    params = dict(state.params)
    prior = engine.arena.member_of["prior"].parameter("statistic")
    changed_prior = np.full(8, NEG)
    changed_prior[:2] = np.log([0.8, 0.2])
    params[prior.key] = params[prior.key].at[prior.index].set(changed_prior)
    changed = _replace(state, params=params)
    reused = engine.run(changed, 400, tol=1e-13)

    fresh = Engine(compile_template(template, arrays=arrays))
    reference = fresh.run(
        _replace(fresh.init_state(), params=params), 400, tol=1e-13
    )
    assert reused.status is reference.status is RunStatus.CONVERGED
    assert reused.solver_compilations == 0
    assert reference.solver_compilations == 1
    for name, marginal in fresh.marginals(reference.state).items():
        np.testing.assert_allclose(
            engine.marginals(reused.state)[name], marginal, atol=2e-12
        )
    assert not np.allclose(
        engine.marginals(reused.state)["z0"],
        engine.marginals(cold.state)["z0"],
    )


def test_outer_kernel_cache_keys_exact_support_and_dtype():
    template, arrays = build_tmaze()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()

    def chart(st):
        return build_outer_chart(
            engine.arena,
            OuterArrays(st.messages, st.g_eta, st.g_lam, st.ef_eta),
            engine._outer_eliminated,
        )

    full = chart(state)
    assert engine._outer_kernel_plan(full, state, []).compiled
    assert not engine._outer_kernel_plan(full, state, []).compiled

    block = next(item for item in full.categorical if item.coordinates.size)
    sparse = _replace(
        state,
        messages=state.messages.at[int(block.coordinates[-1])].set(NEG),
    )
    sparse_chart = chart(sparse)
    assert full.support_changed(sparse.messages)
    assert outer_chart_signature(sparse_chart) != outer_chart_signature(full)
    assert engine._outer_kernel_plan(sparse_chart, sparse, []).compiled

    float32 = jax.tree_util.tree_map(
        lambda value: (
            value.astype(jnp.float32)
            if jnp.issubdtype(value.dtype, jnp.inexact)
            else value
        ),
        sparse,
    )
    float32_chart = chart(float32)
    assert outer_chart_signature(float32_chart) != outer_chart_signature(
        sparse_chart
    )
    assert engine._outer_kernel_plan(float32_chart, float32, []).compiled


def one_update_corridor_case() -> tuple[Engine, object, np.ndarray]:
    """Freeze the minimal post-conjugate-update state from the retained cycle."""
    length, modes, actions, horizon = 8, 2, 3, 8
    position_prior = np.r_[1.0, np.zeros(length - 1)]
    mode_prior = np.full(modes, 1.0 / modes)
    action_prior = np.full(actions, 1.0 / actions)
    transition_prior = np.full(
        (modes * actions, length, length), 1.0 / length
    )
    sensor = np.eye(length * modes)
    utility = np.zeros(length * modes)
    utility[(length - 1) * modes :] = 3.0
    preference = np.exp(utility - utility.max())
    preference /= preference.sum()

    def builder(*, past, steps, **kwargs):
        return build_factored_corridor(
            past=past,
            steps=steps,
            length=length,
            modes=modes,
            position_prior=position_prior,
            mode_prior=mode_prior,
            observation_matrix=sensor,
            observation_source="state",
            preference=preference,
            actions=action_prior,
            dtype=np.float64,
            gfe=True,
            **kwargs,
        )

    agent = RollingAgent(
        builder,
        horizon,
        learn=("position_transition",),
        priors={"position_transition": transition_prior},
        damp=0.5,
        n_sweeps=400,
        select="sample",
        seed=0,
    )
    alpha = transition_prior.copy()
    alpha[5, 0, 0] += 1.0
    agent.learnables.params["position_transition"]["alpha"] = alpha
    agent.beliefs = {
        "position": position_prior.copy(),
        "mode": np.array([0.0, 1.0]),
    }
    return agent.plan, agent._with_params(agent.plan), alpha


@pytest.mark.slow
def test_one_update_corridor_is_globalized_without_changing_the_equations():
    engine, state, alpha = one_update_corridor_case()
    assert engine.schedule.kind == "mixed"
    assert len(engine.arena.gfe_groups) == 1
    assert len(engine.arena.gfe_groups[0].members) == 8
    assert not engine.arena.gfe_tm_groups

    baseline = np.full_like(alpha, 1.0 / 8.0)
    changed = np.argwhere(alpha != baseline)
    np.testing.assert_array_equal(changed, [[5, 0, 0]])
    np.testing.assert_allclose(alpha[5, 0, 0], 9.0 / 8.0)

    result = engine.run(state, 400, tol=1e-10)
    assert result.status is RunStatus.CONVERGED
    assert result.inner_status is RunStatus.CONVERGED
    assert result.inner_residual < 1e-10
    assert result.outer_residual < 1e-10
    assert np.isfinite(result.outer_merit) and result.outer_merit >= 0.0
    assert np.isfinite(result.outer_gradient_norm)
    assert result.accepted_steps > 0
    assert result.termination_reason.value == "root_converged"
    assert np.isfinite(result.free_energies[-1])

    with pytest.raises(ValueError, match="damp does not apply"):
        engine.run(state, 1, tol=1e-10, damp=0.5)

    legacy = engine.run(
        state, 400, tol=1e-10, damp=0.5, outer_solver="picard"
    )
    assert legacy.status is RunStatus.BUDGET_EXHAUSTED
    assert legacy.inner_status is RunStatus.CONVERGED
    assert 1.5 < legacy.residual < 1.9
    assert legacy.termination_reason.value == "legacy_picard"

    energy_tail = np.asarray(legacy.free_energies[-128:])
    delta_tail = np.asarray(legacy.message_deltas[-128:])
    assert np.isfinite(energy_tail).all() and np.isfinite(delta_tail).all()
    assert np.ptp(energy_tail) > 20.0
    assert delta_tail.min() > 1.0
