"""A serialized learned goal has one posterior across execution batches."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import digamma, gammaln
from test_goal_hyperprior import _arrays, _model

from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import CompileError, compile_template
from cffg.core import BlobRef, VariableSpec
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine, RunStatus
from cffg.execution.synchronous import make_sweep
from cffg.outer import OuterArrays, build_outer_chart, outer_chart_signature
from cffg.schedule import _arena_signature


def mixed_goals(states=(2, 3), owners=None, observations=3):
    template = _model(slices=len(states), observations=observations)
    arrays = _arrays(observations=observations)
    owners = owners or ("Cc",) * len(states)
    for index, (cardinality, owner) in enumerate(zip(states, owners)):
        template.variables[f"z{index}"] = VariableSpec(f"z{index}", cardinality)
        local = _arrays(observations=observations, states=cardinality)
        for factor, role, key, value in (
            (f"obs{index}", "matrix", f"A{index}", local["A"]),
            (f"state{index}", "statistic", f"d{index}", local["d"]),
            (f"cprior{index}", "counts", owner, arrays["Cc"].copy()),
        ):
            template.factors[factor] = replace(
                template.factors[factor], param_blobs={role: key}
            )
            template.blobs[key] = BlobRef(key, key + ".npy")
            arrays[key] = value
    return template, arrays


def constant_channels(arrays, states=(2, 3)):
    predictions = (np.array([0.65, 0.3, 0.05]), np.array([0.05, 0.2, 0.75]))
    for index, cardinality in enumerate(states):
        arrays[f"A{index}"] = np.tile(predictions[index][:, None], (1, cardinality))
    return sum(predictions)


def test_mixed_shape_step_pools_one_shared_goal():
    template, arrays = mixed_goals()
    engine = Engine(compile_template(template, arrays=arrays))
    assert len(engine.arena.gfe_groups) == 2
    state = engine.step(engine.init_state())
    marginals = engine.marginals(state)
    expected = arrays["Cc"] + marginals["x0"] + marginals["x1"]
    for posterior in engine.goal_marginals(state).values():
        np.testing.assert_allclose(posterior.alpha, expected, atol=1e-7)
    np.testing.assert_allclose(engine.goal_prior_updates(state)["Cc"], expected)


def test_off_coordinate_energy_charges_one_shared_parameter_term():
    template, arrays = mixed_goals()
    predicted = constant_channels(arrays)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.step(engine.init_state())
    counts = jnp.array([0.1, 0.4, 0.3])
    messages = state.messages
    for group in engine.arena.gfe_groups:
        indices = engine.arena.message_indices(group.goal_count_scatter, 3)
        messages = messages.at[indices].set(counts)
        predictive_indices = engine.arena.message_indices(group.scatter_x, 3)
        messages = messages.at[predictive_indices].set(
            jnp.log(jnp.array([0.8, 0.1, 0.1]))
        )
    state = eqx.tree_at(lambda value: value.messages, state, messages)
    alpha0 = jnp.asarray(arrays["Cc"])
    alpha = alpha0 + counts

    def log_beta(a):
        return gammaln(a).sum() - gammaln(a.sum())

    expected = (
        log_beta(alpha0)
        - log_beta(alpha)
        + (counts - predicted) @ (digamma(alpha) - digamma(alpha.sum()))
    )
    np.testing.assert_allclose(engine.bfe(state), expected, atol=1e-7)


def test_equal_prior_values_do_not_tie_different_blob_names():
    template, arrays = mixed_goals(owners=("Cc", "independent"))
    constant_channels(arrays)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.step(engine.init_state())
    marginals = engine.marginals(state)
    for index, owner in enumerate(("Cc", "independent")):
        np.testing.assert_allclose(
            engine.goal_prior_updates(state)[owner],
            arrays[owner] + marginals[f"x{index}"],
        )
    assert not np.array_equal(*engine.goal_prior_updates(state).values())


@pytest.mark.parametrize("schedule", ["flood", "auto"])
@pytest.mark.parametrize("solver", ["picard", "trust_region"])
def test_mixed_goal_convergence_and_rollover(schedule, solver):
    template, arrays = mixed_goals()
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    if schedule == "auto":
        assert engine.schedule.kind == "mixed"
    result = engine.run(
        engine.init_state(), 100, tol=1e-9, harden_last=0, outer_solver=solver
    )
    assert result.status is RunStatus.CONVERGED
    marginals = engine.marginals(result.state)
    posterior = engine.goal_prior_updates(result.state)["Cc"]
    expected = arrays["Cc"] + marginals["x0"] + marginals["x1"]
    np.testing.assert_allclose(posterior, expected, atol=2e-7)
    assert np.isfinite(result.free_energies).all()
    fresh = engine.with_goal_priors(engine.init_state(), {"Cc": posterior})
    next_state = engine.step(fresh)
    next_marginals = engine.marginals(next_state)
    np.testing.assert_allclose(
        engine.goal_prior_updates(next_state)["Cc"],
        posterior + next_marginals["x0"] + next_marginals["x1"],
    )
    with pytest.raises(ValueError, match="fresh"):
        engine.with_goal_priors(result.state, {"Cc": posterior})


def test_outer_trial_coordinates_keep_goal_aliases_equal():
    template, arrays = mixed_goals(states=(2, 3, 2))
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.step(engine.init_state())
    base = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
    chart = build_outer_chart(engine.arena, base)
    assert len(chart.vectors) == 2
    coordinates = chart.pack(base) + 0.03
    trial = jax.jit(chart.unpack)(coordinates)
    np.testing.assert_allclose(chart.pack(trial), coordinates)
    _, tangent = jax.jvp(chart.unpack, (coordinates,), (jnp.ones_like(coordinates),))
    (plan,) = engine.arena.goal_parameter_plans
    for slots in (plan.prior_scatter, plan.count_scatter):
        values = np.asarray(trial.messages)[engine.arena.message_indices(slots, 3)]
        np.testing.assert_array_equal(values, np.broadcast_to(values[0], values.shape))
        derivatives = np.asarray(tangent.messages)[
            engine.arena.message_indices(slots, 3)
        ]
        np.testing.assert_array_equal(
            derivatives, np.broadcast_to(derivatives[0], derivatives.shape)
        )
    broken = replace(
        chart, vectors=(replace(chart.vectors[0], aliases=None), *chart.vectors[1:])
    )
    assert outer_chart_signature(chart) != outer_chart_signature(broken)


def test_global_plan_reconstruction_and_fingerprint(tmp_path):
    template, arrays = mixed_goals(states=(2, 3, 2))
    arena = compile_template(template, arrays=arrays)
    (plan,) = arena.goal_parameter_plans
    assert plan.prior_blob == "Cc"
    assert plan.group_rows == (("gfeC:3x2", (0, 1)), ("gfeC:3x3", (0,)))
    assert plan.parameter_edges == ("cparam0", "cparam2", "cparam1")
    with pytest.raises(TypeError):
        plan.count_scatter[0] = 0
    restored = from_json(to_json(template))
    assert structure_hash(restored) == structure_hash(template)
    path = tmp_path / "shared-goals.cffgz"
    save_bundle(path, restored, arrays)
    bundled, payloads, _ = load_bundle(path)
    rebuilt = compile_template(bundled, arrays=payloads)
    assert _arena_signature(rebuilt) == _arena_signature(arena)
    assert _arena_signature(
        replace(arena, goal_parameter_plans=())
    ) != _arena_signature(arena)
    engine = Engine(rebuilt)
    assert engine.goal_prior_updates(engine.step(engine.init_state()))["Cc"].shape == (
        3,
    )


def test_group_order_damping_and_vmapped_instances():
    template, arrays = mixed_goals(states=(2, 3, 2))
    arena = compile_template(template, arrays=arrays)
    engine = Engine(arena)
    state = engine.step(engine.init_state())
    reversed_engine = Engine(
        replace(arena, gfe_groups=tuple(reversed(arena.gfe_groups)))
    )
    next_state = engine.step(state, damp=0.25)
    np.testing.assert_allclose(
        next_state.messages, reversed_engine.step(state, damp=0.25).messages
    )
    np.testing.assert_allclose(engine.bfe(next_state), reversed_engine.bfe(next_state))
    for posterior in engine.goal_marginals(next_state).values():
        np.testing.assert_array_equal(
            posterior.alpha, engine.goal_prior_updates(next_state)["Cc"]
        )

    raw_step = make_sweep(arena)

    def advance(messages, params):
        return raw_step(
            messages,
            state.g_eta,
            state.g_lam,
            state.g_obs,
            state.g_mask,
            state.ef_eta,
            state.ef_obs,
            state.ef_mask,
            params,
        )[0]

    another = engine.step(
        engine.with_goal_priors(engine.init_state(), {"Cc": arrays["Cc"] * 2})
    )
    messages = jnp.stack([state.messages, another.messages])
    params = jax.tree.map(lambda a, b: jnp.stack([a, b]), state.params, another.params)
    batched = jax.jit(jax.vmap(advance))(messages, params)
    for index, single in enumerate((state, another)):
        np.testing.assert_allclose(
            batched[index], advance(single.messages, single.params), atol=1e-7
        )
    assert not np.array_equal(batched[0], batched[1])


def test_float32_sparse_support_with_32_shared_composites():
    with jax.enable_x64(False):
        template, arrays = mixed_goals(states=(2, 3, 4, 5) * 8, observations=32)
        for index in range(32):
            matrix = arrays[f"A{index}"]
            matrix[-1] = 0
            if index % 2 == 0:
                matrix[0] = 0
            arrays[f"A{index}"] = (matrix / matrix.sum(axis=0)).astype(np.float32)
        arrays["Cc"] = arrays["Cc"].astype(np.float32)
        arrays["Cc"][-1] = np.finfo(np.float32).tiny * 16
        engine = Engine(compile_template(template, arrays=arrays))
        state = engine.step(engine.init_state())
        posterior = engine.goal_prior_updates(state)["Cc"]
        marginals = engine.marginals(state)
        predicted = sum(marginals[f"x{i}"] for i in range(32))
        assert posterior.dtype == np.float32
        np.testing.assert_allclose(posterior, arrays["Cc"] + predicted, rtol=2e-6)
        assert posterior[-1] == arrays["Cc"][-1]
        assert posterior[0] > arrays["Cc"][0]
        for index in range(0, 32, 2):
            assert marginals[f"x{index}"][0] == 0
        (plan,) = engine.arena.goal_parameter_plans
        counts = np.asarray(state.messages)[
            engine.arena.message_indices(plan.count_scatter, 32)
        ]
        assert np.all(counts[:, -1] == 0)
        assert np.all(counts[:, 0] > 0)
        assert np.isfinite(engine.bfe(state))


@pytest.mark.parametrize(
    "bad_prior", [np.array([0.0, 1.0, 1.0]), np.array([np.nan, 1.0, 1.0]), np.ones(4)]
)
def test_shared_goal_rejects_improper_or_incompatible_prior(bad_prior):
    template, arrays = mixed_goals()
    arrays["Cc"] = bad_prior
    with pytest.raises(CompileError, match="prior"):
        compile_template(template, arrays=arrays)
