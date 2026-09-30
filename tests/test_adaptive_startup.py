"""Admission guards do not evaluate deferred functions or invent convergence."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import Engine, RunStatus, compile_template
from cffg.core import BlobRef, FactorSpec, FunctionSpec, PortRole, PortSpec, VariableSpec
from cffg.execution.transport import _set_gauss_block
from cffg.execution.startup import make_startup_program
from cffg.execution.synchronous import make_sweep
from cffg.failure_facts import FailureReason
from cffg.mean_functions import MEAN_IMPLEMENTATIONS, MeanImplementation
from cffg.runtime_types import _replace
from nonlinear_fixture import model


def args(state):
    return (state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.ef_eta, state.ef_obs, state.ef_mask, state.params)


def counter_model(monkeypatch, *, method="linearization", members=1, nonfinite_function=False):
    calls = []

    def evaluate(x, parameters, configuration):
        jax.debug.callback(lambda value: calls.append(float(value)), parameters[0], ordered=True)
        return jnp.full_like(x, jnp.nan) if nonfinite_function else x + parameters[0]

    identifier = "cffg.test.startup_counter"
    monkeypatch.setitem(MEAN_IMPLEMENTATIONS, identifier,
                        MeanImplementation(identifier, "1.0.0", lambda di, do, c: (1,), evaluate))
    template, arrays = model(method, dim=2)
    for index in range(members):
        suffix = str(index)
        for variable in ("x", "y"):
            template.variables[variable + suffix] = VariableSpec(variable + suffix, dim=2)
        for name in ("prior", "evidence", "sensor"):
            factor = template.factors[name]
            params = dict(factor.param_blobs)
            if name == "sensor":
                blob = "theta" + suffix
                arrays[blob] = np.array([.1 + .1 * index])
                template.blobs[blob] = BlobRef(blob, blob + ".npy")
                params["parameters"] = blob
            template.factors[name + suffix] = replace(
                factor, name=name + suffix, edges={r: v + suffix for r, v in factor.edges.items()},
                param_blobs=params,
                function=FunctionSpec(identifier, "1.0.0", method=method) if name == "sensor" else None,
            )
    for name in ("prior", "evidence", "sensor"):
        del template.factors[name]
    for name in ("x", "y"):
        del template.variables[name]
    template.constraints = [replace(c, target="sensor" + str(i))
                            for i in range(members) for c in template.constraints]
    engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
    group = next(g for g in engine.arena.gauss_groups if g.kernel.binding.iterative)
    return engine, group, calls


def input_blocks(engine, group, state, precisions, etas=None):
    precisions = jnp.asarray(precisions, dtype=state.g_lam.dtype)
    etas = jnp.zeros((len(group.members), 2), state.g_eta.dtype) if etas is None else jnp.asarray(etas, state.g_eta.dtype)
    e, p = _set_gauss_block(engine.arena, state.g_eta, state.g_lam, group.gather[:, 1], 2, etas, precisions)
    return _replace(state, g_eta=e, g_lam=p)


@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_only_ready_members_evaluate_even_under_an_outer_vmap(monkeypatch, method, dtype):
    with jax.enable_x64(dtype is np.float64):
        engine, group, calls = counter_model(monkeypatch, method=method, members=2)
        state = input_blocks(engine, group, engine.init_state(), [np.eye(2), np.diag([1., 0.])])
        program = engine._schedule_program.startup
        eligible = program.eligibility(state.g_eta, state.g_lam)
        admitted = tuple(jnp.zeros_like(v) for v in eligible)
        result = program.step(*args(state), eligible, admitted)
        jax.block_until_ready(result)
        assert int(result.sweep[5]) == 0
        assert calls and np.allclose(calls, .1)
        np.testing.assert_array_equal(result.admitted[0], [True, False])
        calls.clear()
        partial = input_blocks(engine, group, state, [np.diag([1., 0.])] * 2)
        batched = jax.tree.map(lambda a, b: jnp.stack([a, b]), state, partial)
        actual = jax.jit(jax.vmap(lambda s: program.step(*args(s), eligible, admitted)))(batched)
        jax.block_until_ready(actual)
        assert calls and np.allclose(calls, .1)
        np.testing.assert_array_equal(actual.sweep[5], [0, 0])
        np.testing.assert_array_equal(actual.admitted[0], [[True, False], [False, False]])


@pytest.mark.parametrize("role", [0, 1])
@pytest.mark.parametrize("coordinate", ["eta", "precision", "subnormal"])
def test_nonzero_owned_coordinates_on_either_role_forbid_deferral(monkeypatch, role, coordinate):
    engine, group, calls = counter_model(monkeypatch)
    state = input_blocks(engine, group, engine.init_state(), [np.diag([1., 0.])])
    slot = group.scatter[0, role]
    if coordinate == "precision":
        state = _replace(state, g_lam=state.g_lam.at[engine.arena.g_lam_indices(slot, 2)[0, 0]].set(1.))
    else:
        value = np.finfo(state.g_eta.dtype).tiny / 2 if coordinate == "subnormal" else 1.
        state = _replace(state, g_eta=state.g_eta.at[engine.arena.g_eta_indices(slot, 2)[0]].set(value))
    program = engine._schedule_program.startup
    eligible = program.eligibility(state.g_eta, state.g_lam)
    assert not eligible[0][0]
    check = program.inspect(state.g_eta, state.g_lam, state.g_obs, state.g_mask, state.params,
                            eligible, (jnp.array([False]),))
    assert int(check.failure.reason) == FailureReason.IMPROPER_CAVITY
    result = engine.run(state, 3, tol=1e-6)
    assert result.status is RunStatus.DIVERGED and result.failure.reason == "improper_cavity"
    np.testing.assert_array_equal(result.state.g_eta, state.g_eta)
    np.testing.assert_array_equal(result.state.g_lam, state.g_lam)
    assert not calls


def test_admission_latch_forbids_later_partial_information(monkeypatch):
    engine, group, calls = counter_model(monkeypatch)
    state = input_blocks(engine, group, engine.init_state(), [np.diag([1., 0.])])
    program = engine._schedule_program.startup
    eligible = program.eligibility(state.g_eta, state.g_lam)
    check = program.inspect(state.g_eta, state.g_lam, state.g_obs, state.g_mask, state.params,
                            eligible, (jnp.array([True]),))
    assert int(check.failure.reason) == FailureReason.IMPROPER_CAVITY
    assert not calls


def test_unrelated_nonfinite_state_does_not_invent_an_adaptive_origin(monkeypatch):
    engine, group, calls = counter_model(monkeypatch)
    state = input_blocks(engine, group, engine.init_state(), [np.eye(2)])
    for corrupt_input in (False, True):
        current = input_blocks(engine, group, state, [-np.eye(2)]) if corrupt_input else state
        current = _replace(current, params={**current.params, "unrelated": jnp.array([jnp.nan])})
        result = engine.run(current, 5, tol=1e-6)
        assert result.status is RunStatus.DIVERGED
        assert result.failure.reason == "nonfinite_state" and result.failure.factor is None
        assert result.failure.phase == "initialization" and not calls


def test_indefinite_output_potentials_and_conditioning_keep_the_declared_tilt_contract(monkeypatch):
    from cffg.kernel_runtime import parameters

    engine, group, calls = counter_model(monkeypatch)
    state = engine.init_state()
    eta = jnp.zeros((1, 2, 2), state.g_eta.dtype)
    precision = jnp.zeros((1, 2, 2, 2), state.g_lam.dtype).at[0, 1].set(jnp.diag(jnp.array([1., 0.])))
    callback = group.kernel.binding.admission
    par = parameters(group, state.params)
    small = precision.at[0, 0].set(-jnp.eye(2) * .1)
    _, reason = callback(eta, small, par, group.dims, None, None, None)
    assert int(reason[0]) == FailureReason.NONE
    impossible = precision.at[0, 0].set(-jnp.eye(2) * 100)
    _, reason = callback(eta, impossible, par, group.dims, None, None, None)
    assert int(reason[0]) == FailureReason.IMPROPER_TILT
    _, reason = callback(eta, impossible, par, group.dims, 0, jnp.zeros((1, 2)), jnp.ones(1))
    assert int(reason[0]) == FailureReason.NONE and not calls


def test_admission_supports_bindings_without_optional_solver_diagnostics(monkeypatch):
    engine, group, calls = counter_model(monkeypatch, members=2)
    message = group.kernel.binding.message
    binding = replace(group.kernel.binding, message=lambda *xs: message(*xs).outgoing)
    replacement = replace(group, kernel=replace(group.kernel, binding=binding))
    arena = replace(engine.arena, gauss_groups=tuple(replacement if g is group else g for g in engine.arena.gauss_groups))
    program = make_startup_program(arena, make_sweep)
    state = input_blocks(engine, group, engine.init_state(), [np.eye(2), np.diag([1., 0.])])
    eligible = program.eligibility(state.g_eta, state.g_lam)
    result = program.step(*args(state), eligible, tuple(jnp.zeros_like(v) for v in eligible))
    jax.block_until_ready(result)
    assert int(result.sweep[5]) == 0 and calls and np.allclose(calls, .1)


def test_failed_admitted_function_and_required_sigma_geometry_remain_failures(monkeypatch):
    engine, group, calls = counter_model(monkeypatch, nonfinite_function=True)
    state = input_blocks(engine, group, engine.init_state(), [np.eye(2)])
    result = engine.run(state, 5, tol=1e-6)
    assert result.status is RunStatus.DIVERGED
    assert result.failure.reason == "nonfinite_function" and result.failure.factor == "sensor0"
    assert calls

    engine, group, calls = counter_model(monkeypatch, method="sigma_point")
    state = input_blocks(engine, group, engine.init_state(), [np.eye(2) * 1e40], [[1e42, 1e42]])
    result = engine.run(state, 5, tol=1e-6)
    assert result.status is RunStatus.DIVERGED
    assert result.failure.reason == "unrepresentable_sigma_points" and result.failure.factor == "sensor0"


def test_functional_failure_after_startup_retains_consumed_work(monkeypatch):
    from cffg.outer_strategy import OuterServices, OuterTrustRegionStrategy
    from cffg.outer_structure import OuterStructure
    from cffg import run_results

    engine, _, calls = counter_model(monkeypatch)
    strategy = OuterTrustRegionStrategy(
        structure=OuterStructure.from_arena(engine.arena), hardening=(), eliminated={},
        kernel_provider=lambda *xs: pytest.fail("functional failure must precede trust solving"),
        iter_step=engine._iter_step, prime=engine._prime, reconstruct=engine._reconstruct,
        services=OuterServices(engine._state_is_finite, engine._has_zero_support, engine.bfe,
                               lambda state: (state, 0., 2, 3.)),
        adaptive_gaussian=True, startup=engine._schedule_program.startup,
    )
    outcome = strategy.run(engine.init_state(), 5, tol=1e-6, harden_last=0)
    assert outcome.iterations == 1 and len(outcome.message_deltas) == 1
    result = run_results.finalize_outer(outcome)
    assert result.status is RunStatus.DIVERGED
    assert result.failure.reason == "functional_divergence" and result.failure.iteration == 2
    assert not calls


@pytest.mark.parametrize("problem,reason", [
    ("parameters", FailureReason.NONFINITE_FUNCTION), ("noise", FailureReason.INVALID_NOISE),
    ("nonfinite_noise", FailureReason.INVALID_NOISE), ("nonfinite_input", FailureReason.IMPROPER_CAVITY),
    ("nonfinite_output", FailureReason.IMPROPER_CAVITY),
    ("output", FailureReason.IMPROPER_TILT), ("asymmetry", FailureReason.IMPROPER_CAVITY),
    ("null_information", FailureReason.IMPROPER_CAVITY), ("negative_input", FailureReason.IMPROPER_CAVITY),
])
def test_invalid_models_fail_before_partial_input_can_wait(monkeypatch, problem, reason):
    engine, group, calls = counter_model(monkeypatch)
    state = input_blocks(engine, group, engine.init_state(), [np.diag([1., 0.])])
    if problem in ("parameters", "noise", "nonfinite_noise"):
        key = group.name + (":theta" if problem == "parameters" else ":W")
        params = dict(state.params)
        params[key] = -params[key] if problem == "noise" else jnp.full_like(params[key], jnp.nan)
        state = _replace(state, params=params)
    elif problem == "nonfinite_input":
        state = input_blocks(engine, group, state, [np.diag([1., 0.])], [[jnp.nan, 0.]])
    elif problem in ("null_information", "negative_input"):
        state = input_blocks(engine, group, state,
                             [np.diag([1., -1. if problem == "negative_input" else 0.])], [[0., 1.]])
    else:
        precision = (-jnp.eye(2) * 100 if problem == "output" else
                     jnp.full((2, 2), jnp.nan) if problem == "nonfinite_output" else jnp.array([[1., 1.], [0., 1.]]))
        e, p = _set_gauss_block(engine.arena, state.g_eta, state.g_lam, group.gather[:, 0], 2,
                                jnp.zeros((1, 2)), precision[None])
        state = _replace(state, g_eta=e, g_lam=p)
    program = engine._schedule_program.startup
    eligible = program.eligibility(state.g_eta, state.g_lam)
    check = program.inspect(state.g_eta, state.g_lam, state.g_obs, state.g_mask, state.params,
                            eligible, (jnp.array([False]),))
    assert int(check.failure.reason) == reason
    result = engine.run(state, 5, tol=1e-6)
    assert result.status is RunStatus.DIVERGED
    assert result.failure.factor == "sensor0" and result.failure.reason == reason.name.lower()
    assert not calls


@pytest.mark.parametrize("budget", [0, 1, 5])
def test_permanently_partial_input_exhausts_exactly_the_requested_budget(monkeypatch, budget):
    engine, _, calls = counter_model(monkeypatch)
    # Replace the input prior with a rank-one observation: no proper forward cavity can arrive.
    template, arrays = model(affine=True, dim=2)
    template.variables["z"] = VariableSpec("z", dim=1)
    template.ports["z"] = PortSpec("z", "z", PortRole.DATA)
    arrays.update(A=np.array([[1., 0.]]), b=np.zeros(1), S=np.eye(1))
    template.blobs.update({n: BlobRef(n, n + ".npy") for n in ("A", "b", "S")})
    template.factors["prior"] = FactorSpec("prior", "gaussian_linear", {"in": "x", "out": "z"},
                                          param_blobs={"matrix": "A", "offset": "b", "cov": "S"})
    template.factors["sensor"] = replace(template.factors["sensor"],
                                         function=FunctionSpec("cffg.test.startup_counter", "1.0.0"))
    arrays["theta"] = np.array([.1])
    engine = Engine(compile_template(template, arrays=arrays))
    initial = engine.bind_data(engine.init_state(), {"z": np.zeros(1)})
    result = engine.run(initial, budget, tol=1e-6)
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.iterations == budget
    assert not result.free_energies and not calls
    if budget:
        assert result.failure.reason == "local_budget"
        assert result.failure.factor == "sensor" and result.failure.phase == "startup"
        assert result.failure.iteration == budget
    else:
        assert result.failure.reason == "outer_iteration_budget"
        assert result.failure.factor is None and result.failure.iteration == 0
        np.testing.assert_array_equal(result.state.g_eta, initial.g_eta)
        np.testing.assert_array_equal(result.state.g_lam, initial.g_lam)


def test_float32_affine_chain_with_partial_goals_matches_dense_evidence():
    from cstr_planning_fixture import planning_fixture
    from test_cstr_planning import affine_reference

    with jax.enable_x64(False):
        template, arrays, data = planning_fixture(past=0, horizon=2, dtype=np.float32, affine=True)
        engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
        result = engine.run(engine.bind_data(engine.init_state(), data), 40, tol=1e-5)
        state = result.require_converged()
        mean, covariance, energy = affine_reference({k: v.astype(np.float64) for k, v in arrays.items()}, data)
        tolerance = 512 * np.finfo(np.float32).eps
        np.testing.assert_allclose(engine.bfe(state), energy, rtol=tolerance, atol=tolerance)
        for index in range(3):
            actual_mean, actual_covariance = engine.gaussian_marginals(state)[f"s{index}"]
            np.testing.assert_allclose(actual_mean, mean[index*4:(index+1)*4], rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(actual_covariance, covariance[index*4:(index+1)*4, index*4:(index+1)*4],
                                       rtol=tolerance, atol=tolerance)


@pytest.mark.slow
def test_tenfold_affine_horizon_startup_remains_budgeted_without_finite_energy():
    from cstr_planning_fixture import planning_fixture

    template, arrays, data = planning_fixture(past=0, horizon=150, affine=True)
    engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
    result = engine.run(engine.bind_data(engine.init_state(), data), 4, tol=1e-6)
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.iterations == 4 and not result.free_energies
    assert result.failure.reason == "local_budget" and result.failure.phase == "startup"
    assert result.failure.factor.startswith("tr")
