"""Provenance follows actual writes without entering numerical decisions."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nonlinear_fixture import model
from test_constraint_data import constraint_model
from test_dirichlet_param_edges import _make_template
from test_ef_engine import gamma_poisson
from test_failure_facts import assert_fact
from test_hybrid_engine import mixlin_template
from test_schedule_program import program_arena, state_args, state_in_dtype
from test_shared_goals import mixed_goals

from cffg import Engine, compile_template
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.execution.contracts import DirichletMessages, ExecutionDelta
from cffg.execution.failures import FailureIndex, pooled_goal_failure
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.execution.synchronous import make_sweep
from cffg.failure_facts import FailurePhase, FailureReason
from cffg.schedule import BatchStage, ScanElem, ScanStage
from cffg.schedule_program import make_exact_unit, make_propagate


def propagation_args(state):
    return (state.messages, state.g_eta, state.g_lam, state.ef_eta,
            state.g_obs, state.g_mask, state.ef_obs, state.ef_mask, state.params)


def independent_priors(count=40):
    names = [f"p{i:03d}" for i in range(count)]
    template = Template(
        "independent", variables={n: VariableSpec(n, 2) for n in names},
        factors={n: FactorSpec(n, "categorical", {"out": n},
                               param_blobs={"statistic": "p"}) for n in names},
        blobs={"p": BlobRef("p", "p.npy")},
        ports={n: PortSpec(n, n, PortRole.DATA) for n in names},
    )
    return compile_template(template, arrays={"p": np.array([1., 0.])})


def marked_executors():
    def discrete(plan, incoming, params, op):
        delta = SCHEDULED_EXECUTORS.discrete(plan, incoming, params, op)
        return delta._replace(status=jnp.ones(incoming.shape[0], jnp.int32))
    return replace(SCHEDULED_EXECUTORS, discrete=discrete)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("mode", ["batch", "registerized", "flat"])
def test_ordered_updates_keep_first_cause_despite_finite_messages(dtype, mode):
    arena = independent_priors()
    engine = Engine(arena)
    state = state_in_dtype(engine, dtype)
    group = arena.groups[0]
    assert len(group.members) == 40
    rows = np.arange(39, -1, -1).reshape(-1, 1)
    if mode == "batch":
        stages = tuple(BatchStage("disc", 0, row) for row in rows)
    else:
        stages = (ScanStage(40, (ScanElem("disc", 0, rows),)),)
    if mode == "flat":
        from cffg.schedule_program import _flat_scan_fn
        from jax.scipy.special import logsumexp
        run = _flat_scan_fn(arena, stages[0], logsumexp, marked_executors())
    else:
        run = make_propagate(arena, stages, marked_executors())
        if mode == "registerized":
            assert run.scan_registerized == (True,)
    direct = run(*propagation_args(state))
    compiled = eqx.filter_jit(run)(*propagation_args(state))
    index = FailureIndex(arena)
    expected = int(index.ranks([group.members[39]])[0])
    for result in (direct, compiled):
        assert_fact(result[4], expected, FailureReason.LOCAL_BUDGET)
        assert np.isfinite(result[0]).all()
    reference = make_propagate(arena, stages)(*propagation_args(state))
    for actual, expected_array in zip(compiled[:4], reference[:4], strict=True):
        np.testing.assert_array_equal(actual, expected_array)
    assert_fact(reference[4], -1, FailureReason.NONE)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_real_nonlinear_cavity_reason_is_independent_in_batched_sweeps(dtype):
    template, arrays = model()
    engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
    state = state_in_dtype(engine, dtype)
    group = next(g for g in engine.arena.gauss_groups if g.type == "gaussian_nonlinear")
    indices = engine.arena.g_lam_indices(group.gather, 1)

    def sweep(precision):
        lam = state.g_lam.at[indices].set(jnp.ones(indices.shape, dtype))
        lam = lam.at[indices[:, 1]].set(precision)
        return engine._step(*state_args(replace(state, g_lam=lam)), 0., False)

    inputs = jnp.asarray([1., 0., -1.] * 10, dtype)
    result = eqx.filter_jit(jax.vmap(sweep))(inputs)
    expected = int(FailureIndex(engine.arena).ranks(["sensor"])[0])
    np.testing.assert_array_equal(result[6].factor, [-1, -1, expected] * 10)
    np.testing.assert_array_equal(result[6].reason,
                                  [0, 0, FailureReason.IMPROPER_CAVITY] * 10)
    np.testing.assert_array_equal(result[5], [0, 0, 2] * 10)


@pytest.mark.parametrize("propagation_fails", [False, True])
def test_exact_unit_distinguishes_propagation_and_certificate_origin(propagation_fails):
    arena = program_arena("two_pass")
    engine = Engine(arena)
    executors = marked_executors() if propagation_fails else SCHEDULED_EXECUTORS
    propagate = make_propagate(arena, engine.schedule.stages, executors)
    raw = make_sweep(arena)

    def certificate(*args):
        result = raw(*args)
        return (*result[:6], result[6]._replace(
            factor=jnp.asarray(0, jnp.int32),
            reason=jnp.asarray(FailureReason.LOCAL_DIVERGENCE, jnp.int32),
        ))

    unit = make_exact_unit(arena, propagate, certificate, lambda *args: jnp.asarray(37.))
    result = eqx.filter_jit(unit)(*state_args(engine.init_state()))
    assert_fact(result[9], 0,
                FailureReason.LOCAL_BUDGET if propagation_fails else FailureReason.LOCAL_DIVERGENCE,
                FailurePhase.PROPAGATION if propagation_fails else FailurePhase.CERTIFICATION)
    assert float(result[6]) == 37.
    assert not bool(result[7]) and not bool(result[8])


def test_parameter_counts_are_not_log_support_and_priors_have_their_own_names():
    template = _make_template("categorical", (2,))
    arena = compile_template(template, arrays={"alpha0": np.ones(2)})
    plan = arena.dirichlet_groups[0]
    index = FailureIndex(arena)
    out = DirichletMessages(jnp.array([[[0., -jnp.inf]]]),
                            jnp.ones((1, 2)), jnp.ones((1, 2)))
    delta = ExecutionDelta(out)
    assert_fact(index.delta(plan, delta, "disc"), -1, FailureReason.NONE)
    prior = index.delta(plan, delta._replace(outgoing=out._replace(
        prior=out.prior.at[0, 0].set(-jnp.inf))), "disc")
    assert_fact(prior, int(index.ranks(plan.priors)[0]), FailureReason.NONFINITE_OUTPUT)
    counts = index.delta(plan, delta._replace(outgoing=out._replace(
        counts=out.counts.at[0, 0].set(jnp.inf))), "disc")
    assert_fact(counts, int(index.ranks(plan.members)[0]), FailureReason.NONFINITE_OUTPUT)


def test_shared_pooled_count_reports_unknown_owner_instead_of_guessing():
    template, arrays = mixed_goals()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    assert_fact(pooled_goal_failure(engine.arena, state.messages), -1, FailureReason.NONE)
    plan = engine.arena.goal_parameter_plans[0]
    idx = engine.arena.message_indices(plan.count_scatter[0], plan.cardinality)
    messages = state.messages.at[idx].set(jnp.inf)
    assert_fact(jax.jit(lambda m: pooled_goal_failure(engine.arena, m))(messages),
                -1, FailureReason.NONFINITE_SHARED_GOAL)


def test_constraint_singleton_and_categorical_padding_only_check_written_coordinates():
    template, arrays = constraint_model("at_most_k", 1)
    arena = compile_template(template, arrays=arrays)
    plan = arena.constraint_groups[0]
    index = FailureIndex(arena)
    # A singleton constraint omits the member axis; padding is not scattered.
    outgoing = jnp.array([[0., -jnp.inf, jnp.nan], [-jnp.inf, 0., jnp.inf]])
    delta = ExecutionDelta(outgoing)
    assert_fact(index.delta(plan, delta, "disc"), -1, FailureReason.NONE)
    failed = delta._replace(outgoing=outgoing.at[1, 0].set(jnp.nan))
    assert_fact(index.delta(plan, failed, "disc"), int(index.ranks(plan.members)[0]),
                FailureReason.NONFINITE_OUTPUT)


def test_synchronous_selection_is_global_rank_not_group_execution_order(monkeypatch):
    from cffg.execution import synchronous
    arena = program_arena("mixed")
    engine = Engine(arena, schedule="flood")
    index = FailureIndex(arena)
    executed = []
    original = synchronous._execute_discrete

    def marked(plan, *args):
        executed.extend(plan.members)
        delta = original(plan, *args)
        return delta._replace(status=jnp.full(len(plan.members), 2, jnp.int32))

    monkeypatch.setattr(synchronous, "_execute_discrete", marked)
    result = make_sweep(arena)(*state_args(engine.init_state()), 0., False)
    assert len(executed) > 1
    assert_fact(result[6], int(index.ranks([min(executed)])[0]), FailureReason.LOCAL_DIVERGENCE)


@pytest.mark.parametrize("mode", ["flood", "batch", "registerized", "flat"])
def test_finite_ef_update_overflowing_float32_storage_keeps_its_owner(mode):
    from cffg.schedule_program import _flat_scan_fn
    from jax.scipy.special import logsumexp
    template, arrays = gamma_poisson(1)
    arena = compile_template(template, arrays=arrays)
    engine = Engine(arena)
    state = state_in_dtype(engine, jnp.float32)
    group_id, group = next((i, g) for i, g in enumerate(arena.ef_groups) if g.type == "gamma")
    params = {**state.params, group.name + ":eta": jnp.asarray([[1., -1e40]], jnp.float64)}
    state = replace(state, params=params)
    assert all(np.isfinite(v).all() for v in params.values())
    if mode == "flood":
        result = eqx.filter_jit(make_sweep(arena))(*state_args(state), 0., False)
        fact = result[6]
    else:
        stage = (BatchStage("ef", group_id, np.array([0])) if mode == "batch"
                 else ScanStage(1, (ScanElem("ef", group_id, np.array([[0]])),)))
        run = (_flat_scan_fn(arena, stage, logsumexp) if mode == "flat"
               else make_propagate(arena, (stage,)))
        result = eqx.filter_jit(run)(*propagation_args(state))
        fact = result[4]
    assert np.isneginf(result[3][group.scatter[0, 0], 1])
    assert_fact(fact, int(FailureIndex(arena).ranks(group.members)[0]),
                FailureReason.NONFINITE_OUTPUT)


def test_origin_survives_dependent_nonfinite_update_with_lower_factor_rank():
    template = Template(
        "dependent", variables={n: VariableSpec(n, 2) for n in ("x", "y")},
        factors={
            "z_prior": FactorSpec("z_prior", "categorical", {"out": "x"},
                                   param_blobs={"statistic": "p"}),
            "a_conditional": FactorSpec("a_conditional", "matrix_cat", {"in": "x", "out": "y"},
                                        param_blobs={"matrix": "A"}),
        },
        ports={"y": PortSpec("y", "y", PortRole.DATA)},
        blobs={n: BlobRef(n, n + ".npy") for n in ("p", "A")},
    )
    arena = compile_template(template, arrays={"p": np.array([.3, .7]), "A": np.eye(2)})
    engine = Engine(arena)
    groups = {g.type: i for i, g in enumerate(arena.groups)}
    stages = tuple(BatchStage("disc", groups[kind], np.array([0]))
                   for kind in ("categorical", "matrix_cat"))

    def failing_prior(plan, *args):
        delta = SCHEDULED_EXECUTORS.discrete(plan, *args)
        if plan.type == "categorical":
            return delta._replace(
                outgoing=jnp.full_like(delta.outgoing, jnp.nan),
                status=jnp.array([2], jnp.int32),
                failure_reason=jnp.array([FailureReason.IMPROPER_CAVITY], jnp.int32),
            )
        return delta

    result = eqx.filter_jit(make_propagate(
        arena, stages, replace(SCHEDULED_EXECUTORS, discrete=failing_prior)
    ))(*propagation_args(engine.init_state()))
    index = FailureIndex(arena)
    assert_fact(result[4], int(index.ranks(["z_prior"])[0]), FailureReason.IMPROPER_CAVITY)
    downstream = eqx.filter_jit(make_propagate(arena, stages[1:]))(
        *result[:4], *propagation_args(engine.init_state())[4:])
    assert_fact(downstream[4], int(index.ranks(["a_conditional"])[0]), FailureReason.NONFINITE_OUTPUT)


def test_real_hybrid_nonfinite_output_has_owner_without_local_status():
    from cffg.execution.cross_domain import execute_mixture
    from cffg.execution.parameters import mixture_parameters
    from cffg.execution.transport import _gauss_rows
    template, arrays = mixlin_template()
    arena = compile_template(template, arrays=arrays)
    engine = Engine(arena, schedule="flood")
    state = engine.step(engine.init_state())
    group = arena.mix_groups[0]
    key = group.name + ":b"
    state = replace(state, params={**state.params, key: state.params[key].at[0, 0, 0].set(jnp.nan)})
    selector = state.messages[arena.message_indices(group.s_gather, group.k_sel)]
    eta, precision = _gauss_rows(arena, state.g_eta, state.g_lam, group.g_gather, group.dims)
    local = execute_mixture(group, selector, eta, precision, mixture_parameters(group, state.params))
    assert local.status is None
    result = engine._step(*state_args(state), 0., False)
    assert_fact(result[6], int(FailureIndex(arena).ranks(["mix"])[0]), FailureReason.NONFINITE_OUTPUT)
    for role, outgoing in enumerate(local.outgoing.gaussian):
        indices = arena.g_eta_indices(group.g_scatter[:, role], group.dims[role])
        np.testing.assert_allclose(result[1][indices], outgoing.eta, equal_nan=True)
    assert int(result[5]) == 0
