"""Public explanations preserve status, causality, and per-instance ownership."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_engine_contracts import assert_result_equal, ordinary_case
from test_hybrid_engine import mixlin_template
from test_schedule_program import program_arena, state_in_dtype

from cffg import Engine, FailureDiagnostic, RunStatus, compile_template
from cffg.colony import Colony
from cffg.failure_facts import FailureFact, FailurePhase, FailureReason
from cffg.runtime_types import InferenceError


def fact(rank=0, reason=FailureReason.IMPROPER_CAVITY, phase=FailurePhase.NONE):
    return FailureFact(*(jnp.asarray(value, jnp.int32) for value in (rank, reason, phase)))


@pytest.mark.parametrize("schedule", ["auto", "flood"])
@pytest.mark.parametrize("problem,reason", [(jnp.nan, "nonfinite_state"), (-1e30, "empty_support")])
def test_initial_global_rejections_do_not_invent_factor_ownership(schedule, problem, reason):
    from cffg.compiled import NEG
    engine = Engine(program_arena("two_pass"), schedule=schedule)
    state = engine.init_state()
    messages = (state.messages.at[0].set(problem) if reason == "nonfinite_state"
                else jnp.full_like(state.messages, NEG))
    result = engine.run(replace(state, messages=messages), 5, tol=1e-8)
    assert result.failure == FailureDiagnostic(reason, None, "initialization", 0)
    assert result.iterations == 0
    with pytest.raises(FrozenInstanceError):
        result.failure.factor = "invented"
    with pytest.raises(InferenceError) as caught:
        result.require_converged()
    assert caught.value.result.failure is result.failure


@pytest.mark.parametrize("phase", [FailurePhase.PROPAGATION, FailurePhase.CERTIFICATION])
def test_exact_failure_retains_phase_and_logical_iteration(phase):
    engine = Engine(program_arena("two_pass"))
    original = engine._exact_unit

    def failed(*args):
        result = original(*args)
        return (*result[:4], jnp.asarray(jnp.nan), *result[5:9], fact(phase=phase))

    engine._exact_unit = failed
    result = engine.run(engine.init_state(), 10, tol=1e-8)
    assert result.status is RunStatus.DIVERGED
    assert result.failure == FailureDiagnostic("improper_cavity", "prior", phase.name.lower(), 1)


def test_success_discards_recovered_auxiliary_facts_and_budget_has_generic_reason():
    engine = Engine(program_arena("two_pass"))
    initial = engine.init_state()
    zero = engine.run(initial, 0)
    assert zero.failure == FailureDiagnostic("iteration_budget", None, "soft", 0)
    original = engine._exact_unit

    def recovered(*args):
        result = original(*args)
        return (*result[:9], fact(phase=FailurePhase.PROPAGATION))

    engine._exact_unit = recovered
    result = engine.run(initial, 5)
    assert result.status is RunStatus.CONVERGED
    assert result.failure is None


@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_actual_hybrid_overflow_identifies_its_model_local_factor(schedule):
    template, arrays = mixlin_template()
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    state = state_in_dtype(engine, jnp.float32)
    group = engine.arena.mix_groups[0]
    key = group.name + ":b"
    state = replace(state, params={**state.params, key: jnp.full_like(state.params[key], 1e30)})
    assert engine._state_is_finite(state)
    result = engine.run(state, 5, tol=1e-5)
    assert result.status is RunStatus.DIVERGED
    assert result.failure == FailureDiagnostic(
        "nonfinite_output", "mix", "soft", 1 if schedule == "auto" else 2)


def test_priming_origin_survives_later_lower_rank_failure():
    template, arrays, observations = ordinary_case(0)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), observations)
    prime, step = engine._prime, engine._iter_step
    assert prime is not None
    rank = len(engine._failure_names) - 1

    def primed(*args):
        result = prime(*args)
        return (*result[:4], fact(rank, FailureReason.LOCAL_BUDGET))

    def failed(*args):
        result = step(*args)
        return (*result[:5], jnp.asarray(2), fact(0, FailureReason.NONFINITE_OUTPUT))

    engine._prime, engine._iter_step = primed, failed
    result = engine.run(state, 5, tol=1e-8)
    assert result.status is RunStatus.DIVERGED
    assert result.failure == FailureDiagnostic("local_budget", engine._failure_names[rank], "priming", 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_larger_colony_keeps_failed_members_separate_from_healthy_results(dtype):
    template, arrays, observations = ordinary_case(0)
    colony = Colony()
    ids = [colony.spawn(template, arrays, observations=observations) for _ in range(24)]
    engine = colony.engines[colony.instances[ids[0]].key]
    original = engine._iter_step

    def marked(messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, *args):
        result = original(messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, *args)
        bad = params["diagnostic_test_failure"]
        local = FailureFact(
            jnp.where(bad, jnp.int32(0), jnp.int32(-1)),
            jnp.where(bad, jnp.int32(FailureReason.IMPROPER_CAVITY), jnp.int32(0)),
            jnp.asarray(0, jnp.int32),
        )
        return (*result[:5], jnp.where(bad, 2, result[5]), local)

    engine._iter_step = marked
    states = []
    for i, iid in enumerate(ids):
        state = jax.tree.map(lambda x: x.astype(dtype) if x.dtype.kind == "f" else x,
                             colony.instances[iid].state)
        state = replace(state, params={**state.params, "diagnostic_test_failure": jnp.asarray(i % 3 == 0)})
        states.append(state)
        colony.instances[iid].state = state
    expected = [engine.run(state, 8, tol=1e-6, harden_last=8) for state in states]
    actual = colony.run(8, tol=1e-6)
    for i, (iid, reference) in enumerate(zip(ids, expected, strict=True)):
        assert_result_equal(actual[iid], reference)
        if i % 3 == 0:
            assert actual[iid].failure == FailureDiagnostic(
                "improper_cavity", engine._failure_names[0], "soft", 1)
            assert actual[iid].iterations == 1
        else:
            assert actual[iid].status is RunStatus.CONVERGED
            assert actual[iid].failure is None


def test_scalar_ledger_failure_has_no_fabricated_factor():
    engine = Engine(program_arena("two_pass"), schedule="flood")
    engine.bfe = lambda state: float("nan")
    result = engine.run(engine.init_state(), 3, tol=1e-8)
    assert result.failure == FailureDiagnostic("nonfinite_objective", None, "ledger", 1)


def test_checkpoint_metadata_excludes_run_local_failure():
    from cffg._agent_mu3 import _load_result, _result_metadata
    engine = Engine(program_arena("two_pass"), schedule="flood")
    result = engine.run(engine.init_state(), 0)
    assert result.failure is not None
    metadata = _result_metadata(result)
    assert "failure" not in metadata
    restored = _load_result(result.state, metadata)
    assert restored.failure is None
    assert_result_equal(restored, replace(result, failure=None))


def test_hard_phase_keeps_one_based_numbering_across_soft_updates():
    from cffg.core import ConstraintKind, ConstraintSpec
    from test_schedule import unit_categorical
    template = unit_categorical([ConstraintSpec(ConstraintKind.DELTA, "x")])
    engine = Engine(compile_template(template, arrays={"p": np.array([.2, .3, .5])}))
    original = engine._iter_step

    def marked(*args):
        result = original(*args)
        if args[-1]:
            return (*result[:5], jnp.asarray(2), fact())
        return result

    engine._iter_step = marked
    result = engine.run(engine.init_state(), 4, harden_last=2)
    assert result.status is RunStatus.DIVERGED
    assert result.iterations == 3
    assert result.failure == FailureDiagnostic("improper_cavity", "prior", "hard", 3)


def test_functional_failure_precedes_the_first_counted_message_update():
    template, arrays, observations = ordinary_case(0)
    engine = Engine(compile_template(template, arrays=arrays))
    engine._update_mu3 = lambda state: (state, 0., 2, float("inf"))
    result = engine.run(engine.bind_data(engine.init_state(), observations), 4)
    assert result.iterations == 0
    assert result.failure == FailureDiagnostic("functional_divergence", None, "functional", 1)


def test_zero_budget_delta_colony_does_not_invent_a_hard_phase():
    from cffg.core import ConstraintKind, ConstraintSpec
    from test_schedule import unit_categorical
    template = unit_categorical([ConstraintSpec(ConstraintKind.DELTA, "x")])
    arrays = {"p": np.array([.2, .3, .5])}
    colony = Colony()
    ids = [colony.spawn(template, arrays) for _ in range(2)]
    engine = colony.engines[colony.instances[ids[0]].key]
    expected = engine.run(colony.instances[ids[0]].state, 0, harden_last=0)
    for result in colony.run(0).values():
        assert_result_equal(result, expected)
        assert result.failure == FailureDiagnostic("iteration_budget", None, "soft", 0)


def test_functional_budget_cause_survives_undefined_ledger_status_reclassification():
    from test_exact_colony import gaussian_goal_chain
    template, arrays, observations = gaussian_goal_chain(2)
    engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
    state = engine.bind_data(engine.init_state(), observations)
    codes = iter([0, 1])
    engine._update_mu3 = lambda state: (state, 0., next(codes), 0.5)
    engine.bfe = lambda state: float("nan")
    result = engine.run(state, 3)
    # Preserve the existing undefined-ledger decision, but not at the cost of its cause.
    assert result.status is RunStatus.DIVERGED
    assert result.inner_status is RunStatus.BUDGET_EXHAUSTED
    assert result.iterations == 1
    assert result.failure == FailureDiagnostic("functional_budget", None, "functional", 2)


@pytest.mark.parametrize("initial", [False, True])
def test_outer_terminal_fallback_never_invents_local_trial_ownership(initial):
    from cffg.outer_outcome import OuterOutcome
    from cffg.run_results import finalize_outer
    from cffg.runtime_types import OuterTerminationReason
    state = Engine(program_arena("two_pass")).init_state()
    outcome = OuterOutcome(
        state, OuterTerminationReason.NONFINITE_EVALUATION,
        iterations=0 if initial else 2, initial_rejection=initial,
    )
    result = finalize_outer(outcome)
    assert result.status is RunStatus.DIVERGED
    assert result.failure == (FailureDiagnostic("nonfinite_state", None, "initialization", 0)
                              if initial else FailureDiagnostic("outer_nonfinite_evaluation", None, "outer", 2))
