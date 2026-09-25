"""Exact oracle for the paper-literal constraint flip and rolling windows.

A one-slice before/after pair records the literal planning-to-DATA flip without
changing its logical variables.  A two-slice post-flip mixed history then
provides the paper-shaped second decision to compare with a re-anchored rolling
window and its exact filtered boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
import pytest

import cffg.agent as agent_module
from cffg import rules
from cffg.agent import RollingAgent
from cffg.compiler import _log_of, compile_template
from cffg.core.ir import ConstraintKind, PortRole
from cffg.engine import Engine, InferenceError, RunStatus
from cffg.models.rollout import build_rollout


_GRID_POINTS = 300_001
_FIRST_ACTION = 0
_FIRST_OBSERVATION = 1


@dataclass(frozen=True)
class LiteralSnapshot:
    variables: tuple[str, ...]
    generative_factors: tuple[str, ...]
    planning_factors: tuple[str, ...]
    p_substituted_factors: tuple[str, ...]
    data: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PlanReference:
    state: np.ndarray
    action: np.ndarray
    local_gfe: float
    objective: float
    tolerance: float


def _literal_snapshot(
    first_step_data: tuple[int, int] | None,
) -> LiteralSnapshot:
    common = LiteralSnapshot(
        variables=(
            "z0", "u1", "zm1", "zo1", "zt1", "x1",
        ),
        generative_factors=(
            "prior", "tm1", "eq1", "obs1",
        ),
        planning_factors=("e1", "goal1"),
        p_substituted_factors=("obs1",),
        data=(),
    )
    if first_step_data is None:
        return common
    action, observation = first_step_data
    return LiteralSnapshot(
        variables=common.variables,
        generative_factors=common.generative_factors,
        planning_factors=(),
        p_substituted_factors=(),
        data=(("act1", action), ("obs1", observation)),
    )


def _parameters(dtype=np.float64):
    return {
        "d": np.asarray([0.65, 0.35], dtype=dtype),
        "A": np.asarray([[0.9, 0.2], [0.1, 0.8]], dtype=dtype),
        "c": np.asarray([0.25, 0.75], dtype=dtype),
        "e": np.asarray([0.6, 0.4], dtype=dtype),
        "Bs": np.asarray(
            [
                [[0.85, 0.15], [0.15, 0.85]],
                [[0.20, 0.75], [0.80, 0.25]],
            ],
            dtype=dtype,
        ),
    }


def _builder_for(parameters):
    def builder(*, past, steps, **kwargs):
        kwargs.pop("learned_a", None)
        kwargs.pop("learned_c", None)
        kwargs.pop("learn", None)
        return build_rollout(
            "literal-flip",
            past=past,
            steps=steps,
            gfe=True,
            **parameters,
        )

    return builder


def _rho(A: np.ndarray, c: np.ndarray, state: np.ndarray) -> np.ndarray:
    column_entropy = -np.sum(
        np.where(A > 0, A * np.log(np.where(A > 0, A, 1.0)), 0.0),
        axis=0,
    )
    predictive_observation = A @ state
    with np.errstate(divide="ignore"):
        log_ratio = np.log(c) - np.log(predictive_observation)
    weighted = np.zeros_like(A)
    np.multiply(A, log_ratio[:, None], out=weighted, where=A > 0)
    return weighted.sum(axis=0) - column_entropy


def _numeric_tolerance(parameters, *values, grid: bool = False) -> float:
    dtype = np.asarray(parameters["A"]).dtype
    positive = np.concatenate(
        [np.asarray(value, dtype=np.float64).ravel() for value in values]
    )
    positive = positive[positive > 0]
    conditioning = 1.0 if not len(positive) else max(1.0, 1.0 / positive.min())
    grid_error = 8.0 / (_GRID_POINTS - 1) if grid else 0.0
    return grid_error + 128 * np.finfo(dtype).eps * conditioning


def _plan_reference(parameters, boundary: np.ndarray) -> PlanReference:
    """Enumerate Part II Fig. 5's binary simplex objective independently."""
    A = np.asarray(parameters["A"], dtype=np.float64)
    c = np.asarray(parameters["c"], dtype=np.float64)
    e = np.asarray(parameters["e"], dtype=np.float64)
    Bs = np.asarray(parameters["Bs"], dtype=np.float64)
    boundary = np.asarray(boundary, dtype=np.float64)
    predictive = np.einsum("uij,j,u->i", Bs, boundary, e)

    supported = predictive > 0
    if not np.any(supported):
        raise ValueError("empty planning support")
    if np.count_nonzero(supported) == 1:
        state = supported.astype(np.float64)
    else:
        first = np.linspace(0.0, 1.0, _GRID_POINTS)
        candidates = np.stack([first, 1.0 - first])
        predictive_observation = A @ candidates
        column_entropy = -np.sum(
            np.where(A > 0, A * np.log(np.where(A > 0, A, 1.0)), 0.0),
            axis=0,
        )
        rho = (
            A.T
            @ (np.log(c)[:, None] - np.log(predictive_observation))
            - column_entropy[:, None]
        )
        log_predictive = np.where(supported, np.log(predictive), 0.0)
        negative_entropy = np.sum(
            np.where(
                candidates > 0,
                candidates * np.log(np.where(candidates > 0, candidates, 1.0)),
                0.0,
            ),
            axis=0,
        )
        objective = -np.sum(candidates * (rho + log_predictive[:, None]), axis=0)
        objective += negative_entropy
        objective[np.any(candidates[~supported] > 0, axis=0)] = np.inf
        state = candidates[:, int(np.argmin(objective))]

    rho = _rho(A, c, state)
    rho = np.where(supported, rho, -np.inf)
    backward = np.exp(rho - np.max(rho))
    joint = np.einsum("u,j,uij,i->uji", e, boundary, Bs, backward)
    action = joint.sum(axis=(1, 2))
    action /= action.sum()
    reconstructed_state = joint.sum(axis=(0, 1))
    reconstructed_state /= reconstructed_state.sum()

    tolerance = _numeric_tolerance(parameters, predictive, A, c, grid=True)
    np.testing.assert_allclose(reconstructed_state, state, atol=tolerance)

    negative_entropy = np.sum(
        np.where(state > 0, state * np.log(np.where(state > 0, state, 1.0)), 0.0)
    )
    local_gfe = float(-state[supported] @ rho[supported] + negative_entropy)
    objective = float(local_gfe - state[supported] @ np.log(predictive[supported]))
    return PlanReference(state, action, local_gfe, objective, tolerance)


def _filter_reference(parameters, boundary, action: int, observation: int):
    predicted = np.asarray(parameters["Bs"])[action] @ np.asarray(boundary)
    unnormalized = np.asarray(parameters["A"])[observation] * predicted
    evidence = float(unnormalized.sum())
    tolerance = _numeric_tolerance(
        parameters,
        predicted,
        np.asarray(parameters["A"])[observation],
        np.asarray([evidence]),
    )
    if evidence <= 0:
        return None, evidence, tolerance
    return unnormalized / evidence, evidence, tolerance


def _run_plan(agent: RollingAgent):
    state = agent._with_params(agent.plan)
    result = agent.plan.run(state, agent.n_sweeps, tol=1e-10)
    if result.status is not RunStatus.CONVERGED:
        return result, None, None
    marginals = agent.plan.marginals(result.state)
    return result, marginals["u1"], marginals["zm1"]


def _run_filter(agent: RollingAgent, action: int, observation: int):
    state = agent._with_params(agent.filter)
    state = agent.filter.bind_data(state, {"act1": action, "obs1": observation})
    return agent.filter.run(state, 60, tol=1e-12)


def test_literal_snapshots_record_the_constraint_change_without_rewriting_variables():
    parameters = _parameters()
    before = _literal_snapshot(None)
    after = _literal_snapshot((_FIRST_ACTION, _FIRST_OBSERVATION))
    planning, _ = _builder_for(parameters)(past=0, steps=1)
    observed, _ = _builder_for(parameters)(past=1, steps=1)

    assert after.variables == before.variables
    assert after.generative_factors == before.generative_factors
    assert before.data == ()
    assert after.data == (
        ("act1", _FIRST_ACTION),
        ("obs1", _FIRST_OBSERVATION),
    )
    assert set(planning.factors) == set(
        before.generative_factors + before.planning_factors
    )
    assert set(observed.factors) == set(
        after.generative_factors + after.planning_factors
    )
    for name in before.generative_factors:
        assert observed.factors[name] == planning.factors[name]
    assert set(planning.variables) == set(before.variables)
    assert set(observed.variables) == set(after.variables)

    before_p = tuple(
        constraint.target
        for constraint in planning.constraints
        if constraint.kind is ConstraintKind.P_SUBSTITUTION
    )
    after_p = tuple(
        constraint.target
        for constraint in observed.constraints
        if constraint.kind is ConstraintKind.P_SUBSTITUTION
    )
    assert before_p == before.p_substituted_factors
    assert after_p == after.p_substituted_factors
    assert [
        (constraint.kind, constraint.target, constraint.detail)
        for constraint in planning.constraints
    ] == [
        (
            ConstraintKind.P_SUBSTITUTION,
            "obs1",
            (("substituted_role", "out"),),
        )
    ]
    assert observed.constraints == []
    assert tuple(
        name for name in before.planning_factors if name in planning.factors
    ) == before.planning_factors
    assert tuple(
        name for name in before.planning_factors if name in observed.factors
    ) == after.planning_factors
    assert {
        name: (port.variable, port.role)
        for name, port in planning.ports.items()
    } == {"horizon": ("zt1", PortRole.GLUE)}
    assert {
        name: (port.variable, port.role)
        for name, port in observed.ports.items()
    } == {
        "act1": ("u1", PortRole.DATA),
        "obs1": ("x1", PortRole.DATA),
        "horizon": ("zt1", PortRole.GLUE),
    }


@pytest.mark.slow
def test_two_window_agent_matches_two_step_enumerator_and_local_ledgers(monkeypatch):
    parameters = _parameters()
    compile_calls = []
    original_compile = agent_module.compile_template

    def counted_compile(*args, **kwargs):
        compile_calls.append(args[0].name)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(agent_module, "compile_template", counted_compile)
    agent = RollingAgent(
        _builder_for(parameters), horizon=1, n_sweeps=400, select="map"
    )
    assert len(compile_calls) == 2
    plan_engine = agent.plan
    filter_engine = agent.filter
    plan_arena = agent.plan.arena
    filter_arena = agent.filter.arena

    first_reference = _plan_reference(parameters, parameters["d"])
    first_result, first_action_q, first_state_q = _run_plan(agent)
    assert first_result.status is RunStatus.CONVERGED
    np.testing.assert_allclose(
        first_action_q, first_reference.action, atol=first_reference.tolerance
    )
    np.testing.assert_allclose(
        first_state_q, first_reference.state, atol=first_reference.tolerance
    )
    first_action = int(np.argmax(first_reference.action))
    assert first_action == _FIRST_ACTION
    assert agent.act() == first_action

    log_predictive = _log_of(
        np.einsum("uij,j,u->i", parameters["Bs"], parameters["d"], parameters["e"])
    )
    local = rules.gfe_energy(
        log_predictive[None, :],
        np.log(parameters["A"])[None, :, :],
        np.log(parameters["c"])[None, :],
    )[0]
    np.testing.assert_allclose(
        local, first_reference.local_gfe, atol=first_reference.tolerance
    )
    np.testing.assert_allclose(
        agent.plan.bfe(first_result.state),
        first_reference.objective,
        atol=first_reference.tolerance,
    )

    filtered, evidence, filter_tolerance = _filter_reference(
        parameters, parameters["d"], first_action, _FIRST_OBSERVATION
    )
    filter_result = _run_filter(agent, first_action, _FIRST_OBSERVATION)
    assert filter_result.status is RunStatus.CONVERGED
    np.testing.assert_allclose(
        agent.filter.marginals(filter_result.state)["zm1"],
        filtered,
        atol=filter_tolerance,
    )
    np.testing.assert_allclose(
        agent.filter.bfe(filter_result.state),
        -np.log(evidence),
        atol=filter_tolerance,
    )
    np.testing.assert_allclose(
        agent.observe(first_action, _FIRST_OBSERVATION),
        filtered,
        atol=filter_tolerance,
    )

    second_reference = _plan_reference(parameters, filtered)
    second_result, second_action_q, second_state_q = _run_plan(agent)
    assert second_result.status is RunStatus.CONVERGED
    np.testing.assert_allclose(
        second_action_q, second_reference.action, atol=second_reference.tolerance
    )
    np.testing.assert_allclose(
        second_state_q, second_reference.state, atol=second_reference.tolerance
    )
    np.testing.assert_allclose(
        agent.plan.bfe(second_result.state),
        second_reference.objective,
        atol=second_reference.tolerance,
    )

    second_predictive = np.einsum(
        "uij,j,u->i", parameters["Bs"], filtered, parameters["e"]
    )
    second_local = rules.gfe_energy(
        _log_of(second_predictive)[None, :],
        np.log(parameters["A"])[None, :, :],
        np.log(parameters["c"])[None, :],
    )[0]
    np.testing.assert_allclose(
        second_local, second_reference.local_gfe, atol=second_reference.tolerance
    )

    mixed_template, mixed_arrays = _builder_for(parameters)(past=1, steps=2)
    rolled_template, _ = _builder_for(parameters)(past=0, steps=1)
    for mixed_name, rolled_name in (
        ("tm2", "tm1"),
        ("eq2", "eq1"),
        ("obs2", "obs1"),
        ("e2", "e1"),
        ("goal2", "goal1"),
    ):
        mixed_factor = mixed_template.factors[mixed_name]
        rolled_factor = rolled_template.factors[rolled_name]
        assert mixed_factor.type == rolled_factor.type
        assert mixed_factor.param_blobs == rolled_factor.param_blobs
        assert mixed_factor.edges.keys() == rolled_factor.edges.keys()
        for role in mixed_factor.edges:
            mixed_variable = mixed_template.variables[mixed_factor.edges[role]]
            rolled_variable = rolled_template.variables[rolled_factor.edges[role]]
            assert mixed_variable.domain_key == rolled_variable.domain_key

    mixed_engine = Engine(compile_template(mixed_template, arrays=mixed_arrays))
    mixed_state = mixed_engine.bind_data(
        mixed_engine.init_state(),
        dict(_literal_snapshot((first_action, _FIRST_OBSERVATION)).data),
    )
    mixed_result = mixed_engine.run(mixed_state, 400, tol=1e-10)
    assert mixed_result.status is RunStatus.CONVERGED
    mixed_marginals = mixed_engine.marginals(mixed_result.state)
    np.testing.assert_allclose(
        mixed_marginals["u2"],
        second_reference.action,
        atol=second_reference.tolerance,
    )
    np.testing.assert_allclose(
        mixed_marginals["zm2"],
        second_reference.state,
        atol=second_reference.tolerance,
    )
    np.testing.assert_allclose(
        mixed_engine.bfe(mixed_result.state) - agent.plan.bfe(second_result.state),
        -np.log(evidence),
        atol=max(second_reference.tolerance, filter_tolerance),
    )

    second_action = int(np.argmax(second_reference.action))
    second_filtered, _, second_filter_tolerance = _filter_reference(
        parameters, filtered, second_action, 0
    )
    np.testing.assert_allclose(
        agent.observe(second_action, 0),
        second_filtered,
        atol=second_filter_tolerance,
    )

    for _ in range(14):
        action = agent.act()
        agent.observe(action, 0)
    assert agent.tick == 16
    assert agent.learnables.params == {}
    assert len(compile_calls) == 2
    assert agent.plan is plan_engine and agent.filter is filter_engine
    assert agent.plan.arena is plan_arena and agent.filter.arena is filter_arena


def test_exact_zero_rollover_survives_and_empty_support_is_infeasible():
    parameters = {
        "d": np.asarray([1.0, 0.0]),
        "A": np.eye(2),
        "c": np.asarray([0.5, 0.5]),
        "e": np.asarray([0.5, 0.5]),
        "Bs": np.stack([np.eye(2), np.eye(2)]),
    }
    agent = RollingAgent(_builder_for(parameters), horizon=1, n_sweeps=100)

    reference = _plan_reference(parameters, parameters["d"])
    plan_result, action_q, state_q = _run_plan(agent)
    assert plan_result.status is RunStatus.CONVERGED
    np.testing.assert_array_equal(state_q, reference.state)
    np.testing.assert_allclose(action_q, reference.action, atol=reference.tolerance)
    np.testing.assert_array_equal(agent.observe(0, 0), [1.0, 0.0])
    next_result, _, next_state_q = _run_plan(agent)
    assert next_result.status is RunStatus.CONVERGED
    np.testing.assert_array_equal(next_state_q, [1.0, 0.0])

    impossible_reference, evidence, _ = _filter_reference(
        parameters, agent.belief, 0, 1
    )
    assert impossible_reference is None and evidence == 0.0
    impossible_filter = _run_filter(agent, 0, 1)
    assert impossible_filter.status is RunStatus.INFEASIBLE
    belief_before = agent.belief.copy()
    tick_before = agent.tick
    with pytest.raises(InferenceError, match="infeasible") as error:
        agent.observe(0, 1)
    assert error.value.result.status is RunStatus.INFEASIBLE
    np.testing.assert_array_equal(agent.belief, belief_before)
    assert agent.tick == tick_before

    agent.belief = np.zeros(2)
    empty_plan, _, _ = _run_plan(agent)
    assert empty_plan.status is RunStatus.INFEASIBLE
    with pytest.raises(InferenceError, match="infeasible") as error:
        agent.act()
    assert error.value.result.status is RunStatus.INFEASIBLE


def test_vacuous_float32_boundary_matches_or_reports_truthful_exhaustion():
    with jax.enable_x64(False):
        parameters = _parameters(np.float32)
        parameters["d"] = np.full(2, 0.5, dtype=np.float32)
        agent = RollingAgent(
            _builder_for(parameters), horizon=1, n_sweeps=400, select="map"
        )
        reference = _plan_reference(parameters, parameters["d"])
        result, action_q, state_q = _run_plan(agent)

        if result.status is RunStatus.CONVERGED:
            np.testing.assert_allclose(
                action_q, reference.action, atol=reference.tolerance
            )
            np.testing.assert_allclose(
                state_q, reference.state, atol=reference.tolerance
            )
            assert agent.act() == int(np.argmax(reference.action))
        else:
            assert result.status is RunStatus.BUDGET_EXHAUSTED
            assert np.isfinite(np.asarray(result.state.messages)).all()
            assert np.isfinite(result.residual)
            with pytest.raises(InferenceError, match="budget_exhausted") as error:
                agent.act()
            assert error.value.result.status is RunStatus.BUDGET_EXHAUSTED

        action = int(np.argmax(reference.action))
        filtered, evidence, filter_tolerance = _filter_reference(
            parameters, parameters["d"], action, _FIRST_OBSERVATION
        )
        filter_result = _run_filter(agent, action, _FIRST_OBSERVATION)
        assert filter_result.status is RunStatus.CONVERGED
        np.testing.assert_allclose(
            agent.filter.marginals(filter_result.state)["zm1"],
            filtered,
            atol=filter_tolerance,
        )
        np.testing.assert_allclose(
            agent.filter.bfe(filter_result.state),
            -np.log(evidence),
            atol=filter_tolerance,
        )
        np.testing.assert_allclose(
            agent.observe(action, _FIRST_OBSERVATION),
            filtered,
            atol=filter_tolerance,
        )

        second_reference = _plan_reference(parameters, filtered)
        second_result, second_action_q, second_state_q = _run_plan(agent)
        if second_result.status is RunStatus.CONVERGED:
            np.testing.assert_allclose(
                second_action_q,
                second_reference.action,
                atol=second_reference.tolerance,
            )
            np.testing.assert_allclose(
                second_state_q,
                second_reference.state,
                atol=second_reference.tolerance,
            )
        else:
            assert second_result.status is RunStatus.BUDGET_EXHAUSTED
            assert np.isfinite(np.asarray(second_result.state.messages)).all()
            assert np.isfinite(second_result.residual)

        mixed_template, mixed_arrays = _builder_for(parameters)(past=1, steps=2)
        mixed_engine = Engine(compile_template(mixed_template, arrays=mixed_arrays))
        mixed_state = mixed_engine.bind_data(
            mixed_engine.init_state(),
            dict(_literal_snapshot((action, _FIRST_OBSERVATION)).data),
        )
        mixed_result = mixed_engine.run(mixed_state, 400, tol=1e-10)
        if mixed_result.status is RunStatus.CONVERGED:
            mixed_marginals = mixed_engine.marginals(mixed_result.state)
            np.testing.assert_allclose(
                mixed_marginals["u2"],
                second_reference.action,
                atol=second_reference.tolerance,
            )
            np.testing.assert_allclose(
                mixed_engine.bfe(mixed_result.state)
                - second_reference.objective,
                -np.log(evidence),
                atol=max(second_reference.tolerance, filter_tolerance),
            )
        else:
            assert mixed_result.status is RunStatus.BUDGET_EXHAUSTED
            assert np.isfinite(np.asarray(mixed_result.state.messages)).all()
            assert np.isfinite(mixed_result.residual)
