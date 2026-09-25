"""Production integration checks for D9 recurrence pair regions."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import recurrence
from cffg.compiler import NEG, compile_template
from cffg.core.ir import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine, RunStatus, _replace
from cffg.experiments.recurrence_smoothing import (
    _case,
    probability_tolerance,
    recurrence_chain_template,
    stick_transition,
)
from cffg.experiments.structured_recurrence import smooth_log_chain


def _chain(card, dtype, schedule="auto"):
    initial, evidence, drives, weights, offsets = _case(card, dtype=dtype)
    template, arrays, observations = recurrence_chain_template(
        initial, evidence, drives, weights, offsets, dtype=dtype
    )
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    state = engine.bind_data(engine.init_state(), observations)
    state = jax.tree_util.tree_map(
        lambda value: (
            value.astype(dtype)
            if jnp.issubdtype(value.dtype, jnp.inexact)
            else value
        ),
        state,
    )
    tolerance = probability_tolerance(dtype, card)
    result = engine.run(state, 1000, tol=tolerance / 8)

    effective_initial = initial * evidence[0]
    effective_initial /= effective_initial.sum()
    log_evidence = np.log(evidence)
    log_evidence[0] = 0.0
    log_transition = np.log(
        np.stack(
            [stick_transition(drive, weights, offsets) for drive in drives]
        )
    )
    oracle = smooth_log_chain(
        np.log(effective_initial), log_evidence, log_transition
    )
    return engine, result, oracle, weights, offsets, tolerance


def _pair_regions(engine, state, weights, offsets):
    gaussian = engine.gaussian_node_stats(state)
    rows = []
    for group in engine.arena.recurrence_groups:
        card = group.cards[0]
        for index, member in enumerate(group.members):
            out_slot = group.d_gather_both[index, 0, 1]
            in_slot = group.d_gather_both[index, 1, 1]
            out_cavity = np.asarray(
                state.messages[engine.arena.message_indices(out_slot, card)]
            )
            in_cavity = np.asarray(
                state.messages[engine.arena.message_indices(in_slot, card)]
            )
            out_cavity = np.where(out_cavity != NEG, out_cavity, -np.inf)
            in_cavity = np.where(in_cavity != NEG, in_cavity, -np.inf)
            moments = gaussian[member]
            potential = recurrence.structured_pair_potential(
                moments["drive_mean"],
                moments["drive_covariance"],
                weights,
                offsets,
            )
            pair = recurrence.structured_pair_belief(
                in_cavity, out_cavity, potential.log_potential
            )
            rows.append((int(member.removeprefix("switch")), pair))
    return [pair for _, pair in sorted(rows)]


def test_wrapper_keeps_asymmetric_source_destination_axes():
    source = jnp.log(jnp.asarray([[0.72, 0.19, 0.09]], dtype=jnp.float64))
    destination = jnp.log(
        jnp.asarray([[0.11, 0.27, 0.62]], dtype=jnp.float64)
    )
    weights = jnp.asarray(
        [[[[0.7], [-0.2]], [[-0.5], [0.4]], [[0.15], [0.9]]]],
        dtype=jnp.float64,
    )
    offsets = jnp.asarray(
        [[[0.3, -0.8], [0.1, 0.6], [-0.4, 0.2]]],
        dtype=jnp.float64,
    )
    mean = jnp.asarray([0.65], dtype=jnp.float64)
    covariance = jnp.asarray([[0.4]], dtype=jnp.float64)
    precision = jnp.linalg.inv(covariance)[None]
    eta = (precision[0] @ mean)[None]
    expected_potential = recurrence.structured_pair_potential(
        mean, covariance, weights[0], offsets[0]
    )
    expected = recurrence.structured_pair_belief(
        source[0], destination[0], expected_potential.log_potential
    )
    swapped = recurrence.structured_pair_belief(
        destination[0], source[0], expected_potential.log_potential
    )

    actual = recurrence.structured_recurrence_update(
        destination, source, eta, precision, weights, offsets
    )
    assert int(actual.status[0]) == 0
    np.testing.assert_allclose(actual.to_out[0], expected.to_out)
    np.testing.assert_allclose(actual.to_in[0], expected.to_in)
    np.testing.assert_allclose(actual.pair_probability[0], expected.probability)
    assert np.max(
        np.abs(np.asarray(actual.pair_probability[0] - swapped.probability))
    ) > 1e-2


@pytest.mark.parametrize("card", (2, 3))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_engine_pair_regions_and_bfe_match_fixed_drive_chain(card, dtype):
    engine, result, oracle, weights, offsets, tolerance = _chain(card, dtype)
    assert result.status is RunStatus.CONVERGED
    assert result.inner_status is RunStatus.CONVERGED
    pairs = _pair_regions(engine, result.state, weights, offsets)
    actual = np.stack([np.asarray(pair.probability) for pair in pairs])
    np.testing.assert_allclose(actual, oracle.consecutive, atol=tolerance)
    for step, pair in enumerate(pairs):
        assert bool(pair.proper)
        np.testing.assert_allclose(
            pair.source_probability,
            oracle.singleton[step],
            atol=tolerance,
        )
        np.testing.assert_allclose(
            pair.destination_probability,
            oracle.singleton[step + 1],
            atol=tolerance,
        )
    assert engine.bfe(result.state) == pytest.approx(
        -oracle.log_normalizer, abs=tolerance
    )

    assert len(engine.arena.recurrence_components) == 1
    component = engine.arena.recurrence_components[0]
    assert component.acyclic
    assert component.members == tuple(
        f"switch{step}" for step in range(len(pairs))
    )
    for group in engine.arena.recurrence_groups:
        np.testing.assert_array_equal(group.component_ids, 0)


def test_engine_gaussian_write_uses_joint_pair_not_singleton_product():
    engine, result, _, weights, offsets, tolerance = _chain(3, np.float64)
    assert result.status is RunStatus.CONVERGED
    gaussian = engine.gaussian_node_stats(result.state)
    structured_gaps = []
    product_gaps = []
    for group in engine.arena.recurrence_groups:
        card = group.cards[0]
        actual_eta = np.asarray(
            result.state.g_eta[
                engine.arena.g_eta_indices(group.drive_scatter, group.dim)
            ]
        )
        actual_precision = np.asarray(
            result.state.g_lam[
                engine.arena.g_lam_indices(group.drive_scatter, group.dim)
            ]
        )
        for index, member in enumerate(group.members):
            out_slot = group.d_gather_both[index, 0, 1]
            in_slot = group.d_gather_both[index, 1, 1]
            destination = np.asarray(
                result.state.messages[
                    engine.arena.message_indices(out_slot, card)
                ]
            )
            source = np.asarray(
                result.state.messages[
                    engine.arena.message_indices(in_slot, card)
                ]
            )
            moments = gaussian[member]
            potential = recurrence.structured_pair_potential(
                moments["drive_mean"],
                moments["drive_covariance"],
                weights,
                offsets,
            )
            pair = recurrence.structured_pair_belief(
                source, destination, potential.log_potential
            )
            structured = recurrence.structured_gaussian_message(
                pair.probability, potential.omega, weights, offsets
            )
            independent = recurrence.structured_gaussian_message(
                np.outer(
                    np.asarray(pair.source_probability),
                    np.asarray(pair.destination_probability),
                ),
                potential.omega,
                weights,
                offsets,
            )
            structured_gaps.append(
                max(
                    np.max(np.abs(actual_eta[index] - structured.eta)),
                    np.max(
                        np.abs(actual_precision[index] - structured.precision)
                    ),
                )
            )
            product_gaps.append(
                max(
                    np.max(np.abs(actual_eta[index] - independent.eta)),
                    np.max(
                        np.abs(actual_precision[index] - independent.precision)
                    ),
                )
            )
    assert max(structured_gaps) <= tolerance
    assert max(product_gaps) > 100 * tolerance


def test_flood_and_mixed_schedules_share_pair_regions_messages_and_bfe():
    mixed_engine, mixed, _, weights, offsets, tolerance = _chain(
        3, np.float64, "auto"
    )
    flood_engine, flood, _, _, _, _ = _chain(3, np.float64, "flood")
    assert mixed_engine.schedule.kind == "mixed"
    assert flood_engine.schedule.kind == "flood"
    assert mixed.status is flood.status is RunStatus.CONVERGED
    np.testing.assert_allclose(
        mixed.state.messages, flood.state.messages, atol=tolerance
    )
    np.testing.assert_allclose(mixed.state.g_eta, flood.state.g_eta, atol=tolerance)
    np.testing.assert_allclose(mixed.state.g_lam, flood.state.g_lam, atol=tolerance)
    mixed_pairs = _pair_regions(mixed_engine, mixed.state, weights, offsets)
    flood_pairs = _pair_regions(flood_engine, flood.state, weights, offsets)
    np.testing.assert_allclose(
        np.stack([pair.probability for pair in mixed_pairs]),
        np.stack([pair.probability for pair in flood_pairs]),
        atol=tolerance,
    )
    assert mixed_engine.bfe(mixed.state) == pytest.approx(
        flood_engine.bfe(flood.state), abs=tolerance
    )


def test_flood_and_mixed_preserve_exact_support_and_low_finite_values():
    initial, evidence, drives, weights, offsets = _case(3, dtype=np.float64)
    initial = np.asarray([1.0, 0.0, 0.0])
    template, arrays, observations = recurrence_chain_template(
        initial, evidence, drives, weights, offsets, dtype=np.float64
    )
    runs = []
    for schedule in ("auto", "flood"):
        engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
        state = engine.bind_data(engine.init_state(), observations)
        group = engine.arena.recurrence_groups[0]
        low_slot = group.d_gather_both[0, 1, 1]
        low_messages = state.messages.at[
            engine.arena.message_indices(low_slot, 3)
        ].set(-1.0e20)
        low_step = engine.step(_replace(state, messages=low_messages))
        assert np.all(np.isfinite(np.asarray(low_step.messages)))

        run = engine.run(state, 1000, tol=1e-10)
        assert run.status is RunStatus.CONVERGED
        pairs = _pair_regions(engine, run.state, weights, offsets)
        np.testing.assert_array_equal(np.asarray(pairs[0].probability)[1:], 0.0)
        runs.append((engine, run, pairs))

    mixed_engine, mixed, mixed_pairs = runs[0]
    flood_engine, flood, flood_pairs = runs[1]
    np.testing.assert_allclose(mixed.state.messages, flood.state.messages)
    np.testing.assert_allclose(
        np.stack([pair.probability for pair in mixed_pairs]),
        np.stack([pair.probability for pair in flood_pairs]),
    )
    assert mixed_engine.bfe(mixed.state) == pytest.approx(
        flood_engine.bfe(flood.state), abs=1e-10
    )


def _loop_template():
    blobs = {
        name: BlobRef(name, f"{name}.npy")
        for name in ("weights", "offsets", "mean", "covariance")
    }
    template = Template(
        name="recurrence-loop",
        variables={
            "z0": VariableSpec("z0", cardinality=2),
            "z1": VariableSpec("z1", cardinality=2),
            "d0": VariableSpec("d0", dim=1),
            "d1": VariableSpec("d1", dim=1),
        },
        factors={
            "switch0": FactorSpec(
                "switch0",
                "recurrence_link",
                edges={"out": "z1", "in": "z0", "drive": "d0"},
                param_blobs={"weights": "weights", "offsets": "offsets"},
            ),
            "switch1": FactorSpec(
                "switch1",
                "recurrence_link",
                edges={"out": "z0", "in": "z1", "drive": "d1"},
                param_blobs={"weights": "weights", "offsets": "offsets"},
            ),
            "drive0": FactorSpec(
                "drive0",
                "gaussian",
                edges={"out": "d0"},
                param_blobs={"mean": "mean", "cov": "covariance"},
            ),
            "drive1": FactorSpec(
                "drive1",
                "gaussian",
                edges={"out": "d1"},
                param_blobs={"mean": "mean", "cov": "covariance"},
            ),
        },
        blobs=blobs,
    )
    arrays = {
        "weights": np.asarray([[[0.4]], [[-0.3]]]),
        "offsets": np.asarray([[0.1], [-0.2]]),
        "mean": np.zeros(1),
        "covariance": np.eye(1),
    }
    return template, arrays


def test_compiler_marks_categorical_recurrence_loop_without_exact_claim():
    template, arrays = _loop_template()
    arena = compile_template(template, arrays=arrays)
    assert len(arena.recurrence_components) == 1
    component = arena.recurrence_components[0]
    assert component.members == ("switch0", "switch1")
    assert component.categorical_factors == ("switch0", "switch1")
    assert not component.acyclic
    assert set(arena.recurrence_groups[0].component_ids) == {0}


def test_compiler_marks_recurrence_self_loop_without_exact_claim():
    template, arrays = _loop_template()
    template = dataclasses.replace(
        template,
        name="recurrence-self-loop",
        variables={
            name: variable
            for name, variable in template.variables.items()
            if name in {"z0", "d0"}
        },
        factors={
            "switch0": dataclasses.replace(
                template.factors["switch0"],
                edges={"out": "z0", "in": "z0", "drive": "d0"},
            ),
            "drive0": template.factors["drive0"],
        },
    )
    arena = compile_template(template, arrays=arrays)
    assert len(arena.recurrence_components) == 1
    assert arena.recurrence_components[0].members == ("switch0",)
    assert not arena.recurrence_components[0].acyclic


def _equality_branch_template(mean_field=False):
    blobs = {
        name: BlobRef(name, f"{name}.npy")
        for name in ("weights", "offsets", "mean", "covariance", "prior")
    }
    template = Template(
        name="recurrence-equality-branch",
        variables={
            name: VariableSpec(name, cardinality=2)
            for name in ("z0", "z1", "z1a", "z1b")
        }
        | {"drive": VariableSpec("drive", dim=1)},
        factors={
            "switch": FactorSpec(
                "switch",
                "recurrence_link",
                edges={"out": "z1", "in": "z0", "drive": "drive"},
                param_blobs={"weights": "weights", "offsets": "offsets"},
            ),
            "start": FactorSpec(
                "start",
                "categorical",
                edges={"out": "z0"},
                param_blobs={"statistic": "prior"},
            ),
            "branch": FactorSpec(
                "branch",
                "equality",
                edges={"a": "z1", "b": "z1a", "c": "z1b"},
            ),
            "leaf_a": FactorSpec(
                "leaf_a",
                "categorical",
                edges={"out": "z1a"},
                param_blobs={"statistic": "prior"},
            ),
            "leaf_b": FactorSpec(
                "leaf_b",
                "categorical",
                edges={"out": "z1b"},
                param_blobs={"statistic": "prior"},
            ),
            "drive_prior": FactorSpec(
                "drive_prior",
                "gaussian",
                edges={"out": "drive"},
                param_blobs={"mean": "mean", "cov": "covariance"},
            ),
        },
        blobs=blobs,
        constraints=(
            [ConstraintSpec(ConstraintKind.MEAN_FIELD, "leaf_a")]
            if mean_field
            else []
        ),
    )
    arrays = {
        "weights": np.asarray([[[0.4]], [[-0.3]]]),
        "offsets": np.asarray([[0.1], [-0.2]]),
        "mean": np.zeros(1),
        "covariance": np.eye(1),
        "prior": np.asarray([0.55, 0.45]),
    }
    return template, arrays


def test_equality_branch_is_interior_and_mean_field_is_a_boundary():
    template, arrays = _equality_branch_template()
    ordinary = compile_template(template, arrays=arrays)
    assert ordinary.recurrence_components[0].categorical_factors == (
        "branch",
        "leaf_a",
        "leaf_b",
        "start",
        "switch",
    )
    assert ordinary.recurrence_components[0].acyclic

    mean_field, arrays = _equality_branch_template(mean_field=True)
    bounded = compile_template(mean_field, arrays=arrays)
    assert bounded.recurrence_components[0].categorical_factors == (
        "branch",
        "leaf_b",
        "start",
        "switch",
    )
    assert bounded.recurrence_components[0].acyclic


def test_supported_gfe_composite_is_a_recurrence_topology_boundary():
    blobs = {
        name: BlobRef(name, f"{name}.npy")
        for name in (
            "weights",
            "offsets",
            "mean",
            "covariance",
            "prior",
            "matrix",
            "goal",
        )
    }
    template = Template(
        name="recurrence-gfe-boundary",
        variables={
            "z0": VariableSpec("z0", cardinality=2),
            "z1": VariableSpec("z1", cardinality=2),
            "x": VariableSpec("x", cardinality=2),
            "drive": VariableSpec("drive", dim=1),
        },
        factors={
            "switch": FactorSpec(
                "switch",
                "recurrence_link",
                edges={"out": "z1", "in": "z0", "drive": "drive"},
                param_blobs={"weights": "weights", "offsets": "offsets"},
            ),
            "start": FactorSpec(
                "start",
                "categorical",
                edges={"out": "z0"},
                param_blobs={"statistic": "prior"},
            ),
            "g_obs": FactorSpec(
                "g_obs",
                "matrix_cat",
                edges={"out": "x", "in": "z1"},
                param_blobs={"matrix": "matrix"},
            ),
            "g_goal": FactorSpec(
                "g_goal",
                "categorical",
                edges={"out": "x"},
                param_blobs={"statistic": "goal"},
            ),
            "drive_prior": FactorSpec(
                "drive_prior",
                "gaussian",
                edges={"out": "drive"},
                param_blobs={"mean": "mean", "cov": "covariance"},
            ),
        },
        blobs=blobs,
        constraints=[
            ConstraintSpec(
                ConstraintKind.P_SUBSTITUTION,
                "g_obs",
                (("substituted_role", "out"),),
            )
        ],
    )
    arrays = {
        "weights": np.asarray([[[0.4]], [[-0.3]]]),
        "offsets": np.asarray([[0.1], [-0.2]]),
        "mean": np.zeros(1),
        "covariance": np.eye(1),
        "prior": np.asarray([0.55, 0.45]),
        "matrix": np.asarray([[0.85, 0.2], [0.15, 0.8]]),
        "goal": np.asarray([0.5, 0.5]),
    }
    arena = compile_template(template, arrays=arrays)
    assert arena.gfe_groups
    assert arena.recurrence_components[0].categorical_factors == (
        "start",
        "switch",
    )
    assert arena.recurrence_components[0].acyclic


def test_compiler_assigns_disconnected_recurrences_to_separate_components():
    template, arrays = _loop_template()
    template = dataclasses.replace(
        template,
        variables={
            **template.variables,
            "z2": VariableSpec("z2", cardinality=2),
            "z3": VariableSpec("z3", cardinality=2),
            "d2": VariableSpec("d2", dim=1),
        },
        factors={
            **template.factors,
            "switch2": FactorSpec(
                "switch2",
                "recurrence_link",
                edges={"out": "z3", "in": "z2", "drive": "d2"},
                param_blobs={"weights": "weights", "offsets": "offsets"},
            ),
            "z2_prior": FactorSpec(
                "z2_prior",
                "categorical",
                edges={"out": "z2"},
                param_blobs={"statistic": "initial"},
            ),
            "z3_prior": FactorSpec(
                "z3_prior",
                "categorical",
                edges={"out": "z3"},
                param_blobs={"statistic": "initial"},
            ),
            "drive2": FactorSpec(
                "drive2",
                "gaussian",
                edges={"out": "d2"},
                param_blobs={"mean": "mean", "cov": "covariance"},
            ),
        },
        blobs={
            **template.blobs,
            "initial": BlobRef("initial", "initial.npy"),
        },
    )
    arrays["initial"] = np.asarray([0.55, 0.45])
    arena = compile_template(template, arrays=arrays)
    assert len(arena.recurrence_components) == 2
    assert [component.members for component in arena.recurrence_components] == [
        ("switch0", "switch1"),
        ("switch2",),
    ]
    assert [component.acyclic for component in arena.recurrence_components] == [
        False,
        True,
    ]
    assert set(arena.recurrence_groups[0].component_ids) == {0, 1}


def test_vacuous_drive_retains_valid_categorical_pair_and_checks_parameters():
    _, _, _, weights, offsets = _case(2)
    destination = jnp.log(jnp.asarray([[0.25, 0.75]], dtype=jnp.float64))
    source = jnp.asarray([[0.0, -jnp.inf]], dtype=jnp.float64)
    eta = jnp.zeros((1, 1), dtype=jnp.float64)
    precision = jnp.zeros((1, 1, 1), dtype=jnp.float64)
    result = recurrence.structured_recurrence_update(
        destination,
        source,
        eta,
        precision,
        jnp.asarray(weights)[None],
        jnp.asarray(offsets)[None],
    )
    assert int(result.status[0]) == 0
    np.testing.assert_allclose(
        result.pair_probability[0],
        np.asarray([[0.25, 0.75], [0.0, 0.0]]),
    )
    np.testing.assert_allclose(result.source_probability[0], [1.0, 0.0])
    np.testing.assert_allclose(result.destination_probability[0], [0.25, 0.75])

    poisoned = recurrence.structured_recurrence_update(
        destination,
        source,
        eta,
        precision,
        jnp.asarray(weights)[None].at[0, 0, 0, 0].set(jnp.nan),
        jnp.asarray(offsets)[None],
    )
    assert int(poisoned.status[0]) == 2
    assert np.all(np.isnan(poisoned.pair_probability[0]))


def test_empty_support_and_nonfinite_potential_have_distinct_statuses():
    initial, evidence, drives, weights, offsets = _case(2)
    vacuous_empty = recurrence.structured_recurrence_update(
        jnp.zeros((1, 2), dtype=jnp.float64),
        jnp.full((1, 2), -jnp.inf, dtype=jnp.float64),
        jnp.zeros((1, 1), dtype=jnp.float64),
        jnp.zeros((1, 1, 1), dtype=jnp.float64),
        jnp.asarray(weights)[None],
        jnp.asarray(offsets)[None],
    )
    assert int(vacuous_empty.status[0]) == 2
    assert np.all(np.isnan(vacuous_empty.pair_probability[0]))

    template, arrays, observations = recurrence_chain_template(
        initial, evidence, drives, weights, offsets, dtype=np.float64
    )
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), observations)
    group = engine.arena.recurrence_groups[0]
    empty_slot = group.d_gather_both[0, 1, 1]
    empty_messages = state.messages.at[
        engine.arena.message_indices(empty_slot, 2)
    ].set(NEG)
    empty_state = _replace(state, messages=empty_messages)
    empty = engine.run(empty_state, 4, tol=1e-8)
    assert empty.status is RunStatus.INFEASIBLE
    with pytest.raises(RuntimeError, match="factor-local inference diverged"):
        engine.step(empty_state)
    empty_outputs = engine._step(
        empty_state.messages,
        empty_state.g_eta,
        empty_state.g_lam,
        empty_state.g_obs,
        empty_state.g_mask,
        empty_state.ef_eta,
        empty_state.ef_obs,
        empty_state.ef_mask,
        empty_state.params,
    )
    assert int(empty_outputs[5]) == 2

    primed = engine.step(state)
    params = dict(primed.params)
    params[group.name + ":weights"] = params[
        group.name + ":weights"
    ].at[0, 0, 0, 0].set(jnp.nan)
    poisoned = _replace(primed, params=params)
    divergent = engine.run(poisoned, 4, tol=1e-8)
    assert divergent.status is RunStatus.DIVERGED
    assert divergent.inner_status is RunStatus.DIVERGED
    with pytest.raises(RuntimeError, match="factor-local inference diverged"):
        engine.step(poisoned)

    outputs = engine._step(
        poisoned.messages,
        poisoned.g_eta,
        poisoned.g_lam,
        poisoned.g_obs,
        poisoned.g_mask,
        poisoned.ef_eta,
        poisoned.ef_obs,
        poisoned.ef_mask,
        poisoned.params,
    )
    assert int(outputs[5]) == 2
    one_step = engine.run(state, 1, tol=1e-14)
    assert one_step.status is RunStatus.BUDGET_EXHAUSTED

    flood_engine = Engine(
        compile_template(template, arrays=arrays), schedule="flood"
    )
    flood_state = flood_engine.bind_data(flood_engine.init_state(), observations)
    flood_primed = flood_engine.step(flood_state)
    flood_group = flood_engine.arena.recurrence_groups[0]
    flood_params = dict(flood_primed.params)
    flood_params[flood_group.name + ":weights"] = flood_params[
        flood_group.name + ":weights"
    ].at[0, 0, 0, 0].set(jnp.nan)
    flood_poisoned = _replace(flood_primed, params=flood_params)
    flood_divergent = flood_engine.run(flood_poisoned, 4, tol=1e-8)
    assert flood_divergent.status is RunStatus.DIVERGED
    assert flood_divergent.inner_status is RunStatus.DIVERGED
    with pytest.raises(RuntimeError, match="factor-local inference diverged"):
        flood_engine.step(flood_poisoned)


def test_recurrence_and_supported_gfe_coexist_in_mixed_ledger():
    template, arrays = _loop_template()
    template = dataclasses.replace(
        template,
        name="recurrence-with-gfe",
        variables={
            **template.variables,
            "g_z": VariableSpec("g_z", cardinality=2),
            "g_x": VariableSpec("g_x", cardinality=2),
        },
        factors={
            **template.factors,
            "g_prior": FactorSpec(
                "g_prior",
                "categorical",
                edges={"out": "g_z"},
                param_blobs={"statistic": "g_d"},
            ),
            "g_obs": FactorSpec(
                "g_obs",
                "matrix_cat",
                edges={"out": "g_x", "in": "g_z"},
                param_blobs={"matrix": "g_A"},
            ),
            "g_goal": FactorSpec(
                "g_goal",
                "categorical",
                edges={"out": "g_x"},
                param_blobs={"statistic": "g_c"},
            ),
        },
        blobs={
            **template.blobs,
            "g_d": BlobRef("g_d", "g_d.npy"),
            "g_A": BlobRef("g_A", "g_A.npy"),
            "g_c": BlobRef("g_c", "g_c.npy"),
        },
        constraints=[
            ConstraintSpec(
                ConstraintKind.P_SUBSTITUTION,
                "g_obs",
                (("substituted_role", "out"),),
            )
        ],
    )
    arrays.update(
        {
            "g_d": np.asarray([0.6, 0.4]),
            "g_A": np.asarray([[0.85, 0.2], [0.15, 0.8]]),
            "g_c": np.asarray([0.5, 0.5]),
        }
    )
    engine = Engine(compile_template(template, arrays=arrays))
    assert engine.schedule.kind == "mixed"
    assert engine.arena.gfe_groups
    assert engine.arena.recurrence_groups
    result = engine.run(
        engine.init_state(),
        100,
        tol=1e-8,
        outer_solver="picard",
    )
    assert result.status is RunStatus.CONVERGED
    assert np.isfinite(engine.bfe(result.state))


def test_k8_t100_categorical_pass_is_quadratic_not_cubic():
    card, time, dimension = 8, 100, 1
    initial = np.full(card, 1.0 / card)
    evidence = np.full((time, card), 1.0 / card)
    drives = np.linspace(-0.5, 0.5, time - 1)[:, None]
    weights = np.linspace(
        -0.25, 0.3, card * (card - 1) * dimension
    ).reshape(card, card - 1, dimension)
    offsets = np.broadcast_to(
        -np.log(np.arange(card - 1, 0, -1)), (card, card - 1)
    ).copy()
    template, arrays, _ = recurrence_chain_template(
        initial, evidence, drives, weights, offsets, dtype=np.float32
    )
    arena = compile_template(template, arrays=arrays)
    group = arena.recurrence_groups[0]
    assert len(group.members) == time - 1
    assert arena.recurrence_components[0].acyclic

    result = jax.jit(recurrence.structured_recurrence_update)(
        jnp.zeros((time - 1, card), dtype=jnp.float32),
        jnp.zeros((time - 1, card), dtype=jnp.float32),
        jnp.asarray(drives, dtype=jnp.float32),
        jnp.broadcast_to(
            jnp.eye(dimension, dtype=jnp.float32),
            (time - 1, dimension, dimension),
        ),
        jnp.asarray(arrays["weights"])[None].repeat(time - 1, axis=0),
        jnp.asarray(arrays["offsets"])[None].repeat(time - 1, axis=0),
    )
    assert result.pair_probability.shape == (time - 1, card, card)
    assert result.pair_probability.nbytes == (
        (time - 1) * card * card * np.dtype(np.float32).itemsize
    )
    assert not any(
        value.shape == (time - 1, card, card, card)
        for value in result
        if hasattr(value, "shape")
    )
