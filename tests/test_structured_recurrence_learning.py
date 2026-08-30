"""Public D9 pair readout and pair-derived learning regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import recurrence
from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus
from cffg.experiments.recurrence_smoothing import (
    _case,
    forward_backward,
    probability_tolerance,
    recurrence_chain_template,
    recurrence_statistics,
    stick_transition,
)


def _run_chain(card, dtype):
    initial, evidence, drives, weights, offsets = _case(card, dtype=dtype)
    template, arrays, observations = recurrence_chain_template(
        initial, evidence, drives, weights, offsets, dtype=dtype
    )
    engine = Engine(compile_template(template, arrays=arrays))
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
    assert result.status is RunStatus.CONVERGED
    transitions = np.stack(
        [stick_transition(drive, weights, offsets) for drive in drives]
    )
    exact, _ = forward_backward(initial, evidence, transitions)
    return engine, result, exact, weights, offsets, tolerance


@pytest.mark.parametrize("card", (2, 3))
def test_public_pair_and_learning_match_independent_chain_oracle(card):
    engine, result, exact, weights, offsets, tolerance = _run_chain(
        card, np.float64
    )
    pairs = engine.recurrence_pair_beliefs(result.state)
    gaussian = engine.gaussian_node_stats(result.state)
    learning = engine.recurrence_weight_stats(result.state)

    for step in range(len(exact.consecutive)):
        name = f"switch{step}"
        pair = pairs[name]
        np.testing.assert_allclose(
            pair["pair_probability"], exact.consecutive[step], atol=tolerance
        )
        np.testing.assert_allclose(
            pair["source_probability"], exact.singleton[step], atol=tolerance
        )
        np.testing.assert_allclose(
            pair["destination_probability"],
            exact.singleton[step + 1],
            atol=tolerance,
        )
        for key in pair:
            np.testing.assert_array_equal(gaussian[name][key], pair[key])

        independent = recurrence_statistics(
            exact.consecutive[step],
            pair["drive_mean"],
            pair["drive_covariance"],
            weights,
            offsets,
        )
        for key in (
            "omega",
            "exposure",
            "signed_success",
            "linear",
            "quadratic",
        ):
            np.testing.assert_allclose(
                learning[name][key], independent[key], atol=tolerance
            )
        np.testing.assert_array_equal(
            learning[name]["exposure"], pair["exposure"]
        )
        np.testing.assert_array_equal(
            learning[name]["signed_success"], pair["signed_success"]
        )


def test_gaussian_message_and_learning_share_pair_masses_bit_for_bit():
    engine, result, _exact, _weights, _offsets, tolerance = _run_chain(
        3, np.float64
    )
    pairs = engine.recurrence_pair_beliefs(result.state)
    learning = engine.recurrence_weight_stats(result.state)

    for group in engine.arena.recurrence_groups:
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
        weights = np.asarray(result.state.params[group.name + ":weights"])
        offsets = np.asarray(result.state.params[group.name + ":offsets"])
        for index, name in enumerate(group.members):
            pair = pairs[name]
            expected = recurrence.structured_gaussian_message(
                pair["pair_probability"],
                pair["omega"],
                weights[index],
                offsets[index],
            )
            np.testing.assert_allclose(
                actual_eta[index], expected.eta, atol=tolerance
            )
            np.testing.assert_allclose(
                actual_precision[index], expected.precision, atol=tolerance
            )
            np.testing.assert_array_equal(expected.exposure, pair["exposure"])
            np.testing.assert_array_equal(
                expected.signed_success, pair["signed_success"]
            )
            np.testing.assert_array_equal(
                learning[name]["exposure"], expected.exposure
            )
            np.testing.assert_array_equal(
                learning[name]["signed_success"], expected.signed_success
            )


def test_float32_k8_t100_pair_learning_is_jittable_and_finite():
    rng = np.random.default_rng(271)
    time, card, dimension = 100, 8, 2
    pair = rng.random((time - 1, card, card), dtype=np.float32)
    pair[:, -1] = 0.0
    pair /= pair.sum(axis=(1, 2), keepdims=True)
    mean = rng.normal(size=(time - 1, dimension)).astype(np.float32)
    raw = rng.normal(size=(time - 1, dimension, dimension)).astype(np.float32)
    covariance = np.einsum("tij,tkj->tik", raw, raw)
    covariance += np.eye(dimension, dtype=np.float32)[None] * np.float32(0.25)
    weights = rng.normal(
        size=(time - 1, card, card - 1, dimension)
    ).astype(np.float32)
    offsets = rng.normal(size=(time - 1, card, card - 1)).astype(np.float32)

    with jax.enable_x64(False):
        stats = jax.jit(recurrence.weight_stats)(
            jnp.asarray(pair),
            jnp.asarray(mean),
            jnp.asarray(covariance),
            jnp.asarray(weights),
            jnp.asarray(offsets),
        )
    assert stats.linear.shape == (time - 1, card, card - 1, dimension + 1)
    assert stats.quadratic.shape == (
        time - 1,
        card,
        card - 1,
        dimension + 1,
        dimension + 1,
    )
    assert stats.linear.dtype == jnp.float32
    assert np.all(np.asarray(stats.proper))
    assert np.all(np.isfinite(np.asarray(stats.linear)))
    assert np.all(np.isfinite(np.asarray(stats.quadratic)))
    np.testing.assert_array_equal(np.asarray(stats.linear)[:, -1], 0.0)
    np.testing.assert_array_equal(np.asarray(stats.quadratic)[:, -1], 0.0)
