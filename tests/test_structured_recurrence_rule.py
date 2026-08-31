"""Executable D9 checks for the independently callable recurrence rules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import gauss, recurrence
from cffg.experiments.recurrence_smoothing import recurrence_statistics
from cffg.experiments.structured_recurrence import (
    fixed_drive_potentials,
    probability_tolerance,
    registered_case,
    smooth_log_chain,
    structured_free_energy,
    uncertain_drive_potentials,
)


def _max_error(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


@pytest.mark.parametrize("card", (2, 3))
def test_fixed_drive_rules_match_exact_chain_oracle(card):
    log_initial, log_evidence, log_transition, drives, weights, offsets = (
        fixed_drive_potentials(card)
    )
    oracle = smooth_log_chain(log_initial, log_evidence, log_transition)
    tolerance = probability_tolerance(np.float64, card)

    for step, drive in enumerate(drives):
        potential = recurrence.structured_pair_potential(
            drive, np.zeros((drive.size, drive.size)), weights, offsets
        )
        assert bool(potential.proper)
        np.testing.assert_allclose(
            potential.log_potential,
            log_transition[step],
            atol=tolerance,
        )
        result = recurrence.structured_pair_belief(
            oracle.forward[step],
            log_evidence[step + 1] + oracle.backward[step + 1],
            potential.log_potential,
        )
        assert bool(result.proper)
        np.testing.assert_allclose(
            result.probability, oracle.consecutive[step], atol=tolerance
        )
        np.testing.assert_allclose(
            result.source_probability, oracle.singleton[step], atol=tolerance
        )
        np.testing.assert_allclose(
            result.destination_probability,
            oracle.singleton[step + 1],
            atol=tolerance,
        )
        np.testing.assert_allclose(
            np.exp(result.to_out), oracle.to_out[step], atol=tolerance
        )
        np.testing.assert_allclose(
            np.exp(result.to_in), oracle.to_in[step], atol=tolerance
        )


def test_pair_masses_and_gaussian_message_use_joint_responsibility():
    log_initial, log_evidence, log_transition, drives, weights, offsets = (
        fixed_drive_potentials(3)
    )
    oracle = smooth_log_chain(log_initial, log_evidence, log_transition)
    step = 1
    covariance = np.asarray([[0.35]])
    potential = recurrence.structured_pair_potential(
        drives[step], covariance, weights, offsets
    )
    actual = recurrence.structured_gaussian_message(
        oracle.consecutive[step], potential.omega, weights, offsets
    )
    expected = recurrence_statistics(
        oracle.consecutive[step],
        drives[step],
        covariance,
        weights,
        offsets,
    )
    assert bool(actual.proper)
    np.testing.assert_allclose(actual.exposure, expected["exposure"])
    np.testing.assert_allclose(
        actual.signed_success, expected["linear"][..., -1]
    )
    expected_eta = np.einsum(
        "il,ild->d",
        actual.signed_success - actual.exposure * potential.omega * offsets,
        weights,
    )
    expected_precision = np.einsum(
        "il,il,ild,ile->de",
        actual.exposure,
        potential.omega,
        weights,
        weights,
    )
    np.testing.assert_allclose(actual.eta, expected_eta)
    np.testing.assert_allclose(actual.precision, expected_precision)

    product = (
        oracle.singleton[step, :, None]
        * oracle.singleton[step + 1, None, :]
    )
    assert _max_error(product, oracle.consecutive[step]) > 1e-3
    product_message = recurrence.structured_gaussian_message(
        product, potential.omega, weights, offsets
    )
    assert _max_error(product_message.exposure, actual.exposure) > 1e-3


@pytest.mark.parametrize("card", (2, 3))
def test_pair_node_and_separator_energy_equal_negative_log_normalizer(card):
    log_initial, log_evidence, log_transition, *_ = fixed_drive_potentials(card)
    log_initial = log_initial.copy()
    log_evidence = log_evidence[:3].copy()
    log_transition = log_transition[:2].copy()
    oracle = smooth_log_chain(log_initial, log_evidence, log_transition)
    pair_energy = sum(
        float(
            recurrence.structured_pair_energy(
                oracle.consecutive[step], log_transition[step]
            )
        )
        for step in range(2)
    )
    separator = float(
        recurrence.structured_separator_energy(oracle.singleton[1])
    )
    unary_expectation = np.einsum(
        "i,i->", oracle.singleton[0], log_initial
    ) + np.einsum("ti,ti->", oracle.singleton, log_evidence)
    actual = pair_energy + separator - unary_expectation
    tolerance = probability_tolerance(np.float64, card)
    assert actual == pytest.approx(-oracle.log_normalizer, abs=tolerance)

    precision = np.asarray([[1.7]])
    node = recurrence.structured_node_energy(
        oracle.consecutive[0], log_transition[0], precision
    )
    assert float(node) == pytest.approx(
        float(
            recurrence.structured_pair_energy(
                oracle.consecutive[0], log_transition[0]
            )
            + gauss._neg_entropy(jnp.asarray(precision))
        )
    )


def test_uncertain_drive_preserves_jj_boundary():
    log_initial, log_evidence, exact_transition, bounded_transition = (
        uncertain_drive_potentials()
    )
    _, _, drives, weights, offsets = registered_case(2)
    means = drives + np.asarray([[0.15], [-0.1], [0.2], [0.05]])
    covariances = np.asarray([[[0.9]], [[0.55]], [[1.1]], [[0.7]]])
    actual = np.stack(
        [
            np.asarray(
                recurrence.structured_pair_potential(
                    mean, covariance, weights, offsets
                ).log_potential
            )
            for mean, covariance in zip(means, covariances, strict=True)
        ]
    )
    np.testing.assert_allclose(actual, bounded_transition, atol=1e-12)
    assert np.all(actual <= exact_transition + 1e-12)
    assert np.max(exact_transition - actual) > 1e-3

    bounded = smooth_log_chain(log_initial, log_evidence, actual)
    exact = smooth_log_chain(log_initial, log_evidence, exact_transition)
    assert structured_free_energy(
        bounded, log_initial, log_evidence, actual
    ) >= structured_free_energy(
        exact, log_initial, log_evidence, exact_transition
    )


def test_literal_zero_support_and_vacuous_cavities_are_preserved():
    potential = np.asarray([[0.0, -np.inf], [-np.inf, 0.0]])
    result = recurrence.structured_pair_belief(
        np.asarray([0.0, -np.inf]), np.zeros(2), potential
    )
    assert bool(result.proper)
    np.testing.assert_array_equal(result.probability, [[1.0, 0.0], [0.0, 0.0]])
    assert np.isneginf(np.asarray(result.to_out)[1])
    np.testing.assert_allclose(np.exp(result.to_in), [0.5, 0.5])
    message = recurrence.structured_gaussian_message(
        result.probability,
        np.full((2, 1), 0.25),
        np.ones((2, 1, 1)),
        np.zeros((2, 1)),
    )
    np.testing.assert_array_equal(message.exposure, [[1.0], [0.0]])
    np.testing.assert_array_equal(message.signed_success, [[0.5], [0.0]])

    vacuous = recurrence.structured_pair_belief(
        np.zeros(2), np.zeros(2), np.zeros((2, 2))
    )
    assert bool(vacuous.proper)
    np.testing.assert_allclose(vacuous.probability, 0.25)


@pytest.mark.parametrize(
    "source,destination,potential",
    (
        ([-np.inf, -np.inf], [0.0, 0.0], np.zeros((2, 2))),
        ([0.0, 0.0], [0.0, 0.0], np.full((2, 2), -np.inf)),
        ([0.0, np.nan], [0.0, 0.0], np.zeros((2, 2))),
        ([0.0, 0.0], [0.0, 0.0], np.asarray([[0.0, np.inf], [0.0, 0.0]])),
    ),
)
def test_empty_or_nonfinite_categorical_inputs_fail_loudly(
    source, destination, potential
):
    result = recurrence.structured_pair_belief(
        np.asarray(source), np.asarray(destination), potential
    )
    assert not bool(result.proper)
    assert np.isnan(result.probability).all()


@pytest.mark.parametrize(
    "covariance,weights,offsets",
    (
        (np.asarray([[-1.0]]), np.ones((2, 1, 1)), np.zeros((2, 1))),
        (np.asarray([[1.0]]), np.asarray([[[np.nan]], [[0.0]]]), np.zeros((2, 1))),
        (np.asarray([[np.inf]]), np.ones((2, 1, 1)), np.zeros((2, 1))),
    ),
)
def test_improper_moments_or_parameters_fail_loudly(
    covariance, weights, offsets
):
    result = recurrence.structured_pair_potential(
        np.zeros(1), covariance, weights, offsets
    )
    assert not bool(result.proper)
    assert np.isnan(result.log_potential).all()
    assert np.isnan(result.omega).all()


def test_invalid_pair_statistics_and_energies_fail_loudly():
    invalid_pair = np.asarray([[0.4, 0.1], [0.1, 0.1]])
    message = recurrence.structured_gaussian_message(
        invalid_pair,
        np.full((2, 1), 0.25),
        np.ones((2, 1, 1)),
        np.zeros((2, 1)),
    )
    assert not bool(message.proper)
    assert np.isnan(message.eta).all()
    assert np.isinf(
        recurrence.structured_pair_energy(invalid_pair, np.zeros((2, 2)))
    )
    assert np.isinf(
        recurrence.structured_separator_energy(np.asarray([0.3, 0.3]))
    )
    assert np.isinf(
        recurrence.structured_node_energy(
            np.full((2, 2), 0.25), np.zeros((2, 2)), np.zeros((1, 1))
        )
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_dtype_scaled_guards_and_jax_transforms(dtype):
    _, _, drives, weights, offsets = registered_case(3, dtype=dtype)
    means = np.stack([drives[0], drives[1]])
    covariances = np.broadcast_to(
        np.eye(1, dtype=dtype) * dtype(0.2), (2, 1, 1)
    )
    vmapped = jax.jit(
        jax.vmap(recurrence.structured_pair_potential, in_axes=(0, 0, None, None))
    )(
        jnp.asarray(means),
        jnp.asarray(covariances),
        jnp.asarray(weights),
        jnp.asarray(offsets),
    )
    assert np.asarray(vmapped.log_potential).dtype == dtype
    assert np.asarray(vmapped.proper).all()

    belief = jax.jit(recurrence.structured_pair_belief)(
        jnp.zeros(3, dtype=dtype),
        jnp.zeros(3, dtype=dtype),
        vmapped.log_potential[0],
    )
    tolerance = probability_tolerance(dtype, 3)
    assert bool(belief.proper)
    assert float(belief.probability.sum()) == pytest.approx(
        1.0, abs=tolerance
    )
    eager_message = recurrence.structured_gaussian_message(
        belief.probability, vmapped.omega[0], weights, offsets
    )
    jitted_message = jax.jit(recurrence.structured_gaussian_message)(
        belief.probability, vmapped.omega[0], weights, offsets
    )
    assert bool(jitted_message.proper)
    np.testing.assert_allclose(jitted_message.eta, eager_message.eta)
    np.testing.assert_allclose(
        jitted_message.precision, eager_message.precision
    )
    pair_energy = jax.jit(recurrence.structured_pair_energy)(
        belief.probability, vmapped.log_potential[0]
    )
    separator = jax.jit(recurrence.structured_separator_energy)(
        belief.source_probability
    )
    node_energy = jax.jit(recurrence.structured_node_energy)(
        belief.probability,
        vmapped.log_potential[0],
        jnp.eye(1, dtype=dtype),
    )
    assert np.isfinite([pair_energy, separator, node_energy]).all()


def test_k8_rule_has_quadratic_categorical_state_not_cubic():
    card, dimension = 8, 3
    weights = jnp.reshape(
        jnp.linspace(-0.4, 0.5, card * (card - 1) * dimension),
        (card, card - 1, dimension),
    )
    potential = jax.jit(recurrence.structured_pair_potential)(
        jnp.linspace(-0.2, 0.3, dimension),
        jnp.eye(dimension) * 0.4,
        weights,
        jnp.zeros((card, card - 1)),
    )
    belief = jax.jit(recurrence.structured_pair_belief)(
        jnp.zeros(card), jnp.zeros(card), potential.log_potential
    )
    assert potential.log_potential.shape == (card, card)
    assert belief.probability.shape == (card, card)
    assert not any(
        value.shape == (card, card, card)
        for value in (*potential, *belief)
        if hasattr(value, "shape")
    )


@pytest.mark.parametrize(
    "call,match",
    (
        (
            lambda: recurrence.structured_pair_potential(
                np.zeros(2), np.eye(1), np.zeros((2, 1, 2)), np.zeros((2, 1))
            ),
            "drive moments require",
        ),
        (
            lambda: recurrence.structured_pair_belief(
                np.zeros(2), np.zeros(3), np.zeros((2, 2))
            ),
            "source/destination cavities require",
        ),
        (
            lambda: recurrence.structured_gaussian_message(
                np.zeros((2, 3)),
                np.zeros((2, 1)),
                np.zeros((2, 1, 1)),
                np.zeros((2, 1)),
            ),
            "pair probability requires",
        ),
    ),
)
def test_shape_errors_are_explicit(call, match):
    with pytest.raises(ValueError, match=match):
        call()
