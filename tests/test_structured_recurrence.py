"""Executable anchors for the structured recurrence derivation."""

import itertools

import numpy as np
import pytest

from cffg.experiments.structured_recurrence import (
    chain_entropy,
    enumerate_log_chain,
    fixed_drive_potentials,
    local_pair_from_messages,
    probability_tolerance,
    registered_case,
    smooth_log_chain,
    structured_free_energy,
    structured_recurrence_statistics,
    transition_bethe_energy,
    uncertain_drive_potentials,
)


def _max_error(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


@pytest.mark.parametrize("card", (2, 3))
def test_structured_messages_beliefs_and_free_energy_match_enumeration(card):
    log_initial, log_evidence, log_transition, *_ = fixed_drive_potentials(card)
    smoothed = smooth_log_chain(log_initial, log_evidence, log_transition)
    enumerated = enumerate_log_chain(log_initial, log_evidence, log_transition)
    tolerance = probability_tolerance(np.float64, card)

    assert _max_error(smoothed.singleton, enumerated.singleton) <= tolerance
    assert _max_error(smoothed.consecutive, enumerated.consecutive) <= tolerance
    assert smoothed.log_normalizer == pytest.approx(
        enumerated.log_normalizer, abs=tolerance
    )
    assert structured_free_energy(
        smoothed, log_initial, log_evidence, log_transition
    ) == pytest.approx(-smoothed.log_normalizer, abs=tolerance)


@pytest.mark.parametrize("card", (2, 3))
def test_time_local_pair_regions_are_globally_consistent(card):
    log_initial, log_evidence, log_transition, *_ = fixed_drive_potentials(card)
    result = smooth_log_chain(log_initial, log_evidence, log_transition)
    tolerance = probability_tolerance(np.float64, card)
    for step, pair in enumerate(result.consecutive):
        local = local_pair_from_messages(
            result.forward[step],
            log_evidence[step + 1] + result.backward[step + 1],
            log_transition[step],
        )
        assert _max_error(local, pair) <= tolerance
        assert _max_error(pair.sum(axis=1), result.singleton[step]) <= tolerance
        assert _max_error(pair.sum(axis=0), result.singleton[step + 1]) <= tolerance


def test_pair_regions_and_edge_corrections_equal_chain_entropy_and_energy():
    log_initial, log_evidence, log_transition, *_ = fixed_drive_potentials(3)
    result = smooth_log_chain(log_initial, log_evidence, log_transition)
    expected_transition = np.einsum(
        "tij,tij->", result.consecutive, log_transition
    )
    direct = -chain_entropy(result) - expected_transition
    assert transition_bethe_energy(result, log_transition) == pytest.approx(
        direct, abs=probability_tolerance(np.float64, 3)
    )


def test_pair_responsibilities_drive_all_recurrence_statistics():
    log_initial, log_evidence, log_transition, drives, weights, offsets = (
        fixed_drive_potentials(2)
    )
    result = smooth_log_chain(log_initial, log_evidence, log_transition)
    covariances = np.zeros((len(drives), 1, 1))
    statistics = structured_recurrence_statistics(
        result, drives, covariances, weights, offsets
    )
    for step, row in enumerate(statistics):
        pair = result.consecutive[step]
        at_risk = pair.sum(axis=1)[:, None]
        signed = pair[:, :1] - 0.5 * at_risk
        np.testing.assert_allclose(row["exposure"], at_risk)
        np.testing.assert_allclose(row["linear"][..., -1], signed)
    product = result.singleton[:-1, :, None] * result.singleton[1:, None, :]
    assert _max_error(product, result.consecutive) > 1e-3


def test_uncertain_drive_retains_the_jj_bound_boundary():
    log_initial, log_evidence, exact_transition, bounded_transition = (
        uncertain_drive_potentials()
    )
    exact = smooth_log_chain(log_initial, log_evidence, exact_transition)
    bounded = smooth_log_chain(log_initial, log_evidence, bounded_transition)
    assert structured_free_energy(
        bounded, log_initial, log_evidence, bounded_transition
    ) >= structured_free_energy(
        exact, log_initial, log_evidence, exact_transition
    )
    assert _max_error(bounded.consecutive, exact.consecutive) > 1e-3


def test_literal_zero_support_and_vacuous_evidence_are_exact():
    log_initial = np.asarray([0.0, -np.inf])
    log_evidence = np.zeros((3, 2))
    with np.errstate(divide="ignore"):
        log_transition = np.log(
            np.asarray(
                [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
            )
        )
    result = smooth_log_chain(log_initial, log_evidence, log_transition)
    np.testing.assert_array_equal(
        result.singleton, [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    assert np.isfinite(result.log_normalizer)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_dtype_scaled_tolerance_covers_normalization(dtype):
    log_initial, log_evidence, log_transition, *_ = fixed_drive_potentials(
        2, dtype=dtype
    )
    result = smooth_log_chain(log_initial, log_evidence, log_transition)
    tolerance = probability_tolerance(dtype, 2)
    np.testing.assert_allclose(
        result.singleton.sum(axis=1), 1.0, atol=tolerance
    )
    np.testing.assert_allclose(
        result.consecutive.sum(axis=(1, 2)), 1.0, atol=tolerance
    )


@pytest.mark.parametrize(
    "replacement,match",
    (
        (("initial", np.asarray([np.nan, 0.0])), "log initial"),
        (("evidence", np.asarray([[0.0, 0.0], [np.inf, 0.0]])), "log evidence"),
        (
            ("transition", np.asarray([[[0.0, 0.0], [0.0, np.nan]]])),
            "log transition",
        ),
    ),
)
def test_nonfinite_potentials_fail_loudly(replacement, match):
    values = {
        "initial": np.zeros(2),
        "evidence": np.zeros((2, 2)),
        "transition": np.zeros((1, 2, 2)),
    }
    values[replacement[0]] = replacement[1]
    with pytest.raises(ValueError, match=match):
        smooth_log_chain(values["initial"], values["evidence"], values["transition"])


def test_inconsistent_empty_support_fails_loudly():
    with pytest.raises(ValueError, match="empty support at step 1"):
        smooth_log_chain(
            np.zeros(2),
            np.zeros((2, 2)),
            np.full((1, 2, 2), -np.inf),
        )


def test_k8_t100_uses_dynamic_programming_not_path_enumeration():
    card, time = 8, 100
    log_initial = np.log(np.arange(1, card + 1))
    log_evidence = np.log(
        1.0 + (np.arange(time)[:, None] + np.arange(card)[None, :]) % 9
    )
    log_transition = np.log(
        np.broadcast_to(
            1.0 + (np.arange(card)[:, None] + np.arange(card)[None, :]) % 7,
            (time - 1, card, card),
        )
    )
    result = smooth_log_chain(log_initial, log_evidence, log_transition)
    assert result.singleton.shape == (time, card)
    assert result.consecutive.shape == (time - 1, card, card)
    with pytest.raises(ValueError, match="restricted"):
        enumerate_log_chain(log_initial, log_evidence, log_transition)


def test_path_entropy_matches_literal_enumeration():
    log_initial, log_evidence, log_transition, *_ = fixed_drive_potentials(2)
    result = smooth_log_chain(log_initial, log_evidence, log_transition)
    path_probability = []
    for path in itertools.product(range(2), repeat=len(log_evidence)):
        value = log_initial[path[0]] + log_evidence[0, path[0]]
        for step in range(len(path) - 1):
            value += log_transition[step, path[step], path[step + 1]]
            value += log_evidence[step + 1, path[step + 1]]
        path_probability.append(np.exp(value - result.log_normalizer))
    path_probability = np.asarray(path_probability)
    direct_entropy = -np.sum(path_probability * np.log(path_probability))
    assert chain_entropy(result) == pytest.approx(
        direct_entropy, abs=probability_tolerance(np.float64, 2)
    )
