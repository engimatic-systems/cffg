"""Independent exact-chain diagnosis for recurrence-link responsibilities."""

import json
from pathlib import Path

import numpy as np
import pytest

from cffg.experiments.recurrence_smoothing import (
    _case,
    diagnose_case,
    diagnose_hostile_inputs,
    diagnose_uncertain_drive,
    enumerate_paths,
    forward_backward,
    integrated_transition,
    mean_field_chain,
    probability_tolerance,
    recurrence_statistics,
    stick_transition,
    write_report,
)


@pytest.mark.parametrize("card", (2, 3))
def test_enumerator_and_engine_comparison_issues_variational_ownership(card):
    result = diagnose_case(card)
    tolerance = result["tolerance"]
    assert result["cffg_status"] == "converged"
    assert result["enumerator_vs_forward_backward"] <= tolerance
    assert result["independent_coordinate_message_error"] > 100 * tolerance
    assert result["sum_product_message_error"] <= tolerance
    assert result["singleton_error"] <= tolerance
    assert result["factor_local_joint_error"] <= tolerance
    assert result["learning_readout_error"] <= tolerance
    assert result["product_joint_correlated_gap"] > 100 * tolerance
    assert result["product_joint_error"] > 100 * tolerance
    assert result["learning_structured_responsibility_error"] > 100 * tolerance


def test_uncertain_drive_separates_jj_and_categorical_factorization():
    result = diagnose_uncertain_drive()
    tolerance = result["tolerance"]
    assert result["pg_jj_singleton_error"] > 100 * tolerance
    assert result["pg_jj_pair_error"] > 100 * tolerance
    assert result["categorical_factorization_singleton_error"] > 100 * tolerance
    assert result["categorical_factorization_pair_error"] > 100 * tolerance


def test_exact_zeros_deterministic_and_vacuous_evidence_preserve_support():
    initial = np.array([1.0, 0.0])
    evidence = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    transitions = np.array([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])
    posterior, _ = forward_backward(initial, evidence, transitions)
    np.testing.assert_array_equal(posterior.singleton, [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(posterior.consecutive[0], [[0.0, 1.0], [0.0, 0.0]])
    enumerated = enumerate_paths(initial, evidence, transitions)
    np.testing.assert_array_equal(enumerated.singleton, posterior.singleton)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_dtype_tolerance_and_engine_readout(dtype):
    result = diagnose_case(2, dtype=dtype)
    assert result["sum_product_message_error"] <= result["tolerance"]
    assert result["learning_readout_error"] <= result["tolerance"]
    assert result["drive_clamp_error"] <= result["tolerance"]
    assert result["tolerance"] == probability_tolerance(dtype, 2)


def test_forward_backward_scales_without_path_enumeration():
    rng = np.random.default_rng(91)
    time, card = 100, 8
    initial = rng.random(card)
    evidence = rng.random((time, card))
    transitions = rng.random((time - 1, card, card))
    posterior, _ = forward_backward(initial, evidence, transitions)
    assert posterior.singleton.shape == (time, card)
    assert posterior.consecutive.shape == (time - 1, card, card)
    np.testing.assert_allclose(posterior.singleton.sum(axis=1), 1.0, atol=probability_tolerance(np.float64, card))
    np.testing.assert_allclose(posterior.consecutive.sum(axis=(1, 2)), 1.0, atol=probability_tolerance(np.float64, card))
    with pytest.raises(ValueError, match="restricted"):
        enumerate_paths(initial, evidence, transitions)


def test_manifested_hostile_inputs_cover_vacuous_and_scaled_paths():
    result = diagnose_hostile_inputs()
    assert result["vacuous_message_status"] == 0
    np.testing.assert_allclose(np.exp(result["vacuous_message_to_out"]), [0.5, 0.5])
    assert len(result["nonfinite_or_improper_rejections"]) == 3
    assert result["scale_case"]["cardinality"] == 8
    assert result["scale_case"]["time"] == 100
    assert not result["scale_case"]["path_enumeration_used"]


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("initial", [np.nan, 1.0], "initial must be finite"),
        ("evidence", [[1.0, 0.0], [np.inf, 1.0]], "evidence must be finite"),
        ("transitions", [[[1.0, 0.0], [np.nan, 1.0]]], "transitions must be finite"),
    ),
)
def test_nonfinite_chain_inputs_fail_loudly(field, value, match):
    inputs = {
        "initial": [0.5, 0.5],
        "evidence": [[1.0, 1.0], [1.0, 1.0]],
        "transitions": [[[0.7, 0.3], [0.2, 0.8]]],
    }
    inputs[field] = value
    with pytest.raises(ValueError, match=match):
        forward_backward(**inputs)


def test_nonfinite_improper_and_zero_statistics_inputs_fail_loudly():
    weights = np.array([[[0.5]], [[-0.4]]])
    offsets = np.zeros((2, 1))
    with pytest.raises(ValueError, match="finite"):
        stick_transition([np.inf], weights, offsets)
    with pytest.raises(ValueError, match="nonnegative variance"):
        integrated_transition([0.0], [[-1.0]], weights, offsets)
    with pytest.raises(ValueError, match="nonempty support"):
        recurrence_statistics(np.zeros((2, 2)), [0.0], [[1.0]], weights, offsets)


def test_product_statistics_differ_from_correlated_joint_statistics():
    initial, evidence, drives, weights, offsets = _case(2)
    transitions = np.stack([stick_transition(drive, weights, offsets) for drive in drives])
    exact, _ = forward_backward(initial, evidence, transitions)
    step = 1
    product = np.outer(exact.singleton[step], exact.singleton[step + 1])
    exact_stats = recurrence_statistics(exact.consecutive[step], drives[step], np.zeros((1, 1)), weights, offsets)
    product_stats = recurrence_statistics(product, drives[step], np.zeros((1, 1)), weights, offsets)
    assert np.max(np.abs(exact_stats["linear"] - product_stats["linear"])) > 1e-4


def test_mean_field_rejects_nonfinite_and_report_is_manifested(tmp_path):
    initial, evidence, drives, weights, offsets = _case(2)
    log_potentials = np.stack([np.log(stick_transition(drive, weights, offsets)) for drive in drives])
    broken = log_potentials.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="transitions must be finite|log potentials must be finite"):
        mean_field_chain(initial, evidence, broken)

    output = tmp_path / "results.json"
    write_report(output, Path(__file__).resolve().parents[1])
    document = json.loads(output.read_text())
    assert document["schema_version"] == 1
    assert document["ticket"] == "cff-nbo9"
    assert document["ownership"]["decision"] == "learning_readout_defect"
    assert len(document["oracle_sha256"]) == 64
    assert document["replay_command"].startswith("uv run python")
