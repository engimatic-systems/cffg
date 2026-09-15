"""Shared tiny validation and model-bound local diagnostic summaries."""

import json

import numpy as np
import pytest

from test_d12_moderate import complete_report
from d12_moderate_controls import analytic_model, refresh_arithmetic, synthetic  # noqa: F401
from d12_compile_scaling import digest, write_json
from d12_moderate_payload import (
    close,
    output_contract,
    read_snapshot,
    validate_outputs,
    write_snapshot,
)
from d12_moderate_plan import solve_cases
from d12_moderate_reconstruct import local_summaries, scaled_blocks
from d12_moderate_reference import validate_moderate_outputs
from d12_moderate_report import validate_report


def tiny_point():
    """Analytic T2/K2/d1 zero-message point, not stationary or moderate evidence.

    Zero logits and A=b=0,Q=1 make the factor (1/2)*N(y;0,1).
    Identity unary precisions give region covariance diag(1,1/2), while
    zero-message separators have variance1. Thus the local moment summary
    is1/2 and the outgoing destination precision (quotient summary) is1.
    """
    case = dict(T=2, K=2, d=1, xi=0, messages="raw-zero")
    arrays = {
        key: np.zeros(shape, dtype)
        for key, (shape, dtype) in output_contract(2, 2, 1).items()
    }
    parameters = {
        name: np.zeros(shape, np.float32)
        for name, shape in dict(
            weights=(1, 2, 1, 1),
            offsets=(1, 2, 1),
            xi=(1, 2, 1),
            matrix=(1, 2, 1, 1),
            bias=(1, 2, 1),
            noise=(1, 2, 1, 1),
        ).items()
    }
    parameters["noise"].fill(1)
    for name, probability, covariance, normalizer in (
        ("regions", 0.25, np.diag([1, 0.5]), np.log(2 * np.sqrt(np.pi))),
        ("separators", 0.5, np.eye(1), np.log(2 * np.sqrt(2 * np.pi))),
    ):
        prefix = "state.evaluation.local." + name + "."
        arrays[prefix + "probability"].fill(probability)
        arrays[prefix + "log_probability"].fill(np.log(probability))
        arrays[prefix + "log_normalizer"].fill(normalizer)
        arrays[prefix + "covariance"][...] = covariance
        for field in ("full_covariance", "second"):
            arrays[name + "." + field][...] = covariance
    arrays["state.evaluation.local.targets.1.precision"].fill(1)
    dual = -np.log(2 * np.sqrt(np.pi))
    arrays["state.evaluation.local.dual_energy"][()] = dual
    arrays["state.evaluation.local.free_energy"][()] = dual + 0.25
    arrays["state.status"][()] = 3
    arrays["state.reason"][()] = 1
    arrays["state.evaluations"][()] = 1
    arrays["state.relative_energy_change"][()] = np.inf
    refresh_arithmetic(arrays, case, parameters)
    return arrays, case, parameters


def test_tiny_status_payload_retains_shared_validation():
    arrays, case, parameters = tiny_point()
    validate_outputs(arrays, case, initial=True, parameters=parameters)
    assert local_summaries(arrays, case) == dict(
        moment_residual=0.5, quotient_residual=1
    )
    with pytest.raises(
        ValueError, match="independent reference shape outside contract"
    ):
        validate_moderate_outputs(arrays, case, initial=True, parameters=parameters)
    arrays["state.evaluation.residual"][0] += 1
    with pytest.raises(ValueError, match="reconstructed residual"):
        validate_outputs(arrays, case, initial=True, parameters=parameters)


def test_local_covariance_summary_uses_rounded_readouts():
    arrays, case, parameters = tiny_point()
    # Match both endpoint projections to their separators at the high word.
    arrays["state.evaluation.local.regions.covariance"][..., 1, 1] = 1
    arrays["state.evaluation.local.regions.covariance_correction"][..., 1, 1] = (
        np.finfo(np.float32).eps / 4
    )
    for field in ("full_covariance", "second"):
        arrays["regions." + field][..., 1, 1] = 1
    refresh_arithmetic(arrays, case, parameters)
    validate_outputs(arrays, case, initial=True, parameters=parameters)
    assert local_summaries(arrays, case)["moment_residual"] == 0
    assert scaled_blocks(arrays, case, parameters)[2][3] > 0


def test_local_quotient_summary_uses_each_incidence_gauge_and_scale():
    arrays, case, parameters = tiny_point()
    arrays["messages.left.constant"][0, 1] = 4
    arrays["state.evaluation.local.targets.0.constant"][0] = [100, 106]
    # Destination precision differs by1 at magnitude4; constants dominate at2/6.
    arrays["messages.right.precision"].fill(3)
    arrays["state.evaluation.local.targets.1.precision"].fill(4)
    expected = np.float32(2) / 6
    np.testing.assert_equal(
        local_summaries(arrays, case)["quotient_residual"], expected
    )
    arrays["state.evaluation.local.targets.0.constant"] -= 100
    np.testing.assert_equal(
        local_summaries(arrays, case)["quotient_residual"], expected
    )


@pytest.mark.usefixtures("analytic_model")
@pytest.mark.parametrize("field", ["moment_residual", "quotient_residual"])
@pytest.mark.parametrize("initial", [False, True])
def test_local_summary_corruption_from_valid_point(field, initial):
    case = solve_cases()[1 if initial else 0]
    arrays = synthetic(case, initial=initial)
    validate_moderate_outputs(arrays, case, initial=initial)
    arrays["state.evaluation.local." + field][()] = 100
    with pytest.raises(ValueError, match="local " + field):
        validate_moderate_outputs(arrays, case, initial=initial)


@pytest.mark.usefixtures("analytic_model")
@pytest.mark.parametrize("field", ["moment_residual", "quotient_residual"])
@pytest.mark.parametrize(
    "value", [-1e-5, 0.0, -0.0], ids=["negative", "zero", "signed-zero"]
)
def test_local_summary_sign_inside_arithmetic_allowance(field, value):
    case = solve_cases()[0]
    arrays = synthetic(case)
    validate_moderate_outputs(arrays, case)
    summary = arrays["state.evaluation.local." + field]
    summary[()] = np.float32(value)
    close(summary, local_summaries(arrays, case)[field], "local " + field)
    if value < 0:
        with pytest.raises(ValueError, match="negative local " + field):
            validate_moderate_outputs(arrays, case)
    else:
        validate_moderate_outputs(arrays, case)


@pytest.mark.usefixtures("analytic_model")
@pytest.mark.parametrize("field", ["moment_residual", "quotient_residual"])
def test_negative_local_summary_aggregate_with_refreshed_hashes(tmp_path, field):
    report = complete_report(tmp_path)
    folder = tmp_path / "t32"
    worker = json.loads((folder / "worker.json").read_text())
    row = worker["cases"][0]
    arrays = read_snapshot(folder, row["terminal"])
    summary = arrays["state.evaluation.local." + field]
    summary[()] = np.float32(-1e-5)
    close(summary, local_summaries(arrays, row["case"])[field], "local " + field)
    row["terminal"] = write_snapshot(folder, "negative-summary", arrays)
    write_json(folder / "worker.json", worker)
    report["jobs"][0]["worker_sha256"] = digest(folder / "worker.json")
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="negative local " + field):
        validate_report(tmp_path)


@pytest.mark.usefixtures("analytic_model")
@pytest.mark.parametrize("field", ["moment_residual", "quotient_residual", "both"])
@pytest.mark.parametrize("point", ["initial", "terminal"])
def test_local_summary_aggregate_with_refreshed_hashes(tmp_path, field, point):
    report = complete_report(tmp_path)
    folder = tmp_path / "t32"
    worker = json.loads((folder / "worker.json").read_text())
    row = worker["cases"][1 if point == "initial" else 0]
    arrays = read_snapshot(folder, row[point])
    fields = ("moment_residual", "quotient_residual") if field == "both" else (field,)
    for name in fields:
        arrays["state.evaluation.local." + name][()] = 100
    row[point] = write_snapshot(folder, "changed-summary", arrays)
    write_json(folder / "worker.json", worker)
    report["jobs"][0]["worker_sha256"] = digest(folder / "worker.json")
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="local (moment|quotient)_residual"):
        validate_report(tmp_path)
