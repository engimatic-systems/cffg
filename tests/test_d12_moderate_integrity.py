"""Independent corruption mechanisms; each starts from a separately accepted control."""

from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from test_d12_moderate import ROOT, complete_report, synthetic
from d12_compile_scaling import child_environment, digest, write_json
from d12_moderate_layouts import contract
from d12_moderate_plan import solve_cases
from d12_moderate_payload import read_snapshot, validate_outputs, write_snapshot
from d12_moderate_reconstruct import physical_learning
from d12_moderate_report import (
    validate_fixture,
    validate_logical,
    validate_report,
    validate_runtime,
)


def corrupt(arrays, kind):
    evaluation = "state.evaluation."
    auxiliary = evaluation + "auxiliary."
    separator = evaluation + "local.separators."
    if kind == "natural_scale":
        arrays[evaluation + "local.targets.0.constant"][0, 1] = 1
        arrays[evaluation + "residual"][0] = 1
        arrays[evaluation + "scales"][0] = 1e6
        arrays[evaluation + "certificates.natural"][()] = 1e-6
    elif kind == "mean":
        arrays[separator + "displacement"][0, 0, 0] = 10
        arrays["separators.mean"][0, 0, 0] = 10
        arrays["separators.second"][0, 0, 0, 0] += 100
    elif kind == "covariance":
        arrays[separator + "covariance"][0, 0, 0, 0] += 10
        arrays["separators.full_covariance"][0, 0, 0, 0] += 10
        arrays["separators.second"][0, 0, 0, 0] += 10
    elif kind == "mass":
        arrays[separator + "probability"][0] = np.float32([0.1, 0.2, 0.3, 0.4])
        arrays[separator + "log_probability"][0] = np.log(
            arrays[separator + "probability"][0]
        )
    elif kind == "auxiliary_residual":
        arrays[auxiliary + "residual"].fill(1)
    elif kind == "auxiliary_target":
        arrays[auxiliary + "target"].fill(100)
    elif kind == "auxiliary_scale":
        arrays[evaluation + "scales"][-1] = 1e6
    elif kind == "certificate_summary":
        arrays[evaluation + "certificates.natural"][()] = np.float32(0.0001)
    elif kind == "primal_dual":
        arrays[evaluation + "local.dual_energy"][()] += 1
    elif kind == "physical_precision":
        arrays["learning.physical_precision"][0, 0, 0, 0, 0] = -100
    elif kind == "physical_linear":
        arrays["learning.physical_linear"][0, 0, 0, 0] = 100
    elif kind in ("centered_precision", "centered_linear", "center"):
        field = kind.removeprefix("centered_")
        arrays[auxiliary + field].flat[0] += 10
        values = physical_learning(
            *(arrays[auxiliary + name] for name in ("linear", "precision", "center"))
        )
        for name, value in values.items():
            arrays["learning." + name] = value
    elif kind == "exposure":
        arrays[auxiliary + "exposure"].fill(1)
    elif kind == "energy":
        arrays["state.relative_energy_change"][()] = -np.inf
    else:
        raise AssertionError(kind)


CORRUPTIONS = (
    "natural_scale",
    "mean",
    "covariance",
    "mass",
    "auxiliary_residual",
    "auxiliary_target",
    "auxiliary_scale",
    "primal_dual",
    "certificate_summary",
    "physical_precision",
    "physical_linear",
    "centered_precision",
    "centered_linear",
    "center",
    "exposure",
    "energy",
)


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_saved_relationship_corruption(kind):
    case = solve_cases()[0]
    arrays = synthetic(case)
    validate_outputs(arrays, case)
    corrupt(arrays, kind)
    with pytest.raises(
        ValueError, match="reconstructed|prescribed|learning|convergence|summary"
    ):
        validate_outputs(arrays, case)


@pytest.mark.parametrize("field", ["center", "status", "precision"])
def test_target_coordinate_metadata(field):
    case = solve_cases()[0]
    arrays = synthetic(case)
    validate_outputs(arrays, case)
    value = arrays["state.evaluation.local.targets.0." + field]
    if field == "precision":
        value[0, 0, 1, 0] = 1
    else:
        value.flat[0] = 1
    with pytest.raises(ValueError, match="target metadata"):
        validate_outputs(arrays, case)


@pytest.mark.parametrize("change", [-np.inf, -1, np.nan, np.inf])
def test_terminal_energy_is_finite_nonnegative_but_initial_infinity_is_valid(change):
    case = solve_cases()[0]
    validate_outputs(synthetic(case, initial=True), case, initial=True)
    arrays = synthetic(case)
    validate_outputs(arrays, case)
    arrays["state.relative_energy_change"][()] = change
    with pytest.raises(ValueError, match="convergence"):
        validate_outputs(arrays, case)


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_aggregate_cannot_certify_corrupt_terminal(tmp_path, kind):
    report = complete_report(
        tmp_path
    )  # Asserts the complete twelve-case control passes.
    folder = tmp_path / "t32"
    worker = json.loads((folder / "worker.json").read_text())
    record = worker["cases"][0]
    arrays = read_snapshot(folder, record["terminal"])
    corrupt(arrays, kind)
    record["terminal"] = write_snapshot(folder, "corrupted-terminal", arrays)
    write_json(folder / "worker.json", worker)
    report["jobs"][0]["worker_sha256"] = digest(folder / "worker.json")
    write_json(tmp_path / "report.json", report)
    with pytest.raises(
        ValueError, match="reconstructed|prescribed|learning|convergence|summary"
    ):
        validate_report(tmp_path)


@pytest.mark.parametrize(
    "group", ["metadata", "context", "trial_state", "derivative_captures"]
)
def test_complete_inventory_rejects_scalar_placeholders(group):
    entry = deepcopy(contract()["cases"][0])
    inventory, shape = entry["inventory"], entry["shape"]
    validate_logical(inventory, shape)
    inventory["groups"][group] = [
        dict(path=".bogus", shape=[], dtype="float32", elements=1, bytes=4),
        dict(path=".index", shape=[], dtype="int32", elements=1, bytes=4),
    ]
    with pytest.raises(ValueError, match=group):
        validate_logical(inventory, shape)


@pytest.mark.parametrize(
    "group,leaf_kind",
    [
        ("metadata", "index"),
        ("context", "index"),
        ("trial_state", "correction"),
        ("derivative_captures", "index"),
        ("accepted_evaluation", "correction"),
        ("terminal_state", "correction"),
    ],
)
def test_inventory_cannot_omit_retained_leaf(group, leaf_kind):
    entry = deepcopy(contract()["cases"][0])
    inventory, shape = entry["inventory"], entry["shape"]
    validate_logical(inventory, shape)
    leaves = inventory["groups"][group]
    index = next(
        i
        for i, leaf in enumerate(leaves)
        if (
            np.dtype(leaf["dtype"]).kind in "iu"
            if leaf_kind == "index"
            else "correction" in leaf["path"]
        )
    )
    leaves.pop(index)
    with pytest.raises(ValueError, match=group):
        validate_logical(inventory, shape)


@pytest.mark.parametrize("kind", ["source", "runtime", "worker"])
def test_inventory_source_and_runtime_binding(kind):
    entry = deepcopy(contract()["cases"][0])
    inventory, shape = entry["inventory"], entry["shape"]
    runtime = deepcopy(inventory["binding"]["runtime"])
    validate_logical(inventory, shape, runtime)
    if kind == "source":
        inventory["binding"]["sources"]["src/cffg/hybrid_joint.py"] = "0" * 64
    elif kind == "runtime":
        inventory["binding"]["runtime"]["jax"] = "different"
    else:
        runtime["trace_config"]["jax_enable_x64"] = True
    with pytest.raises(ValueError, match="binding|runtime"):
        validate_logical(inventory, shape, runtime)


def test_fixed_input_inventory_and_process_diagnostics(tmp_path):
    complete_report(tmp_path)
    folder = tmp_path / "t32"
    worker = json.loads((folder / "worker.json").read_text())
    arrays = read_snapshot(folder, worker["inputs"])
    validate_fixture(arrays, 32)
    for name in ("origin", "support", "active", "constants", "first", "auxiliary"):
        bad = deepcopy(arrays)
        bad["metadata." + name] = bad["metadata." + name].flat[0:1].reshape(())
        with pytest.raises(ValueError, match="fixture"):
            validate_fixture(bad, 32)
        del bad["metadata." + name]
        with pytest.raises(ValueError, match="inventory"):
            validate_fixture(bad, 32)
    process = json.loads((folder / "process.json").read_text())
    validate_runtime(worker, process)
    for name in ("worker_seconds", "rss_samples"):
        bad = deepcopy(process)
        del bad[name]
        with pytest.raises(ValueError, match="elapsed|RSS"):
            validate_runtime(worker, bad)


def test_native_local_reconstruction_rounding():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/d12_moderate_checks.py"), "--arithmetic"],
        cwd=ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout.splitlines()[-1])
    assert record["native"] and record["complete_program_calls"] == 0
    assert record["retained_low_word"] > 0 and record["local_cases"] == 2


class ProjectionDiscrepancy(AssertionError):
    pass


@pytest.mark.xfail(
    strict=True,
    raises=ProjectionDiscrepancy,
    reason="cdh-urn0: native marginal disagrees with saved-input covariance replay",
)
@pytest.mark.parametrize("probe", ["weighted", "rare"])
def test_native_projection_discrepancy_remains_visible(probe):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/d12_moderate_checks.py"),
            "--projection-probe",
            probe,
        ],
        cwd=ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode:
        assert "ValueError: projection covariance," in result.stderr, result.stderr
        raise ProjectionDiscrepancy(result.stderr.splitlines()[-1])
