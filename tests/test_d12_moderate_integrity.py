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


@pytest.mark.parametrize("probe", ["weighted", "rare"])
def test_native_projection_matches_saved_input_covariance(probe):
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
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout.splitlines()[-1])
    assert record["native"] and record["complete_program_calls"] == 0
    assert record["retained_low_word"] > 0 and record["local_cases"] == 2


def sparse_control(case):
    """Analytic independent-label control with exact inactive support."""
    from d12_compile_fixture import fixture_arrays

    arrays = synthetic(case)
    q = np.float32([0, 0.25, 0.25, 0.5])
    for name, mass in (("regions", q[:, None] * q[None, :]), ("separators", q)):
        prefix = "state.evaluation.local." + name + "."
        p = arrays[prefix + "probability"]
        p[...] = mass
        arrays[prefix + "log_probability"][...] = np.log(np.where(p > 0, p, 1))
        arrays[prefix + "log_probability"][p == 0] = -np.inf
        arrays[prefix + "covariance"][p == 0] = 0
        for field in ("full_covariance", "second"):
            arrays[name + "." + field][...] = arrays[prefix + "covariance"]
    p = arrays["state.evaluation.local.regions.probability"]
    exposure = np.cumsum(p[:, :, ::-1], axis=2, dtype=np.float32)[:, :, ::-1][:, :, :-1]
    parameters, _ = fixture_arrays(case["T"])
    target = np.where(
        exposure > 0,
        np.sum(parameters["weights"] ** 2, axis=-1) + parameters["offsets"] ** 2,
        np.float32(0),
    )
    arrays["xi"][...] = target
    arrays["state.coordinate"][-target.size :] = target.ravel()
    arrays["state.evaluation.scales"][-target.size :] = (1 + target).ravel()
    arrays["state.evaluation.auxiliary.target"][...] = target
    arrays["state.evaluation.auxiliary.exposure"][...] = exposure
    for name in ("state.evaluation.auxiliary.linear", "learning.physical_linear"):
        arrays[name][..., -1] = p[:, :, :-1] - exposure / 2
    roots = np.sqrt(np.where(target > 0, target, 1).astype(np.float64))
    omega = np.where(target > 0, np.tanh(roots / 2) / (2 * roots), 0.25).astype(
        np.float32
    )
    for name in ("state.evaluation.auxiliary.precision", "learning.physical_precision"):
        arrays[name][...] = (omega * exposure)[..., None, None] * np.eye(
            case["d"] + 1, dtype=np.float32
        )
    validate_outputs(arrays, case)
    return arrays


def replace_terminal(directory, report, folder, worker, arrays, name):
    """Refresh every affected hash so semantic rejection is the tested mechanism."""
    worker["cases"][0]["terminal"] = write_snapshot(folder, name, arrays)
    write_json(folder / "worker.json", worker)
    report["jobs"][0]["worker_sha256"] = digest(folder / "worker.json")
    write_json(directory / "report.json", report)


@pytest.mark.parametrize(
    "kind", ["oversized_low_word", "finite_log_zero", "zero_updates"]
)
def test_rereview_corruptions_reject_from_valid_aggregate(tmp_path, kind):
    from d12_moderate_report import validate_work

    report = complete_report(tmp_path)
    folder = tmp_path / "t32"
    worker = json.loads((folder / "worker.json").read_text())
    case = worker["cases"][0]["case"]
    arrays = sparse_control(case) if kind == "finite_log_zero" else synthetic(case)
    if kind == "finite_log_zero":
        # Start comparisons require the same exact support face within a horizon.
        for row in worker["cases"][1:]:
            row["terminal"] = write_snapshot(
                folder, "sparse-" + row["case"]["id"], sparse_control(row["case"])
            )
    replace_terminal(tmp_path, report, folder, worker, arrays, "positive-control")
    assert validate_report(tmp_path)["complete"]
    if kind == "oversized_low_word":
        for name in ("regions", "separators"):
            prefix = "state.evaluation.local." + name + "."
            high = arrays[prefix + "covariance"]
            arrays[prefix + "covariance_correction"][...] = -high
            high *= 2  # The rounded physical covariance is unchanged.
        message = "invalid covariance low word"
    elif kind == "finite_log_zero":
        for name in ("regions", "separators"):
            prefix = "state.evaluation.local." + name + "."
            arrays[prefix + "log_probability"][
                arrays[prefix + "probability"] == 0
            ] = -1000
        message = "support/finite-log underflow"
    else:
        arrays["state.iterations"][()] = 0
        arrays["state.evaluations"][()] = 2
        # Terminal-only counts and certificates pass; the initial point falsifies the history.
        validate_outputs(arrays, case)
        validate_work(arrays)
        message = "zero accepted updates changed coordinate"
    replace_terminal(
        tmp_path, report, folder, worker, arrays, "corrupted-rereview-terminal"
    )
    with pytest.raises(ValueError, match=message):
        validate_report(tmp_path)


def test_signed_correction_words_are_valid_without_being_positive_definite():
    case = solve_cases()[0]
    arrays = synthetic(case)
    validate_outputs(arrays, case)
    for name in ("regions", "separators"):
        low = arrays[f"state.evaluation.local.{name}.covariance_correction"]
        signs = np.resize(np.float32([-1, 1]), low.shape[-1])
        low[...] = np.diag(signs * np.finfo(np.float32).eps / 4)
    validate_outputs(arrays, case)


@pytest.mark.parametrize(
    "field,kind",
    [
        ("covariance", "asymmetric"),
        ("covariance_correction", "asymmetric"),
        ("covariance_correction", "nonfinite"),
        ("covariance_correction", "subnormal"),
        ("covariance_correction", "dtype"),
        ("covariance_correction", "shape"),
    ],
)
def test_belief_word_validity(field, kind):
    from d12_moderate_payload import validate_belief

    arrays = synthetic(solve_cases()[0])
    prefix = "state.evaluation.local.separators."
    high = arrays[prefix + "covariance"]
    high[..., 0, 1] = high[..., 1, 0] = np.float32(0.25)
    for name in ("full_covariance", "second"):
        arrays["separators." + name][...] = high
    validate_belief(arrays, "separators")
    word = arrays[prefix + field]
    if kind == "asymmetric":
        word[0, 0, 0, 1] = (
            np.nextafter(word[0, 0, 0, 1], np.float32(np.inf))
            if field == "covariance"
            else np.finfo(np.float32).eps / 8
        )
    elif kind == "nonfinite":
        word[0, 0, 0, 0] = np.inf
    elif kind == "subnormal":
        word[0, 0, 0, 0] = np.nextafter(np.float32(0), np.float32(1))
    elif kind == "dtype":
        arrays[prefix + field] = word.astype(np.float64)
    else:
        arrays[prefix + field] = word[..., 0]
    with pytest.raises(ValueError, match="word"):
        validate_belief(arrays, "separators")


@pytest.mark.parametrize(
    "logs", [-1000, -100, np.float32(np.log(np.finfo(np.float32).tiny))]
)
def test_positive_probability_cannot_hide_native_exp_underflow(logs):
    from d12_moderate_payload import validate_belief

    arrays = sparse_control(solve_cases()[0])
    prefix = "state.evaluation.local.separators."
    validate_belief(arrays, "separators")
    # Even a forged positive normal probability must not hide an underflowed exp(log p).
    arrays[prefix + "log_probability"][:, 0] = logs
    arrays[prefix + "probability"][:, 0] = np.finfo(np.float32).tiny
    arrays[prefix + "covariance"][:, 0] = np.eye(2, dtype=np.float32)
    for field in ("full_covariance", "second"):
        arrays["separators." + field][:, 0] = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="native probability underflow"):
        validate_belief(arrays, "separators")


def test_unchanged_zero_update_and_single_update_coordinate_history():
    from d12_moderate_report import validate_work

    case = solve_cases()[0]
    terminal = synthetic(case)
    terminal["state.iterations"][()] = 0
    terminal["state.evaluations"][()] = 2
    validate_outputs(terminal, case)
    # The explicit-coordinate loop can start at this analytic stationary point.
    initial = {"state.coordinate": terminal["state.coordinate"].copy()}
    validate_work(terminal, initial)
    initial["state.coordinate"][-1] = np.nextafter(
        terminal["state.coordinate"][-1], np.float32(np.inf)
    )
    with pytest.raises(ValueError, match="zero accepted updates changed"):
        validate_work(terminal, initial)
    terminal["state.iterations"][()] = 1
    terminal["state.evaluations"][()] = 3
    initial["state.coordinate"][-1] += np.float32(1)
    validate_work(terminal, initial)
    initial["state.coordinate"][...] = terminal["state.coordinate"]
    with pytest.raises(ValueError, match="one accepted update did not move"):
        validate_work(terminal, initial)
    # Endpoints alone do not determine movement after multiple updates.
    terminal["state.iterations"][()] = 2
    terminal["state.evaluations"][()] = 4
    validate_work(terminal, initial)


def xla_control(record):
    record["xla_memory"] = dict(
        available=True,
        estimates=dict(
            argument_size_in_bytes=64,
            output_size_in_bytes=32,
            temp_size_in_bytes=8,
            alias_size_in_bytes=16,
        ),
    )
    return record


@pytest.mark.parametrize(
    "kind", ["missing", "negative", "float", "bool", "argument", "output", "zero"]
)
def test_xla_allocation_byte_relationships(kind):
    from test_d12_moderate import compilation_record
    from d12_moderate_report import validate_compile

    record = xla_control(compilation_record(solve_cases()[0], "initialize", 0))
    validate_compile(record, 0)
    estimates = record["xla_memory"]["estimates"]
    if kind == "missing":
        del estimates["temp_size_in_bytes"]
    elif kind in ("negative", "float", "bool"):
        estimates["temp_size_in_bytes"] = {"negative": -1, "float": 8.0, "bool": True}[
            kind
        ]
    elif kind in ("argument", "output"):
        estimates[kind + "_size_in_bytes"] = 8
    else:
        estimates.update(
            argument_size_in_bytes=0,
            output_size_in_bytes=0,
            alias_size_in_bytes=1000000,
        )
    with pytest.raises(ValueError, match="XLA"):
        validate_compile(record, 0)


def test_impossible_xla_aliases_rejected_in_complete_storage_report(tmp_path):
    from test_d12_moderate import compilation_record
    from d12_moderate_plan import SCHEMA, identities, load_spec

    spec, trusted, identity = load_spec(), contract(), identities()
    identity["git_status"] = ""
    runtime = deepcopy(trusted["binding"]["runtime"])
    runtime["config"] = {
        k: v for k, v in runtime["trace_config"].items() if "cache" in k or "x64" in k
    }
    runtime["clock"] = dict(monotonic=True, adjustable=False)
    report = dict(
        schema=SCHEMA, phase="storage", status="complete", identity=identity, jobs=[]
    )
    write_json(tmp_path / "spec.json", spec)
    for case in [dict(id="inventory"), *spec["storage"]["cases"]]:
        folder = tmp_path / case["id"]
        folder.mkdir()
        worker = dict(status="complete", identity=identity, runtime=runtime)
        if case["id"] == "inventory":
            worker["complete_program_calls"] = dict(lower=0, compile=0, execute=0)
            worker["shape_inventory"] = deepcopy(trusted["cases"])
        else:
            worker["case"] = case
            worker["compilation"] = xla_control(
                compilation_record(case, case["program"], 0)
            )
            shape = {k: case[k] for k in ("T", "K", "d")}
            worker["logical"] = next(
                entry["inventory"]
                for entry in trusted["cases"]
                if entry["shape"] == shape
            )
        (folder / "compiler.log").write_text(
            "Synthetic storage control; no compilation.\n"
        )
        process = dict(
            exit_code=0,
            stop=None,
            worker_seconds=1.0,
            rss_samples=dict(count=0, maximum_bytes=None),
            cache_evidence=dict(log_sha256=digest(folder / "compiler.log")),
            process_high_water=dict(raw_unit="bytes", raw_ru_maxrss=1, bytes=1),
        )
        write_json(folder / "worker.json", worker)
        write_json(folder / "process.json", process)
        report["jobs"].append(
            dict(
                id=case["id"],
                worker_sha256=digest(folder / "worker.json"),
                process_sha256=digest(folder / "process.json"),
            )
        )
    write_json(tmp_path / "report.json", report)
    assert validate_report(tmp_path)["complete"]
    folder = tmp_path / report["jobs"][1]["id"]
    worker = json.loads((folder / "worker.json").read_text())
    worker["compilation"]["xla_memory"]["estimates"].update(
        argument_size_in_bytes=0, output_size_in_bytes=0, alias_size_in_bytes=1000000
    )
    write_json(folder / "worker.json", worker)
    report["jobs"][1]["worker_sha256"] = digest(folder / "worker.json")
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="XLA alias bytes"):
        validate_report(tmp_path)
