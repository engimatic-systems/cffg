"""Bounded harness checks; synthetic report records are never numerical evidence."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from cffg import hybrid_joint_solver as solver
from d12_compile_fixture import fixture_arrays
from d12_compile_scaling import child_environment, digest, write_json
from d12_moderate_plan import identities, load_spec, solve_cases, storage_cases
from d12_moderate_payload import (
    CERTIFICATES,
    output_contract,
    read_snapshot,
    validate_outputs,
    write_snapshot,
)
from d12_moderate_report import (
    compare_starts,
    fixed_metadata,
    program_contract,
    validate_fixture,
    validate_program,
    validate_report,
    validate_work,
)


def test_prospective_inventory_and_frozen_defaults():
    spec = load_spec()
    assert spec["cases"] == solve_cases() and len(spec["cases"]) == 12
    assert spec["storage"]["cases"] == storage_cases() and len(storage_cases()) == 10
    assert spec["budgets"] == asdict(solver.Budgets())
    assert tuple(solver.joint.Certificates._fields) == CERTIFICATES
    for t in (32, 100):
        cases = [c for c in storage_cases() if c["T"] == t]
        outer = [c for c in cases if c["budgets"]["outer"] != 200]
        assert len(outer) == 1 and outer[0]["program"] == "solve_from_coordinate"
        assert outer[0]["budgets"]["linear"] == 128


def test_frozen_formula_and_axis_mapping():
    for t in (32, 100):
        p, s = fixture_arrays(t)
        validate_fixture(
            {
                **{f"parameters.{k}": v for k, v in p.items()},
                **{f"sites.{k}": v for k, v in s.items()},
                **{f"metadata.{k}": v for k, v in fixed_metadata(t).items()},
            },
            t,
        )
        for i in range(4):
            for l in range(3):
                np.testing.assert_array_equal(
                    p["weights"][:, i, l],
                    np.broadcast_to(
                        np.float32(
                            [
                                0.12 * (i - 1.5) + 0.04 * (l - 1),
                                -0.08 * (i - 1.5) + 0.03 * (l - 1),
                            ]
                        ),
                        (t - 1, 2),
                    ),
                )
            for time in range(t):
                np.testing.assert_array_equal(
                    s["linear"][time, i],
                    np.float32(
                        [
                            0.1 * (i - 1.5) + 0.05 * (-1) ** time,
                            -0.08 * (i - 1.5) + 0.03 * ((time % 3) - 1),
                        ]
                    ),
                )
        assert p["weights"].shape == (t - 1, 4, 3, 2) and p["noise"].shape == (
            t - 1,
            4,
            2,
            2,
        )


def synthetic(case, *, initial=False):
    """Analytic independent-label Gaussian controls for report relationships.

    These are internally consistent payloads, not solves of the frozen model.
    Terminal records model one accepted update and the final certification.
    Uniform pair mass, zero means and identity covariance give closed-form
    projected moments and source-conditioned learning without a report helper.
    """
    arrays = {
        key: np.zeros(shape, dtype)
        for key, (shape, dtype) in output_contract(
            case["T"], case["K"], case["d"]
        ).items()
    }
    arrays["state.evaluation.scales"].fill(1)
    for name in ("regions", "separators"):
        prefix = f"state.evaluation.local.{name}"
        probability = arrays[f"{prefix}.probability"]
        probability.fill(1 / np.prod(probability.shape[1:]))
        arrays[f"{prefix}.log_probability"][...] = np.log(probability)
        covariance = arrays[f"{prefix}.covariance"]
        covariance[...] = np.eye(covariance.shape[-1], dtype=np.float32)
        for field in ("full_covariance", "second"):
            arrays[f"{name}.{field}"][...] = covariance
    parameters, _ = fixture_arrays(case["T"])
    target = np.sum(parameters["weights"] ** 2, axis=-1) + parameters["offsets"] ** 2
    arrays["xi"][...] = case["xi"] if initial else target
    arrays["state.coordinate"][-arrays["xi"].size :] = arrays["xi"].ravel()
    k, d = case["K"], case["d"]
    exposure = (k - np.arange(k - 1, dtype=np.float32)) / np.float32(k * k)
    arrays["state.evaluation.auxiliary.target"][...] = target
    arrays["state.evaluation.auxiliary.residual"][...] = arrays["xi"] - target
    arrays["state.evaluation.auxiliary.exposure"][...] = exposure
    for name in ("state.evaluation.auxiliary.linear", "learning.physical_linear"):
        arrays[name][..., -1] = np.float32(1 / (k * k)) - exposure / 2
    xi = arrays["xi"].astype(np.float64)
    root = np.sqrt(np.where(xi > 0, xi, 1))
    omega = np.where(xi > 0, np.tanh(root / 2) / (2 * root), 0.25).astype(np.float32)
    for name in ("state.evaluation.auxiliary.precision", "learning.physical_precision"):
        arrays[name][...] = (omega * exposure)[..., None, None] * np.eye(
            d + 1, dtype=np.float32
        )
    arrays["state.evaluation.residual"][-xi.size :] = (arrays["xi"] - target).ravel()
    arrays["state.evaluation.scales"][-xi.size :] = (1 + arrays["xi"]).ravel()
    arrays["state.evaluation.certificates.auxiliary"][()] = np.max(
        abs(arrays["xi"] - target) / (1 + arrays["xi"])
    )
    arrays["state.iterations"][()] = 0 if initial else 1
    arrays["state.evaluations"][()] = 1 if initial else 3
    arrays["state.status"][()] = 3 if initial else 0
    arrays["state.reason"][()] = 1 if initial else 0
    if initial:
        arrays["state.relative_energy_change"][()] = np.inf
    return arrays


def test_lossless_snapshots_and_tampering(tmp_path):
    arrays = synthetic(solve_cases()[0], initial=True)
    record = write_snapshot(tmp_path, "initial", arrays)
    loaded = read_snapshot(tmp_path, record)
    validate_outputs(loaded, solve_cases()[0], initial=True)
    assert np.isposinf(loaded["state.relative_energy_change"])
    with pytest.raises(FileExistsError):
        write_snapshot(tmp_path, "initial", arrays)
    record["leaves"]["xi"]["bytes"] += 4
    with pytest.raises(ValueError, match="manifest"):
        read_snapshot(tmp_path, record)
    record = write_snapshot(tmp_path, "second", arrays)
    with (tmp_path / "second.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        read_snapshot(tmp_path, record)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_learning",
        "missing_low_word",
        "dtype",
        "raw_feasible",
        "xi",
        "support",
        "certificate",
        "nonfinite",
        "counter",
        "histogram",
        "status",
    ),
)
def test_malformed_or_inconsistent_outputs_rejected(mutation):
    case = solve_cases()[0]
    initial = mutation in ("raw_feasible", "xi")
    arrays = synthetic(case, initial=initial)
    if mutation == "missing_learning":
        del arrays["learning.physical_precision"]
    elif mutation == "missing_low_word":
        del arrays["state.evaluation.local.regions.covariance_correction"]
    elif mutation == "dtype":
        arrays["xi"] = arrays["xi"].astype(np.float64)
    elif mutation == "raw_feasible":
        arrays["state.coordinate"][0] = 1
        arrays["messages.left.constant"][0, 1] = 1
    elif mutation == "xi":
        arrays["xi"][0, 0, 0] = 1
        arrays["state.coordinate"][-arrays["xi"].size] = 1
    elif mutation == "support":
        arrays["messages.left.constant"][0, 1] = -np.inf
    elif mutation == "certificate":
        arrays["state.evaluation.certificates.conditional_mean"][()] = 1
    elif mutation == "nonfinite":
        arrays["state.evaluation.local.free_energy"][()] = np.nan
    elif mutation == "counter":
        arrays["state.iterations"][()] = 201
    elif mutation == "histogram":
        arrays["state.rejection_counts"][2] = 1
    elif mutation == "status":
        arrays["state.reason"][()] = 3
    with pytest.raises(ValueError):
        validate_outputs(arrays, case, initial=initial)
        validate_work(arrays)


def test_start_comparisons_include_all_groups_and_all_pairs():
    terminals = {c["id"]: synthetic(c) for c in solve_cases()}
    rows = compare_starts(terminals)
    assert len(rows) == 30
    key = solve_cases()[1]["id"]
    terminals[key]["learning.physical_precision"][0, 0, 0, 0, 0] = 1
    rows = compare_starts(terminals)
    assert any(
        row["material"] and row["scaled_maximum"]["learning"] > 0.1 for row in rows
    )
    assert all(
        set(row["scaled_maximum"])
        == {"messages", "xi", "energy", "regions", "separators", "learning"}
        for row in rows
    )


def compilation_record(case, program, calls):
    arguments, outputs = program_contract(case, program)

    def leaves(contract):
        return [
            dict(
                path=name,
                shape=list(shape),
                dtype=dtype,
                elements=int(np.prod(shape)),
                bytes=int(np.prod(shape)) * np.dtype(dtype).itemsize,
            )
            for name, (shape, dtype) in contract.items()
        ]

    return dict(
        calls=dict(lower=1, compile=1, execute=calls),
        lower_seconds=1.0,
        compile_seconds=1.0,
        arguments=leaves(arguments),
        lowered_outputs=leaves(outputs),
        compiled_outputs=leaves(outputs),
        stablehlo_dtypes=dict(value_type_occurrences=dict(f32=1)),
        xla_memory=dict(
            available=False, substitute="logical leaves and process high-water"
        ),
    )


@pytest.mark.parametrize(
    "program", ("initialize", "solve_from_coordinate", "iteration")
)
def test_complete_compilation_descriptors_and_call_counts(program):
    case = solve_cases()[0]
    record = compilation_record(case, program, 0)
    validate_program(record, case, program, 0)
    bad = deepcopy(record)
    bad["compiled_outputs"] = bad["compiled_outputs"][:1]
    with pytest.raises(ValueError, match="complete dynamic program"):
        validate_program(bad, case, program, 0)
    bad = deepcopy(record)
    bad["calls"]["execute"] = 1
    with pytest.raises(ValueError, match="count"):
        validate_program(bad, case, program, 0)
    bad = deepcopy(record)
    bad["arguments"][0]["dtype"] = "float64"
    with pytest.raises(ValueError, match="native"):
        validate_program(bad, case, program, 0)


def set_messages(arrays, vector):
    """Populate gauge-reduced schema arrays for the synthetic serialization test."""
    l, k, d = arrays["messages.left.linear"].shape
    nc, nh, np_ = l * (k - 1), l * k * d, l * k * d * (d + 1) // 2
    parts = np.split(vector, [2 * nc, 2 * nc + 2 * nh])
    row, col = np.triu_indices(d)
    for index, side in enumerate(("left", "right")):
        arrays[f"messages.{side}.constant"][:, 1:] = parts[0][
            index * nc : (index + 1) * nc
        ].reshape(l, k - 1)
        arrays[f"messages.{side}.linear"][...] = parts[1][
            index * nh : (index + 1) * nh
        ].reshape(l, k, d)
        precision = arrays[f"messages.{side}.precision"]
        values = parts[2][index * np_ : (index + 1) * np_].reshape(l, k, -1)
        precision[..., row, col] = values
        precision[..., col, row] = values
    arrays["state.coordinate"][: vector.size] = vector
    arrays["state.evaluation.residual"][: vector.size] = -vector
    arrays["state.evaluation.scales"][: vector.size] = np.maximum(1, abs(vector))
    arrays["state.evaluation.certificates.natural"][()] = np.max(
        abs(vector) / np.maximum(1, abs(vector))
    )


def complete_report(tmp_path):
    """Synthetic I/O acceptance exercises aggregation, never solver correctness."""
    identity = identities()
    identity["git_status"] = ""
    report = dict(
        schema="cffg.d12-moderate-report.v1",
        phase="solves",
        identity=identity,
        status="complete",
        jobs=[],
    )
    write_json(tmp_path / "spec.json", load_spec())
    for horizon in (32, 100):
        name = f"t{horizon}"
        folder = tmp_path / name
        folder.mkdir()
        cases = [c for c in solve_cases() if c["T"] == horizon]
        p, s = fixture_arrays(horizon)
        inputs = write_snapshot(
            folder,
            "inputs",
            dict(
                parameters=p,
                sites=s,
                metadata=fixed_metadata(horizon),
            ),
        )
        worker = dict(
            identity=identity,
            status="complete",
            inputs=inputs,
            cases=[],
            initialization_compilation=compilation_record(cases[0], "initialize", 6),
            solve_compilation=compilation_record(cases[0], "solve_from_coordinate", 6),
            runtime=dict(
                backend="cpu",
                config=dict(
                    jax_enable_x64=False,
                    jax_explicit_x64_dtypes="ERROR",
                    jax_enable_compilation_cache=False,
                    jax_persistent_cache_enable_xla_caches="",
                ),
                python="synthetic",
                executable="synthetic",
                platform="synthetic",
                machine="synthetic",
                jax="synthetic",
                jaxlib="synthetic",
                numpy="synthetic",
                devices=["synthetic"],
                clock=dict(monotonic=True, adjustable=False),
            ),
        )
        for case in cases:
            initial = synthetic(case, initial=True)
            if case["messages"] == "perturbed":
                size = initial["state.coordinate"].size - initial["xi"].size
                set_messages(
                    initial,
                    np.float32(0.01) * np.sin(np.arange(size, dtype=np.float32) + 1),
                )
            worker["cases"].append(
                dict(
                    case=case,
                    initial=write_snapshot(folder, case["id"] + "-initial", initial),
                    terminal=write_snapshot(
                        folder, case["id"] + "-terminal", synthetic(case)
                    ),
                    initialization_seconds=1.0,
                    solve_seconds=1.0,
                    export_seconds=1.0,
                )
            )
        (folder / "compiler.log").write_text("synthetic schema test; no compilation\n")
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
                id=name,
                worker_sha256=digest(folder / "worker.json"),
                process_sha256=digest(folder / "process.json"),
            )
        )
    write_json(tmp_path / "report.json", report)
    result = validate_report(tmp_path)
    assert result["complete"] and len(result["comparisons"]) == 30
    return report


def test_complete_report_roundtrip_and_missing_start(tmp_path):
    report = complete_report(tmp_path)
    folder = tmp_path / "t100"
    worker = json.loads((folder / "worker.json").read_text())
    worker["cases"].pop()
    write_json(folder / "worker.json", worker)
    report["jobs"][-1]["worker_sha256"] = digest(folder / "worker.json")
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="six-start inventory"):
        validate_report(tmp_path)


@pytest.mark.parametrize(
    "status,jobs",
    (
        ("started", []),
        ("failed", []),
        ("complete", []),
        ("complete", [dict(id="t32"), dict(id="t32")]),
        ("complete", [dict(id="t100"), dict(id="t32")]),
    ),
)
def test_incomplete_report_never_qualifies(tmp_path, status, jobs):
    identity = identities()
    identity["git_status"] = ""
    write_json(
        tmp_path / "report.json",
        dict(
            schema="cffg.d12-moderate-report.v1",
            phase="solves",
            identity=identity,
            status=status,
            jobs=jobs,
        ),
    )
    with pytest.raises(ValueError, match="incomplete|inventory"):
        validate_report(tmp_path)


def test_native_process_setup_and_abstract_storage():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/d12_moderate_checks.py")],
        cwd=ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr[-12000:]
    assert json.loads(result.stdout)["moderate_optimizer_calls"] == 0


@pytest.mark.slow
def test_explicit_host_jit_single_vmap_status():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/d12_moderate_checks.py"), "--status"],
        cwd=ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr[-12000:]
    assert json.loads(result.stdout)["status_variants"] == 4


@pytest.mark.slow
def test_opt_in_complete_moderate_evidence():
    """Execute the standalone reviewed plan explicitly, then validate both phases."""
    solves, storage = (
        os.getenv("CFFG_D12_MODERATE_REPORT"),
        os.getenv("CFFG_D12_STORAGE_REPORT"),
    )
    if not solves or not storage:
        pytest.skip(
            "requires explicit paths to complete A3WZ solve and storage evidence"
        )
    assert validate_report(solves)["phase"] == "solves"
    assert validate_report(storage)["phase"] == "storage"
