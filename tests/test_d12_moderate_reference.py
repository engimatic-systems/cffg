"""Independent model anchoring; analytic reports are not moderate qualification."""

from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from test_d12_moderate import ROOT, complete_report
from d12_moderate_controls import (
    analytic_model,  # noqa: F401
    refresh_arithmetic,
    set_messages,
    synthetic,
)  # noqa: F401
from d12_compile_scaling import child_environment, digest, write_json
from d12_moderate_payload import read_snapshot, validate_outputs, write_snapshot
from d12_moderate_plan import identities, solve_cases
from d12_moderate_reference import decode, evaluate
from d12_moderate_report import validate_report


def mutate(arrays, case, kind):
    local = "state.evaluation.local."
    if kind in ("paired_constant", "paired_linear", "paired_precision"):
        field = kind.removeprefix("paired_")
        key = "messages.left." + field
        index = (0, 1) + (0,) * (arrays[key].ndim - 2)
        arrays[key][index] += 10
        arrays[local + "targets.0." + field][index] += 10
        from d12_moderate_reconstruct import fields, packed_messages

        vector = packed_messages(
            [fields(arrays, "messages." + side) for side in ("left", "right")]
        )
        set_messages(arrays, vector)
    elif kind == "equal_energy_shift":
        # Exceeds the frozen absolute allowance but would pass a relative allowance.
        shift = 2 * 512 * max(4, 5 * case["T"]) * np.finfo(np.float32).eps
        for name in ("free_energy", "dual_energy"):
            arrays[local + name][()] += shift
    elif kind == "normalizer_accumulation":
        # Each local difference is allowed; their sum violates the energy identity.
        arrays[local + "regions.log_normalizer"] += np.float32(0.0005)
    elif kind.endswith("normalizer"):
        name = kind.removesuffix("_normalizer")
        arrays[local + name + ".log_normalizer"] += 100
    elif kind == "paired_beliefs":
        for name in ("regions", "separators"):
            prefix = local + name + "."
            arrays[prefix + "covariance"] *= 2
            arrays[prefix + "covariance_correction"] *= 2
            for field in ("full_covariance", "second"):
                arrays[name + "." + field] *= 2
    else:
        raise AssertionError(kind)
    # Rebuild every saved relationship; rejection must come from the model anchor.
    refresh_arithmetic(arrays, case)


CORRUPTIONS = (
    "paired_constant",
    "paired_linear",
    "paired_precision",
    "equal_energy_shift",
    "regions_normalizer",
    "separators_normalizer",
    "paired_beliefs",
    "normalizer_accumulation",
)


@pytest.mark.usefixtures("analytic_model")
@pytest.mark.parametrize("kind", CORRUPTIONS)
@pytest.mark.parametrize("initial", [False, True])
def test_independent_anchor_rejects_coordinated_changes(kind, initial):
    case = solve_cases()[1 if initial else 0]
    arrays = synthetic(case, initial=initial)
    validate_outputs(arrays, case, initial=initial)
    mutate(arrays, case, kind)
    with pytest.raises(
        ValueError, match="independent reference|normalizer dual identity"
    ):
        validate_outputs(arrays, case, initial=initial)


@pytest.mark.usefixtures("analytic_model")
@pytest.mark.parametrize("kind", CORRUPTIONS)
@pytest.mark.parametrize("point", ["initial", "terminal"])
def test_aggregate_anchor_with_refreshed_hashes(tmp_path, kind, point):
    report = complete_report(tmp_path)
    folder = tmp_path / "t32"
    worker = json.loads((folder / "worker.json").read_text())
    row = worker["cases"][1 if point == "initial" else 0]
    arrays = read_snapshot(folder, row[point])
    mutate(arrays, row["case"], kind)
    row[point] = write_snapshot(folder, "changed-point", arrays)
    write_json(folder / "worker.json", worker)
    report["jobs"][0]["worker_sha256"] = digest(folder / "worker.json")
    write_json(tmp_path / "report.json", report)
    with pytest.raises(
        ValueError, match="independent reference|normalizer dual identity"
    ):
        validate_report(tmp_path)


@pytest.mark.usefixtures("analytic_model")
def test_analytic_reduction_and_target_gauge():
    for case in (solve_cases()[0], solve_cases()[6]):
        arrays = synthetic(case)
        regions, separators, targets, primal, dual = evaluate(
            arrays["state.coordinate"], case["T"]
        )
        q = np.array([0.5, 0.25, 0.125, 0.125])
        for t, belief in enumerate(regions):
            source = np.full(4, 0.25) if t == 0 else q
            np.testing.assert_allclose(
                belief.probability, source[:, None] * q[None, :], rtol=1e-6
            )
            np.testing.assert_allclose(belief.mean, 0, atol=1e-12)
            expected = np.diag([1 if t == 0 else 0.5] * 2 + [0.5] * 2)
            np.testing.assert_allclose(
                belief.covariance,
                np.broadcast_to(expected, belief.covariance.shape),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                targets[t][1].constant, np.log(q / q[0]), atol=1e-6
            )
        expected_energy = (case["T"] - 1) * np.log(2) - np.log(8 * np.pi)
        np.testing.assert_allclose([primal, dual], expected_energy, atol=1e-6)
        np.testing.assert_allclose(
            separators[0].log_normalizer, np.log(8 * np.pi), atol=1e-6
        )
        for side in (0, 1):
            arrays[f"state.evaluation.local.targets.{side}.constant"] += np.float32(0.5)
        validate_outputs(arrays, case)  # Only target constants have this gauge.


def test_analytic_report_cannot_qualify_frozen_model(tmp_path, analytic_model):  # noqa: F811
    complete_report(tmp_path)
    analytic_model.undo()
    with pytest.raises(ValueError, match="frozen fixture mismatch"):
        validate_report(tmp_path)


def test_reference_sources_are_part_of_prospective_identity():
    sources = identities()["sources"]
    for name in (
        "tools/d12_moderate_reference.py",
        "src/cffg/experiments/hybrid_separator.py",
        "src/cffg/experiments/regime_pair.py",
    ):
        assert sources[name] == digest(ROOT / name)


def test_reference_evaluation_does_not_import_candidate_or_jax():
    script = """
import sys
sys.path.insert(0, 'tools')
import numpy as np
from d12_moderate_reference import evaluate
from d12_moderate_payload import output_contract
evaluate(np.zeros(output_contract(32)['state.coordinate'][0], np.float32), 32)
assert not any(n == 'jax' or n.startswith('jax.') or n == 'cffg' or n.startswith('cffg.') for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.usefixtures("analytic_model")
def test_independent_coordinate_order_and_reference_inputs():
    case = solve_cases()[0]
    arrays = synthetic(case)
    vector = np.arange(arrays["state.coordinate"].size, dtype=np.float32) / 100
    messages, xi = decode(vector, 32)
    nc, nh, nj = 31 * 3, 31 * 4 * 2, 31 * 4 * 3
    assert messages[0][0].constant.tolist() == [0, *vector[:3]]
    np.testing.assert_array_equal(messages[0][1].constant[1:], vector[nc : nc + 3])
    np.testing.assert_array_equal(
        messages[0][1].linear.ravel(), vector[2 * nc + nh : 2 * nc + nh + 8]
    )
    np.testing.assert_array_equal(
        messages[0][1].precision[0],
        vector[2 * nc + 2 * nh + nj : 2 * nc + 2 * nh + nj + 3][[0, 1, 1, 2]].reshape(
            2, 2
        ),
    )
    np.testing.assert_array_equal(xi.ravel(), vector[2 * (nc + nh + nj) :])
    # The reference cannot consume another saved field: its API has only x and T.
    original = evaluate(arrays["state.coordinate"], 32)
    changed = deepcopy(arrays)
    for name in changed:
        if name.startswith("state.evaluation"):
            changed[name].fill(42)
    repeated = evaluate(changed["state.coordinate"], 32)
    assert original[3:] == repeated[3:]


def test_frozen_reference_scope_and_properness(monkeypatch):
    import d12_moderate_reference as anchor
    from d12_compile_fixture import fixture_arrays
    from d12_moderate_payload import output_contract

    vector = np.zeros(output_contract(32)["state.coordinate"][0], np.float32)
    with pytest.raises(ValueError, match="horizon"):
        decode(vector, 3)
    with pytest.raises(ValueError, match="shape/dtype"):
        decode(vector.astype(np.float64), 32)
    negative = vector.copy()
    negative[-1] = -1
    with pytest.raises(ValueError, match="negative xi"):
        decode(negative, 32)
    p, s = fixture_arrays(32)
    s["center"][0, 0, 0] = 1
    monkeypatch.setattr(anchor, "fixture_arrays", lambda horizon: (p, s))
    with pytest.raises(ValueError, match="zero origins"):
        anchor.evaluate(vector, 32)
    s["center"].fill(0)
    s["precision"] *= -1
    with pytest.raises(ValueError, match="improper"):
        anchor.evaluate(vector, 32)


def test_native_frozen_fixed_points_and_reference_overhead():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/d12_moderate_anchor_checks.py")],
        cwd=ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout.splitlines()[-1])
    assert record["native"] and record["optimizer_calls"] == 0
    assert record["complete_program_calls"] == dict(lower=0, compile=0, execute=0)
    assert [row["T"] for row in record["fixed_points"]] == [32, 100]
    assert all(row["reference_seconds"] >= 0 for row in record["fixed_points"])
