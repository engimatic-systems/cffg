"""Independent fixed-coordinate D12 arithmetic for the frozen moderate models.

The NumPy experiment implements local EC regions, projections and both energies;
its only mathematical dependency is experiments.regime_pair. This adapter never
imports candidate mathematics or consumes saved beliefs, targets or energies as
reference inputs. Stored binary32 model values and coordinates are promoted to
binary64, with c = sqrt(xi). Work is O(T K² d³), without optimization or temporal
path enumeration. The older experiment uses world coordinates: this entry is
restricted to the registered zero-origin, full-support T32/T100, K4, d2 cases.
The separate tiny joint reference retains its T2/T3 restriction.
"""

import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

from d12_compile_fixture import fixture_arrays
from d12_moderate_payload import close, require, validate_outputs


def _load_reference():
    """Load the NumPy experiment without importing the cffg/JAX public facade."""
    name = "_d12_numpy_reference"
    directory = Path(__file__).resolve().parents[1] / "src/cffg/experiments"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, directory / "__init__.py", submodule_search_locations=[str(directory)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return importlib.import_module(name + ".hybrid_separator")


reference = _load_reference()


def validate_moderate_outputs(arrays, case, *, initial=False, parameters=None):
    """Mandatory model-specific entry for every moderate evidence payload.

    The shared validator also serves tiny status mechanisms; it makes no model
    qualification claim. This entry has no option to disable the frozen anchor.
    """
    require(
        case["T"] in (32, 100) and (case["K"], case["d"]) == (4, 2),
        "independent reference shape outside contract",
    )
    counts = validate_outputs(arrays, case, initial=initial, parameters=parameters)
    if int(arrays["state.status"]) != 2:
        validate_reference(arrays, case)
    return counts


def decode(vector, horizon):
    """Decode the frozen full-support schema, independently of Coordinate.unpack.

    Constants for label zero are gauge anchors. Banks are ordered left then
    right within constants, linear coefficients and upper-triangle precision.
    The final block contains all active xi values; none are inferred from output.
    """
    require(horizon in (32, 100), "independent reference horizon outside contract")
    l, k, d = horizon - 1, 4, 2
    nc, nh, nj, nx = 2 * l * (k - 1), 2 * l * k * d, 2 * l * k * 3, l * k * (k - 1)
    vector = np.asarray(vector)
    require(
        vector.dtype == np.dtype("float32")
        and vector.shape == (nc + nh + nj + nx,)
        and np.all(np.isfinite(vector)),
        "independent reference coordinate shape/dtype/finiteness",
    )
    a, h, j, xi = np.split(vector.astype(np.float64), np.cumsum([nc, nh, nj]))
    constant = np.zeros((2, l, k), np.float64)
    constant[..., 1:] = a.reshape(2, l, k - 1)
    linear = h.reshape(2, l, k, d)
    precision = np.zeros((2, l, k, d, d), np.float64)
    row, col = np.triu_indices(d)
    precision[..., row, col] = j.reshape(2, l, k, 3)
    precision[..., col, row] = j.reshape(2, l, k, 3)
    xi = xi.reshape(l, k, k - 1)
    require(np.all(xi >= 0), "independent reference negative xi")
    messages = [
        tuple(
            reference.Natural(constant[s, t], linear[s, t], precision[s, t])
            for s in (0, 1)
        )
        for t in range(l)
    ]
    return messages, xi


def evaluate(vector, horizon):
    """Evaluate only trusted frozen inputs and the independently decoded point."""
    messages, xi = decode(vector, horizon)
    parameters, sites = fixture_arrays(horizon)
    require(
        np.all(sites["center"] == 0)
        and np.all(sites["status"] == 0)
        and np.all(np.isfinite(sites["constant"])),
        "independent reference requires zero origins and full support",
    )
    require(
        all(value.dtype == np.dtype("float32") for value in parameters.values())
        and all(
            sites[name].dtype == np.dtype("float32")
            for name in ("constant", "linear", "precision")
        ),
        "independent reference requires stored native float32 model values",
    )
    parameters = {name: value.astype(np.float64) for name, value in parameters.items()}
    unary = [
        reference.Natural(
            *(
                sites[name][t].astype(np.float64)
                for name in ("constant", "linear", "precision")
            )
        )
        for t in range(horizon)
    ]
    factors = [
        reference.transition(
            parameters["weights"][t],
            parameters["offsets"][t],
            np.sqrt(xi[t]),
            parameters["matrix"][t],
            parameters["bias"][t],
            parameters["noise"][t],
        )
        for t in range(horizon - 1)
    ]
    return reference.evaluate(factors, unary, messages)


def validate_reference(arrays, case):
    """Anchor raw outputs with the existing local and absolute-energy allowances.

    Covariances compare the represented high+low value, never an independently
    positive-definite low word. Readout rounding is checked separately by
    validate_belief. Target constants are compared modulo their common gauge.
    Energies use the frozen absolute allowance, with no relative enlargement.
    """
    require(
        (case["K"], case["d"]) == (4, 2), "independent reference shape outside contract"
    )
    regions, separators, targets, primal, dual = evaluate(
        arrays["state.coordinate"], case["T"]
    )
    for name, beliefs in (("regions", regions), ("separators", separators)):
        prefix = f"state.evaluation.local.{name}."
        require(
            np.all(arrays[prefix + "center"] == 0),
            "independent reference belief origin changed",
        )
        probability = np.stack([b.probability for b in beliefs])
        require(
            np.array_equal(
                np.isfinite(arrays[prefix + "log_probability"]), probability > 0
            ),
            "independent reference support mismatch",
        )
        for field, expected in (
            ("probability", probability),
            ("log_probability", np.log(probability)),
            ("displacement", np.stack([b.mean for b in beliefs])),
            ("log_normalizer", np.asarray([b.log_normalizer for b in beliefs])),
        ):
            close(
                arrays[prefix + field],
                expected,
                f"independent reference {name} {field}",
            )
        covariance = arrays[prefix + "covariance"].astype(np.float64) + arrays[
            prefix + "covariance_correction"
        ].astype(np.float64)
        close(
            covariance,
            np.stack([b.covariance for b in beliefs]),
            f"independent reference {name} represented covariance",
        )
    for side in (0, 1):
        prefix = f"state.evaluation.local.targets.{side}."
        for field in ("constant", "linear", "precision"):
            expected = np.stack([getattr(pair[side], field) for pair in targets])
            actual = arrays[prefix + field].astype(np.float64)
            if field == "constant":
                actual = actual - actual[:, :1]
                expected = expected - expected[:, :1]
            close(actual, expected, f"independent reference target {side} {field}")
    atol = 512 * max(4, 5 * case["T"]) * np.finfo(np.float32).eps
    for name, expected in (("free_energy", primal), ("dual_energy", dual)):
        actual = arrays[f"state.evaluation.local.{name}"]
        require(
            np.isfinite(expected) and abs(float(actual) - expected) <= atol,
            f"independent reference absolute {name} mismatch",
        )
    saved_dual = -np.sum(
        arrays["state.evaluation.local.regions.log_normalizer"], dtype=np.float64
    ) + np.sum(
        arrays["state.evaluation.local.separators.log_normalizer"][1:-1],
        dtype=np.float64,
    )
    require(
        abs(float(arrays["state.evaluation.local.dual_energy"]) - saved_dual) <= atol,
        "saved normalizer dual identity mismatch",
    )
