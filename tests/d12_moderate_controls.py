"""Analytic model controls for report tests, never frozen moderate evidence.

With zero recurrence weights/offsets and zero dynamics matrix/bias, each label
has probability (1/2,1/4,1/8,1/8), independently of its source. Each destination
gets N(0,I) from its transition and an identity-precision unary site. The exact
stationary messages are zero to the source and (log q,0,I) to the destination.
This is a closed-form reduction, with no optimizer. Fixture replacement and
memoization below are test-only; production fixture/provenance checks stay strict.
"""

from functools import lru_cache
import sys

import numpy as np
import pytest

from d12_compile_fixture import fixture_arrays as frozen_fixture
from d12_moderate_payload import CERTIFICATES, output_contract
from d12_moderate_reconstruct import scaled_blocks
import d12_moderate_reference as anchor

_evaluate = anchor.evaluate


@lru_cache(maxsize=32)
def _analytic_evaluation(vector, horizon):
    assert anchor.fixture_arrays is analytic_fixture
    return _evaluate(np.frombuffer(vector, dtype=np.float32), horizon)


def analytic_fixture(horizon):
    p, s = frozen_fixture(horizon)
    for name in ("weights", "offsets", "xi", "matrix", "bias"):
        p[name].fill(0)
    p["noise"][...] = np.eye(2, dtype=np.float32)
    s["constant"].fill(0)
    s["linear"].fill(0)
    s["precision"][...] = np.eye(2, dtype=np.float32)
    return p, s


@pytest.fixture
def analytic_model(monkeypatch):
    """Explicit dependency injection for analytic schema tests only."""
    for name in (
        "d12_compile_fixture",
        "d12_moderate_report",
        "d12_moderate_reconstruct",
        "d12_moderate_reference",
        "test_d12_moderate",
    ):
        module = sys.modules.get(name)
        if module is not None:
            monkeypatch.setattr(module, "fixture_arrays", analytic_fixture)
    monkeypatch.setattr(
        anchor,
        "evaluate",
        lambda vector, horizon: _analytic_evaluation(vector.tobytes(), horizon),
    )
    return monkeypatch


def set_messages(arrays, vector):
    """Populate message schema only; callers recompute arithmetic when intended."""
    t = arrays["messages.left.linear"].shape[0] + 1
    x = arrays["state.coordinate"]
    x[: vector.size] = vector
    messages, _ = anchor.decode(x, t)
    for side, name in enumerate(("left", "right")):
        for field in ("constant", "linear", "precision"):
            arrays[f"messages.{name}.{field}"][...] = np.stack(
                [getattr(pair[side], field) for pair in messages]
            )


def refresh_arithmetic(arrays, case):
    """Recompute saved internal relationships after a deliberate test mutation."""
    r, scale, certificates, learning = scaled_blocks(arrays, case)
    arrays["state.evaluation.residual"][...] = r
    arrays["state.evaluation.scales"][...] = scale
    # Summaries follow the once-stored residual, as they do in native output.
    widths = (
        arrays["state.coordinate"].size - arrays["xi"].size,
        2 * (case["T"] - 1) * case["K"],
        2 * (case["T"] - 1) * case["K"] * case["d"],
        (case["T"] - 1) * case["K"] * case["d"] * (case["d"] + 1),
    )
    blocks = np.split(
        arrays["state.evaluation.residual"] / arrays["state.evaluation.scales"],
        np.cumsum(widths),
    )
    certificates[:-1] = [np.max(abs(b)) for b in blocks]
    for name, value in zip(CERTIFICATES, certificates, strict=True):
        arrays["state.evaluation.certificates." + name][()] = value
    for name in ("target", "residual", "exposure", "linear", "precision", "center"):
        arrays["state.evaluation.auxiliary." + name][...] = learning[name]
    for name in ("physical_linear", "physical_precision"):
        arrays["learning." + name][...] = learning[name]


def populate(arrays, case):
    """Save one independent fixed point, with native represented/readout semantics."""
    regions, separators, targets, primal, dual = anchor.evaluate(
        arrays["state.coordinate"], case["T"]
    )
    for name, beliefs in (("regions", regions), ("separators", separators)):
        prefix = "state.evaluation.local." + name + "."
        for field in ("probability", "log_normalizer"):
            arrays[prefix + field][...] = np.stack([getattr(b, field) for b in beliefs])
        arrays[prefix + "log_probability"][...] = np.log(
            np.stack([b.probability for b in beliefs])
        )
        mean = np.stack([b.mean for b in beliefs])
        covariance = np.stack([b.covariance for b in beliefs])
        high = covariance.astype(np.float32)
        low = (covariance - high.astype(np.float64)).astype(np.float32)
        # Native CPU does not retain subnormal correction words.
        low[abs(low) < np.finfo(np.float32).tiny] = 0
        arrays[prefix + "covariance"][...] = high
        arrays[prefix + "covariance_correction"][...] = low
        arrays[prefix + "displacement"][...] = mean
        arrays[name + ".mean"][...] = mean
        arrays[name + ".full_covariance"][...] = high + low
        m = mean.astype(np.float32)
        arrays[name + ".second"][...] = high + low + m[..., :, None] * m[..., None, :]
    for side in (0, 1):
        for field in ("constant", "linear", "precision"):
            arrays[f"state.evaluation.local.targets.{side}.{field}"][...] = np.stack(
                [getattr(pair[side], field) for pair in targets]
            )
    arrays["state.evaluation.local.free_energy"][()] = primal
    arrays["state.evaluation.local.dual_energy"][()] = dual
    refresh_arithmetic(arrays, case)


def synthetic(case, *, initial=False):
    """Analytic stationary terminal or proper nonstationary initial point."""
    arrays = {
        key: np.zeros(shape, dtype)
        for key, (shape, dtype) in output_contract(
            case["T"], case["K"], case["d"]
        ).items()
    }
    l, k, d = case["T"] - 1, case["K"], case["d"]
    a = np.zeros((2, l, k - 1), np.float32)
    a[1] = np.log(np.float32([0.5, 0.25, 0.25]))
    h = np.zeros((2, l, k, d), np.float32)
    j = np.zeros((2, l, k, 3), np.float32)
    j[1, ..., (0, 2)] = 1
    vector = np.concatenate((a.ravel(), h.ravel(), j.ravel()))
    if initial:
        vector *= 0 if case["messages"] == "raw-zero" else 0.5
        if case["messages"] == "perturbed":
            vector += np.float32(0.01) * np.sin(
                np.arange(vector.size, dtype=np.float32) + 1
            )
        arrays["xi"].fill(case["xi"])
    arrays["state.coordinate"][-arrays["xi"].size :] = arrays["xi"].ravel()
    set_messages(arrays, vector)
    populate(arrays, case)
    arrays["state.iterations"][()] = 0 if initial else 1
    arrays["state.evaluations"][()] = 1 if initial else 3
    arrays["state.status"][()] = 3 if initial else 0
    arrays["state.reason"][()] = 1 if initial else 0
    if initial:
        arrays["state.relative_energy_change"][()] = np.inf
    return arrays
