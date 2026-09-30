"""Switching Kalman filter through the engine (cff-pksr acceptance tier).

Two oracles from tests/swkf_reference.py, per the D3 test design:
- `collapsed_fixed_point` implements the SAME D3 collapse rules in plain
  numpy — the engine must match it tightly (same fixed point; the extra
  equality nodes the FFG discipline requires are exact and change nothing).
- `enumerate_exact` is exact posterior inference (exponential enumeration) —
  the collapse gap against it is MEASURED (see test_swkf_reference.py's
  module docstring) and asserted with ~2x headroom, honestly, per
  LIMITATIONS S1d: assumed-density inference does not match enumeration.
"""

import numpy as np
import pytest

from cffg.compiler import compile_template
from cffg.core import (
    BlobRef,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine

from swkf_reference import collapsed_fixed_point, enumerate_exact
from test_swkf_reference import CASE_1D_YS, CASE_2D_YS, case_1d_model, case_2d_model


def _swkf_template(model, T: int) -> tuple[Template, dict]:
    """The switching-KF CFFG: regime chain (categorical + matrix_cat, with
    equality fan-outs), state chain (gaussian_mixture_linear per step, with
    equality fan-outs), shared gaussian_linear emissions with DATA ports.
    Parameters are tied across time via shared blob names."""
    t = Template(name=f"swkf_T{T}")
    d, dy, K = model.d, model.dy, model.K
    arrays = {
        "m0": model.m0, "P0": model.P0, "A": model.A, "boff": model.b,
        "Q": model.Q, "H": model.H, "h0": np.zeros(dy), "R": model.R,
        "pi": model.pi, "P": model.P,
    }
    for n in arrays:
        t.blobs[n] = BlobRef(n, f"blobs/{n}.npy")

    t.variables["x0"] = VariableSpec("x0", dim=d)
    t.factors["x0prior"] = FactorSpec(
        "x0prior", "gaussian", edges={"out": "x0"}, param_blobs={"mean": "m0", "cov": "P0"}
    )
    for i in range(1, T + 1):
        last = i == T
        # regime variable s_i and its chain factor
        t.variables[f"s{i}"] = VariableSpec(f"s{i}", K)
        if i == 1:
            t.factors["s1prior"] = FactorSpec(
                "s1prior", "categorical", edges={"out": "s1"}, param_blobs={"statistic": "pi"}
            )
        else:
            t.factors[f"strans{i}"] = FactorSpec(
                f"strans{i}", "matrix_cat",
                edges={"out": f"s{i}", "in": f"s{i - 1}n"}, param_blobs={"matrix": "P"},
            )
        if last:
            sel = f"s{i}"
        else:  # fan out: mixture selector branch + next chain link
            sel = f"s{i}b"
            t.variables[sel] = VariableSpec(sel, K)
            t.variables[f"s{i}n"] = VariableSpec(f"s{i}n", K)
            t.factors[f"seq{i}"] = FactorSpec(
                f"seq{i}", "equality", edges={"a": f"s{i}", "b": sel, "c": f"s{i}n"}
            )
        # state transition through the mixture node
        t.variables[f"x{i}"] = VariableSpec(f"x{i}", dim=d)
        x_in = "x0" if i == 1 else f"x{i - 1}n"
        t.factors[f"mix{i}"] = FactorSpec(
            f"mix{i}", "gaussian_mixture_linear",
            edges={"out": f"x{i}", "in": x_in, "selector": sel},
            param_blobs={"matrix": "A", "offset": "boff", "cov": "Q"},
        )
        if last:
            x_obs = f"x{i}"
        else:  # fan out: emission branch + next transition input
            x_obs = f"x{i}o"
            t.variables[x_obs] = VariableSpec(x_obs, dim=d)
            t.variables[f"x{i}n"] = VariableSpec(f"x{i}n", dim=d)
            t.factors[f"xeq{i}"] = FactorSpec(
                f"xeq{i}", "equality", edges={"a": f"x{i}", "b": x_obs, "c": f"x{i}n"}
            )
        t.variables[f"y{i}"] = VariableSpec(f"y{i}", dim=dy)
        t.factors[f"em{i}"] = FactorSpec(
            f"em{i}", "gaussian_linear",
            edges={"out": f"y{i}", "in": x_obs},
            param_blobs={"matrix": "H", "offset": "h0", "cov": "R"},
        )
        t.ports[f"obs{i}"] = PortSpec(f"obs{i}", f"y{i}", PortRole.DATA)
    return t, arrays


def _run_engine(model, ys, n_iters=150):
    t, arrays = _swkf_template(model, len(ys))
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.bind_data(
        eng.init_state(), {f"obs{i + 1}": (None if y is None else y) for i, y in enumerate(ys)}
    )
    st, deltas = eng.run(st, n_iters, tol=1e-12).require_converged_with_trace()
    assert deltas[-1] < 1e-9, "engine did not reach the collapse fixed point"
    marg = eng.marginals(st)
    gmarg = eng.gaussian_marginals(st)
    T = len(ys)
    regimes = np.stack([marg[f"s{i}"] for i in range(1, T + 1)])  # [T, K]
    means = np.stack([gmarg["x0"][0]] + [gmarg[f"x{i}"][0] for i in range(1, T + 1)])
    covs = np.stack([gmarg["x0"][1]] + [gmarg[f"x{i}"][1] for i in range(1, T + 1)])
    return eng, st, regimes, means, covs


CASES = [
    ("1d", case_1d_model, CASE_1D_YS, dict(regime=0.06, mean=0.4, cov=0.2)),
    ("2d", case_2d_model, CASE_2D_YS, dict(regime=0.14, mean=0.35, cov=1.3)),
]


@pytest.mark.parametrize("label,mk,ys,gaps",
                         [pytest.param(*c, marks=pytest.mark.slow if c[0] == "2d" else ()) for c in CASES],
                         ids=[c[0] for c in CASES])
def test_engine_matches_collapsed_reference(label, mk, ys, gaps):
    """Tight parity: the engine and the independent numpy implementation of
    the same D3 rules must land on the same fixed point."""
    model = mk()
    _, _, regimes, means, covs = _run_engine(model, ys)
    ref = collapsed_fixed_point(model, ys, n_sweeps=300, tol=1e-13)
    assert ref.converged
    np.testing.assert_allclose(regimes, ref.regime_marginals, atol=1e-7)
    np.testing.assert_allclose(means, ref.state_mean, atol=1e-6)
    np.testing.assert_allclose(covs, ref.state_cov, atol=1e-6)


@pytest.mark.parametrize("label,mk,ys,gaps", CASES, ids=[c[0] for c in CASES])
def test_engine_vs_exact_enumeration_within_measured_gap(label, mk, ys, gaps):
    """Honest bound vs exact inference: the assumed-density gap, measured by
    the reference tier, asserted with ~2x headroom (LIMITATIONS S1d)."""
    model = mk()
    _, _, regimes, means, covs = _run_engine(model, ys)
    ex = enumerate_exact(model, ys)
    assert np.max(np.abs(regimes - ex.regime_marginals)) < gaps["regime"]
    assert np.max(np.abs(means - ex.state_mean)) < gaps["mean"]
    assert np.max(np.abs(covs - ex.state_cov)) < gaps["cov"]


def test_missing_observation(viz_gallery):
    model = case_1d_model()
    ys = [CASE_1D_YS[0], None, CASE_1D_YS[2]]
    t, _ = _swkf_template(model, 3)
    viz_gallery(t)
    _, _, regimes, means, _ = _run_engine(model, ys)
    ref = collapsed_fixed_point(model, ys, n_sweeps=300, tol=1e-13)
    np.testing.assert_allclose(regimes, ref.regime_marginals, atol=1e-7)
    np.testing.assert_allclose(means, ref.state_mean, atol=1e-6)


def test_bfe_finite_and_near_evidence():
    """The reported BFE is a *bounded* ledger (LIMITATIONS S1d), not the
    tree identity: assert it is finite, loud-free, and lands within a
    diagnostic distance of −log evidence (measured ≈0.02 on CASE_1D; the
    generous bound here is a canary against silent ledger regressions, not
    a claimed identity)."""
    model = case_1d_model()
    eng, st, *_ = _run_engine(model, CASE_1D_YS)
    bfe = eng.bfe(st)
    ex = enumerate_exact(model, CASE_1D_YS)
    assert np.isfinite(bfe)
    assert abs(bfe - (-ex.log_evidence)) < 0.5
