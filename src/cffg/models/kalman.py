"""Linear-Gaussian state-space model (LGSSM) template builder (cff-4e5z).

The continuous mirror of the HMM chain: prior → transitions → emissions,
with equality nodes branching each interior state to its emission —

    prior ── x0 ──[eq0]── x0c ──[trans1]── x1 ── ... ── x{T-1} ──[emis]── y
                    │
                   x0o ──[emis0]── y0 : DATA

The last state needs no equality (transition-in + emission = degree 2).
Observation edges are Gaussian DATA ports, compiled to *conditioning*
(cffg.gauss): bind vectors with `engine.bind_data(state, {"y0": np.array([...])})`,
or None for a missing observation. This is the Kalman-parity test tier's
model and the base template for switching extensions (cff-pksr).
"""

from __future__ import annotations

import numpy as np

from cffg.core.ir import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec


def lgssm(
    n_steps: int,
    f: np.ndarray,  # [n, n] transition matrix
    q: np.ndarray,  # [n, n] process noise covariance
    h: np.ndarray,  # [m, n] emission matrix
    r: np.ndarray,  # [m, m] observation noise covariance
    m0: np.ndarray,  # [n] prior mean
    p0: np.ndarray,  # [n, n] prior covariance
) -> tuple[Template, dict[str, np.ndarray]]:
    """Returns (template, arrays) ready for compile_template(t, arrays=...).
    State marginals live on edges "x{t}"; DATA ports are "y{t}"."""
    n, m = f.shape[0], h.shape[0]
    t = Template(name=f"lgssm_{n_steps}")
    arrays = {
        "F": np.asarray(f), "Q": np.asarray(q), "H": np.asarray(h),
        "R": np.asarray(r), "m0": np.asarray(m0), "P0": np.asarray(p0),
        "zn": np.zeros(n), "zm": np.zeros(m),
    }
    for name in arrays:
        t.blobs[name] = BlobRef(name, f"blobs/{name}.safetensors")

    def var(name: str, dim: int) -> str:
        t.variables[name] = VariableSpec(name, dim=dim)
        return name

    t.factors["prior"] = FactorSpec(
        "prior", "gaussian", edges={"out": var("x0", n)},
        param_blobs={"mean": "m0", "cov": "P0"},
    )
    for k in range(n_steps):
        last = k == n_steps - 1
        if not last:
            t.factors[f"eq{k}"] = FactorSpec(
                f"eq{k}", "equality",
                edges={"a": f"x{k}", "b": var(f"x{k}c", n), "c": var(f"x{k}o", n)},
            )
            t.factors[f"trans{k + 1}"] = FactorSpec(
                f"trans{k + 1}", "gaussian_linear",
                edges={"out": var(f"x{k + 1}", n), "in": f"x{k}c"},
                param_blobs={"matrix": "F", "offset": "zn", "cov": "Q"},
            )
        t.factors[f"emis{k}"] = FactorSpec(
            f"emis{k}", "gaussian_linear",
            edges={"out": var(f"y{k}", m), "in": f"x{k}" if last else f"x{k}o"},
            param_blobs={"matrix": "H", "offset": "zm", "cov": "R"},
        )
        t.ports[f"y{k}"] = PortSpec(f"y{k}", f"y{k}", PortRole.DATA)
    return t, arrays


def kalman_smoother(
    ys: list[np.ndarray | None],
    f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray,
    m0: np.ndarray, p0: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Reference forward filter + RTS smoother; None = missing observation.
    Returns (smoothed means, smoothed covs, log likelihood of the bound ys)."""
    means, covs = [], []
    mean, cov, loglik = np.asarray(m0, dtype=float), np.asarray(p0, dtype=float), 0.0
    for k, y in enumerate(ys):
        if k > 0:
            mean, cov = f @ mean, f @ cov @ f.T + q
        if y is not None:
            s = h @ cov @ h.T + r
            resid = y - h @ mean
            gain = cov @ h.T @ np.linalg.inv(s)
            mean = mean + gain @ resid
            cov = cov - gain @ h @ cov
            loglik -= 0.5 * (
                np.linalg.slogdet(2 * np.pi * s)[1] + resid @ np.linalg.inv(s) @ resid
            )
        means.append(mean.copy())
        covs.append(cov.copy())
    for k in range(len(ys) - 2, -1, -1):
        mp, pp = f @ means[k], f @ covs[k] @ f.T + q
        g = covs[k] @ f.T @ np.linalg.inv(pp)
        means[k] = means[k] + g @ (means[k + 1] - mp)
        covs[k] = covs[k] + g @ (covs[k + 1] - pp) @ g.T
    return means, covs, loglik
