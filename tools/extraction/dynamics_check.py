"""Synthetic development checks for the extraction transition and nonlinear filter.

Run with the harness interpreter. The native tangent oracle differentiates
the installed PCGym ODE's CVODES flow. The particle filter integrates a separate
NumPy stage-by-stage implementation and retains the non-Gaussian particle law.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import study


def native_step(m11, u2):
    from dynamics import _cvodes
    return _cvodes()[0](m11, u2).full().ravel()


def _particle_rhs(x, u, kla):
    """Ten-by-batch physical states; explicit stage indexing mirrors the balances."""
    import numpy as np
    rhs = np.empty_like(x)
    for i in range(0, 10, 2):
        transfer = kla * (x[i] - x[i + 1]**2)
        liquid_in = .6 if i == 0 else x[i - 2]
        gas_in = .05 if i == 8 else x[i + 3]
        rhs[i] = u[0] * (liquid_in - x[i]) / 5. - transfer
        rhs[i + 1] = u[1] * (gas_in - x[i + 1]) / 5. + transfer
    return rhs


def batch_step(particles, u2, *, substeps=8):
    """Independent RK4 propagation on the study domain; leaves each log(Kla) intact."""
    import numpy as np
    from plant import DT

    z = np.array(particles, dtype=np.float64, copy=True)
    if z.ndim != 2 or z.shape[1] != 11 or substeps < 1:
        raise ValueError("Expected rows of 11 particle coordinates and a positive RK4 budget")
    h = DT / substeps
    for start in range(0, len(z), 4096):
        block = z[start:start+4096]
        x = np.ascontiguousarray(block[:, :10].T)
        kla = np.exp(block[:, 10])
        for _ in range(substeps):
            k1 = _particle_rhs(x, u2, kla)
            k2 = _particle_rhs(x + h/2*k1, u2, kla)
            k3 = _particle_rhs(x + h/2*k2, u2, kla)
            k4 = _particle_rhs(x + h*k3, u2, kla)
            x += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        block[:, :10] = x.T
    if not np.isfinite(z).all():
        raise FloatingPointError("Nonfinite RK4 particle prediction")
    return z


def _psd_root(cov):
    import numpy as np
    if not np.isfinite(cov).all():
        raise ValueError("Particle covariance must be finite")
    values, vectors = np.linalg.eigh(cov)
    tol = len(cov) * np.finfo(float).eps * np.max(abs(cov))
    if np.max(abs(cov-cov.T)) > tol or values[0] < -tol:
        raise ValueError("Particle covariance must be positive semidefinite")
    return vectors * np.sqrt(np.maximum(values, 0.))


def _prefix_posterior(m0, prior_root, flows, readings, process_root, R, count, rng, substeps):
    """Importance-sample all initial/process innovations using this observation prefix."""
    import numpy as np
    from scipy.special import logsumexp
    from dynamics import _cvodes
    from plant import OUTLETS

    # Build the joint Gaussian observation proposal around the noise-free path.
    # Exact nonlinear likelihood weights below remove this approximation.
    width = 11 * (1 + len(flows))
    sensitivity = np.zeros((11, width))
    sensitivity[:, :11] = prior_root
    center = m0
    designs, innovations = [], []
    for t, (u, y) in enumerate(zip(flows, readings, strict=True)):
        A, _, nxt = _cvodes()[1](center, u)
        center = nxt.full().ravel()
        sensitivity = A.full() @ sensitivity
        sensitivity[:, 11*(t+1):11*(t+2)] = process_root
        if y is not None:
            designs.append(sensitivity[OUTLETS])
            innovations.append(np.asarray(y) - center[OUTLETS])

    noise_root = np.linalg.cholesky(R)
    whiten = np.linalg.solve(noise_root, np.eye(2))
    log_z = 0.
    if designs:
        design = np.vstack(designs)
        innovation = np.concatenate(innovations)
        marginal = design @ design.T + np.kron(np.eye(len(designs)), R)
        chol = np.linalg.cholesky(marginal)
        gain = np.linalg.solve(marginal, design).T
        residual = np.linalg.solve(chol, innovation)
        log_z = -.5*(len(innovation)*np.log(2*np.pi) + residual@residual) - np.log(chol.diagonal()).sum()

    endpoints = np.empty((count, 11))
    log_weights = np.zeros(count)
    for first in range(0, count, 4096):
        last = min(first + 4096, count)
        n = last - first
        draws = rng.standard_normal((n, width))
        if designs:
            # Conditional simulation of standard normal innovations given all
            # observations in *this prefix*. No future reading enters a filter.
            errors = rng.standard_normal((n, len(designs), 2)) @ noise_root.T
            draws += (innovation - draws @ design.T - errors.reshape(n, -1)) @ gain.T
            linear = (innovation - draws @ design.T).reshape(n, -1, 2) @ whiten.T
            log_weights[first:last] = .5*np.sum(linear**2, axis=(1,2))
        x = m0 + draws[:, :11] @ prior_root.T
        for t, (u, y) in enumerate(zip(flows, readings, strict=True)):
            x = batch_step(x, u, substeps=substeps)
            x += draws[:,11*(t+1):11*(t+2)] @ process_root.T
            if y is not None:
                actual = (np.asarray(y) - x[:,OUTLETS]) @ whiten.T
                log_weights[first:last] -= .5*np.sum(actual**2, axis=1)
        endpoints[first:last] = x
    total = logsumexp(log_weights)
    weights = np.exp(log_weights - total)
    mean = weights @ endpoints
    centered = endpoints - mean
    return (mean, (centered.T * weights) @ centered, log_z + total - np.log(count),
            1. / (weights @ weights))


def particle_reference(m0, P0, flows, observations, Q, R, *, count, seed, substeps=8):
    """Estimate filtering moments and evidence under the full nonlinear transition.

    Start from the supplied Gaussian. Each flow is followed by an [X5,Y1]
    reading or None. Each prefix has its own Gaussian trajectory proposal and
    nonlinear importance correction. The returned evidence is per transition.
    """
    import numpy as np
    if count < 2 or len(flows) == 0 or len(observations) != len(flows):
        raise ValueError("Expected a nonempty matched window and at least two particles")
    initial, process = _psd_root(np.asarray(P0)), _psd_root(np.asarray(Q))
    seeds = np.random.SeedSequence(seed).spawn(len(flows))
    out = [_prefix_posterior(np.asarray(m0), initial, flows[:i], observations[:i], process,
                             np.asarray(R), count, np.random.default_rng(s), substeps)
           for i, s in enumerate(seeds, 1)]
    means, covs, log_joint, ess = map(np.array, zip(*out))
    # A missing observation integrates to one; retain the last observed log
    # evidence instead of differencing two independent Monte Carlo estimates.
    for t, y in enumerate(observations):
        if y is None:
            log_joint[t] = log_joint[t-1] if t else 0.
    return dict(mean=means, cov=covs, log_evidence=np.diff(log_joint, prepend=0.), ess=ess)


def transition_check():
    import numpy as np
    from dynamics import _cvodes, linearize, step
    from plant import FLOW_HIGH, FLOW_LOW, NOMINAL_FLOW, nominal_profile

    profiles = (nominal_profile(), np.linspace(.02,.98,10), np.linspace(.98,.02,10),
                np.tile([.99,.01],5), np.tile([.01,.99],5))
    grid = list(product((FLOW_LOW[0], NOMINAL_FLOW[0], FLOW_HIGH[0]),
                        (FLOW_LOW[1], NOMINAL_FLOW[1], FLOW_HIGH[1])))
    groups = {}
    for name, controls, constants in (("study", grid, (3.,4.5,5.,8.)),
                                     ("published_flow_stress", list(product((5.,500.), (10.,1000.))), (3.,8.))):
        maxima = dict(state=0., A=0., B=0., centered=0.)
        latencies = []
        for x, u, kla in product(profiles, controls, constants):
            z = np.r_[x, np.log(kla)]
            u = np.array(u)
            start = perf_counter()
            A, B, b = linearize(z, u)
            latencies.append(perf_counter()-start)
            nxt = step(z, u)
            ref_A, ref_B, ref_x = (out.full() for out in _cvodes()[1](z, u))
            for key, err in (("state", nxt-ref_x.ravel()), ("A", A-ref_A), ("B", B-ref_B),
                             ("centered", A @ z + B @ u + b - nxt)):
                maxima[key] = max(maxima[key], float(np.max(abs(err))))
            np.testing.assert_array_equal(A[10], np.eye(11)[10])
            np.testing.assert_array_equal(B[10], np.zeros(2))
        if max(maxima.values()) > 3e-8:
            raise ValueError(f"{name} native-flow parity failed: {maxima}")
        groups[name] = dict(windows=len(latencies), maximum_absolute_error=maxima,
                            median_linearize_ms=1000*float(np.median(latencies)))
    # A direct centered difference of native predictions also checks CVODES sensitivities.
    z = np.r_[profiles[1],np.log(4.5)]
    zu = np.r_[z,FLOW_HIGH]
    tangent = np.column_stack(linearize(z,FLOW_HIGH)[:2])
    errors = []
    for i in range(13):
        h = 2e-5 * max(1., abs(zu[i]))
        dz = h * np.eye(13)[i]
        hi, lo = zu+dz, zu-dz
        fd = (native_step(hi[:11],hi[11:])-native_step(lo[:11],lo[11:])) / (2*h)
        errors.append(float(np.max(abs(fd-tangent[:,i]))))
    if max(errors) > 2e-6:
        raise ValueError("Finite-difference native derivative check failed")
    groups["native_central_difference_max"] = max(errors)
    return groups


def development_window():
    """Synthetic four-transition window, with an informative initial prior and no future goals."""
    import numpy as np
    import plant

    m0 = np.r_[plant.nominal_profile(), np.log(plant.NOMINAL_KLA)]
    P0 = np.diag(np.r_[[plant.INITIAL_SD**2]*10, plant.LOG_KLA_SD**2])
    Q = np.diag(np.r_[[plant.PROCESS_SD**2]*10, study.MOTION_SD**2])
    R = np.diag(plant.SENSOR_SD**2)
    rng = np.random.default_rng(5487)
    truth = rng.multivariate_normal(m0, P0)
    flows = np.array([[105.,18.], [45.,45.], [75.,30.], [45.,18.]])
    readings = []
    for u in flows:
        truth = native_step(truth,u) + rng.multivariate_normal(np.zeros(11),Q)
        readings.append(truth[plant.OUTLETS] + rng.multivariate_normal(np.zeros(2),R))
    return dict(m0=m0, P0=P0, flows=flows, observations=readings, Q=Q, R=R)


def difference(candidate, ref):
    import numpy as np
    from plant import HIDDEN
    sd = np.sqrt(ref["cov"].diagonal(axis1=-2, axis2=-1))
    dm = np.abs(candidate["mean"]-ref["mean"])
    dc = np.abs(candidate["cov"]-ref["cov"])
    return dict(hidden_mean_max=float(dm[:,HIDDEN].max()), log_kla_mean_max=float(dm[:,10].max()),
                mean_max_posterior_sd=float((dm/sd).max()),
                cov_max_scaled=float((dc/sd[:,:,None]/sd[:,None,:]).max()),
                one_step_log_evidence_max=float(np.max(abs(candidate["log_evidence"]-ref["log_evidence"]))))


def check_native_backend(backend, window, reference):
    """Apply the real library update to exactly the nonlinear reference's data."""
    import numpy as np
    from dynamics import linearize
    from plant import OUTLETS

    m, P, Q, R = (window[k] for k in ("m0", "P0", "Q", "R"))
    H = np.eye(11)[OUTLETS]
    result = dict(mean=[], cov=[], log_evidence=[])
    for u, y in zip(window["flows"], window["observations"], strict=True):
        A, B, b = linearize(m,u)
        m, P = backend.update(m,P,A,B@u+b,Q,H,R,y)
        result["mean"].append(m)
        result["cov"].append(P)
        result["log_evidence"].append(-backend.last_innovation_nll)
    posterior = {k: np.array(v) for k, v in result.items()}
    return dict(difference=difference(posterior, reference), posterior=posterior)


def particle_check(backends, counts):
    import importlib
    import numpy as np
    w = development_window()
    estimates = []
    for count in counts:
        estimates.append(particle_reference(**w, count=count, seed=4719))
    ref = estimates[-1]
    repeat = particle_reference(**w, count=counts[-1], seed=12137)
    # Common random numbers isolate the independent RK4 integration error.
    doubled = particle_reference(**w, count=counts[0], seed=4719, substeps=16)
    changes = [difference(v,ref) for v in estimates[:-1]]
    replicate = difference(repeat,ref)
    integration = difference(doubled,estimates[0])
    score = {}
    for name in backends:
        backend = importlib.import_module(name+"_native").Backend()
        try:
            score[name] = check_native_backend(backend,w,ref)
        finally:
            if hasattr(backend, "close"):
                backend.close()
    stability_limits = dict(mean_max_posterior_sd=.12, cov_max_scaled=.15)
    native_limits = dict(mean_max_posterior_sd=.25, cov_max_scaled=.25,
                         one_step_log_evidence_max=.25)
    stable = all(d[k] <= limit for d in (changes[-1], replicate)
                 for k, limit in stability_limits.items())
    for row in score.values():
        row["passed"] = all(row["difference"][k] <= limit for k, limit in native_limits.items())
    return dict(counts=counts, substeps=8, budget_differences_to_largest=changes,
                independent_largest_replicate=replicate, doubled_RK4_substeps=integration,
                reference_stable=stable, largest_ess=ref["ess"], native_comparisons=score,
                stability_limits=stability_limits, native_limits=native_limits, window=w, reference=ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="*", choices=("cffg", "rxinfer"), default=["cffg", "rxinfer"])
    parser.add_argument("--counts", type=int, nargs="+", default=[16384,65536,262144])
    args = parser.parse_args()
    if len(args.counts) < 2 or any(a >= b for a,b in zip(args.counts,args.counts[1:])):
        parser.error("Expected at least two increasing particle budgets")
    study.configure()
    transition = transition_check()
    results = particle_check(args.backends,args.counts)
    passed = (results["reference_stable"] and all(
        results["native_comparisons"].get(name, {}).get("passed", False) for name in ("cffg","rxinfer")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=passed, transition=transition, particles=results),
                                      default=lambda v: v.tolist(), indent=2, allow_nan=False)+"\n")
    if not passed:
        raise SystemExit("Short nonlinear reference or native-filter qualification failed")
