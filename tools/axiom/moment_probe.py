"""Qualify public centered and legacy raw moments against Gaussian conditioning."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cffg import mixg
from cffg.compiler import compile_template
from cffg.core.ir import BlobRef, FactorSpec, Template, VariableSpec
from cffg.engine import Engine, _replace


def metadata():
    source = Path(mixg.__file__).resolve().parent
    files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(source.rglob("*.py"))}
    return {
        "source_directory": str(source),
        "python_source_tree_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "executable": sys.executable,
        "packages": {name: version(name) for name in ("jax", "jaxlib", "numpy", "scipy", "equinox")},
        "platform": platform.platform(), "device": str(jax.devices()[0]),
        "x64_enabled": bool(jax.config.x64_enabled),
        "clock": vars(time.get_clock_info("perf_counter")),
    }


def model(arrays):
    k, d = arrays["m"].shape
    return Template(
        name="mixed_feature_checkpoint",
        variables={"x": VariableSpec("x", dim=d), "s": VariableSpec("s", cardinality=k)},
        factors={
            "class": FactorSpec("class", "categorical", {"out": "s"}, param_blobs={"statistic": "p"}),
            "sensor": FactorSpec("sensor", "gaussian", {"out": "x"}, param_blobs={"mean": "y", "cov": "R"}),
            "object": FactorSpec("object", "gaussian_mixture", {"out": "x", "selector": "s"},
                                 param_blobs={"mean": "m", "cov": "V"}),
        },
        blobs={k: BlobRef(k, k + ".npy") for k in arrays},
    )


def local_state(arrays):
    engine = Engine(compile_template(model(arrays), arrays=arrays))
    # Both incoming messages at the mixture now come from normalized leaves.
    return engine, engine.step(engine.init_state())


def oracle(mean, covariance, prior, h, y, noise):
    """Independent covariance-form conditioning and innovation likelihood in float64."""
    means, covariances, scores = [], [], []
    for m, v, p in zip(mean.astype(float), covariance.astype(float), prior, strict=True):
        innovation = y - h @ m
        s = h @ v @ h.T + noise
        gain = np.linalg.solve(s, h @ v).T
        means.append(m + gain @ innovation)
        covariances.append(v - gain @ h @ v)
        cost = innovation @ np.linalg.solve(s, innovation) + np.linalg.slogdet(s)[1]
        scores.append(np.log(p) - .5 * cost if p > 0 else -np.inf)
    weight = np.exp(scores - np.max(scores))
    return weight / weight.sum(), np.array(means), np.array(covariances)


def verify(raw, centered, expected, dtype):
    w, m, v = expected
    eps = np.finfo(dtype).eps
    k, d = m.shape
    expected_shapes = ((k,), (k, d), (k, d, d))
    for stats, names in ((raw, ("responsibility", "weighted_mean", "weighted_second")),
                         (centered, ("responsibility", "component_mean", "component_covariance"))):
        assert set(stats) == {*names, "proper"} and bool(stats["proper"])
        assert np.asarray(stats["proper"]).shape == () and np.asarray(stats["proper"]).dtype == np.bool_
        for key, shape in zip(names, expected_shapes, strict=True):
            assert stats[key].shape == shape and stats[key].dtype == dtype
        np.testing.assert_allclose(stats["responsibility"], w, atol=16*eps, rtol=16*eps)
        np.testing.assert_array_equal(np.array(stats["responsibility"])[w == 0], 0)
    np.testing.assert_allclose(centered["component_mean"], m, rtol=4*eps, atol=16*eps)
    np.testing.assert_allclose(centered["component_covariance"], v, rtol=16*eps, atol=8*eps)
    # The raw route can represent a large second moment accurately but lose its small variance.
    np.testing.assert_allclose(raw["weighted_mean"], w[:, None]*m, atol=32*eps, rtol=32*eps)
    raw_second = w[:, None, None] * (v + m[:, :, None]*m[:, None, :])
    np.testing.assert_allclose(raw["weighted_second"], raw_second, atol=32*eps, rtol=32*eps)


def check(origin, component_variance, dtype, *, displaced=False):
    offset = np.array([0., .125]) if displaced else np.zeros(2)
    y = origin + (1. if displaced else 0.)
    noise = .25 if displaced else component_variance
    variances = np.array([.5, 1.]) if displaced else np.full(2, component_variance)
    arrays = {k: np.array(v, dtype=dtype) for k, v in {
        "m": (origin + offset)[:, None], "V": variances[:, None, None],
        "y": [y], "R": [[noise]], "p": [.5, .5],
    }.items()}
    engine, state = local_state(arrays)
    raw = engine.gaussian_node_stats(state)
    centered = engine.gaussian_node_stats(state, centered_mixtures=True)
    for name in raw["sensor"]:
        np.testing.assert_array_equal(centered["sensor"][name], raw["sensor"][name])
    raw, centered = raw["object"], centered["object"]
    expected = oracle(arrays["m"], arrays["V"], arrays["p"], np.eye(1), np.array([y]), np.array([[noise]]))
    verify(raw, centered, expected, dtype)
    w = raw["responsibility"].astype(float)
    first = raw["weighted_mean"][:, 0].astype(float)
    second = raw["weighted_second"][:, 0, 0].astype(float)
    return dict(dtype=dtype.__name__, origin=origin, displaced=displaced,
                prior_variances=variances.tolist(), proper=True, weights=w.tolist(),
                exact_weights=expected[0].tolist(), weighted_mean=first.tolist(),
                weighted_second=second.tolist(), recovered_variance=(second/w-(first/w)**2).tolist(),
                exact_variance=expected[2][:, 0, 0].tolist(),
                centered_variance=centered["component_covariance"][:, 0, 0].tolist(),
                exact_mean=expected[1][:, 0].tolist(), centered_mean=centered["component_mean"][:, 0].tolist())


def require_improper(stats):
    assert not stats["proper"]
    for key, value in stats.items():
        if key != "proper":
            assert np.isnan(value).all(), key


def matrix_checks(dtype):
    arrays = {k: np.array(v, dtype=dtype) for k, v in {
        "p": [1., 0.], "m": [[.4, -.8], [.75, 1.25]],
        "V": [[[1.5, .25], [.25, .75]], [[1., -.375], [-.375, 1.5]]],
        "y": [1., -.25], "R": [[1., .25], [.25, 1.5]],
    }.items()}
    engine, state = local_state(arrays)
    group, = engine.arena.mix_groups
    eta_ix = engine.arena.g_eta_indices(group.g_gather[:, 0], 2)[0]
    lam_ix = engine.arena.g_lam_indices(group.g_gather[:, 0], 2)[0]
    sel_ix = engine.arena.message_indices(group.s_gather, group.k_sel)[0]
    precision_key = group.name + ":W"
    args_by_rank, expected_by_rank = [], []
    # Incoming precision has rank 2, 1 and 0 respectively.
    for h, y, r in ((np.eye(2), arrays["y"], arrays["R"]),
                    (np.array([[1., -.5]]), np.array([.25]), np.array([[.75]])),
                    (np.zeros((1, 2)), np.zeros(1), np.eye(1))):
        eta = h.T @ np.linalg.solve(r.astype(float), y)
        lam = h.T @ np.linalg.solve(r.astype(float), h)
        probe = _replace(state, g_eta=state.g_eta.at[eta_ix].set(eta),
                         g_lam=state.g_lam.at[lam_ix].set(lam))
        raw = engine.gaussian_node_stats(probe)["object"]
        centered = engine.gaussian_node_stats(probe, centered_mixtures=True)["object"]
        expected = oracle(arrays["m"], arrays["V"], arrays["p"], h, y, r)
        verify(raw, centered, expected, dtype)
        expected_by_rank.append(expected)
        args_by_rank.append((probe.g_eta[eta_ix], probe.g_lam[lam_ix], probe.messages[sel_ix],
                             state.params[group.name + ":m"][0], state.params[precision_key][0]))

    for members in (1, 3):
        args = tuple(jnp.stack(items) for items in zip(*args_by_rank[:members], strict=True))
        for compiled in (False, True):
            raw_fn, centered_fn = mixg.mix_stats, mixg.mix_centered_stats
            if compiled:
                raw_fn, centered_fn = jax.jit(raw_fn), jax.jit(centered_fn)
            raw, centered = raw_fn(*args)._asdict(), centered_fn(*args)._asdict()
            for i in range(members):
                verify({k: v[i] for k, v in raw.items()}, {k: v[i] for k, v in centered.items()},
                       expected_by_rank[i], dtype)

    params = state.params
    singular = {**params, precision_key: params[precision_key].at[0, 0].set(jnp.ones((2, 2)))}
    for bad in (_replace(state, params=singular),
                _replace(state, g_eta=state.g_eta.at[eta_ix[0]].set(jnp.nan)),
                _replace(state, messages=state.messages.at[sel_ix].set(-jnp.inf))):
        for centered in (True, False):
            require_improper(engine.gaussian_node_stats(bad, centered_mixtures=centered)["object"])
    return dict(dimension=2, regimes=2, prior=[1., 0.], incoming_ranks=[2, 1, 0],
                kernel_members=[1, 3], execution=["public", "kernel eager", "kernel jit"],
                improper=["singular component precision", "nonfinite cavity", "empty selector"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dtype", choices=("float32", "float64"))
    dtype = np.dtype(parser.parse_args().dtype).type
    jax.config.update("jax_enable_x64", dtype is np.float64)
    rows = [check(origin, v, dtype) for origin in (0., 1., 128., 10000.) for v in (.25, 1.)]
    rows.extend(check(origin, .5, dtype, displaced=True) for origin in (0., 10000.))
    matrix = matrix_checks(dtype)
    print(json.dumps({"repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      **metadata(), "rows": rows, "matrix_checks": matrix, "passed": True},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
