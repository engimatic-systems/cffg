"""Qualify persistent mixed Gaussian/count evidence on a pinned source tree."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import json
from pathlib import Path
import sys
from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np

from cffg import mixed_learning
from cffg.mixed_learning import MixedBank, MixedBatch, score, update

FIELDS = ("mu", "kappa", "psi", "nu", "alpha", "active")
UPDATE = jax.jit(update)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime():
    root = Path(mixed_learning.__file__).resolve().parent
    hashes = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))}
    return dict(
        source_root=str(root),
        source_digest=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        probe_sha256=sha(__file__),
        python=sys.version,
        device=str(jax.devices()[0]),
        x64=bool(jax.config.x64_enabled),
        packages={key: version(key) for key in ("jax", "jaxlib", "numpy", "scipy")},
    )


def visible(b):
    return {
        **{key: np.asarray(getattr(b, key)) for key in FIELDS},
        **{f"category_{i}": np.asarray(h) for i, h in enumerate(b.categories)},
    }


def payload(b, mode):
    return b.state_dict(exact=True) if mode == "exact" else b.state_dict()


def read_file(file):
    with np.load(file, allow_pickle=False) as data:
        return MixedBank.from_state_dict({k: data[k] for k in data.files})


def restart(b, mode):
    with tempfile.TemporaryFile() as stream:
        np.savez(stream, **payload(b, mode))
        stream.seek(0)
        return read_file(stream)


def same_tree(a, b):
    x, s = jax.tree.flatten(a)
    y, t = jax.tree.flatten(b)
    return s == t and all(
        np.asarray(p).dtype == np.asarray(q).dtype
        and np.asarray(p).shape == np.asarray(q).shape
        and np.asarray(p).tobytes() == np.asarray(q).tobytes()
        for p, q in zip(x, y, strict=True)
    )


def bank(dtype, m, kappa, psi, nu, alpha, categories, active):
    vals = [np.asarray(a, dtype=dtype) for a in (m, kappa, psi, nu, alpha)]
    b = MixedBank(*vals, tuple(np.asarray(x, dtype=dtype) for x in categories), np.asarray(active, bool))
    return jax.tree.map(jnp.asarray, b)


def batch(mean, covariance, observed, head, dtype):
    return MixedBatch(
        np.asarray(mean, dtype),
        np.asarray(covariance, dtype),
        np.asarray(observed, bool),
        tuple(np.asarray(a, np.int32) for a in head),
    )


def oracle(b, x, w):
    """Independent float64 NIW in local coordinates, prior mass as an extra observation."""
    out = {key: a.astype(float).copy() if key != "active" else a.copy() for key, a in visible(b).items()}
    w = np.asarray(w, float)
    any_seen = np.asarray(x.gaussian_observed).copy()
    for h in x.categories:
        any_seen |= h >= 0
    for j in np.flatnonzero(out["active"]):
        g = x.gaussian_observed & (w[:, j] > 0)
        mass = w[g, j]
        if len(mass):
            prior_k = out["kappa"][j]
            delta = x.mean[g].astype(float) - out["mu"][j]
            k = prior_k + mass.sum()
            shift = np.einsum("n,nd->d", mass, delta) / k
            residual = delta - shift
            out["psi"][j] += (
                np.einsum("n,nij->ij", mass, x.covariance[g].astype(float))
                + np.sum(mass[:, None, None] * residual[:, :, None] * residual[:, None, :], axis=0)
                + prior_k * np.outer(shift, shift)
            )
            out["mu"][j] += shift
            out["nu"][j] += mass.sum()
            out["kappa"][j] = k
        out["alpha"][j] += w[any_seen, j].sum()
        for i, head in enumerate(x.categories):
            keep = head >= 0
            counts = out[f"category_{i}"][j]
            counts += np.bincount(head[keep], weights=w[keep, j], minlength=len(counts))
    return out


def metrics(b, ref):
    actual = visible(b)
    errors = {
        k: float(np.max(np.abs(actual[k].astype(float) - v))) for k, v in ref.items() if k != "active"
    }
    return errors


def accurate(b, ref, eps):
    a = visible(b)
    # Use the scale of the actual posterior, including very small SPD scales.
    for k, v in ref.items():
        if k == "active":
            np.testing.assert_array_equal(a[k], v)
        elif k == "mu":
            rounded = v.astype(a[k].dtype)
            # One neighboring representable real plus roundoff about zero.
            tol = np.maximum(
                abs(np.spacing(rounded)), eps * np.min(np.sqrt(ref["psi"].diagonal(axis1=-2, axis2=-1)))
            )
            if np.any(abs(a[k] - rounded) > tol):
                return False
        elif not np.allclose(a[k], v, rtol=64 * eps, atol=64 * eps * float(np.abs(v).max(initial=0))):
            return False
    return True


def run_stream(b, x, w, *, mode=None, every=1):
    ok = True
    sequence = []
    for i in range(len(w)):
        step = jax.tree.map(lambda a: a[i : i + 1], x)
        b, v = UPDATE(b, step, w[i : i + 1])
        ok &= bool(v)
        if mode and (i + 1) % every == 0:
            b = restart(b, mode)
        sequence.append(bool(v))
    assert ok, f"invalid update at {sequence.index(False)}"
    return b


def scalar_fixture(dtype, name):
    m, k, psi, n, y, variance = {
        "translated": (10000.0, 4096.0, 3.0, 1024, 10001.0, 0.125),
        "origin": (0.0, 4096.0, 3.0, 1024, 1.0, 0.125),
        "two_ulp": (
            4096.0,
            1.0,
            float(np.spacing(dtype(4096.0))) ** 2,
            2,
            float(dtype(4096.0) + np.spacing(dtype(4096.0))),
            0.0,
        ),
        "sub_ulp_scale": (0.0, 1.0, 2.0**24, 16, 0.0, 0.25),
    }[name]
    b = bank(dtype, [[m]], [k], [[[psi]]], [4.0], [2.0], ([[1.0, 1.0, 1.0]],), [True])
    x = batch(
        np.full((n, 1), y), np.full((n, 1, 1), variance), np.ones(n, bool), (np.zeros(n, int),), dtype
    )
    return b, x, np.ones((n, 1), dtype)


def counts_fixture(dtype):
    m = 2.0**22
    b = bank(
        dtype,
        np.zeros((3, 1)),
        [m] * 3,
        np.ones((3, 1, 1)),
        [m] * 3,
        [m] * 3,
        (np.full((3, 3), m), np.full((3, 5), m)),
        [True, True, False],
    )
    x = batch(
        np.ones((4, 1)), np.zeros((4, 1, 1)), np.ones(4, bool), (np.zeros(4, int), np.full(4, 4)), dtype
    )
    return b, x, np.tile(np.array([[0.25, 0.75, 0]], dtype), (4, 1))


def case(name, dtype, is_new):
    b, x, w = counts_fixture(dtype) if name == "large_counts" else scalar_fixture(dtype, name)
    ref = oracle(b, x, w)
    if name == "two_ulp":
        np.testing.assert_allclose(
            ref["psi"].item(), 5 / 3 * float(np.spacing(dtype(4096.0))) ** 2, rtol=1e-15
        )
    if name == "translated":
        np.testing.assert_allclose(ref["psi"].item(), 950.2, rtol=1e-14)
    eps = np.finfo(dtype).eps
    full, ok = UPDATE(b, x, w)
    assert ok
    last = run_stream(b, x, w)
    routes = {"batch": full, "stream": last}
    # Frequent/rare restarts use real serialized NPZ arrays. The translated
    # drift test uses the full 1,024 ticks. Other tiny controls can reset each tick.
    if name != "origin":
        routes["compact_every_1"] = run_stream(b, x, w, mode="compact")
        routes["compact_rare"] = run_stream(b, x, w, mode="compact", every=128 if len(w) >= 128 else len(w))
        if is_new:
            routes["exact_every_1"] = run_stream(b, x, w, mode="exact")
    rows = {}
    for label, post in routes.items():
        vals = visible(post)
        rows[label] = dict(
            accurate=accurate(post, ref, eps),
            error=metrics(post, ref),
            mean=vals["mu"].tolist(),
            psi=vals["psi"].tolist(),
            increments={
                f: (vals[f].astype(float) - visible(b)[f].astype(float)).tolist()
                for f in ("alpha", "kappa", "nu", "category_0")
            },
        )
    if name in ("large_counts", "sub_ulp_scale"):
        fields = (
            ("alpha", "kappa", "nu", "category_0", "category_1") if name == "large_counts" else ("psi",)
        )
        for label, post in routes.items():
            obs = visible(post)
            exactly_represented = all(np.array_equal(obs[key], ref[key]) for key in fields)
            rows[label]["target_increments_exact"] = exactly_represented
            rows[label]["accurate"] &= exactly_represented
    exact = same_tree(routes["exact_every_1"], last) if "exact_every_1" in routes else None
    return dict(
        passed=rows["batch"]["accurate"] and rows["stream"]["accurate"] and exact is not False,
        exact_restarts_bitwise=exact,
        expected={k: v.tolist() for k, v in ref.items()},
        rows=rows,
    )


def invalid(dtype, is_new):
    b, x, w = counts_fixture(dtype)
    first = jax.tree.map(lambda a: a[:1], x)
    state, ok = UPDATE(b, first, w[:1])
    assert ok
    bad_states = []
    for weights in (np.full_like(w[:1], np.nan), w[:1, ::-1]):
        result, valid = UPDATE(state, first, weights)
        bad_states.append(not bool(valid) and same_tree(result, state))
    empty = first._replace(
        mean=np.full_like(first.mean, np.nan),
        covariance=np.full_like(first.covariance, np.nan),
        gaussian_observed=np.zeros(1, bool),
        categories=(np.array([-1], np.int32), np.array([-1], np.int32)),
    )
    result, valid = UPDATE(state, empty, w[:1])
    scores, weights, sv = score(state, empty)
    assert sv and np.all(np.asarray(scores)[0, :2] == 0) and np.asarray(weights)[0, 2] == 0
    partial = empty._replace(categories=(np.array([1], np.int32), np.array([-1], np.int32)))
    n = 4
    masked_batch = jax.tree.map(lambda a: np.repeat(a, n, axis=0), partial)
    post = run_stream(b, masked_batch, w)
    mask_correct = accurate(post, oracle(b, masked_batch, w), np.finfo(dtype).eps)
    for name in ("mu", "psi", "kappa", "nu", "category_1"):
        np.testing.assert_array_equal(visible(post)[name], visible(b)[name])
    mask_counts = np.asarray(post.categories[0])[:, 1].astype(float) - np.asarray(b.categories[0])[:, 1]
    mask_count_exact = np.array_equal(mask_counts, [1.0, 3.0, 0.0])
    # Baseline is expected to exhibit the same representability failure.
    return dict(
        passed=all(bad_states)
        and bool(valid)
        and same_tree(result, state)
        and mask_correct
        and mask_count_exact,
        invalid_returns_input=bad_states,
        no_evidence_unchanged=same_tree(result, state),
        observed_head_increments=mask_counts.tolist(),
        mask_count_exact=mask_count_exact,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", choices=("old", "new"), required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), required=True)
    p.add_argument("--output", type=Path, required=True)
    opts = p.parse_args()
    dtype = np.dtype(opts.dtype).type
    jax.config.update("jax_enable_x64", dtype == np.float64)
    new = opts.route == "new"
    results = {}
    for name in ("translated", "origin", "two_ulp", "large_counts", "sub_ulp_scale"):
        results[name] = case(name, dtype, new)
    results["mask_atomic"] = invalid(dtype, new)
    report = dict(
        route=opts.route,
        passed=all(v["passed"] for v in results.values()),
        checks=results,
        provenance=runtime(),
    )
    opts.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if new and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
