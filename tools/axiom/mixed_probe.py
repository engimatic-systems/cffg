"""Independent mixed-evidence qualification and synchronized CPU timing.

Run the plan command before measurement. The qualification and timing commands write
strict JSON to the supplied filename. Timing starts a fresh worker per shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.special import digamma, logsumexp

SHAPES = [(n, k) for k in (32, 500, 5000) for n in (1, 32)]
ENV = dict(
    JAX_PLATFORMS="cpu",
    JAX_ENABLE_COMPILATION_CACHE="0",
    JAX_DISABLE_JIT="0",
    OPENBLAS_NUM_THREADS="1",
    VECLIB_MAXIMUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)
BLOCKS, CALLS = 12, 3


def plan():
    return dict(
        shapes=SHAPES,
        dimension=6,
        benchmark_dtype="float32",
        qualification_dtypes=["float32", "float64"],
        host_reference_max_k=500,
        blocks=BLOCKS,
        calls_per_block=CALLS,
        warmups=3,
        environment=ENV,
        comparison="jit(public score + update) / checked per-component SciPy ELL + NIW.update_centered",
        boundary="same immutable NumPy inputs through full host scores, weights, posterior and both valid flags",
        assignments="likelihood only, recomputed every call from original bank",
        cold="one first call per arm per shape, after imports, fixture and JIT wrapper construction",
        interval="median paired block-mean ratio; 10000 whole-block bootstrap draws, seed 910",
        material_slowdown=1.2,
        qualification="car-8rgl rational posterior and absolute scores; scalar NIW",
    )


def provenance():
    import cffg
    import jax

    src = Path(cffg.__file__).resolve().parent
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            src / "mixed_learning.py",
            src / "mixed_scoring.py",
            src / "conj.py",
            Path(__file__).resolve(),
        )
    }
    tree = {
        str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(src.rglob("*.py"))
    }
    return dict(
        module_sha256=hashes,
        source_root=str(src),
        tree_sha256=hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        packages={p: version(p) for p in ("numpy", "scipy", "jax", "jaxlib", "equinox")},
        device=str(jax.devices()[0]),
        x64=jax.config.x64_enabled,
        clock=vars(time.get_clock_info("perf_counter")),
    )


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def matrix_check(a, *, positive):
    """Validate covariance in dimensionless coordinates at the caller's precision."""
    if not a.size:
        return
    eps = np.finfo(a.dtype).eps
    x = np.asarray(a, dtype=float)
    require(np.isfinite(x).all(), "nonfinite covariance")
    diag = x.diagonal(axis1=-2, axis2=-1)
    require(np.all(diag > 0 if positive else diag >= 0), "invalid diagonal")
    scale = np.sqrt(diag)[..., :, None] * np.sqrt(diag)[..., None, :]
    require(np.all((scale > 0) | (x == 0)), "nonzero covariance with zero diagonal")
    corr = x / np.where(scale > 0, scale, 1.0)
    tol = 16 * x.shape[-1] * eps
    require(np.max(np.abs(corr - corr.swapaxes(-1, -2))) <= tol, "asymmetric covariance")
    eigen = np.linalg.eigvalsh(corr)
    require(np.all(eigen > 0 if positive else eigen >= -tol), "invalid correlation spectrum")


def check_bank(b):
    k, d = b.mu.shape
    require(b.active.shape == (k,) and b.active.dtype == np.bool_ and b.active.any(), "active set")
    require(
        b.psi.shape == (k, d, d) and b.kappa.shape == b.nu.shape == b.alpha.shape == (k,), "bank layout"
    )
    require(np.isfinite(b.mu).all(), "nonfinite location")
    for v in (b.kappa, b.nu, b.alpha, *b.categories):
        require(np.isfinite(v).all() and (v > 0).all() and v.shape[0] == k, "invalid prior count")
    require(np.all(b.nu > d - 1), "improper NIW")
    matrix_check(b.psi, positive=True)


def check_batch(b, x):
    n, d = x.mean.shape
    require(d == b.mu.shape[1] and x.covariance.shape == (n, d, d), "feature dimensions")
    require(x.gaussian_observed.shape == (n,) and x.gaussian_observed.dtype == np.bool_, "observation mask")
    require(np.isfinite(x.mean[x.gaussian_observed]).all(), "nonfinite observed location")
    matrix_check(x.covariance[x.gaussian_observed], positive=False)
    for head, obs in zip(b.categories, x.categories, strict=True):
        require(
            obs.shape == (n,) and obs.dtype.kind in "iu" and np.all((obs >= -1) & (obs < head.shape[1])),
            "invalid categorical observation",
        )


def host_score(b, x, *, include_mixing=False):
    """Absolute Gaussian expectation and observed Dirichlet log means, component by component."""
    check_bank(b)
    check_batch(b, x)
    n, d = x.mean.shape
    scores = np.full((n, len(b.mu)), -np.inf)
    g = x.gaussian_observed
    mean, covariance = x.mean[g].astype(float), x.covariance[g].astype(float)
    for j in np.flatnonzero(b.active):
        psi = b.psi[j].astype(float)
        inv = np.linalg.solve(psi, np.eye(d))
        df = float(b.nu[j])
        log_precision = (
            digamma(0.5 * (df - np.arange(d))).sum() + d * np.log(2.0) - np.linalg.slogdet(psi)[1]
        )
        diff = mean - b.mu[j]
        quadratic = np.einsum("ni,ij,nj->n", diff, inv, diff) + np.einsum("ij,nji->n", inv, covariance)
        scores[:, j] = 0
        scores[g, j] = 0.5 * (
            log_precision - d * np.log(2 * np.pi) - d / b.kappa[j].item() - df * quadratic
        )
        for head, obs in zip(b.categories, x.categories, strict=True):
            seen = obs != -1
            counts = head[j].astype(float)
            scores[seen, j] += digamma(counts[obs[seen]]) - digamma(counts.sum())
    logits = scores.copy()
    if include_mixing:
        mass = b.alpha[b.active].astype(float)
        logits[:, b.active] += digamma(mass) - digamma(mass.sum())
    weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    require(
        np.isfinite(scores[:, b.active]).all() and np.isfinite(weights).all(), "nonfinite scoring result"
    )
    return scores.astype(b.mu.dtype), weights.astype(b.mu.dtype)


def host_update(b, x, weights):
    """Pool scalar weighted NIWs and add each observed field's own counts."""
    require(weights.shape == (len(x.mean), len(b.mu)), "weight shape")
    require(np.all(np.isfinite(weights) & (weights >= 0)) and np.all(weights[:, ~b.active] == 0), "support")
    require(
        np.allclose(weights.sum(axis=1, dtype=float), 1.0, atol=16 * np.finfo(weights.dtype).eps, rtol=0),
        "not normalized",
    )
    return host_fold(b, x, weights)


def host_fold(b, x, weights):
    from cffg.conj import NIW

    post = type(b)(*(v.copy() for v in b[:5]), tuple(v.copy() for v in b.categories), b.active.copy())
    observed = x.gaussian_observed.copy()
    for obs in x.categories:
        observed |= obs >= 0
    for j in np.flatnonzero(b.active):
        w = weights[:, j].astype(float)
        post.alpha[j] = b.alpha[j] + w[observed].sum()
        g = x.gaussian_observed
        niw = NIW(b.mu[j], b.kappa[j], b.psi[j], b.nu[j]).update_centered(x.mean[g], x.covariance[g], w[g])
        post.mu[j], post.kappa[j], post.psi[j], post.nu[j] = niw.mu, niw.kappa, niw.psi, niw.nu
        for output, prior, obs in zip(post.categories, b.categories, x.categories, strict=True):
            seen = obs >= 0
            output[j] = prior[j] + np.bincount(obs[seen], weights=w[seen], minlength=output.shape[1])
    check_bank(post)
    return post


def host(b, x):
    scores, weights = host_score(b, x)
    return scores, weights, host_update(b, x, weights), np.bool_(True), np.bool_(True)


def native(b, x):
    from cffg.mixed_learning import score, update

    scores, weights, sv = score(b, x)
    post, uv = update(b, x, weights)
    return scores, weights, post, sv, uv


def fix(dtype):
    from cffg.mixed_learning import MixedBank, MixedBatch

    def repeat(value):
        return np.repeat(np.array(value, dtype=dtype)[None], 3, axis=0)

    b = MixedBank(
        repeat([1, -1]),
        repeat(2),
        repeat([[2, 1], [1, 2]]),
        repeat(4),
        np.array([2, 5, 0.1], dtype),
        (repeat([2, 3]), repeat([1, 2, 1])),
        np.array([True, True, False]),
    )
    x = MixedBatch(
        np.array([[2, 0], [0, 0]], dtype),
        np.array([np.zeros((2, 2)), [[0.5, -0.25], [-0.25, 0.5]]], dtype),
        np.array([True, True]),
        (np.array([0, 1], np.int32), np.array([-1, 1], np.int32)),
    )
    return b, x


def fixture(n, k, dtype):
    from cffg.mixed_learning import MixedBank, MixedBatch

    # Eight profiles repeat, so the large bank can also be checked against
    # a tiny scalar reference with analytically expanded normalized masses.
    d = 6
    group = np.arange(k) % 8
    dim = np.arange(d)
    mu = (group[:, None] - dim) / 8
    psi = (2 * np.eye(d) + 0.125) * (1 + group[:, None, None] / 8)
    alpha = (2 + group).astype(dtype)
    heads = tuple((1 + (group[:, None] + np.arange(c)) % 5).astype(dtype) for c in (3, 5))
    b = MixedBank(
        mu.astype(dtype),
        (2 + group / 8).astype(dtype),
        psi.astype(dtype),
        (8 + group / 4).astype(dtype),
        alpha,
        heads,
        group != 7,
    )
    t = np.arange(n)
    x = MixedBatch(
        np.sin((t[:, None] + dim) / 9).astype(dtype),
        ((0.125 * np.eye(d) + 0.03125) * (1 + t[:, None, None] % 3)).astype(dtype),
        t % 7 != 4,
        tuple(np.where(t % p == 1, -1, t % c).astype(np.int32) for c, p in ((3, 4), (5, 7))),
    )
    return b, x


def host_arrays(tree):
    import jax

    return jax.tree.map(np.asarray, tree)


def close(actual, expected, dtype, *, factor=48):
    import jax

    atol = rtol = factor * np.finfo(dtype).eps
    flat, shape = jax.tree.flatten(actual)
    reference, other = jax.tree.flatten(expected)
    assert shape == other
    for i, (a, b) in enumerate(zip(flat, reference, strict=True)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, (i, a.shape, b.shape, a.dtype, b.dtype)
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=f"leaf {i}")


def stress_oracle(b, x):
    """Compute all K absolute scores from the eight profiles and normalize over all K."""
    import jax

    small = jax.tree.map(lambda a: a[:8], b)
    score, _ = host_score(small, x)
    group = np.arange(len(b.mu)) % 8
    scores = score[:, group].copy()
    logits = scores.astype(float)
    weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True)).astype(b.mu.dtype)
    # Each representative receives only its own expanded weight. Use scalar
    # NIW on that mass directly; the reduced 8 columns do not sum to one.
    post = host_fold(small, x, weights[:, :8])
    return scores, weights, jax.tree.map(lambda a: a[group], post), np.bool_(True), np.bool_(True)


def qualify(dtype):
    import jax
    from cffg.mixed_learning import score, update

    b, x = fix(dtype)
    fixed = np.array([[0.25, 0.75, 0.0], [0.5, 0.5, 0.0]], dtype)
    eps = np.finfo(dtype).eps
    report = {}
    compiled_score = jax.jit(score, static_argnames=("include_mixing",))
    compiled_update = jax.jit(update)

    def run(name, fn):
        try:
            report[name] = {"passed": True, **(fn() or {})}
        except (AssertionError, ValueError, FloatingPointError) as err:
            report[name] = {"passed": False, "error": str(err) or type(err).__name__}

    def pinned():
        absolute = np.array([-4.381065542, -9.047732209])
        for include in (False, True):
            expected_score, expected_weight = host_score(b, x, include_mixing=include)
            np.testing.assert_allclose(expected_score[:, 0], absolute, atol=max(6e-10, 10 * eps), rtol=0)
            for fn in (score, compiled_score):
                s, w, ok = host_arrays(fn(b, x, include_mixing=include))
                assert ok and np.isneginf(s[:, 2]).all() and np.all(w[:, 2] == 0)
                close((s, w), (expected_score, expected_weight), dtype)
        expected = host_update(b, x, fixed)
        m = np.array([[10 / 11, -8 / 11], [14 / 13, -8 / 13], [1, -1]], dtype)
        psi = np.array(
            [
                np.array([[262, 61], [61, 246]]) / 88,
                np.array([[362, 107], [107, 314]]) / 104,
                [[2, 1], [1, 2]],
            ],
            dtype,
        )
        close((expected.mu, expected.psi), (m, psi), dtype)
        for fn in (update, compiled_update):
            post, ok = host_arrays(fn(b, x, fixed))
            assert ok
            close(post, expected, dtype)
            single = b
            for i in range(2):
                single, ok = host_arrays(
                    fn(single, jax.tree.map(lambda v: v[i : i + 1], x), fixed[i : i + 1])
                )
                assert ok
            close(single, post, dtype)
            for a, prior in zip(jax.tree.leaves(post), jax.tree.leaves(b), strict=True):
                np.testing.assert_array_equal(a[2], prior[2])
        return dict(
            scores=s[:, 0].tolist(), with_mixing=w[0].tolist(), mu=post.mu.tolist(), psi=post.psi.tolist()
        )

    def unequal_mass_scores():
        # Same observations after a fractional update, with unequal kappa, nu,
        # mixture counts, full Gaussian scale and category counts.
        post = host_update(b, x, fixed)
        assert post.kappa[0] != post.kappa[1] and post.nu[0] != post.nu[1]
        reference_scores, _ = host_score(post, x)
        assert not np.array_equal(reference_scores[:, 0], reference_scores[:, 1])
        for mixing in (False, True):
            expected = host_score(post, x, include_mixing=mixing)
            s, w, ok = host_arrays(compiled_score(post, x, include_mixing=mixing))
            assert ok
            close((s, w), expected, dtype)
            close(s, reference_scores, dtype)
        return dict(
            kappa=post.kappa[:2].tolist(),
            nu=post.nu[:2].tolist(),
            absolute_scores=s[:, :2].tolist(),
            with_mixing=w[:, :2].tolist(),
        )

    def additive():
        # Each live component first receives two hard observations at x=1/A.
        # A soft 1/2 observation at x=3/B has additive new mass 2.5 per component.
        from cffg.mixed_learning import MixedBank, MixedBatch

        one = np.ones
        prior = MixedBank(
            np.zeros((2, 1), dtype),
            one(2, dtype),
            one((2, 1, 1), dtype),
            one(2, dtype) * 3,
            np.full(2, 0.1, dtype),
            (one((2, 2), dtype),),
            one(2, bool),
        )
        batch = MixedBatch(
            np.array([[1], [1], [1], [1], [3]], dtype),
            np.zeros((5, 1, 1), dtype),
            one(5, bool),
            (np.array([0, 0, 0, 0, 1], np.int32),),
        )
        w0 = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0.5, 0.5]], dtype)
        block, ok = host_arrays(compiled_update(prior, batch, w0))
        assert ok
        close(block, host_update(prior, batch, w0), dtype)
        np.testing.assert_allclose(block.alpha, 2.6, rtol=8 * eps)
        first_sum = block.kappa * block.mu[:, 0]
        second_sum = block.psi[:, 0, 0] - prior.psi[:, 0, 0] + block.kappa * block.mu[:, 0] ** 2
        np.testing.assert_allclose(first_sum, 3.5, rtol=8 * eps)
        np.testing.assert_allclose(second_sum, 6.5, rtol=8 * eps)
        np.testing.assert_allclose(block.categories[0] - 1, [[2, 0.5], [2, 0.5]], rtol=8 * eps)
        return dict(
            additive_alpha=block.alpha.tolist(),
            prior_excluded_mass_first_second=[2.5, 3.5, 6.5],
            released_reference_only=[1.5, 2.5, 5.5],
            released_reference_alpha=1.6,
        )

    def missing():
        masked = x._replace(
            mean=np.full_like(x.mean, np.nan),
            covariance=np.full_like(x.covariance, np.nan),
            gaussian_observed=np.zeros(2, bool),
            categories=(np.array([0, -1], np.int32), np.array([-1, -1], np.int32)),
        )
        scores, weights, post, sv, uv = host_arrays(jax.jit(native)(b, masked))
        assert sv and uv
        close((scores, weights, post), host(b, masked)[:3], dtype)
        np.testing.assert_array_equal(scores[1, :2], 0)
        np.testing.assert_array_equal(post.psi, b.psi)
        np.testing.assert_array_equal(post.kappa, b.kappa)
        np.testing.assert_array_equal(post.nu, b.nu)
        return dict(scores=scores[:, :2].tolist(), alpha=post.alpha.tolist())

    def large():
        mass = 2**22
        alpha = b.categories[0].copy()
        alpha[:] = [mass, 1]
        high = b._replace(categories=(alpha, *b.categories[1:]))
        masked = x._replace(
            gaussian_observed=np.zeros(2, bool),
            categories=(np.zeros(2, np.int32), np.full(2, -1, np.int32)),
        )
        s, w, ok = host_arrays(compiled_score(high, masked))
        assert ok
        # psi(a) - psi(a+1) = -1/a. Require nonzero at source precision.
        np.testing.assert_allclose(s[:, :2], -1 / mass, atol=0.0, rtol=32 * eps)
        return dict(score=float(s[0, 0]), exact=-1 / mass)

    def count_precision(mass):
        one = jax.tree.map(lambda value: value[:1], x)
        prior = b._replace(
            kappa=np.full(3, mass, dtype), nu=np.full(3, mass, dtype), alpha=np.full(3, mass, dtype)
        )
        count = 4
        batch = jax.tree.map(lambda value: np.repeat(value, count, axis=0), one)
        weights = np.repeat(fixed[:1], count, axis=0)

        def observation(post, valid, intended):
            deltas = {
                name: (np.asarray(getattr(post, name), dtype=float) - mass).tolist()
                for name in ("alpha", "kappa", "nu")
            }
            exact = bool(valid and all(np.array_equal(v, intended) for v in deltas.values()))
            return dict(
                public_valid=bool(valid),
                increments=deltas,
                exact_target=intended.tolist(),
                preserves_fractional_counts=exact,
            )

        first, ok = host_arrays(compiled_update(prior, one, fixed[:1]))
        result = dict(first=observation(first, ok, fixed[0]))
        batched, ok = host_arrays(compiled_update(prior, batch, weights))
        result["one_batch"] = observation(batched, ok, weights.sum(axis=0))
        single = prior
        all_valid = True
        for _ in range(count):
            single, ok = host_arrays(compiled_update(single, one, fixed[:1]))
            all_valid &= bool(ok)
        result["repeated_singles"] = observation(single, all_valid, weights.sum(axis=0))
        exact = all(row["preserves_fractional_counts"] for row in result.values())
        return dict(
            mass=mass,
            source_spacing=float(np.spacing(dtype(mass))),
            observations=count,
            preserves_fractional_counts=exact,
            results=result,
        )

    def resolved_counts():
        result = count_precision(8192)
        assert result["preserves_fractional_counts"], result
        return result

    def translation():
        shift = np.array([10000.0, -10000.0], dtype)
        translated = b._replace(mu=b.mu + shift)
        xx = x._replace(mean=x.mean + shift)
        post, ok = host_arrays(compiled_update(translated, xx, fixed))
        control, good = host_arrays(compiled_update(b, x, fixed))
        assert ok and good
        close(post.psi, control.psi, dtype)
        close(post, host_update(translated, xx, fixed), dtype)
        s, w, ok = host_arrays(compiled_score(translated, xx))
        assert ok
        close((s, w), host_score(b, x), dtype)
        return dict(scatter_max_abs_delta=float(np.max(np.abs(post.psi - control.psi))))

    def hostile():
        cases = dict(
            empty_active=(b._replace(active=np.zeros(3, bool)), x),
            singular_scale=(b._replace(psi=np.repeat(np.ones((1, 2, 2), dtype), 3, axis=0)), x),
            invalid_dof=(b._replace(nu=np.ones(3, dtype)), x),
            invalid_alpha=(b._replace(alpha=-b.alpha), x),
            observed_nan=(b, x._replace(mean=np.full_like(x.mean, np.nan))),
            indefinite_covariance=(
                b,
                x._replace(covariance=np.repeat(np.array([[[1, 2], [2, 1]]], dtype), 2, axis=0)),
            ),
            wrong_category=(b, x._replace(categories=(np.array([0, 2], np.int32), x.categories[1]))),
        )
        for key, (bank, batch) in cases.items():
            _, _, sv = compiled_score(bank, batch)
            post, uv = host_arrays(compiled_update(bank, batch, fixed))
            assert not sv and not uv, key
            close(post, bank, dtype, factor=0)
            try:
                host(bank, batch)
            except ValueError:
                pass
            else:
                raise AssertionError("host oracle accepted " + key)
        for bad in (fixed + 1, np.full_like(fixed, np.nan), fixed[:, ::-1], -fixed):
            post, ok = host_arrays(compiled_update(b, x, bad))
            assert not ok
            close(post, b, dtype, factor=0)
        return dict(invalid_inputs=list(cases), invalid_weight_cases=4)

    for name, fn in (
        ("pinned_fractional", pinned),
        ("unequal_mass_scores", unequal_mass_scores),
        ("release_vs_additive", additive),
        ("masked_fields", missing),
        ("large_dirichlet", large),
        ("resolved_quarter_count_updates", resolved_counts),
        ("translated", translation),
        ("atomic_rejection", hostile),
    ):
        run(name, fn)
    boundary = count_precision(2**22)
    fixed_cases = all(item["passed"] for item in report.values())
    return dict(
        all_accuracy_cases_passed=fixed_cases and boundary["preserves_fractional_counts"],
        fixed_case_gates_passed=fixed_cases,
        count_resolution_at_2_to_22=boundary,
        checks=report,
        provenance=provenance(),
    )


def stress(dtype):
    import jax
    from cffg.mixed_learning import update

    compiled = jax.jit(native)
    update_one = jax.jit(update)
    rows = []
    for n, k in SHAPES:
        b, x = fixture(n, k, dtype)
        actual = host_arrays(compiled(b, x))
        expected = stress_oracle(b, x)
        close(actual, expected, dtype, factor=64)
        # The absolute allclose bound alone could miss a lost small weight.
        np.testing.assert_allclose(actual[1], expected[1], rtol=64 * np.finfo(dtype).eps, atol=0)
        if k <= 500:
            close(actual, host(b, x), dtype, factor=64)
        single = b
        if n == 32:
            for i in range(n):
                single, valid = update_one(
                    single, jax.tree.map(lambda a: a[i : i + 1], x), actual[1][i : i + 1]
                )
                assert valid, (i, n, k)
            single = host_arrays(single)
            close(single, actual[2], dtype, factor=64)
        rows.append(
            dict(
                n=n,
                k=k,
                single_batch_max_abs_scale=(
                    float(np.max(np.abs(single.psi - actual[2].psi))) if n == 32 else None
                ),
                max_abs_score=float(np.max(np.abs(actual[0][:, b.active] - expected[0][:, b.active]))),
                max_abs_weight=float(np.max(np.abs(actual[1] - expected[1]))),
                max_abs_scale=float(np.max(np.abs(actual[2].psi - expected[2].psi))),
            )
        )
    return dict(passed=True, shapes=rows, provenance=provenance())


def measure(fn, *args):
    start = time.perf_counter_ns()
    result = host_arrays(fn(*args))
    return (time.perf_counter_ns() - start) / 1e6, result


def timing_cell(n, k):
    import jax

    # Populate imports and device discovery before the first public call.
    import_module("cffg.conj")
    import_module("cffg.mixed_learning")

    metadata = provenance()
    start = time.perf_counter()
    b, x = fixture(n, k, np.float32)
    compiled = jax.jit(native)
    methods = dict(jax=compiled)
    if k <= 500:
        methods["host"] = host
    setup_s = time.perf_counter() - start
    index = SHAPES.index((n, k))
    arms = list(methods)[:: 1 if index % 2 == 0 else -1]
    cold = {}
    for arm in arms:
        cold[arm], first = measure(methods[arm], b, x)
        if arm == "jax":
            first_jax = first
        else:
            first_host = first
    if "host" in methods:
        close(first_jax, first_host, np.float32, factor=64)
    expected = stress_oracle(b, x)
    close(first_jax, expected, np.float32, factor=64)
    np.testing.assert_allclose(first_jax[1], expected[1], rtol=64 * np.finfo(np.float32).eps, atol=0)
    for _ in range(3):
        for arm in arms:
            measure(methods[arm], b, x)
    blocks = []
    for block in range(BLOCKS):
        row = {}
        for arm in arms[:: 1 if block % 2 == 0 else -1]:
            row[arm] = [measure(methods[arm], b, x)[0] for _ in range(CALLS)]
        blocks.append(row)
    summary = {}
    for arm in arms:
        calls = [t for row in blocks for t in row[arm]]
        means = [statistics.mean(row[arm]) for row in blocks]
        summary[arm] = dict(
            cold_ms=cold[arm],
            warm_median_ms=statistics.median(calls),
            block_mean_range_ms=[min(means), max(means)],
        )
    if k <= 500:
        ratios = [statistics.mean(row["jax"]) / statistics.mean(row["host"]) for row in blocks]
        samples = np.random.default_rng(910).choice(ratios, size=(10000, BLOCKS))
        ci = np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()
        median = statistics.median(ratios)
        summary["jax_over_host"] = dict(
            median=median,
            ci95=ci,
            range=[min(ratios), max(ratios)],
            material_slowdown=median > 1.2,
            ci_above_bound=ci[0] > 1.2,
        )
    return dict(
        n=n,
        k=k,
        d=6,
        setup_s=setup_s,
        first_call_order=arms,
        summary=summary,
        blocks_ms=blocks,
        input_bytes=sum(a.nbytes for a in jax.tree.leaves((b, x))),
        output_bytes=sum(a.nbytes for a in jax.tree.leaves(first_jax)),
        provenance=metadata,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "qualify", "stress", "timing", "worker"))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--shape", nargs=2, type=int, metavar=("N", "K"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        result = plan()
    elif args.stage == "timing":
        if args.dtype != "float32" or not args.output:
            parser.error("timing requires float32 and --output")
        shapes = []
        result = dict(protocol=plan(), start_utc=time.time(), shapes=shapes)
        for n, k in SHAPES:
            cmd = [sys.executable, __file__, "worker", "--shape", str(n), str(k)]
            output = subprocess.check_output(
                cmd, text=True, env=dict(os.environ, **ENV, JAX_ENABLE_X64="0")
            )
            shapes.append(json.loads(output))
            print(f"N={n} K={k} finished", file=sys.stderr, flush=True)
            args.output.with_suffix(".partial.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )
        result["end_utc"] = time.time()
    else:
        import jax

        dtype = np.dtype(args.dtype).type
        jax.config.update("jax_enable_x64", dtype is np.float64)
        if args.stage == "worker":
            result = timing_cell(*args.shape)
        elif args.stage == "qualify":
            result = qualify(dtype)
        else:
            result = stress(dtype)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n")
        if args.stage == "timing":
            args.output.with_suffix(".partial.json").unlink()
    else:
        print(payload)
    if result.get("fixed_case_gates_passed", result.get("passed")) is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
