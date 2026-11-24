"""Independent Gaussian/Dirichlet scores and weighted NIW on all six timing fixtures."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from scipy.special import digamma, logsumexp

from cffg.mixed_learning import score, update
from online_precision_probe import visible, oracle, runtime, metrics, payload, read_file, same_tree
from online_precision_timing import SHAPES, sequence


def score_oracle(b, x):
    active = np.asarray(b.active)
    mu = np.asarray(b.mu, float)[active]
    d = mu.shape[1]
    psi = np.asarray(b.psi, float)[active]
    nu = np.asarray(b.nu, float)[active]
    kappa = np.asarray(b.kappa, float)[active]
    inv = np.linalg.inv(psi)
    log_det = (
        digamma((nu[:, None] - np.arange(d)) / 2).sum(axis=1) + d * np.log(2.0) - np.linalg.slogdet(psi)[1]
    )
    dx = x.mean[:, None, :].astype(float) - mu
    quadratic = np.einsum("nki,kij,nkj->nk", dx, inv, dx) + np.einsum(
        "kij,nji->nk", inv, x.covariance.astype(float)
    )
    ell = np.where(
        x.gaussian_observed[:, None],
        0.5 * (log_det - d * np.log(2 * np.pi) - d / kappa - nu * quadratic),
        0,
    )
    for obs, counts in zip(x.categories, b.categories, strict=True):
        head = np.asarray(counts, float)[active]
        logp = digamma(head) - digamma(head.sum(axis=1))[:, None]
        ell += np.where(obs[:, None] >= 0, logp[:, np.maximum(obs, 0)].T, 0)
    full = np.full((len(dx), len(active)), -np.inf)
    full[:, active] = ell
    return full, np.exp(full - logsumexp(full, axis=1, keepdims=True))


@jax.jit
def recurrent_valid(bank, tape):
    def step(state, xx):
        _, weights, score_ok = score(state, xx)
        updated, update_ok = update(state, xx, weights)
        return updated, score_ok & update_ok

    _, flags = jax.lax.scan(step, bank, tape)
    return flags


def run(dtype, root, legacy=None):
    compiled = jax.jit(score)
    fold = jax.jit(update)
    rows = []
    eps = np.finfo(dtype).eps
    for n, k in SHAPES:
        bank, batches = sequence(n, k, dtype)
        x = jax.tree.map(lambda a: a[-1], batches)
        s, w, sv = compiled(bank, x)
        assert sv
        s, w = np.asarray(s), np.asarray(w)
        ref_s, ref_w = score_oracle(bank, x)
        active = np.asarray(bank.active)
        assert np.isneginf(s[:, ~active]).all() and np.all(w[:, ~active] == 0)
        np.testing.assert_allclose(s[:, active], ref_s[:, active], rtol=64 * eps, atol=64 * eps)
        # Normalization error scales with absolute log-score roundoff.
        budget = 64 * eps * (1 + abs(ref_s[:, active]).max(axis=1, keepdims=True))
        if np.any(abs(w[:, active] - ref_w[:, active]) > budget * ref_w[:, active]):
            raise AssertionError("normalized weights exceed log-score roundoff budget")
        post, ok = fold(bank, x, w)
        assert ok, f"single batched update rejected at N={n}, K={k}"
        expected = oracle(bank, x, w)
        single = bank
        for i in range(n):
            single, ok = fold(single, jax.tree.map(lambda a: a[i : i + 1], x), w[i : i + 1])
            assert ok, f"fixed-assignment update {i} rejected at N={n}, K={k}"
        flags = np.asarray(recurrent_valid(bank, batches))
        rejected = np.flatnonzero(~flags).tolist()
        assert not rejected, f"recurrent updates rejected at N={n}, K={k}, ticks {rejected}"
        a = visible(post)
        for field, r in expected.items():
            np.testing.assert_allclose(a[field], r, rtol=64 * eps, atol=64 * eps)
            np.testing.assert_allclose(visible(single)[field], r, rtol=64 * eps, atol=64 * eps)
            np.testing.assert_array_equal(a[field][~active], visible(bank)[field][~active])
        rows.append(
            dict(
                n=n,
                k=k,
                recurrent_valid_ticks=int(flags.sum()),
                max_abs_score=float(abs(s[:, active] - ref_s[:, active]).max()),
                max_abs_weight=float(abs(w - ref_w).max()),
                batch_error=metrics(post, expected),
                stream_error=metrics(single, expected),
            )
        )
        if n == 32 and k == 32:
            file = root / f"learned-compact-{np.dtype(dtype).name}.npz"
            np.savez_compressed(file, **payload(post, "compact"))
            if legacy:
                source = legacy / file.name
                old = read_file(source)
                with np.load(source, allow_pickle=False) as record:
                    for key, value in visible(old).items():
                        np.testing.assert_array_equal(value, record[key])
                resumed, ok = fold(old, x, w)
                assert ok
                other = root / f"learned-exact-{np.dtype(dtype).name}.npz"
                np.savez_compressed(other, **payload(resumed, "exact"))
                assert same_tree(resumed, read_file(other))
    return dict(passed=True, rows=rows, runtime=runtime())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", choices=("float32", "float64"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--legacy-directory", type=Path)
    a = p.parse_args()
    dt = np.dtype(a.dtype).type
    jax.config.update("jax_enable_x64", dt == np.float64)
    a.output.write_text(
        json.dumps(run(dt, a.output.parent, a.legacy_directory), indent=2, allow_nan=False) + "\n"
    )
