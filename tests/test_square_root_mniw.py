"""Translated regression precision, direct draws and declared rooted checkpoints."""

from dataclasses import replace
from fractions import Fraction

import jax
import numpy as np
import pytest

from cffg import (
    Engine,
    MNIW,
    SquareRootMNIW,
    compile_template,
    load_bundle,
    save_bundle,
)
from cffg import conj
from cffg.learn import FAMILIES, Learnables
from test_gaussian_learnables import _learning_template
from test_hybrid_learnables import _mixture_template


def rooted_fixture(component):
    prior = SquareRootMNIW.from_dense(MNIW(np.zeros((2, 3)), np.eye(3), np.eye(2), 4.0))
    if component:
        t, a, _ = _mixture_template()
        name, family = "neutral_dynamics", "component_sqrt_mniw"
        priors = {name: (prior,) * 3}
    else:
        t, a = _learning_template()
        name, family = "z_dynamics", "sqrt_mniw"
        priors = {name: prior, "a_state": conj.vague_niw(2)}
    t.learnables[name] = replace(
        t.learnables[name], family=family, semantic_identity=None
    )
    return t, a, priors


def rational_fit(c, n):
    # Identity prior. Each observed [y,x1,x2] has mean [2,c,c] and
    # covariance I with Cov(y,x1)=1/4. The exact column precision is a
    # diagonal matrix plus n v v.T, v=[c,c,1].
    v = [Fraction(c), Fraction(c), Fraction(1)]
    base_inv = [Fraction(1, n + 1), Fraction(1, n + 1), Fraction(1)]
    projected = [a * b for a, b in zip(base_inv, v)]
    rhs = [n * (2 * c + Fraction(1, 4)), 2 * n * c, 2 * n]
    multiplier = sum(a * b for a, b in zip(projected, rhs)) / (
        Fraction(1, n) + sum(a * b for a, b in zip(projected, v))
    )
    m = [a * b - z * multiplier for a, b, z in zip(base_inv, rhs, projected)]
    psi = 1 + 5 * n - sum(a * b for a, b in zip(m, rhs))
    return np.array(m, float), float(psi)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("c", [4096, 2**27])
def test_rooted_contrast_and_inverse_survive_checkpoint(c, dtype, tmp_path):
    prior = MNIW(np.zeros((1, 3)), np.eye(3), np.eye(1), 3.0)
    post = SquareRootMNIW.from_dense(prior)
    v = np.array([1.0, -1.0, 0.0])
    x = np.array([[2, c, c]], dtype)
    cov = np.eye(3, dtype=dtype)[None]
    cov[0, 1, 0] = cov[0, 0, 1] = 0.25
    assert post.update_centered([[np.nan] * 3], cov, [0]) is post
    for n in (1, 2):
        post = post.update_centered(x, cov, [1])
        m, psi = rational_fit(c, n)
        np.testing.assert_allclose(post.m[0], m, rtol=0, atol=8e-15)
        np.testing.assert_allclose(post.psi, psi, rtol=4e-15)
        assert post.precision_quadratic(v) == pytest.approx(2 * (n + 1), rel=3e-15)
        assert v @ post.solve_precision(v) == pytest.approx(2 / (n + 1), rel=4e-15)
        if c == 4096:
            assert v @ post.lam @ v == pytest.approx(2 * (n + 1), abs=5e-8)
        else:
            with pytest.raises(ValueError, match="dense column precision"):
                _ = post.lam
        fields = FAMILIES["sqrt_mniw"].as_params(post)
        path = tmp_path / "root.npz"
        np.savez(path, **fields)
        with np.load(path, allow_pickle=False) as saved:
            restored = SquareRootMNIW(**dict(saved))
        np.testing.assert_array_equal(post.lam_root, restored.lam_root)
        post = restored


def test_rooted_samples_use_the_source_root_and_physical_units():
    c = 2**27
    v = np.array([1.0, -1.0, 0.0])
    post = SquareRootMNIW.from_dense(conj.vague_mniw(1, 2))
    post = post.update_centered([[0, c, c]], np.eye(3)[None], [1])
    key = jax.random.key(485)
    m, cov = post.sample(key, 1024)
    # Conditional draws divided by their own row standard deviation are iid
    # standard normals after normalizing the exact column contrast.
    z = (m[:, 0, :] @ v) / np.sqrt(cov[:, 0, 0] * (2 / 1.001))
    assert abs(z.mean()) < 5 / np.sqrt(len(z))
    assert abs(np.mean(z**2) - 1) < 5 * np.sqrt(2 / len(z))
    p = SquareRootMNIW(np.zeros((1, 2)), np.diag([1e-15, 1e15]), np.eye(1), 3.0)
    np.testing.assert_allclose(p.precision_quadratic([1e15, 1e-15]), 2.0)
    np.testing.assert_allclose(p.solve_precision(np.diag([1e-30, 1e30])), np.eye(2))
    for bad in (
        np.array([[1.0, 1.0], [0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.1, 1.0]]),
        np.array([[1.0, 1.0], [0.0, np.finfo(np.float32).eps]], np.float32),
    ):
        with pytest.raises(ValueError, match="column precision root"):
            SquareRootMNIW(p.m, bad, p.psi, p.nu)


def test_opposite_offsets_and_wider_design():
    dim = 20
    v = np.zeros(dim + 1)
    v[:2] = [1.0, -1.0]
    p = SquareRootMNIW.from_dense(conj.vague_mniw(1, dim))
    before = p.precision_quadratic(v)
    for n, c in enumerate((2**27, -(2**27)), 1):
        p = p.update_centered([[0] + [c] * dim], np.eye(dim + 1)[None], [1])
        assert p.precision_quadratic(v) == pytest.approx(before + 2 * n, rel=3e-15)
    # The legacy float64 words at the first offset cannot distinguish the
    # true diagonals c*c+1 and c*c+2 from c*c.
    assert float(c * c + 1) == float(c * c + 2) == float(c * c)


@pytest.mark.parametrize("component", [False, True])
def test_declared_rooted_folds_bundle_and_next_update(tmp_path, component):
    t, a, priors = rooted_fixture(component)
    if component:
        a.update(
            input_mean=np.full(2, 2.0**27),
            matrix=np.zeros_like(a["matrix"]),
            input_cov=np.eye(2),
        )
    else:
        a.update(m0=np.full(2, 2.0**27), P0=np.eye(2), A=np.zeros((2, 2)), Q=np.eye(2))
    eng = Engine(compile_template(t, arrays=a))
    learn = Learnables(eng, priors=priors)
    state = eng.init_state()
    if not component:
        state = eng.bind_data(state, {"observation": np.array([0.4, -0.1])})
    result = eng.run(state, 100, tol=1e-9)
    result.require_converged()
    learn.accumulate(eng, result)
    name = "neutral_dynamics" if component else "z_dynamics"
    assert set(learn.params[name]) == {"m", "lam_root", "psi", "nu"}
    v = np.array([1.0, -1.0, 0.0])
    gained = learn.params[name]["nu"] - learn.initial_params[name]["nu"]
    # Both predictors have independent unit variance and the same location.
    # Every unit of evidence adds two units to this unnormalized contrast.
    q = np.sum((learn.params[name]["lam_root"] @ v) ** 2, axis=-1)
    np.testing.assert_allclose(q, 2 * (1 + gained), rtol=4e-14)

    if component:
        for field in learn.params[name]:
            np.testing.assert_array_equal(
                learn.params[name][field][-1], learn.initial_params[name][field][-1]
            )
    path = tmp_path / "root.cffgz"
    save_bundle(path, t, a, learnables=learn)
    other = load_bundle(path).learnables
    for target in (other, learn):
        target.accumulate(eng, result)
    resumed_q = np.sum((other.params[name]["lam_root"] @ v) ** 2, axis=-1)
    np.testing.assert_allclose(resumed_q, 2 * (1 + 2 * gained), rtol=4e-14)
    for k, value in learn.params[name].items():
        np.testing.assert_array_equal(other.params[name][k], value)
    np.testing.assert_array_equal(
        other.rollover(name, "matrix"), learn.rollover(name, "matrix")
    )
    for role, blob in other.write(eng, state.params).items():
        assert np.all(np.isfinite(blob)), role


def test_legacy_dense_conversion_and_unrecoverable_precision():
    prior = MNIW(np.zeros((1, 3)), np.eye(3), np.eye(1), 3.0)
    mean = np.array([[0.0, 4096.0, 4096.0]])
    cov = np.eye(3)[None]
    old = prior.update_centered(mean, cov, [1.0])
    post = SquareRootMNIW.from_dense(old).update_centered(mean, cov, [1.0])
    assert post.precision_quadratic([1.0, -1.0, 0.0]) == pytest.approx(6.0, abs=1e-9)
    # A positive unshifted Cholesky pivot alone is insufficient for migration.
    near = np.ones((3, 3)) + np.eye(3) * np.finfo(float).eps
    with pytest.raises(ValueError, match="unresolved"):
        SquareRootMNIW.from_dense(MNIW(prior.m, near, prior.psi, prior.nu))


def test_raw_update_resolves_both_dense_states_and_preserves_zero_weight():
    dense = MNIW(np.zeros((1, 3)), np.eye(3), np.eye(1), 3.0)
    rooted = SquareRootMNIW.from_dense(dense)
    zero = np.zeros((1, 3))
    blocks = (np.eye(1), zero, np.eye(3))
    got = rooted.update(*blocks, 1)
    expected = dense.update(*blocks, 1)
    for field in ("lam", "m", "psi", "nu"):
        np.testing.assert_allclose(getattr(got, field), getattr(expected, field))
    extreme = rooted.update_centered([[0., 2.**27, 2.**27]], np.eye(3)[None], [1])
    with pytest.raises(ValueError, match="dense column precision"):
        extreme.update(*blocks, 1)
    assert extreme.update(np.zeros((1, 1)), zero, np.zeros((3, 3)), 0) is extreme
    with pytest.raises(ValueError, match="zero-weight"):
        extreme.update(*blocks, 0)
    with pytest.raises(ValueError, match="real and finite"):
        rooted.update(np.full((1, 1), np.nan), zero, np.eye(3), 1)
