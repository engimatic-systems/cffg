"""Frame initialization, hard-winner growth and transactional failures."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, softmax

from cffg.experiments.axiom_slots import (
    FrameStatus,
    _fit,
    frame_tokens,
    initial_model,
    initialize_frame,
    observe_frame,
    slot_descriptors,
)
from cffg.experiments.axiom_spatial import Positions
from cffg.experiments.axiom_tracking import DecodedSlots
from cffg.frame_regression import Regression


def same(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def fixture(dtype=jnp.float32, slots=3):
    rgb = np.zeros((5, 4, 3), np.uint8)
    rgb[1:3, 2:] = [180, 20, 40]
    model = initial_model(jax.random.key(9), dtype=dtype, slots=slots)
    return model, frame_tokens(jnp.asarray(rgb), dtype)


def test_encoding_and_seeded_template():
    rgb = jnp.array([[[0, 128, 255], [128, 0, 64]]], jnp.uint8)
    y = frame_tokens(rgb)
    np.testing.assert_array_equal(y, [[-1, -1, -1, 0, 127 / 128], [0, -1, 0, -1, -0.5]])
    m, _ = fixture()
    same(m, fixture()[0])
    np.testing.assert_array_equal(m.appearance_prior.mean[0, :2], np.eye(2, 3))
    np.testing.assert_array_equal(m.prior_counts, 1)
    with pytest.raises(ValueError):
        frame_tokens(rgb.astype(jnp.float32))


def test_forced_write_uses_only_new_row_and_restarts_from_immutable_prior():
    m, xy = fixture()
    w = jnp.zeros((len(xy), 3), dtype=xy.dtype).at[:, 0].set(1)
    w = w.at[1].set(jnp.array([0, 1, 0], xy.dtype))
    post, ok = _fit(m, xy, m.position_prior, w, new_slot=1)
    assert ok
    assert post.counts.tolist() == [1, 1001, 1]
    assert float(post.appearance.precision[1, -1, -1]) == 1000
    for i in (0, 2):
        same(
            jax.tree.map(lambda a: a[i], m.appearance),
            jax.tree.map(lambda a: a[i], post.appearance),
        )
    repeated, ok = _fit(post, xy, m.position_prior, w, new_slot=1)
    assert ok
    same(repeated, post)


@pytest.mark.parametrize("dtype", [jnp.float32, pytest.param(jnp.float64, marks=pytest.mark.slow)])
def test_initialize_observe_growth_budget_and_no_write(dtype):
    m, xy = fixture(dtype)
    trained, r = jax.jit(initialize_frame)(m, xy)
    assert r.valid
    np.testing.assert_allclose(r.assignment.sum(1), 1, rtol=1e-6)
    same(m.appearance_prior, trained.appearance_prior)
    observe = jax.jit(observe_frame, static_argnames=("max_births", "sweeps"))
    before = observe(trained, xy, r.positions, threshold=-1e7, max_births=0)
    assert before.status == FrameStatus.EXPLAINED and before.births == 0
    same(before.model, trained)
    empty = np.flatnonzero(np.asarray(before.readout.occupancy) == 0)
    assert len(empty) > 0
    refused = observe(trained, xy, r.positions, threshold=1e7, max_births=0)
    assert refused.status == FrameStatus.GROWTH_BUDGET
    grown = observe(trained, xy, r.positions, threshold=1e7, max_births=2)
    assert grown.readout.valid
    assert 1 <= int(grown.births) <= 2
    np.testing.assert_array_equal(
        grown.growth[0], [np.argmin(before.readout.best_log_likelihood), empty[0]]
    )
    assert grown.status in (FrameStatus.CAPACITY, FrameStatus.GROWTH_BUDGET)


def test_full_capacity_and_invalid_transaction():
    m, xy = fixture(slots=1)
    observe = jax.jit(observe_frame, static_argnames="max_births")
    single = observe(m, xy, m.position_prior, threshold=1e7, max_births=1)
    assert single.status == FrameStatus.CAPACITY and single.births == 0
    same(single.model, m)
    broken = observe(m, xy.at[0, 0].set(jnp.nan), m.position_prior, max_births=1)
    assert broken.status == FrameStatus.INVALID
    same(broken.model, m)


def test_forward_descriptors_match_expected_log_likelihood_quadratic():
    m, _ = fixture(dtype=jnp.float64)
    reg = m.appearance
    pos = m.position_prior
    mean, cov = slot_descriptors(m, pos)
    design = np.column_stack((pos.mean, np.ones(len(pos.mean))))
    expected = np.stack([np.asarray(w) @ x for w, x in zip(reg.mean, design, strict=True)])
    np.testing.assert_allclose(mean, expected, atol=1e-15)
    np.testing.assert_allclose(cov, [np.diag(1 / t) for t in reg.shape / reg.rate])
    same((mean, cov), slot_descriptors(m, pos._replace(cov=pos.cov * 4)))


def _oracle_positions(model, prior, previous, pixels, weights, *, keep_empty):
    """Solve a separate augmented least-squares problem for each position."""
    pos, cov = (v.copy() for v in previous)
    for i, w in enumerate(weights.T):
        n = w.sum()
        if n == 0 and keep_empty:
            continue
        beta = model.mean[i]
        whiten = np.linalg.inv(np.linalg.cholesky(prior.cov[i]))
        # Uncertainty of the affine coefficients contributes an extra quadratic penalty.
        coef_root = np.linalg.inv(np.linalg.cholesky(model.precision[i]))
        scale = np.sqrt(n * model.shape[i] / model.rate[i])
        penalty = np.sqrt(n * beta.shape[0]) * coef_root
        design = np.vstack((whiten, scale[:, None] * beta[:, :2], penalty[:, :2], [[1e-4, 1e-4]]))
        mean_pixel = w @ pixels / n if n else np.zeros(pixels.shape[1])
        target = np.r_[whiten @ prior.mean[i], scale * (mean_pixel - beta[:, 2]), -penalty[:, 2], 0.]
        pos[i] = np.linalg.lstsq(design, target, rcond=None)[0]
        cov[i] = np.linalg.inv(design.T @ design)
    return Positions(pos, cov)


def _oracle_sweep(model, prior, pos, pixels, counts, *, keep_empty):
    likelihood = np.empty((len(pixels), len(counts)))
    for i, beta in enumerate(model.mean):
        x = np.r_[pos.mean[i], 1.]
        C = np.pad(pos.cov[i], ((0, 1), (0, 1)))
        noise = np.trace(np.linalg.solve(model.precision[i], np.outer(x, x) + C))
        error = (pixels - beta @ x)**2 + np.diag(beta @ C @ beta.T)
        a, b = model.shape[i], model.rate[i]
        likelihood[:, i] = .5 * (digamma(a) - np.log(2*np.pi*b) - a/b*error - noise).sum(axis=1)
    weights = softmax(likelihood + digamma(counts), axis=1)
    return _oracle_positions(model, prior, pos, pixels, weights, keep_empty=keep_empty), weights, likelihood.max(axis=1)


def _oracle_appearance(prior, pos, pixels, weights, previous, counts0, counts, *, birth=None):
    """Conjugate fit from weighted raw observations and the fixed regression prior."""
    fit = Regression(*(v.copy() for v in previous))
    counts = counts.copy()
    for i, w in enumerate(weights.T):
        n = w.sum()
        if n == 0 or (birth is not None and i != birth):
            continue
        x = np.r_[pos.mean[i], 1.]
        C = np.pad(pos.cov[i], ((0, 1), (0, 1)))
        P = prior.precision[i]
        precision = P + n * (np.outer(x, x) + C)
        beta = np.linalg.solve(precision, P @ prior.mean[i].T + np.outer(x, w @ pixels)).T
        delta = beta - prior.mean[i]
        rss = w @ ((pixels - beta @ x)**2) + n*np.diag(beta @ C @ beta.T) + np.diag(delta @ P @ delta.T)
        fit.mean[i], fit.precision[i] = beta, precision
        fit.shape[i], fit.rate[i] = prior.shape[i] + .5*n, prior.rate[i] + .5*rss
        counts[i] = counts0[i] + (1000 if birth is not None else n)
        if birth is not None:
            fit.precision[i, -1, -1] = 1000
            fit.rate[i, :2] *= 2
    return fit, counts


@pytest.mark.parametrize("height,width,k,dtype", [
    (8, 10, 4, jnp.float32),
    pytest.param(210, 160, 32, jnp.float32, marks=pytest.mark.slow),
    pytest.param(210, 160, 32, jnp.float64, marks=pytest.mark.slow),
])
def test_observed_slots_match_independent_frame_and_growth(height, width, k, dtype):
    rgb = np.full((height, width, 3), 64, np.uint8)
    rgb[height//2:, width//2:] = 192
    rgb[height//3, width//3] = [255, 0, 255]
    # Encode the input independently, retaining its actual row-major coordinates.
    y, x = np.indices((height, width))
    pixels = np.dstack((2*x/width - 1, 2*y/height - 1, rgb.astype(float)/128 - 1)).reshape(-1, 5)
    source = initial_model(jax.random.key(3), slots=k, dtype=dtype)
    # Known previous colors and locations. Two foreground/background appearances explain
    # both rectangles; a single unanticipated magenta pixel drives the birth decision.
    beta = np.zeros((k, 5, 3))
    beta[:, :2, :2] = np.eye(2)
    beta[:, 2:, 2] = -.5
    beta[1, 2:, 2] = .5
    appearance = Regression(beta, np.tile(4*np.eye(3), (k, 1, 1)), np.ones((k, 5))*4,
                            np.tile([16, 16, .3, .3, .3], (k, 1)))
    means = np.zeros((k, 2))
    means[:2] = [[-.35, -.15], [.45, .3]]
    prediction = Positions(means, np.tile(.12*np.eye(2), (k, 1, 1)))
    counts = np.ones(k)
    counts[:2] = [30, 20]
    as_device = lambda a: jnp.asarray(a, dtype)
    model = source._replace(appearance=jax.tree.map(as_device, appearance), counts=as_device(counts))
    prior = jax.tree.map(lambda a: np.asarray(a, dtype=float), source.position_prior)
    base = jax.tree.map(lambda a: np.asarray(a, dtype=float), source.appearance_prior)
    pos = prediction
    for _ in range(2):
        pos, w, best = _oracle_sweep(appearance, prior, pos, pixels, counts, keep_empty=True)
    novel = np.ravel_multi_index((height//3, width//3), (height, width))
    assert np.argmin(best) == novel
    second_worst = np.partition(best, 1)[1]
    assert second_worst - best[novel] > 1
    observe = jax.jit(observe_frame, static_argnames=("max_births",))
    args = model, frame_tokens(jnp.asarray(rgb), dtype), jax.tree.map(as_device, prediction)
    before = observe(*args, threshold=best[novel] - 1, max_births=1)
    assert before.readout.valid and before.status == FrameStatus.EXPLAINED and before.births == 0

    def compare(result, pos, w):
        readout = result.readout
        atol = 1e-4 if dtype == jnp.float32 else 1e-9
        np.testing.assert_allclose(readout.assignment, w, rtol=atol, atol=atol)
        np.testing.assert_allclose(readout.positions.mean, pos.mean, rtol=atol, atol=atol)
        np.testing.assert_allclose(readout.positions.cov, pos.cov, rtol=atol, atol=atol)
        winners = w.argmax(axis=1)
        np.testing.assert_array_equal(np.argmax(readout.assignment, axis=1), winners)
        expected_occupancy = np.bincount(winners, minlength=k)
        np.testing.assert_array_equal(readout.occupancy, expected_occupancy)
        moments = slot_descriptors(result.model, readout.positions)
        decoded = DecodedSlots.from_readout(*moments, readout)
        np.testing.assert_array_equal(decoded.occupied, expected_occupancy > 0)
        return expected_occupancy

    occupancy = compare(before, pos, w)
    slot = np.flatnonzero(occupancy == 0)[0]
    threshold = .5 * (best[novel] + second_worst)
    after = observe(*args, threshold=threshold, max_births=1)
    assert after.readout.valid and after.births == 1
    np.testing.assert_array_equal(after.growth[0], [novel, slot])
    forced = w.copy()
    forced[novel] = np.eye(k)[slot]
    pos = _oracle_positions(appearance, prior, prior, pixels, forced, keep_empty=False)
    fit, counts = _oracle_appearance(base, pos, pixels, forced, appearance, np.ones(k), counts, birth=slot)
    pos, w, _ = _oracle_sweep(fit, prior, pos, pixels, np.ones(k), keep_empty=False)
    fit, counts = _oracle_appearance(base, pos, pixels, w, fit, np.ones(k), counts)
    pos, w, best = _oracle_sweep(fit, prior, pos, pixels, counts, keep_empty=False)
    occupied = compare(after, pos, w)
    status = FrameStatus.EXPLAINED if best.min() >= threshold else (
        FrameStatus.CAPACITY if np.all(occupied) else FrameStatus.GROWTH_BUDGET)
    assert after.status == status
