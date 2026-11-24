"""Frame initialization, hard-winner growth and transactional failures."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.experiments.axiom_slots import (
    FrameStatus,
    _fit,
    frame_tokens,
    initial_model,
    initialize_frame,
    observe_frame,
    slot_descriptors,
)


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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
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
