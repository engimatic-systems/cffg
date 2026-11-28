"""Atomic soft updates, missing fields, and resuming a mixed component bank."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import mixed_learning as mixed


def fixture(dtype):
    def f(values):
        return jnp.asarray(values, dtype=dtype)

    bank = mixed.MixedBank(
        f([[1, -1]] * 3), f([2] * 3), f([[[2, 1], [1, 2]]] * 3), f([4] * 3),
        f([2, 5, .1]), (f([[2, 3]] * 3), f([[1, 2, 1]] * 3)),
        jnp.array([True, True, False]),
    )
    batch = mixed.MixedBatch(
        f([[2, 0], [0, 0]]), f([[[0, 0], [0, 0]], [[.5, -.25], [-.25, .5]]]),
        jnp.array([True, True]), (jnp.array([0, 1]), jnp.array([-1, 1])),
    )
    return bank, batch, f([[.25, .75, 0], [.5, .5, 0]])


def same_bank(left, right):
    assert left._fields == right._fields and len(left.categories) == len(right.categories)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_independent_fractional_posterior_and_resume(tmp_path, dtype):
    bank, batch, weights = fixture(dtype)
    update = jax.jit(mixed.update)
    full, valid = update(bank, batch, weights)
    assert valid
    tol = 8 * np.finfo(dtype).eps
    np.testing.assert_allclose(full.mu[:2], [[10/11, -8/11], [14/13, -8/13]], rtol=tol)
    expected = np.array([np.array([[262, 61], [61, 246]]) / 88,
                         np.array([[362, 107], [107, 314]]) / 104])
    np.testing.assert_allclose(full.psi[:2], expected, rtol=tol)
    np.testing.assert_array_equal(full.alpha, bank.alpha + jnp.asarray([.75, 1.25, 0], dtype))
    np.testing.assert_array_equal(full.kappa, [2.75, 3.25, 2])
    np.testing.assert_array_equal(full.nu, [4.75, 5.25, 4])
    np.testing.assert_array_equal(full.categories[0], [[2.25, 3.5], [2.75, 3.5], [2, 3]])
    np.testing.assert_array_equal(full.categories[1], [[1, 2.5, 1], [1, 2.5, 1], [1, 2, 1]])

    first, valid = update(bank, jax.tree.map(lambda x: x[:1], batch), weights[:1])
    assert valid
    path = tmp_path / "posterior.npz"
    np.savez(path, **first.state_dict(exact=True))
    with np.load(path, allow_pickle=False) as state:
        resumed = mixed.MixedBank.from_state_dict(state)
    same_bank(resumed, first)
    tail = jax.tree.map(lambda x: x[1:], batch)
    split, valid = update(resumed, tail, weights[1:])
    assert valid
    same_bank(split, update(first, tail, weights[1:])[0])
    for a, b in zip(jax.tree.leaves(split), jax.tree.leaves(full), strict=True):
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


def test_each_field_updates_only_when_observed():
    bank, batch, weights = fixture(jnp.float32)
    batch = batch._replace(
        mean=jnp.full_like(batch.mean, jnp.nan), covariance=jnp.full_like(batch.covariance, jnp.nan),
        gaussian_observed=jnp.array([False, False]),
        categories=(jnp.array([1, -1]), jnp.array([-1, -1])),
    )
    result, valid = mixed.update(bank, batch, weights)
    assert valid
    np.testing.assert_array_equal(result.alpha, bank.alpha + weights[0])
    np.testing.assert_array_equal(result.categories[0][:, 1], bank.categories[0][:, 1] + weights[0])
    np.testing.assert_array_equal(result.categories[1], bank.categories[1])
    for field in ("mu", "psi", "nu", "kappa"):
        np.testing.assert_array_equal(getattr(result, field), getattr(bank, field))
    empty = jax.tree.map(lambda x: x[:0], batch)
    result, valid = mixed.update(bank, empty, weights[:0])
    assert valid
    same_bank(result, bank)


def test_update_rejects_invalid_batch_without_partial_commit():
    bank, batch, weights = fixture(jnp.float32)
    update = jax.jit(mixed.update)
    invalid = [
        (batch, weights.at[1, 1].set(jnp.nan)),
        (batch, weights.at[0].set(jnp.array([0., .5, .5], weights.dtype))),
        (batch, weights.at[0].set(jnp.array([0., .5, 0.], weights.dtype))),
        (batch._replace(categories=(jnp.array([0, 3]), batch.categories[1])), weights),
        (batch._replace(covariance=batch.covariance.at[1, 0, 0].set(-1.)), weights),
    ]
    for observations, w in invalid:
        result, valid = update(bank, observations, w)
        assert not valid
        same_bank(result, bank)


def test_restore_rejects_invalid_snapshot():
    bank, _, _ = fixture(jnp.float32)
    for extra in ({"version": np.int32(3)}, {"category_9": np.ones(1)},
                  {"psi": np.zeros_like(bank.psi)}):
        with pytest.raises(ValueError):
            mixed.MixedBank.from_state_dict({**bank.state_dict(), **extra})


@pytest.mark.parametrize(("width", "indices", "valid"), [
    (256, np.array([-1, 127], dtype=np.int8), True),
    (257, np.array([0, 255], dtype=np.uint8), True),
    (256, np.array([-2, 0], dtype=np.int8), False),
    (255, np.array([0, 255], dtype=np.uint8), False),
])
def test_narrow_category_indices_score_and_update_only_observed_bins(width, indices, valid):
    bank, batch, weights = fixture(jnp.float32)
    head = jnp.broadcast_to(jnp.arange(1, width + 1, dtype=bank.mu.dtype), (len(bank.mu), width))
    bank = bank._replace(categories=(head,))
    batch = batch._replace(gaussian_observed=jnp.array([False, False]),
                           categories=(jnp.asarray(indices),))
    score = jax.jit(mixed.score)
    scores, _, accepted = score(bank, batch)
    assert bool(accepted) == valid
    reference, _, _ = score(bank, batch._replace(categories=(jnp.asarray(indices, jnp.int32),)))
    np.testing.assert_array_equal(scores, reference)
    updated, accepted = jax.jit(mixed.update)(bank, batch, weights)
    assert bool(accepted) == valid
    counts = np.zeros_like(head)
    if valid:
        for row, index in enumerate(indices):
            if int(index) >= 0:
                counts[:, index] += weights[row]
    same_bank(updated, bank._replace(categories=(head + counts,),
                                     alpha=bank.alpha + counts.sum(axis=1)))


def test_snapshot_checks_ordered_head_inventory_and_reads_legacy():
    bank, _, _ = fixture(jnp.float32)
    state = bank.state_dict()
    assert state["version"] == 2
    np.testing.assert_array_equal(state["category_cardinalities"], [2, 3])
    legacy = {k: v for k, v in state.items() if k != "category_cardinalities"}
    legacy["version"] = np.int32(1)
    same_bank(mixed.MixedBank.from_state_dict(legacy), bank)
    for missing in ("category_0", "category_1", "category_cardinalities"):
        with pytest.raises(ValueError, match="missing"):
            mixed.MixedBank.from_state_dict({k: v for k, v in state.items() if k != missing})
    for widths in ([3, 2], [2], [2, 0], [[2, 3]], [2., 3.]):
        with pytest.raises(ValueError):
            mixed.MixedBank.from_state_dict({**state, "category_cardinalities": np.array(widths)})
    with pytest.raises(ValueError, match="shape"):
        mixed.MixedBank.from_state_dict({**state, "category_1": state["category_1"][:, :-1]})
    empty = bank._replace(categories=())
    same_bank(mixed.MixedBank.from_state_dict(empty.state_dict()), empty)


def test_compact_and_exact_snapshots_have_distinct_continuation_contracts(tmp_path):
    bank, _, _ = fixture(jnp.float32)
    # Represent a quarter vote on a large count. The visible high word is unchanged.
    bank = bank._replace(alpha=bank.alpha.at[0].set(2**22))
    live = bank._replace(carry=bank.carry._replace(alpha=bank.carry.alpha.at[0].set(.25)))
    plain = live.state_dict()
    exact = live.state_dict(exact=True)
    assert plain["version"] == 2 and exact["version"] == 3
    assert set(plain) == set(bank.state_dict())
    assert sum(v.nbytes for v in plain.values()) == sum(v.nbytes for v in bank.state_dict().values())
    assert not any(key.startswith("carry_") for key in plain)
    np.testing.assert_array_equal(plain["alpha"], bank.alpha)
    same_bank(mixed.MixedBank.from_state_dict(plain), bank)
    path = tmp_path / "exact.npz"
    np.savez(path, **exact)
    with np.load(path, allow_pickle=False) as values:
        same_bank(mixed.MixedBank.from_state_dict(values), live)
    # A compact save must leave the data that will drive the next update intact.
    assert float(live.carry.alpha[0]) == .25
    exact["carry_alpha"][0] = 0
    assert float(live.carry.alpha[0]) == .25
    for key in ["carry_mu", "carry_psi", "carry_category_1"]:
        with pytest.raises(ValueError, match="missing"):
            mixed.MixedBank.from_state_dict({k: v for k, v in exact.items() if k != key})


def test_checkpoint_checks_host_dtypes_before_jax_conversion():
    with jax.enable_x64(False):
        bank, _, _ = fixture(jnp.float32)
        wide = np.asarray(bank.alpha, dtype=np.float64)
        with pytest.raises(TypeError, match="alpha must have dtype"):
            bank._replace(alpha=wide).state_dict()
        with pytest.raises(ValueError, match="saved dtype"):
            mixed.MixedBank.from_state_dict({**bank.state_dict(), "alpha": wide})


def test_checkpoint_checks_carry_on_dormant_components():
    bank, _, _ = fixture(jnp.float32)
    low = bank.carry.alpha.at[2].set(jnp.nan)
    invalid = bank._replace(carry=bank.carry._replace(alpha=low))
    for exact in (False, True):
        with pytest.raises(ValueError, match="invalid mixed-learning snapshot"):
            invalid.state_dict(exact=exact)
    snapshot = bank.state_dict(exact=True)
    with pytest.raises(ValueError, match="invalid mixed-learning snapshot"):
        mixed.MixedBank.from_state_dict({**snapshot, "carry_alpha": np.asarray(low)})


def test_checkpoint_preserves_eager_scale_validation_at_rank_boundary():
    from cffg.mixed_scoring import validate_bank

    with jax.enable_x64(True):
        bank, _, _ = fixture(jnp.float64)
        source = bank.state_dict()
        # Scaling this nearly rank-one matrix to correlation units and fusing
        # the subsequent Cholesky check can change the sign of its last pivot.
        matrix = np.array([[120.05345625225891, -9.974864878360373],
                           [-9.974864878360373, .8287802154773474]])
        for direction in (-np.inf, matrix[1, 1], np.inf):
            value = matrix.copy()
            value[1, 1] = np.nextafter(value[1, 1], direction)
            candidate = bank._replace(psi=bank.psi.at[2].set(value))
            snapshot = {**source, "psi": np.asarray(candidate.psi)}
            if validate_bank(candidate):
                np.testing.assert_array_equal(candidate.state_dict()["psi"], snapshot["psi"])
                same_bank(mixed.MixedBank.from_state_dict(snapshot), candidate)
            else:
                with pytest.raises(ValueError, match="invalid mixed-learning snapshot"):
                    candidate.state_dict()
                with pytest.raises(ValueError, match="invalid mixed-learning snapshot"):
                    mixed.MixedBank.from_state_dict(snapshot)


def test_rounded_handoff_scores_primary_values_and_resumes_in_selected_mode():
    bank, batch, weights = fixture(jnp.float32)
    # A small scale makes a half-ULP carried mean observable in the odds.
    h = np.spacing(np.float32(4096))
    mean = jnp.full_like(bank.mu, 4096)
    scale = jnp.broadcast_to(jnp.eye(2, dtype=mean.dtype)*h**2, bank.psi.shape)
    bank = bank._replace(mu=mean, psi=scale)
    bank = bank._replace(carry=bank.carry._replace(mu=bank.carry.mu.at[0, 0].set(h/2)))
    batch = batch._replace(mean=mean[:2].at[:, 0].add(h), covariance=jnp.zeros_like(batch.covariance),
                           categories=tuple(jnp.full_like(index, -1) for index in batch.categories))
    rounded = bank.as_rounded()
    _, live_weights, live_ok = mixed.score(bank, batch)
    _, primary_weights, ok = jax.jit(mixed.score)(rounded, batch)
    assert live_ok and ok and np.all(np.asarray(live_weights)[:, 0] > .5)
    np.testing.assert_array_equal(primary_weights, [[.5, .5, 0], [.5, .5, 0]])

    compact, exact = bank.state_dict(), bank.state_dict(exact=True)
    legacy = {k: v for k, v in compact.items() if k != "category_cardinalities"}
    legacy["version"] = np.int32(1)
    expected, ok = mixed.update(rounded, batch, weights)
    assert ok and isinstance(expected, mixed.RoundedBank)
    for snapshot in (legacy, compact, exact):
        restored = mixed.RoundedBank.from_state_dict(snapshot)
        same_bank(rounded, restored)
        resumed, ok = mixed.update(restored, batch, weights)
        assert ok
        same_bank(expected, resumed)
    for saved in (expected.state_dict(), expected.state_dict(exact=True)):
        assert saved["version"] == 2 and "carry_mu" not in saved
        same_bank(mixed.RoundedBank.from_state_dict(saved), expected)
        same_bank(mixed.MixedBank.from_state_dict(saved), expected.as_compensated())
    with pytest.raises(ValueError, match="invalid mixed-learning snapshot"):
        mixed.RoundedBank.from_state_dict({**exact, "carry_mu": np.full_like(exact["mu"], np.nan)})


def test_rounded_update_masks_missing_fields_and_rejects_invalid_weights_under_vmap():
    bank, batch, weights = fixture(jnp.float32)
    rounded = bank.as_rounded()
    empty = jax.tree.map(lambda v: v[:0], batch)
    seen, ok = mixed.update(rounded, empty, weights[:0])
    assert ok
    same_bank(seen, rounded)
    batch = batch._replace(gaussian_observed=jnp.zeros(2, bool), mean=jnp.full_like(batch.mean, jnp.nan),
                           covariance=jnp.full_like(batch.covariance, jnp.nan))
    mapped = jax.jit(jax.vmap(mixed.update, in_axes=(None, None, 0)))
    result, status = mapped(rounded, batch, jnp.stack([weights, weights.at[0, 0].set(jnp.nan)]))
    np.testing.assert_array_equal(status, [True, False])
    same_bank(jax.tree.map(lambda a: a[1], result), rounded)
    np.testing.assert_array_equal(result.psi[0], rounded.psi)
    np.testing.assert_array_equal(result.categories[0][0, :, 0], rounded.categories[0][:, 0]+weights[0])
