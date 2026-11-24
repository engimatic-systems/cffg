"""Released-history oracle and observed/contextual rMM integration on injected scenes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import component_bank_device as device
from cffg.component_bank import ComponentBank, SlotStatus
from cffg.experiments.axiom_interaction import (
    InteractionConfig, interaction_batch, learn_interactions, score_context,
)
from cffg.experiments.axiom_mixture import AssignmentStatus as S, learn_frame, learn_observation
from cffg.mixed_learning import MixedBank, MixedBatch, score


def bank(active, *, d=1, heads=(), dtype=jnp.float64):
    k = len(active)
    ones = jnp.ones(k, dtype)
    return ComponentBank(MixedBank(jnp.zeros((k,d), dtype), ones,
                                   jnp.tile(jnp.eye(d, dtype=dtype), (k,1,1)),
                                   (d+3)*ones, .1*ones,
                                   tuple(jnp.ones((k, h), dtype) for h in heads), jnp.array(active)))


def points(values, dtype=jnp.float64, categories=()):
    n = len(values)
    return MixedBatch(jnp.asarray(values, dtype)[:, None], jnp.zeros((n,1,1), dtype),
                      jnp.ones(n,bool), tuple(jnp.asarray(head) for head in categories))


def same(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [jnp.float32, pytest.param(jnp.float64, marks=pytest.mark.slow)])
def test_discounted_excess_for_gaussian_and_categorical_heads(dtype):
    p = bank([True, True], heads=(2,3), dtype=dtype)
    history = points([1,1,1,1], dtype, ([0]*4, [2]*4))
    weights = jnp.array([[1,0],[1,0],[0,1],[0,1]], dtype)
    old = p.update(history, weights).bank
    datum = points([3], dtype, ([1], [0]))
    learned = learn_observation(old, datum, threshold=-100.)
    assert learned.valid and learned.status == S.MATCHED and learned.written.all()
    np.testing.assert_array_equal(learned.assignment, [.5,.5])
    # Per component: old records x=1,1 and new x=3 contribute .5 each.
    post = learned.bank.posterior
    np.testing.assert_allclose(post.alpha, [1.6,1.6], rtol=2e-7)
    np.testing.assert_allclose(old.posterior.alpha, [2.1,2.1], rtol=2e-7)
    np.testing.assert_allclose(post.kappa, 2.5)
    np.testing.assert_allclose(post.nu, 5.5)
    np.testing.assert_allclose(post.mu[:,0] * post.kappa, 2.5)
    np.testing.assert_allclose(post.psi[:,0,0] - 1 + post.kappa * post.mu[:,0]**2, 5.5)
    np.testing.assert_allclose(post.categories[0], [[2,1.5]]*2)
    np.testing.assert_allclose(post.categories[1], [[1.5,1,2]]*2)
    assert learned.bank.logical_ids == old.logical_ids
    np.testing.assert_allclose(old.update(datum, jnp.array([[.5,.5]], dtype)).bank.posterior.alpha, 2.6)


def test_winner_history_flag_and_strict_write_thresholds():
    datum = points([1])
    source = bank([True,True])
    old = source.posterior._replace(alpha=jnp.array([1.,5.1]))
    source = source.replace_posterior(old, previous=source.posterior).bank
    # Likelihood tie selects winner 0 with old alpha exactly 1, resetting history for both.
    result = learn_observation(source, datum, threshold=-100.)
    np.testing.assert_allclose(result.bank.posterior.alpha, [.6,.6])
    four = bank([True]*4)
    tie = learn_observation(four, datum, threshold=-100.)
    assert tie.valid and (tie.assignment == .25).all() and not tie.written.any()
    same(tie.bank.posterior, four.posterior)
    full = bank([True])
    full = full.replace_posterior(full.posterior._replace(alpha=jnp.array([31.])),
                                  previous=full.posterior).bank
    capped = learn_observation(full, datum, threshold=100.)
    assert capped.status == S.FULL_TO_ZERO and capped.valid and not capped.written.any()
    same(capped.bank.posterior, full.posterior)


def test_empty_identity_bank_ordered_birth_full_update_and_checkpoint():
    base = bank([False,False], dtype=jnp.float32)
    tape = points([1.,2.,3.,4.], jnp.float32)
    mask = jnp.array([True,False,True,True])
    out = learn_frame(device.from_host(base), tape, mask, threshold=100.)
    assert out.valid
    np.testing.assert_array_equal(out.born, [0,-1,1,-1])
    np.testing.assert_array_equal(out.status, [S.NOVEL,S.SKIPPED,S.NOVEL,S.FULL_TO_ZERO])
    np.testing.assert_array_equal(out.assignment, [[1,0],[0,0],[0,1],[1,0]])
    # The final novelty updates live row 0 at capacity and retains its identity.
    host = base
    for i in [0,2,3]:
        result = learn_observation(host, jax.tree.map(lambda a: a[i:i+1], tape), threshold=100.)
        assert result.valid
        host = result.bank
    assert host.logical_ids == (1,2) and host.generations == (1,1)
    same(out.bank, device.from_host(host))
    restored = ComponentBank.from_state_dict(host.state_dict(exact=True))
    same(device.from_host(restored), out.bank)
    broken = tape._replace(mean=tape.mean.at[-1,0].set(jnp.nan))
    invalid = learn_frame(device.from_host(base), broken, mask, threshold=100.)
    assert not invalid.valid and invalid.status[-1] == S.INVALID
    assert not invalid.written.any() and (invalid.born == -1).all()
    same(invalid.bank, device.from_host(base))


def test_scene_output_heads_change_training_assignment_but_are_hidden_in_prediction():
    cfg = InteractionConfig(actions=2, identities=2, motions=2)
    raw = bank([True,True], d=7, heads=(3,3,2,2,3,2), dtype=jnp.float32)
    x = raw.posterior
    heads = (*x.categories[:-2], jnp.array([[1,1,9],[9,1,1]], jnp.float32),
             jnp.array([[9,1],[1,9]], jnp.float32))
    result = raw.replace_posterior(x._replace(categories=heads), previous=x)
    assert result.status == SlotStatus.OK
    host = result.bank
    scene = jnp.array([[0,0,0,0,0,0,.1,.1], [.5,0,0,0,0,0,.1,.1]], jnp.float32)
    tracked = jnp.array([False,True])
    args = scene, jnp.array([0,1]), tracked, 1, jnp.array([1.201,1.207], jnp.float32)
    context = interaction_batch(*args, config=cfg)
    observed = interaction_batch(*args, config=cfg, reward=1, switches=jnp.array([0,0]))
    _, q, ok = score_context(host.posterior, context)
    assert ok and (q == .5).all()
    assert not score_context(host.posterior, observed)[-1]
    pred = score(host.posterior, observed.records)[1]
    assert pred[1,0] > .99
    learned = learn_interactions(device.from_host(host), observed, tracked, threshold=-1000.)
    np.testing.assert_allclose(learned.assignment[1], pred[1], rtol=1e-6)
    assert learned.valid and learned.written[1,0] and not learned.written[1,1]
    leaked = learn_interactions(device.from_host(host), context, tracked)
    assert not leaked.valid
    same(leaked.bank, device.from_host(host))


def test_ordered_32_slot_frame_matches_independent_natural_statistics():
    # Two identically initialized components keep equal likelihoods. An external
    # oracle can therefore propagate all raw statistics without calling a scorer.
    base = bank([True,True], d=2, heads=(2,), dtype=jnp.float32)
    alpha = base.posterior._replace(alpha=jnp.full(2, 2.1, jnp.float32))
    start = base.replace_posterior(alpha, previous=base.posterior).bank
    coords = np.array([[.2*i,(-1)**i*.125*i] for i in range(32)], np.float32)
    flags = np.arange(32) % 5 != 0
    labels = np.arange(32) % 2
    inputs = MixedBatch(jnp.array(coords), jnp.zeros((32,2,2), jnp.float32),
                        jnp.ones(32, bool), (jnp.array(labels),))
    result = learn_frame(device.from_host(start), inputs, jnp.asarray(flags), threshold=-1000.)
    assert result.valid and (np.asarray(result.written).any(axis=1) == flags).all()
    mass, excess_alpha = 0., 2.
    sx, sxx, cats = np.zeros(2), np.zeros((2,2)), np.zeros(2)
    for x, j, take in zip(coords.astype(float), labels, flags, strict=True):
        if take:
            h = .5 if .1 + excess_alpha > 1 else 0.
            mass = h*mass + .5
            excess_alpha = h*excess_alpha + .5
            sx = h*sx + .5*x
            sxx = h*sxx + .5*np.outer(x,x)
            cats = h*cats + .5*np.eye(2)[j]
    p = result.bank.posterior
    np.testing.assert_allclose(p.mu[0], sx / (1+mass), atol=2e-6)
    np.testing.assert_allclose(p.psi[0], np.eye(2)+sxx-np.outer(sx,sx)/(1+mass), atol=1e-5)
    np.testing.assert_allclose(p.categories[0][0], 1+cats, atol=2e-6)
    np.testing.assert_allclose(p.alpha[0], .1+excess_alpha, atol=2e-6)
