"""Public D6d resample–move posterior, absolute Gaussian evidence, and restart."""

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import multivariate_normal, norm

from cffg import Engine, FunctionSpec, MeanImplementation, compile_template, register_mean_function
from cffg.finite_hypotheses import FiniteHypotheses, _digest
from cffg.static_parameter_particles import ParticleSupportError, StaticParameterParticles, _weight_update


FUNCTION = FunctionSpec("cffg.test.particle_quadratic", "1.0.0")
VARIANCE = .15**2
READINGS = ((1., .63), (1.2, .68), (1., None), (.8, .51), (1.05, .58), (1.1, .61))


def _shape(di, do, configuration):
    assert di == do == 1 and not configuration
    return (1,)


def _mean(x, theta, configuration):
    return x * theta[0] ** 2


register_mean_function(MeanImplementation(FUNCTION.impl_id, FUNCTION.version, _shape, _mean))


def bank(seed=319, *, dtype=np.float64, n=32, variance=VARIANCE, **options):
    return StaticParameterParticles(FUNCTION, [[-1., 1.]], [[2., 2.]], [[variance]], input_dim=1,
                                    n_particles=n, key=jax.random.key(seed), dtype=dtype, **options)


def snapshot(bank, path):
    bank.save(path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        return meta, {name: archive[name].copy() for name in archive.files if name != "metadata"}


def assert_snapshot(bank, path, previous):
    meta, arrays = snapshot(bank, path)
    assert meta == previous[0]
    for name, array in arrays.items():
        np.testing.assert_array_equal(array, previous[1][name])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_no_move_posterior_matches_independent_likelihood(dtype):
    learner = bank(dtype=dtype, ess_fraction=.01, move_steps=0)
    reference = bank(dtype=dtype, ess_fraction=.01, move_steps=0)
    theta = learner.particles[:, 0]
    assert theta.dtype == dtype
    likelihoods = np.zeros(len(theta))
    total_increment = 0.
    tol = 64 * np.finfo(dtype).eps
    for x, y in READINGS:
        outcome = learner.update([x], None if y is None else [y])
        with jax.disable_jit():
            eager = reference.update([x], None if y is None else [y])
        assert outcome.ancestors is None and not outcome.accepted
        assert outcome.log_evidence_increment == pytest.approx(eager.log_evidence_increment, abs=tol)
        if y is not None:
            likelihoods += norm.logpdf(y, loc=x * theta.astype(np.float64)**2, scale=math.sqrt(VARIANCE))
        total_increment += outcome.log_evidence_increment
        joint = likelihoods - math.log(len(theta))
        normalizer = logsumexp(joint)
        np.testing.assert_allclose(learner.log_weights, joint - normalizer, atol=tol, rtol=tol)
        assert learner.log_evidence == pytest.approx(normalizer, abs=tol, rel=tol)
        assert learner.log_evidence == pytest.approx(total_increment, abs=tol)
    np.testing.assert_array_equal(learner.particles[:, 0], theta)
    assert learner.n_observed == 5


def test_correlated_reading_with_vector_parameters():
    function = FunctionSpec("cffg.mean.affine", "1.0.0")
    covariance = [[.75, .3], [.3, .45]]
    x, y = np.array([1.2, -.5]), np.array([-.25, .7])
    bounds = [[-1., 1.], [0., 2.], [-.5, 2.], [-1., 3.], [-4., 1.], [.3, .9]]
    learner = StaticParameterParticles(function, bounds, [[2., 3.]] * 6, covariance,
                                       input_dim=2, n_particles=16, key=jax.random.key(51),
                                       ess_fraction=.01, move_steps=0, dtype=np.float64)
    theta = learner.particles
    means = theta[:, :4].reshape(-1, 2, 2) @ x + theta[:, 4:]
    scores = np.array([multivariate_normal.logpdf(y, m, covariance) for m in means])
    result = learner.update(x, y)
    assert result.ancestors is None
    assert result.log_evidence_increment == pytest.approx(logsumexp(scores) - math.log(len(theta)))
    np.testing.assert_allclose(learner.log_weights, scores - logsumexp(scores), rtol=0, atol=1e-13)


@pytest.mark.parametrize("schedule", ["two_pass", "flood"])
def test_one_particle_native_gaussian_density(schedule):
    learner = bank(n=1, move_steps=0)
    theta = learner.particles[:, 0]
    mean = 1.2 * theta ** 2
    template, arrays = FiniteHypotheses([[theta.item()]], [[VARIANCE]]).conditional_model(0, mean)
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    state = engine.bind_data(engine.init_state(), {"observation": [0.4]})
    solved = engine.run(state, 20, tol=1e-9).require_converged()
    outcome = learner.update([1.2], [0.4])
    assert outcome.log_evidence_increment == pytest.approx(-float(engine.bfe(solved)), abs=1e-12)


def test_high_ess_moves_use_complete_history_and_transformed_prior(tmp_path):
    learner = bank(ess_fraction=.01)
    replay = []
    different_without_history = 0
    for i, (x, y) in enumerate(READINGS):
        if y is None:
            continue
        path = tmp_path / f"before-{i}.npz"
        meta, arrays = snapshot(learner, path)
        z = arrays["z"]
        key = jax.random.wrap_key_data(arrays["key"], impl=meta["key_impl"])
        _, _, move_key = jax.random.split(key, 3)
        proposal_key, accept_key = jax.random.split(jax.random.split(move_key, 1)[0])
        proposed = z + np.asarray(jax.random.normal(proposal_key, z.shape, dtype=jnp.float64))
        log_u = np.log(np.asarray(jax.random.uniform(accept_key, (len(z),), dtype=jnp.float64)))
        old_theta, new_theta = (2 * expit(a[:, 0]) - 1 for a in (z, proposed))
        prior = lambda a: 2 * (log_expit(a[:, 0]) + log_expit(-a[:, 0]))
        prior_delta = prior(proposed) - prior(z)
        replay.append((x, y))
        likelihood_delta = np.stack([
            norm.logpdf(yy, xx * new_theta**2, math.sqrt(VARIANCE))
            - norm.logpdf(yy, xx * old_theta**2, math.sqrt(VARIANCE))
            for xx, yy in replay
        ])
        accept = log_u < prior_delta + likelihood_delta.sum(axis=0)
        different_without_history += np.count_nonzero(accept != (log_u < prior_delta + likelihood_delta[-1]))
        outcome = learner.update([x], [y])
        assert outcome.ancestors is None and outcome.accepted == (int(accept.sum()),)
        _, after = snapshot(learner, tmp_path / f"after-{i}.npz")
        np.testing.assert_array_equal(after["z"], np.where(accept[:, None], proposed, z))
    assert different_without_history > 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_checkpoint_and_resampling_restart_across_history_growth(tmp_path, dtype):
    learner = bank(dtype=dtype, ess_fraction=1., move_steps=2)
    path = tmp_path / "state.npz"
    meta, arrays = snapshot(learner, path)
    key = jax.random.wrap_key_data(arrays["key"], impl=meta["key_impl"])
    _, resample_key, _ = jax.random.split(key, 3)
    theta = learner.particles[:, 0].astype(np.float64)
    scores = norm.logpdf(.68, theta**2, math.sqrt(VARIANCE))
    cdf = np.cumsum(np.exp(scores - logsumexp(scores)))
    positions = (np.arange(len(theta)) + np.asarray(
        jax.random.uniform(resample_key, (len(theta),), dtype=jnp.float64)
    )) / len(theta)
    expected_ancestors = tuple(np.count_nonzero(positions[:, None] >= cdf, axis=1).tolist())
    initial = learner.update([1], [.68])
    assert initial.ancestors == expected_ancestors
    for tick in range(7):
        learner.update([1], [.6 + tick / 100])
    assert learner.n_observed == 8
    snapshot(learner, path)
    resumed = StaticParameterParticles.load(path)
    for x, y in READINGS:
        obs = None if y is None else [y]
        assert learner.update([x], obs) == resumed.update([x], obs)
    assert_snapshot(resumed, path, snapshot(learner, tmp_path / "uninterrupted.npz"))


def test_missing_inputs_and_overflow_fail_without_consuming_an_event_or_key(tmp_path):
    learner = bank(dtype=np.float32)
    learner.update([1], [.6])
    path = tmp_path / "atomic.npz"
    expected = snapshot(learner, path)
    assert learner.update([np.nan], None).log_evidence_increment == 0.
    with pytest.raises(ValueError, match="observation must be finite"):
        learner.update([1.], [np.nan])
    with pytest.raises(FloatingPointError, match="Gaussian likelihood"):
        learner.update([1e20], [0.5])
    assert_snapshot(learner, path, expected)

    unrepresentable = bank(dtype=np.float32, proposal_scale=1000.)
    expected = snapshot(unrepresentable, path)
    with pytest.raises(FloatingPointError, match="representable interior"):
        unrepresentable.update([1], [.6])
    assert_snapshot(unrepresentable, path, expected)


def test_zero_weight_mass_is_distinct_from_invalid_gaussian_evaluation():
    logs = np.log([.5, .5])
    scores = np.array([0., -np.inf])
    updated, evidence = _weight_update(logs, scores)
    np.testing.assert_array_equal(updated, [0., -np.inf])
    assert evidence == math.log(.5)
    with pytest.raises(ParticleSupportError, match="sampled population"):
        _weight_update(updated, np.array([-np.inf, 0.]))
    with pytest.raises(FloatingPointError, match="invalid log likelihood"):
        _weight_update(updated, np.array([0., np.inf]))


def test_rejects_invalid_target_configuration():
    with pytest.raises(ValueError, match="positive definite"):
        bank(variance=0.)
    common = dict(input_dim=1, n_particles=8, key=jax.random.key(7))
    with pytest.raises(ValueError, match="representable open finite interval"):
        StaticParameterParticles(FUNCTION, [[1., 1.]], [[2., 2.]], [[1.]], **common)
    with pytest.raises(ValueError, match="positive beta shapes"):
        StaticParameterParticles(FUNCTION, [[0., 1.]], [[0., 2.]], [[1.]], **common)
    with pytest.raises(ValueError, match="registered FunctionSpec"):
        StaticParameterParticles(_mean, [[0., 1.]], [[2., 2.]], [[1.]], **common)


@pytest.mark.parametrize("mutation", ["legacy", "digest", "key", "function", "covariance", "history", "dtype"])
def test_failed_restore_checks_target_and_carry(tmp_path, mutation):
    learner = bank()
    learner.update([1], [.6])
    path = tmp_path / "checkpoint.npz"
    meta, arrays = snapshot(learner, path)
    if mutation == "legacy":
        meta["version"] = 0
    elif mutation == "digest":
        arrays["z"][0] += .01
    elif mutation == "key":
        arrays["key"] = arrays["key"].astype(np.int64)
    elif mutation == "function":
        meta["function"]["version"] = "99.0.0"
    elif mutation == "covariance":
        arrays["covariance"][0, 0] = 0.
    elif mutation == "dtype":
        meta["dtype"] = None
    else:
        arrays["inputs"][0, 0] = np.inf
    if mutation != "digest":
        del meta["sha256"]
        meta["sha256"] = _digest(meta, arrays)
    np.savez(path, **arrays, metadata=json.dumps(meta))
    with pytest.raises(ValueError):
        StaticParameterParticles.load(path)


def _gauss_oracle():
    def integrate(a, b):
        nodes, weights = np.polynomial.legendre.leggauss(64)
        theta = (a + b + (b - a) * nodes) / 2
        log_likelihood = sum(norm.logpdf(y, loc=x * theta**2, scale=math.sqrt(VARIANCE))
                             for x, y in READINGS if y is not None)
        mass = (b - a) / 2 * weights * .75 * (1 - theta**2) * np.exp(log_likelihood)
        return np.array([sum(mass), mass @ theta, mass @ (theta**2)])

    left, right = integrate(-1, 0), integrate(0, 1)
    evidence, first, second = left + right
    return np.array([evidence, first / evidence, second / evidence, right[0] / evidence])


@pytest.mark.slow
@pytest.mark.parametrize(("dtype", "n"), [(np.float32, 128), (np.float64, 1280)])
def test_sampled_gaussian_evidence_and_bimodal_moments(dtype, n):
    estimates = []
    resampled = 0
    for seed in range(20261128, 20261176):
        # The two modes are about four logit units apart. Fix a wider walk
        # and multiple attempts on each branch rather than adapting to ESS.
        learner = bank(seed, dtype=dtype, n=n, ess_fraction=.8, proposal_scale=2.5, move_steps=4)
        for x, y in READINGS:
            outcome = learner.update([x], None if y is None else [y])
            resampled += outcome.ancestors is not None
        weights = np.exp(learner.log_weights)
        theta = learner.particles[:, 0].astype(np.float64)
        estimates.append([math.exp(learner.log_evidence), weights @ theta,
                          weights @ theta**2, weights @ (theta > 0)])
    assert 0 < resampled < 48 * 5
    estimates = np.asarray(estimates)
    target = _gauss_oracle()
    error = np.abs(estimates.mean(axis=0) - target)
    stderr = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))
    print(dtype.__name__, n, "target", target, "error", error, "standard error", stderr, "resamples", resampled)
    assert np.all(error <= 6 * stderr)
    assert np.all(error <= [.05 * target[0], .03, .03, .05])


@pytest.mark.slow
def test_prior_predictive_posterior_ranks():
    rng = np.random.default_rng(15043)
    ranks = []
    for seed in range(128):
        truth = 2 * rng.beta(2, 2) - 1
        learner = bank(seed, dtype=np.float32, n=128, ess_fraction=.8)
        for x in (1., 1.2, .8):
            learner.update([x], [x * truth**2 + rng.normal(scale=math.sqrt(VARIANCE))])
        ranks.append(np.exp(learner.log_weights) @ (learner.particles[:, 0] < truth))
    assert abs(np.mean(ranks) - .5) <= 4 / np.sqrt(12 * len(ranks))
    assert abs(np.median(ranks) - .5) <= 4 / np.sqrt(4 * len(ranks))
    print("predictive rank mean", np.mean(ranks), "median", np.median(ranks))
