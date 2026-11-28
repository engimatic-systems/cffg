"""Full-density enumeration and conditional Engine oracles for a fixed finite bank."""

import json

import jax
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from cffg import Engine, compile_template, load_bundle, save_bundle
from cffg.finite_hypotheses import FiniteHypotheses, ImpossibleEvidence


CANDIDATES = np.array([[-2.], [0.4], [1.6]])
MEANS = np.array([[[-2., .4], [.5, .2], [1., -.5]], [[-3., .1], [-.2, -.6], [1.5, .8]]])
COV = np.array([[[.8, .25], [.25, 1.]], [[.45, -.2], [-.2, .65]], [[1., .4], [.4, .75]]])
POINTS = np.array([[.3, .6], [-.5, .1]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_absolute_evidence_batch_and_sequential(dtype):
    prior = np.array([.3, .1, .6])
    expected = np.array([[multivariate_normal.logpdf(y, m, cov)
                          for m, cov in zip(means, COV)] for y, means in zip(POINTS, MEANS)])
    joint = np.log(prior) + expected.sum(axis=0)
    bank = FiniteHypotheses(CANDIDATES, COV, prior=prior, dtype=dtype)
    sequence = FiniteHypotheses(CANDIDATES, COV, prior=prior, dtype=dtype)
    tol = 16 * np.finfo(dtype).eps
    batch_energy = bank.update(MEANS, POINTS)
    np.testing.assert_allclose(batch_energy, logsumexp(joint), atol=tol, rtol=tol)
    np.testing.assert_allclose(bank.log_weights, joint - logsumexp(joint), atol=tol, rtol=tol)
    with jax.disable_jit():
        increments = [sequence.update(m, y) for m, y in zip(MEANS, POINTS)]
    assert sequence.n_observed == bank.n_observed == 2
    np.testing.assert_allclose(sequence.log_weights, bank.log_weights, atol=tol, rtol=tol)
    np.testing.assert_allclose(sum(increments), bank.log_evidence, atol=tol)


@pytest.mark.parametrize("schedule", ["two_pass", "flood"])
def test_conditional_template_reproduces_absolute_component_density(tmp_path, schedule):
    bank = FiniteHypotheses(CANDIDATES, COV, prior=[0, 1, 0], dtype=np.float64)
    template, arrays = bank.conditional_model(1, MEANS[0, 1])
    path = tmp_path / "conditional.cffgz"
    save_bundle(path, template, arrays)
    model = load_bundle(path)
    engine = Engine(compile_template(model.template, arrays=model.arrays), schedule=schedule)
    state = engine.bind_data(engine.init_state(), {"observation": POINTS[0]})
    state = engine.run(state, 20, tol=1e-9).require_converged()
    exact = multivariate_normal.logpdf(POINTS[0], MEANS[0, 1], COV[1])
    np.testing.assert_allclose(-engine.bfe(state), exact, atol=1e-12)
    np.testing.assert_allclose(bank.update(MEANS[0], POINTS[0]), exact, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_underflow_recovery_at_100x_candidate_count_and_restart(tmp_path, dtype):
    # Initial posterior suppresses 198 live alternatives below exp underflow.
    candidates = np.arange(200)[:, None]
    prior = np.ones(200)
    prior[-1] = 0.
    bank = FiniteHypotheses(candidates, np.eye(1), prior=prior, dtype=dtype)
    means = np.full((200, 1), 64, dtype=dtype)
    means[0] = 0
    bank.update(means, [0])
    assert np.isfinite(bank.log_weights[1:-1]).all()
    assert np.all(bank.weights[1:] == 0)
    assert bank.support.sum() == 199
    path = tmp_path / "bank.npz"
    bank.save(path)
    resumed = FiniteHypotheses.load(path)
    np.testing.assert_array_equal(resumed.candidates, candidates)
    np.testing.assert_array_equal(resumed.log_weights, bank.log_weights)
    assert resumed.log_evidence == bank.log_evidence
    for b in (bank, resumed):
        b.update(means, [64])
        np.testing.assert_allclose(b.weights, prior / prior.sum(), atol=1e-12)
        assert b.n_observed == 2 and not b.support[-1]
    assert resumed.log_evidence == bank.log_evidence


def test_missing_readings_and_invalid_batch_are_atomic():
    bank = FiniteHypotheses(CANDIDATES, COV)
    means = MEANS.copy()
    points = POINTS.copy()
    means[0] = np.nan
    points[0] = np.nan
    assert bank.update(means, None) == 0.
    assert bank.update(means, points, observed=[False, False]) == 0.
    assert bank.n_observed == 0 and bank.log_evidence == 0.
    bank.update(means, points, observed=[False, True])
    expected = bank.log_weights
    evidence = bank.log_evidence
    with pytest.raises(ValueError, match="finite"):
        bank.update(means[::-1], points[::-1])
    with pytest.raises(ValueError, match="boolean"):
        bank.update(means, points, observed=[0, 1])
    with pytest.raises(ImpossibleEvidence):
        bank.update_log_likelihood([[0., -np.inf, -np.inf], [-np.inf, 0., 0.]])
    with pytest.raises(ValueError, match="negative infinity"):
        bank.update_log_likelihood([[0., 0., np.inf]])
    np.testing.assert_array_equal(bank.log_weights, expected)
    assert bank.log_evidence == evidence and bank.n_observed == 1


def test_prior_replacement_stays_owned_and_resettable(tmp_path):
    candidates = CANDIDATES.copy()
    prior = np.array([1., 3., 0.])
    bank = FiniteHypotheses(candidates, COV)
    bank.reset(prior=prior)
    prior[:] = 0
    candidates[:] = -80
    readout = bank.candidates
    readout[:] = 50
    np.testing.assert_array_equal(bank.candidates, CANDIDATES)
    bank.update_log_likelihood([0., -np.inf, 0.])
    path = tmp_path / "one.npz"
    bank.save(path)
    resumed = FiniteHypotheses.load(path)
    np.testing.assert_array_equal(resumed.support, [True, False, False])
    with pytest.raises(ValueError):
        resumed.reset(prior=[0, 0, 0])
    assert resumed.n_observed == 1
    resumed.reset()
    np.testing.assert_allclose(resumed.weights, [.25, .75, 0.])
    assert resumed.n_observed == 0 and resumed.log_evidence == 0.


def test_large_common_likelihood_preserves_prior_odds_and_numeric_failure_is_loud():
    bank = FiniteHypotheses([[1], [2]], [[1]], prior=[1, 3], dtype=np.float32)
    assert bank.update_log_likelihood([-1e50, -1e50]) == -1e50
    np.testing.assert_allclose(bank.weights, [.25, .75])
    with jax.enable_x64(False), pytest.raises(FloatingPointError, match="Gaussian"):
        bank.update([[0], [1]], [1e30])
    assert bank.n_observed == 1
    with pytest.raises(ValueError, match="positive definite"):
        FiniteHypotheses([[1], [2]], [[1., 1.], [1., 1.]], dtype=np.float32)


def test_changed_candidate_order_and_unknown_version_rejected(tmp_path):
    archive = tmp_path / "bank.npz"
    FiniteHypotheses(CANDIDATES, COV).save(archive)
    with np.load(archive, allow_pickle=False) as snapshot:
        data = dict(snapshot)
    data["candidates"] = data["candidates"][::-1]
    np.savez(archive, **data)
    with pytest.raises(ValueError, match="digest"):
        FiniteHypotheses.load(archive)
    meta = json.loads(data["metadata"].item())
    meta["version"] = 2
    np.savez(archive, **{**data, "metadata": np.array(json.dumps(meta))})
    with pytest.raises(ValueError, match="version"):
        FiniteHypotheses.load(archive)
