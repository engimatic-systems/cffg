"""Same-root qualification of deterministic learned Gaussian initialization."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import neural_mean
from cffg.amortization import GaussianAmortizer, PolynomialFeatures
from cffg.compiled import NEG
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, VariableSpec
from cffg.runtime_types import _replace
from neural_training_fixture import key
from nonlinear_fixture import adaptive_pair
from test_neural import neural_model


@pytest.mark.parametrize("x64", [False, True])
def test_recognition_replay_batching_and_invalid_proposals(tmp_path, x64):
    with jax.enable_x64(x64):
        template, arrays = neural_model()
        template.variables["s"] = VariableSpec("s", cardinality=3)
        for name, probabilities in (("sparse", [.6, .4, 0.]), ("evidence", [1/3, 1/3, 1/3])):
            template.factors[name] = FactorSpec(name, "categorical", {"out": "s"},
                                               param_blobs={"statistic": name})
            template.blobs[name] = BlobRef(name, name + ".npy")
            arrays[name] = np.asarray(probabilities)
        recognition = GaussianAmortizer.fit(
            template, arrays, [{"y": np.array([value])} for value in (-.3, 0., .3)],
            features=PolynomialFeatures(("y",), (1.,), degree=1), reference_index=1)
        dtype = recognition.reference.g_eta.dtype
        values = jnp.asarray([[-.1], [.1]], dtype=dtype)
        actual = jax.jit(jax.vmap(recognition.predict_coordinates))(values)
        np.testing.assert_allclose(actual, jnp.stack([recognition.predict_coordinates(v) for v in values]),
                                   rtol=64 * np.finfo(dtype).eps, atol=64 * np.finfo(dtype).eps)
        initial = recognition.engine.bind_data(recognition.engine.init_state(), {"y": np.array([.1])})
        initial = recognition.engine.step(initial, harden=False)
        assert np.any(np.asarray(initial.messages) == NEG)
        proposal = recognition.propose(initial)
        assert proposal.accepted, proposal.message
        for name in ("messages", "ef_eta", "g_obs", "g_mask", "ef_obs", "ef_mask"):
            np.testing.assert_array_equal(getattr(proposal.state, name), getattr(initial, name))
        for name in initial.params:
            np.testing.assert_array_equal(proposal.state.params[name], initial.params[name])
        saved = tmp_path / "recognition.npz"
        recognition.save(saved)
        restored = GaussianAmortizer.load(saved, template, arrays)
        np.testing.assert_array_equal(restored.predict_coordinates(values[0]), recognition.predict_coordinates(values[0]))
        replay = restored.propose(initial)
        assert replay.accepted, replay.message
        np.testing.assert_array_equal(replay.state.g_eta, proposal.state.g_eta)
        np.testing.assert_array_equal(replay.state.g_lam, proposal.state.g_lam)

        stale = _replace(initial, params={name: value + 1 for name, value in initial.params.items()})
        refused = recognition.propose(stale)
        assert not refused.accepted and refused.state is stale and "parameters changed" in refused.message
        recognition.output_center = jnp.full_like(recognition.output_center, -1e6)
        recognition.coefficients = jnp.zeros_like(recognition.coefficients)
        refused = recognition.propose(initial)
        assert not refused.accepted and refused.state is initial
        assert "improper" in refused.message
        with np.load(saved, allow_pickle=False) as contents:
            corrupt = {name: contents[name].copy() for name in contents.files}
        corrupt["coefficients"][:] = np.nan
        np.savez(saved, **corrupt)
        with pytest.raises(ValueError, match="finite-value"):
            GaussianAmortizer.load(saved, template, arrays)


def paired_model():
    template, arrays = adaptive_pair(variance=.4)
    for i in (1, 2):
        name = f"sensor{i}"
        template.factors[name] = replace(template.factors[name], function=neural_mean((1, 1, 1)))
        del template.factors[f"evidence{i}"]
        template.ports[f"y{i}"] = PortSpec(f"y{i}", f"y{i}", PortRole.DATA)
    arrays["theta"] = np.array([.8, 0., 1.25, 0.])
    return template, arrays


@pytest.mark.slow
def test_frozen_paired_initializer_reduces_work_without_changing_roots(tmp_path):
    template, arrays = paired_model()
    observations = lambda t: {"y1": np.array([t + .15]), "y2": np.array([t - .15])}
    training = [observations(t) for t in np.linspace(-.4, .4, 17)]
    recognition = GaussianAmortizer.fit(template, arrays, training,
        features=PolynomialFeatures(("y1", "y2"), (.5, .5), scale=.4, degree=7),
        reference_index=8, tolerance=1e-8, n_iters=25)
    held = np.asarray(jax.random.uniform(key(3201), (32,), dtype=jnp.float64, minval=-.4, maxval=.4))
    differences, iterations, pairs = [], [], []
    for t in held:
        initial = recognition.engine.bind_data(recognition.engine.init_state(), observations(t))
        baseline = recognition.engine.run(initial, 25, tol=1e-8)
        proposal = recognition.propose(initial)
        assert proposal.accepted, proposal.message
        accelerated = recognition.engine.run(proposal.state, 25, tol=1e-8)
        base_state, proposed_state = baseline.require_converged(), accelerated.require_converged()
        left = recognition.engine.gaussian_marginals(base_state)
        right = recognition.engine.gaussian_marginals(proposed_state)
        for name in left:
            for a, b in zip(left[name], right[name], strict=True):
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        base_energy, proposed_energy = recognition.engine.bfe(base_state), recognition.engine.bfe(proposed_state)
        assert abs(base_energy - proposed_energy) <= 1e-6 * (1 + abs(base_energy))
        differences.append(baseline.outer_map_evaluations - accelerated.outer_map_evaluations)
        iterations.append(baseline.iterations - accelerated.iterations)
        pairs.append(dict(t=float(t), baseline_maps=baseline.outer_map_evaluations,
                          proposed_maps=accelerated.outer_map_evaluations,
                          baseline_iterations=baseline.iterations, proposed_iterations=accelerated.iterations))
    resamples = np.random.default_rng(3202).integers(0, 32, size=(10_000, 32))
    means = np.asarray(differences)[resamples].mean(axis=1)
    interval = np.quantile(means, [.025, .975], method="linear")
    evidence = dict(mean_map_reduction=float(np.mean(differences)), confidence_interval=interval.tolist(),
                    mean_iteration_reduction=float(np.mean(iterations)), pairs=pairs)
    (tmp_path / "amortization.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({name: value for name, value in evidence.items() if name != "pairs"}, sort_keys=True))
    assert interval[0] > 0
