"""Certified optimizer updates, strict continuation and held-out qualification."""

from dataclasses import replace
import json

import jax
import numpy as np
import pytest

from cffg import load_bundle
from cffg.training import NeuralTrainer, TrainingConfig
from neural_training_fixture import affine_baseline, mean_values, regression_model
from test_fixed_point_gradients import coupled_neural_model


@pytest.mark.parametrize("x64", [pytest.param(True, marks=pytest.mark.slow), False])
def test_optimizer_continuation_and_bundle_are_replayable_and_strict(tmp_path, x64):
    with jax.enable_x64(x64):
        template, arrays, observations, _, _ = regression_model(2)
        trainer = NeuralTrainer(template, arrays, observations, config=TrainingConfig(objective_scale=.5))
        first = trainer.step(trainer.initial_state())
        state = first.require_applied()
        checkpoint = tmp_path / "adam.npz"
        trainer.save_checkpoint(checkpoint, state)
        fresh = NeuralTrainer(template, arrays, observations, config=trainer.config)
        restored = fresh.load_checkpoint(checkpoint)
        uninterrupted, resumed = trainer.step(state), fresh.step(restored)
        np.testing.assert_array_equal(uninterrupted.require_applied().weights, resumed.require_applied().weights)
        assert uninterrupted.objective == resumed.objective
        bundle = tmp_path / "trained.cffgz"
        trainer.save_bundle(bundle, state)
        np.testing.assert_array_equal(load_bundle(bundle).arrays["theta"], state.weights)

        with np.load(checkpoint, allow_pickle=False) as saved:
            contents = {name: saved[name].copy() for name in saved.files}
        corruptions = []
        missing = dict(contents)
        missing.pop("metadata")
        corruptions.append((missing, "legacy"))
        invalid = {**contents, "second_moment": -np.ones_like(contents["second_moment"])}
        corruptions.append((invalid, "nonnegative"))
        invalid = {**contents, "weights": np.full_like(contents["weights"], np.nan)}
        corruptions.append((invalid, "finite-value"))
        corruptions.append(({**contents, "step": np.asarray(-1, dtype=np.int64)}, "nonnegative"))
        metadata = json.loads(str(contents["metadata"]))
        metadata["structure"] = "wrong"
        corruptions.append(({**contents, "metadata": np.asarray(json.dumps(metadata))}, "mismatch"))
        for values, message in corruptions:
            np.savez(checkpoint, **values)
            with pytest.raises(ValueError, match=message):
                trainer.load_checkpoint(checkpoint)
            np.testing.assert_array_equal(state.weights, restored.weights)


def test_optimizer_refuses_an_actual_exhausted_adjoint_without_updating():
    template, arrays = coupled_neural_model("linearization")
    trainer = NeuralTrainer(template, arrays, {}, config=TrainingConfig(
        inference_tolerance=1e-10, gradient_tolerance=1e-10, adjoint_iterations=1, krylov_dimension=1))
    state = trainer.initial_state()
    refused = trainer.step(state)
    assert not refused.applied and refused.state is state
    assert "adjoint" in refused.message
    with pytest.raises(ValueError, match="another model"):
        trainer.step(replace(state, identity="other"))


@pytest.mark.slow
def test_frozen_positive_noise_regression_beats_untrained_and_affine_baselines(tmp_path):
    template, arrays, observations, x, y = regression_model()
    trainer = NeuralTrainer(template, arrays, observations, config=TrainingConfig(
        objective_scale=1 / 17, inference_tolerance=1e-10, gradient_tolerance=1e-8))
    state = trainer.initial_state()
    initial_energy = trainer.engine.bfe(trainer.reference.state) / 17
    held_x = np.linspace(-1.5, 1.5, 16)
    truth = np.tanh(2 * held_x)
    rmse = lambda prediction: float(np.sqrt(np.mean((prediction - truth) ** 2)))
    untrained_error = rmse(mean_values(state.weights, held_x))
    slope, intercept = affine_baseline(x, y)
    affine_error = rmse(slope * held_x + intercept)
    for _ in range(32):
        state = trainer.step(state).require_applied()
    final = trainer.infer(state)
    final_energy = trainer.engine.bfe(final.require_converged()) / 17
    trained_error = rmse(mean_values(state.weights, held_x))
    evidence = dict(initial_energy=initial_energy, final_energy=final_energy,
                    untrained_rmse=untrained_error, affine_rmse=affine_error,
                    trained_rmse=trained_error, updates=state.step)
    print(json.dumps(evidence, sort_keys=True))
    (tmp_path / "neural-regression.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert state.step == 32
    assert trained_error <= .075 and trained_error <= .5 * min(untrained_error, affine_error)
    assert initial_energy - final_energy >= 1.
