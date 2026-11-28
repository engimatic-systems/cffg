"""Point training a physical function through the public registration protocol."""

from dataclasses import replace
from pathlib import Path
import runpy

import jax
import numpy as np
import pytest

from cffg import Engine, MeanTrainer, NeuralTrainer, TrainingConfig, compile_template
from cffg import mean_functions


@pytest.fixture
def example(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "physical_mean_training.py"
    example = runpy.run_path(str(path))
    monkeypatch.setattr(mean_functions, "MEAN_IMPLEMENTATIONS",
                        mean_functions.MEAN_IMPLEMENTATIONS.copy())
    return example


def gaussian_oracle(theta, seconds, readings):
    # Integrate out initial excess temperature N(10, .04) analytically.
    rate = np.exp(theta)
    decay = np.exp(-rate * seconds)
    slope = -rate * seconds * decay
    variance = 0.25 + 0.04 * decay**2
    variance_slope = 2 * 0.04 * decay * slope
    residual = readings - 10 * decay
    energy = 0.5 * np.sum(np.log(2 * np.pi * variance) + residual**2 / variance)
    gradient = np.sum(
        0.5 * variance_slope * (1 - residual**2 / variance) / variance
        - residual * 10 * slope / variance
    )
    return energy, gradient


@pytest.mark.parametrize(("x64", "count"), [(True, 2), (False, 20)])
def test_cooling_plate_matches_marginal_likelihood_and_gradient(example, x64, count):
    mean_functions.register_mean_function(example["implementation"])
    with jax.enable_x64(x64):
        # These two readings reproduce the marginal mean and variance at rate 0.5.
        decay = np.exp(-0.5)
        spread = np.sqrt(0.25 + 0.04 * decay**2)
        readings = np.tile([10 * decay - spread, 10 * decay + spread], count // 2)
        problem = example["problem"](readings)
        trainer = MeanTrainer(*problem, config=TrainingConfig(learning_rate=0.1),
                              factors=["trial/0/cool"])
        initial = trainer.initial_state()
        assert initial.weights.size == 1
        assert len(trainer.parameters.blobs[0].rows) == count
        step = trainer.step(initial)
        updated = step.require_applied()
        energy, gradient = gaussian_oracle(initial.weights.item(), 1.0, readings)
        tolerance = max(trainer.config.inference_tolerance, 512 * np.finfo(trainer.dtype).eps)
        np.testing.assert_allclose(step.objective, energy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(step.derivative.gradient, [gradient], rtol=tolerance)
        assert initial.weights.item() < updated.weights.item() < np.log(0.5)
        result = trainer.infer(updated)
        result.require_converged()
        assert result.free_energies[-1] < energy
        for key, row in trainer.parameters.blobs[0].rows:
            np.testing.assert_array_equal(result.state.params[key][row], updated.weights)


def test_zero_elapsed_time_leaves_rate_unidentified(example):
    mean_functions.register_mean_function(example["implementation"])
    trainer = MeanTrainer(*example["problem"]([9.5], seconds=0.0))
    initial = trainer.initial_state()
    for offset in (0.0, 3.0):
        state = replace(initial, weights=initial.weights + offset)
        step = trainer.step(state)
        np.testing.assert_array_equal(step.require_applied().weights, state.weights)
        np.testing.assert_array_equal(step.derivative.gradient, [0.0])
        energy, _ = gaussian_oracle(state.weights.item(), 0.0, np.array([9.5]))
        assert step.objective == pytest.approx(energy)


def test_physical_checkpoint_continues_with_identical_optimizer_state(example, tmp_path):
    mean_functions.register_mean_function(example["implementation"])
    problem = example["problem"]([6.1])
    trainer = MeanTrainer(*problem)
    first = trainer.step(trainer.initial_state()).require_applied()
    path = tmp_path / "physical.npz"
    trainer.save_checkpoint(path, first)
    compatible = NeuralTrainer(*problem)
    restored = compatible.load_checkpoint(path)
    restored_step = compatible.step(restored)
    direct_step = trainer.step(first)
    assert restored_step.objective == direct_step.objective
    assert restored_step.require_applied().step == direct_step.require_applied().step == 2
    for field in ("weights", "first_moment", "second_moment"):
        np.testing.assert_array_equal(getattr(restored_step.state, field), getattr(direct_step.state, field))


def test_registration_opt_out_and_legacy_work_count(example):
    impl = example["implementation"]
    problem = example["problem"]([6.1])
    old_work = lambda di, do, config: {"network_parameters": 1}
    for flag, work, admitted in ((False, old_work, False), (None, None, False),
                                 (None, old_work, True)):
        registry = mean_functions.MEAN_IMPLEMENTATIONS
        registry.pop(impl.impl_id, None)
        mean_functions.register_mean_function(replace(impl, work=work, point_trainable=flag))
        template, arrays, observations = problem
        engine = Engine(compile_template(template, arrays=arrays))
        engine.run(engine.bind_data(engine.init_state(), observations), 50, tol=1e-7).require_converged()
        if admitted:
            assert MeanTrainer(*problem).parameters.size == 1
        else:
            with pytest.raises(ValueError, match="point-trainable"):
                MeanTrainer(*problem)
    with pytest.raises(TypeError, match="point_trainable"):
        mean_functions.register_mean_function(replace(impl, point_trainable=1))


def test_physical_registration_rejects_malformed_parameters(example):
    mean_functions.register_mean_function(example["implementation"])
    template, arrays, observations = example["problem"]([6.1])
    for bad, reason in ((np.array([0.1, 0.2]), "shape"), (np.array([np.nan]), "finite")):
        with pytest.raises(ValueError, match=reason):
            MeanTrainer(template, {**arrays, "log_rate": bad}, observations)
