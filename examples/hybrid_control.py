"""Choose a bounded categorical command and learn only its observed component.

Run with uv run python examples/hybrid_control.py. Set JAX_ENABLE_X64=1
to use float64. The selected arm is categorical DATA. The measured scalar
regressor enters as the center of a positive-variance Gaussian approximation.
"""

import jax
import numpy as np

from cffg import Engine, compile_template, conj
from cffg.core import BlobRef, FactorSpec, LearnableSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.learn import Learnables


COMMANDS = np.array([-.35, 0., .35])
SLOPES = np.array([.7, .8, .9])
OFFSETS = .8 * COMMANDS
MOTION_VARIANCE = .10
SENSOR_VARIANCE = .08


def window(current, input_variance, dtype):
    """Use an ordinary affine DATA emission downstream from mixture dynamics."""
    model = Template("bounded_hybrid_command")
    model.variables = {
        "regressor": VariableSpec("regressor", dim=1),
        "next_state": VariableSpec("next_state", dim=1),
        "reading": VariableSpec("reading", dim=1),
        "selector": VariableSpec("selector", cardinality=len(COMMANDS)),
    }
    model.factors = {
        "context": FactorSpec(
            "context", "gaussian", {"out": "regressor"},
            param_blobs={"mean": "x", "cov": "P"},
        ),
        "motion": FactorSpec(
            "motion", "gaussian_mixture_linear",
            {"in": "regressor", "out": "next_state", "selector": "selector"},
            param_blobs={"matrix": "A", "offset": "b", "cov": "Q"},
        ),
        "emission": FactorSpec(
            "emission", "gaussian_linear", {"in": "next_state", "out": "reading"},
            param_blobs={"matrix": "H", "offset": "zero", "cov": "R"},
        ),
    }
    model.ports = {
        name: PortSpec(name, name, PortRole.DATA) for name in ("selector", "reading")
    }
    model.learnables["arms"] = LearnableSpec(
        "arms", "component_mniw", ("A", "b", "Q"), ("motion",),
    )
    arrays = {
        "x": [current], "P": [[input_variance]], "A": SLOPES[:, None, None],
        "b": OFFSETS[:, None], "Q": np.full((len(COMMANDS), 1, 1), MOTION_VARIANCE),
        "H": [[1.]], "zero": [0.], "R": [[SENSOR_VARIANCE]],
    }
    model.blobs = {name: BlobRef(name, f"{name}.npy") for name in arrays}
    return Engine(compile_template(model, arrays={
        name: np.asarray(value, dtype=dtype) for name, value in arrays.items()
    }))


def infer(engine, arm, reading):
    data = {"selector": arm, "reading": None if reading is None else [reading]}
    return engine.run(engine.bind_data(engine.init_state(), data), 24, tol=1e-6)


def choose_arm(engine, target):
    scores = []
    for arm in range(len(COMMANDS)):
        # Each candidate predicts a reading with its selector fixed. Score
        # the target under this predictive distribution without training.
        state = infer(engine, arm, None).require_converged()
        mean, covariance = engine.gaussian_marginal(state, "reading")
        variance = float(covariance[0, 0])
        if variance <= 0 or not np.isfinite([variance, mean[0]]).all():
            raise ArithmeticError("the arm has no proper predictive reading")
        scores.append(-.5 * (np.log(2*np.pi*variance) + (target - float(mean[0]))**2 / variance))
    return int(np.argmax(scores))


def learn_arm(engine, learner, applied_arm, reading):
    # With a missing reading the motion factor can still predict an outcome.
    # Adding its statistics would count a synthetic transition as new data.
    if reading is None:
        return None
    result = infer(engine, applied_arm, reading)
    state = result.require_converged()
    stats = engine.gaussian_node_stats(state)["motion"]
    responsibility = np.eye(len(COMMANDS))[applied_arm]
    np.testing.assert_array_equal(stats["responsibility"], responsibility)
    learner.accumulate(engine, result)
    return engine.gaussian_marginal(state, "regressor")


def main():
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    current, goal = .3, .95
    variances = (.08, .005)
    engine = window(current, variances[0], dtype)
    selected = choose_arm(engine, goal)
    assert selected == 2
    applied = selected  # Read back the realized selector in a real actuator.
    reading = .9 * current + 1.2 * COMMANDS[applied] + .10
    priors = [conj.vague_mniw(1, 1, scale=.2) for _ in COMMANDS]

    # Fit the same recorded transition separately under two prior variances.
    # Each hypothesis consumes the observation once.
    fitted, posterior_inputs = [], []
    unused = np.arange(len(COMMANDS)) != applied
    for variance in variances:
        engine = window(current, variance, dtype)
        learner = Learnables(engine, priors={"arms": priors})
        before = learner.expected("arms", "matrix").copy()
        assert learn_arm(engine, learner, applied, None) is None
        np.testing.assert_array_equal(learner.expected("arms", "matrix"), before)
        mean, covariance = learn_arm(engine, learner, applied, reading)
        slope = SLOPES[applied]
        predicted_variance = variance * slope**2 + MOTION_VARIANCE + SENSOR_VARIANCE
        gain = variance * slope / predicted_variance
        expected_mean = current + gain * (reading - slope * current - OFFSETS[applied])
        expected_variance = variance - gain * slope * variance
        np.testing.assert_allclose(mean, [expected_mean], atol=2e-6)
        np.testing.assert_allclose(covariance, [[expected_variance]], atol=2e-6)
        for key, value in learner.params["arms"].items():
            expected = np.stack([getattr(prior, key) for prior in priors])
            np.testing.assert_array_equal(value[unused], expected[unused])
        np.testing.assert_array_equal(learner.params["arms"]["nu"], [3., 3., 4.])
        fitted.append(float(learner.expected("arms", "matrix")[applied, 0, 0]))
        posterior_inputs.append(float(mean[0]))

    assert abs(fitted[0] - fitted[1]) > 1e-3
    print("hybrid control: applied command =", float(COMMANDS[applied]))
    print("regressor variances =", variances, "posterior regressor means =", posterior_inputs,
          "fitted slopes =", fitted)


if __name__ == "__main__":
    main()
