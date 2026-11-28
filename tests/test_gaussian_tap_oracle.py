"""Condition the dense joint law of the states and sensors independently of BP."""

import jax
import numpy as np
import pytest
from scipy.linalg import block_diag

from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core import observe_interior
from cffg.engine import Engine
from cffg.models.kalman import lgssm


def dense_joint(steps, arrays):
    """Map independent prior, process and sensor innovations to the joint vector."""
    a = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    transition, sensor = a["F"], a["H"]
    d, s = len(a["m0"]), len(a["zm"])
    state_end = steps * d
    size = state_end + steps * s
    transform = np.eye(size)
    mean = np.zeros(size)
    blocks = {}
    for k in range(steps):
        x = slice(k * d, (k + 1) * d)
        y = slice(state_end + k * s, state_end + (k + 1) * s)
        blocks[f"x{k}"], blocks[f"y{k}"] = x, y
        if k == 0:
            mean[x] = a["m0"]
        else:
            previous = blocks[f"x{k - 1}"]
            mean[x] = transition @ mean[previous] + a["zn"]
            transform[x] += transition @ transform[previous]
        mean[y] = sensor @ mean[x] + a["zm"]
        transform[y] += sensor @ transform[x]
    noise = block_diag(a["P0"], *([a["Q"]] * (steps - 1)), *([a["R"]] * steps))
    return mean, transform @ noise @ transform.T, blocks


def condition(joint, evidence):
    mean, cov, blocks = joint
    if not evidence:
        return mean, cov, 0.
    indices = np.concatenate([np.arange(mean.size)[blocks[name]] for name in evidence])
    observed = np.concatenate(list(evidence.values()))
    obs_cov = cov[np.ix_(indices, indices)]
    cross = cov[:, indices]
    delta = observed - mean[indices]
    sign, logdet = np.linalg.slogdet(obs_cov)
    assert sign > 0
    solution = np.linalg.solve(obs_cov, delta)
    nll = .5 * (indices.size * np.log(2 * np.pi) + logdet + delta @ solution)
    return mean + cross @ solution, cov - cross @ np.linalg.solve(obs_cov, cross.T), nll


def fixture(steps, dtype):
    # Change units along the two axes while keeping full-rank correlated noise.
    units = np.diag([8., .25])
    inv_units = np.diag(1. / units.diagonal())
    template, arrays = lgssm(
        steps, units @ [[.75, .125], [-.25, .875]] @ inv_units,
        units @ [[.25, -.0625], [-.0625, .3125]] @ units,
        np.array([[.5, -.75]]) @ inv_units, np.array([[.375]]),
        units @ [.5, -.25], units @ [[1.25, .375], [.375, .75]] @ units,
    )
    arrays.update(zn=units @ [.125, .25], zm=np.array([-.25]))
    return template, {key: value.astype(dtype) for key, value in arrays.items()}


@pytest.mark.parametrize("schedule,dtype", [
    ("two_pass", np.float32), pytest.param("two_pass", np.float64, marks=pytest.mark.slow),
    pytest.param("flood", np.float32, marks=pytest.mark.slow), ("flood", np.float64),
])
def test_adjacent_taps_match_joint_density_after_rebinding(tmp_path, dtype, schedule):
    template, arrays = fixture(2, dtype)
    joint = dense_joint(2, arrays)
    ports = {}
    copies = {"x0": {"x0", "x0c", "x0o"}, "x1": {"x1"}}
    for latent in copies:
        previous = template
        template, ports[latent] = observe_interior(previous, latent)
        copies[latent].update(template.variables.keys() - previous.variables.keys())

    archive = tmp_path / "adjacent.cffgz"
    save_bundle(archive, template, arrays)
    restored = load_bundle(archive)
    tolerance = 64 * np.finfo(dtype).eps if dtype is np.float32 else 1e-10
    evidence = {}
    updates = (
        {"y0": [.75], "y1": [-.375]},
        {"x0": [6., -.1875]},
        {"x1": [-2., .125]},
        {"x0": None},
        {"x1": [1., -.25], "y1": None},
        {"x1": None, "y1": [-.375]},
    )
    with jax.enable_x64(dtype is np.float64):
        engine = Engine(compile_template(restored.template, arrays=restored.arrays), schedule=schedule)
        state = engine.init_state()
        for update in updates:
            for name, value in update.items():
                if value is None:
                    evidence.pop(name)
                else:
                    evidence[name] = np.asarray(value, dtype=dtype)
            bindings = {ports.get(name, name): value for name, value in update.items()}
            state = engine.run(engine.bind_data(state, bindings), 48, tol=tolerance).require_converged()
            mean, cov, nll = condition(joint, evidence)
            np.testing.assert_allclose(engine.bfe(state), nll, rtol=tolerance, atol=tolerance)
            marginals = engine.gaussian_marginals(state)
            for latent, names in copies.items():
                x = joint[2][latent]
                for edge in names:
                    actual_mean, actual_cov = marginals[edge]
                    if latent in evidence:
                        np.testing.assert_array_equal(actual_mean, evidence[latent])
                        np.testing.assert_array_equal(actual_cov, np.zeros_like(actual_cov))
                    else:
                        np.testing.assert_allclose(actual_mean, mean[x], rtol=tolerance, atol=tolerance)
                        np.testing.assert_allclose(actual_cov, cov[x, x], rtol=tolerance, atol=tolerance)
                selected = engine.gaussian_marginal(state, ports[latent])
                for single, bulk in zip(selected, marginals[ports[latent]], strict=True):
                    np.testing.assert_allclose(single, bulk, rtol=tolerance, atol=tolerance)
            # Each ordinary DATA edge reports its leave-one-sensor-out predictive law.
            for sensor in ("y0", "y1"):
                others = {k: v for k, v in evidence.items() if k != sensor}
                pred, pred_cov, _ = condition(joint, others)
                y = joint[2][sensor]
                actual_mean, actual_cov = marginals[sensor]
                np.testing.assert_allclose(actual_mean, pred[y], rtol=tolerance, atol=tolerance)
                np.testing.assert_allclose(actual_cov, pred_cov[y, y], rtol=tolerance, atol=tolerance)


@pytest.mark.slow
def test_tap_separates_long_gaussian_chain():
    template, arrays = fixture(32, np.float64)
    joint = dense_joint(32, arrays)
    template, port = observe_interior(template, "x16")
    evidence = {"y0": [-.625], "x16": [8., -.25], "y31": [.5]}
    mean, cov, nll = condition(joint, evidence)
    engine = Engine(compile_template(template, arrays=arrays), schedule="two_pass")
    state = engine.bind_data(engine.init_state(), {port: evidence["x16"], "y0": [-.625], "y31": [.5]})
    state = engine.run(state, 1, tol=1e-10).require_converged()
    np.testing.assert_allclose(engine.bfe(state), nll, rtol=1e-10, atol=1e-10)
    marginals = engine.gaussian_marginals(state)
    for k in range(32):
        x = joint[2][f"x{k}"]
        actual_mean, actual_cov = marginals[f"x{k}"]
        np.testing.assert_allclose(actual_mean, mean[x], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual_cov, cov[x, x], rtol=1e-10, atol=1e-10)
