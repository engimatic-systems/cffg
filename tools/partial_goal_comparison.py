"""Compare first-action inference on the retained position-only PCGym fixture."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def _shared_dynamics(data):
    matrices, noises = data["A"], data["Q"]
    habit = data["habit"]
    support = np.asarray(habit) > 0
    if not support.any() or not np.isfinite(habit).all() or (habit < 0).any():
        raise ValueError("the selector needs a finite positive-mass habit")
    if abs(habit.sum() - 1) > 8 * np.finfo(habit.dtype).eps:
        raise ValueError("the selector habit must sum to one")
    first = np.flatnonzero(support)[0]
    if not np.all(matrices[support] == matrices[first]) or not np.all(
        noises[support] == noises[first]
    ):
        raise ValueError(
            "the retained comparison requires common dynamics matrices and covariances"
        )
    if data["C"].shape[0] != 1:
        raise ValueError("the retained comparison requires scalar goals")
    return matrices[first], noises[first], data["C"][0]


def _observation_step(matrix, noise, row, sensor_variance, cov):
    predicted = matrix @ cov @ matrix.T + noise
    cross = predicted @ row
    innovation = sensor_variance + row @ cross
    gain = cross / innovation
    # Joseph form preserves symmetry and accuracy after a sharp scalar goal.
    residual = np.eye(len(row)) - np.outer(gain, row)
    posterior = residual @ predicted @ residual.T + sensor_variance * np.outer(
        gain, gain
    )
    return innovation, gain, posterior


def _result(first_action_logs):
    logs = np.asarray(first_action_logs)
    total = logsumexp(logs)
    return {"first": np.exp(logs - total).tolist(), "log_evidence": float(total)}


def exact_terminal_pair(data, goals, alternative):
    """Sum all supported paths, reusing the prefix when only the final goal changes."""
    matrix, noise, row = _shared_dynamics(data)
    horizon = len(goals)
    if not horizon:
        raise ValueError("a planning horizon must contain a goal")
    support = np.flatnonzero(data["habit"] > 0)
    offsets = data["b"][support]
    priors = np.log(data["habit"][support])
    obs_offset = data["offset"].item()
    cov = data["P0"]
    innovations = []
    for _ in goals:
        variance, gain, cov = _observation_step(
            matrix, noise, row, data["R"].item(), cov
        )
        innovations.append((variance, np.log(2 * np.pi * variance), gain))

    # Cap the expanding suffix at eight steps per prefix batch. Every path
    # remains in the sum, even when its unnormalized density underflows.
    cut = max(1, horizon - 8)
    scores = np.full((2, len(data["habit"])), -np.inf)

    def extend(means, logs, level, steps, final, first=None):
        for t in range(level, steps):
            pred = np.einsum("ij,nj->ni", matrix, means)
            active = (
                slice(first, first + 1) if t == 0 and first is not None else slice(None)
            )
            next_states = pred[:, None, :] + offsets[active]
            pred = next_states.reshape(-1, len(row))
            log_prior = (logs[:, None] + priors[active]).ravel()
            obs = np.einsum("i,ni->n", row, pred) + obs_offset
            error = goals[t] - obs
            variance, normalizer, gain = innovations[t]
            logs = log_prior - 0.5 * (normalizer + error**2 / variance)
            if final and t == horizon - 1:
                changed = log_prior - 0.5 * (
                    normalizer + (alternative - obs) ** 2 / variance
                )
                return np.array([logsumexp(logs), logsumexp(changed)])
            means = pred + error[:, None] * gain
        return means, logs

    for slot, action in enumerate(support):
        if cut == horizon:
            scores[:, action] = extend(
                data["m0"][None], np.zeros(1), 0, cut, True, slot
            )
            continue
        means, logs = extend(data["m0"][None], np.zeros(1), 0, cut, False, slot)
        for start in range(0, len(logs), 16):
            ids = slice(start, start + 16)
            pair = extend(means[ids], logs[ids], cut, horizon, True)
            scores[:, action] = np.logaddexp(scores[:, action], pair)
    return [_result(logs) for logs in scores]


def gaussian_future(data, goals):
    """Keep the first selector exact; match the mean/covariance of future offsets."""
    matrix, noise, row = _shared_dynamics(data)
    weights = data["habit"]
    offset = weights @ data["b"]
    deviations = data["b"] - offset
    future_noise = noise + (deviations.T * weights) @ deviations
    scores = np.full(len(weights), -np.inf)
    for k in np.flatnonzero(weights > 0):
        log_joint = np.log(weights[k])
        mean, cov = data["m0"], data["P0"]
        for t, goal in enumerate(goals):
            predicted = matrix @ mean + (data["b"][k] if t == 0 else offset)
            variance, gain, cov = _observation_step(
                matrix, noise if t == 0 else future_noise, row, data["R"].item(), cov
            )
            error = goal - row @ predicted - data["offset"].item()
            log_joint -= 0.5 * (np.log(2 * np.pi * variance) + error**2 / variance)
            mean = predicted + gain * error
        scores[k] = log_joint
    return _result(scores)


def dense_reference(data, goals):
    """Independent short-horizon joint normal for the entire vector of observations."""
    horizon = len(goals)
    if horizon > 3:
        raise ValueError("dense cross-check is bounded to three goals")
    row = data["C"][0]
    d = len(row)
    support = np.flatnonzero(data["habit"] > 0)
    scores = np.full(len(data["habit"]), -np.inf)
    for path in itertools.product(support, repeat=horizon):
        # Each primitive source is independent N(0,I) in these coordinates.
        influence = np.zeros((d, d * (horizon + 1)))
        influence[:, :d] = np.linalg.cholesky(data["P0"])
        mean = data["m0"]
        obs_mean, obs_influence = [], []
        for t, k in enumerate(path, start=1):
            influence = data["A"][k] @ influence
            influence[:, t * d : (t + 1) * d] = np.linalg.cholesky(data["Q"][k])
            mean = data["A"][k] @ mean + data["b"][k]
            obs_mean.append(row @ mean + data["offset"].item())
            obs_influence.append(row @ influence)
        design = np.stack(obs_influence)
        covariance = design @ design.T + data["R"].item() * np.eye(horizon)
        chol = np.linalg.cholesky(covariance)
        residual = np.linalg.solve(chol, np.asarray(goals) - obs_mean)
        log_joint = np.sum(np.log(data["habit"][list(path)])) - 0.5 * (
            horizon * np.log(2 * np.pi)
            + 2 * np.log(np.diag(chol)).sum()
            + residual @ residual
        )
        scores[path[0]] = np.logaddexp(scores[path[0]], log_joint)
    return _result(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native",
        action="store_true",
        help="also run the public CFFG solver in float64",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from hybrid_planning_fixture import declared_rule_reference, planning_fixture

    if args.native:
        import jax

        from cffg import Engine, compile_template

        jax.config.update("jax_enable_x64", True)

    cases = []
    for horizon in (1, 2, 15):
        template, arrays, observations = planning_fixture(horizon)
        goals = np.array([value.item() for value in observations.values()])
        exact = exact_terminal_pair(arrays, goals, 2.0)
        if args.native:
            engine = Engine(compile_template(template, arrays=arrays))
        for index, final in enumerate((0.0, 2.0)):
            goals[-1] = final
            changed = {**observations, f"g{horizon}": np.array([final])}
            local = declared_rule_reference(arrays, changed)
            record = {
                "horizon": horizon,
                "final": final,
                "paths": 3**horizon,
                "exact": exact[index],
                "gaussian_future": gaussian_future(arrays, goals),
                "current": {"first": local.regimes[0].tolist(), "bfe": local.energy},
            }
            if horizon <= 2:
                dense = dense_reference(arrays, goals)
                np.testing.assert_allclose(
                    dense["first"], exact[index]["first"], rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    dense["log_evidence"],
                    exact[index]["log_evidence"],
                    rtol=0,
                    atol=1e-10,
                )
            if args.native:
                result = engine.run(
                    engine.bind_data(engine.init_state(), changed), 200, tol=1e-9
                )
                state = result.require_converged()
                native = engine.marginals(state, edges=["z0"])["z0"]
                np.testing.assert_allclose(native, local.regimes[0], rtol=0, atol=1e-9)
                np.testing.assert_allclose(
                    result.free_energies[-1], local.energy, rtol=1e-8
                )
                record["native"] = {
                    "first": native.tolist(),
                    "bfe": result.free_energies[-1],
                    "status": result.status.value,
                    "iterations": result.iterations,
                }
            cases.append(record)
    contents = json.dumps(cases, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(contents, end="")
    else:
        args.output.write_text(contents)


if __name__ == "__main__":
    main()
