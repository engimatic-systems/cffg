"""Plain numeric PCGym seed-1 model and an independent declared-rule oracle.

The scalar goals leave backward mixture components improper. Their documented
vacuous fallback makes this fixture a forward collapsed filter, not an exact
long-horizon smoother. The oracle below intentionally exposes that limitation.
"""

from typing import NamedTuple

import numpy as np
from scipy.special import logsumexp

from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec


class PlanningReference(NamedTuple):
    regimes: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    energy: float


def planning_fixture(horizon=15, dtype=np.float64, noise_scale=1., habit=None):
    template = Template(name=f"retained_hybrid_h{horizon}")
    matrix = [[.9872294577831741, .08338833118938865],
              [-.175619222344668, .7290274178747534]]
    covariance = [[.00011554271987053177, 2.0576762787578282e-7],
                  [2.0576762787578282e-7, .00019275601503216264]]
    arrays = {
        "m0": [1., -1.], "P0": np.eye(2) * 1e-4,
        "A": np.tile(matrix, (3, 1, 1)),
        "b": [[-.004707853348770677, -.08696721726375882],
              [-.00017546460387765, -.0004393965675551424],
              [.004356924141015377, .08608842412864852]],
        "Q": np.tile(covariance, (3, 1, 1)) * noise_scale,
        "habit": [1 / 3] * 3 if habit is None else habit,
        "C": [[1., 0.]], "offset": [0.], "R": [[.0025000000000000005]],
    }
    arrays = {name: np.asarray(value, dtype=dtype) for name, value in arrays.items()}
    template.blobs = {name: BlobRef(name, f"{name}.npy") for name in arrays}

    def variable(name, *, discrete=False):
        template.variables[name] = VariableSpec(name, cardinality=3) if discrete else VariableSpec(
            name, dim=1 if name.startswith("g") else 2,
        )
        return name

    template.factors["prior"] = FactorSpec(
        "prior", "gaussian", edges={"out": variable("x0")},
        param_blobs={"mean": "m0", "cov": "P0"},
    )
    for index in range(horizon):
        selector = variable(f"z{index}", discrete=True)
        template.factors[f"hab{index}"] = FactorSpec(
            f"hab{index}", "categorical", edges={"out": selector},
            param_blobs={"statistic": "habit"},
        )
        source = f"x{index}"
        if index:
            source = variable(f"x{index}d")
            template.factors[f"eq{index}"] = FactorSpec(
                f"eq{index}", "equality",
                edges={"a": f"x{index}", "b": source, "c": variable(f"x{index}g")},
            )
        template.factors[f"dyn{index + 1}"] = FactorSpec(
            f"dyn{index + 1}", "gaussian_mixture_linear",
            edges={"in": source, "out": variable(f"x{index + 1}"), "selector": selector},
            param_blobs={"matrix": "A", "offset": "b", "cov": "Q"},
        )
    observations = {}
    for index in range(1, horizon + 1):
        source = f"x{index}" if index == horizon else f"x{index}g"
        goal = variable(f"g{index}")
        template.factors[f"goal{index}"] = FactorSpec(
            f"goal{index}", "gaussian_linear", edges={"in": source, "out": goal},
            param_blobs={"matrix": "C", "offset": "offset", "cov": "R"},
        )
        template.ports[goal] = PortSpec(goal, goal, PortRole.DATA)
        observations[goal] = np.zeros(1, dtype=dtype)
    return template, arrays, observations


def _entropy(covariance):
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return .5 * (len(covariance) * (1 + np.log(2 * np.pi)) + logdet)


def _expected_nll(mean, covariance, target, noise):
    difference = mean - target
    return .5 * (len(mean) * np.log(2 * np.pi) + np.linalg.slogdet(noise)[1]
                 + difference @ np.linalg.solve(noise, difference)
                 + np.trace(np.linalg.solve(noise, covariance)))


def declared_rule_reference(arrays, observations):
    """NumPy moments and local Bethe terms; no production inference/math calls."""
    arrays = {name: np.asarray(value, np.float64) for name, value in arrays.items()}
    mean, covariance = arrays["m0"].copy(), arrays["P0"].copy()
    means, covariances, regimes = [mean], [covariance], []
    weights, c, r = arrays["habit"], arrays["C"], arrays["R"]
    support = weights > 0
    log_weights = np.full(3, -np.inf)
    log_weights[support] = np.log(weights[support])
    # Prior's -H term cancels its degree-two variable entropy correction.
    energy = _expected_nll(mean, covariance, arrays["m0"], arrays["P0"])
    for index in range(1, len(observations) + 1):
        target = np.asarray(observations[f"g{index}"], np.float64)
        component_mean, component_cov, scores, node_terms = [], [], [], []
        for matrix, offset, noise in zip(arrays["A"], arrays["b"], arrays["Q"]):
            predicted_mean = matrix @ mean + offset
            predicted_cov = noise + matrix @ covariance @ matrix.T
            component_mean.append(predicted_mean)
            component_cov.append(predicted_cov)
            innovation = r + c @ predicted_cov @ c.T
            scores.append(-_expected_nll(c @ predicted_mean, np.zeros_like(r), target, innovation))
            joint_mean = np.concatenate((mean, predicted_mean))
            joint_cov = np.block([[covariance, covariance @ matrix.T],
                                  [matrix @ covariance, predicted_cov]])
            h = np.concatenate((np.zeros_like(c), c), axis=1)
            gain = np.linalg.solve(innovation, h @ joint_cov).T
            joint_mean = joint_mean + gain @ (target - h @ joint_mean)
            joint_cov = joint_cov - gain @ h @ joint_cov
            residual = np.concatenate((-matrix, np.eye(2)), axis=1)
            node_terms.append(_expected_nll(residual @ joint_mean,
                                            residual @ joint_cov @ residual.T, offset, noise)
                              - _entropy(joint_cov))
        log_rho = log_weights + scores
        log_rho -= logsumexp(log_rho)
        rho = np.exp(log_rho)
        regimes.append(rho)
        energy += rho @ node_terms + rho[support] @ (log_rho[support] - log_weights[support])
        predicted_mean = weights @ np.asarray(component_mean)
        predicted_cov = sum(weight * (cov + np.outer(mu - predicted_mean, mu - predicted_mean))
                            for weight, mu, cov in zip(weights, component_mean, component_cov))
        gain = np.linalg.solve(r + c @ predicted_cov @ c.T, c @ predicted_cov).T
        mean = predicted_mean + gain @ (target - c @ predicted_mean)
        covariance = predicted_cov - gain @ c @ predicted_cov
        means.append(mean)
        covariances.append(covariance)
        energy += _expected_nll(c @ mean, c @ covariance @ c.T, target, r)
        if index < len(observations):
            # Degree-three state: two entropy corrections minus emission H.
            energy += _entropy(covariance)
    return PlanningReference(np.asarray(regimes), np.asarray(means), np.asarray(covariances),
                             float(energy))
