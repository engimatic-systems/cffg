"""Directed Taylor-chain reference for the retained PR 185 reactor fixture.

Run from this checkout with uv run python tools/benchmark_cstr_chain.py.
The final backward messages include future goals. Only the forward cavity
determines each Taylor expansion. All covariance matrices in this fixture
are proper and its factor graph is a chain with unary evidence.
"""
import argparse
import json
import os
import time

os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax
import jax.numpy as jnp
import numpy as np

from cffg.nonlinear import local_result
from cstr_runtime_fixture import (
    CONFIGURATION, HORIZON, PAST, TOLERANCE, cstr_rk4, model, observations,
)


def symmetric(matrix):
    return (matrix + matrix.T) / 2


def natural(mean, covariance):
    precision = symmetric(np.linalg.solve(covariance, np.eye(len(mean))))
    return precision @ mean, precision


def value_with_aux(state, parameters):
    value = cstr_rk4(state, parameters, CONFIGURATION)
    return value, value


linearize = jax.jit(jax.jacfwd(value_with_aux, argnums=0, has_aux=True))


def solve(arrays, data):
    """Return forward, observation and backward messages plus the frozen tree NLL."""
    mean, covariance = arrays["m0"], arrays["P0"]
    transitions, messages = [], []
    energy = 0.
    for index in range(PAST + HORIZON + 1):
        pred = natural(mean, covariance)
        kind = "past" if index < PAST else "now" if index == PAST else "goal"
        observation = data[f"e{index}"]
        obs = np.zeros_like(mean), np.zeros_like(covariance)
        if observation is not None:
            h, r, offset = (arrays[f"{prefix}_{kind}"] for prefix in ("G", "R", "z"))
            y = observation - offset
            delta = y - h @ mean
            innovation = symmetric(r + h @ covariance @ h.T)
            chol = np.linalg.cholesky(innovation)
            whitened = np.linalg.solve(chol, delta)
            energy += .5 * (len(y) * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum()
                            + whitened @ whitened)
            obs = h.T @ np.linalg.solve(r, y), symmetric(h.T @ np.linalg.solve(r, h))
        filtered_precision = pred[1] + obs[1]
        mean = np.linalg.solve(filtered_precision, pred[0] + obs[0])
        covariance = symmetric(np.linalg.solve(filtered_precision, np.eye(len(mean))))
        messages.append({"pred": pred, "obs": obs})
        if index < PAST + HORIZON:
            a, predicted = map(np.asarray, linearize(jnp.asarray(mean), jnp.asarray(arrays["theta"])))
            q = arrays["Q" if index < PAST else "Qp"]
            transitions.append((a, predicted - a @ mean, q))
            mean, covariance = predicted, symmetric(q + a @ covariance @ a.T)

    eta, precision = np.zeros_like(mean), np.zeros_like(covariance)
    for index in reversed(range(len(messages))):
        site = messages[index]
        site["future"] = eta, precision
        if index:
            eta, precision = eta + site["obs"][0], precision + site["obs"][1]
            a, offset, q = transitions[index - 1]
            # Gaussian integration of a possibly rank-deficient likelihood.
            # I + L Q is invertible for positive Q and positive semidefinite L.
            system = np.eye(len(mean)) + precision @ q
            eta = a.T @ np.linalg.solve(system, eta - precision @ offset)
            precision = symmetric(a.T @ np.linalg.solve(system, precision) @ a)
    return messages, float(energy)


def posterior(site):
    eta = sum(pair[0] for pair in site.values())
    precision = sum(pair[1] for pair in site.values())
    covariance = symmetric(np.linalg.solve(precision, np.eye(len(eta))))
    np.linalg.cholesky(covariance)
    return np.linalg.solve(precision, eta), covariance



def physical_mean(state, parameters):
    return cstr_rk4(state, parameters, CONFIGURATION)


@jax.jit
def transition_checks(left_eta, left_precision, right_eta, right_precision, parameters, noise):
    """Evaluate the engine's D13 rule at all proposed cavities."""
    def check(ei, li, eo, lo, q):
        result = local_result(physical_mean, ei, li, eo, lo, parameters, q, "linearization")
        return {key: result[key] for key in (
            "eta_in", "precision_in", "eta_out", "precision_out", "energy", "proper", "status",
        )}

    return jax.vmap(check)(left_eta, left_precision, right_eta, right_precision, noise)


def joint_transition_checks(left_eta, left_precision, right_eta, right_precision, parameters, noise):
    """Independently integrate each two-state tilted Gaussian in canonical form."""
    result = {key: [] for key in (
        "eta_in", "precision_in", "eta_out", "precision_out", "energy",
    )}
    for ei, li, eo, lo, q in zip(left_eta, left_precision, right_eta, right_precision, noise, strict=True):
        d = len(ei)
        center = np.linalg.solve(li, ei)
        a, value = map(np.asarray, linearize(jnp.asarray(center), jnp.asarray(parameters)))
        offset = value - a @ center
        w = np.linalg.solve(q, np.eye(d))
        precision = symmetric(np.block([[li + a.T @ w @ a, -a.T @ w], [-w @ a, lo + w]]))
        eta = np.concatenate((ei - a.T @ w @ offset, eo + w @ offset))
        chol = np.linalg.cholesky(precision)
        cov = symmetric(np.linalg.solve(precision, np.eye(2 * d)))
        mean = np.linalg.solve(precision, eta)
        # Integrate the joint and divide out each incoming cavity separately.
        for side, block, incoming_eta, incoming_precision in (
            ("in", slice(0, d), ei, li), ("out", slice(d, None), eo, lo),
        ):
            marginal_eta, marginal_precision = natural(mean[block], cov[block, block])
            result["eta_" + side].append(marginal_eta - incoming_eta)
            result["precision_" + side].append(marginal_precision - incoming_precision)
        transform = np.concatenate((-a, np.eye(d)), axis=1)
        entropy = d * (1 + np.log(2 * np.pi)) - np.log(np.diag(chol)).sum()
        result["energy"].append(gaussian_cost(transform @ mean - offset, transform @ cov @ transform.T, q)
                                - entropy)
    return {key: np.asarray(rows) for key, rows in result.items()}


def gaussian_cost(error, covariance, noise):
    chol = np.linalg.cholesky(noise)
    whitened = np.linalg.solve(chol, error)
    return .5 * (len(error) * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum()
                 + whitened @ whitened + np.trace(np.linalg.solve(noise, covariance)))


def check_candidate(arrays, data, messages, beliefs, evidence_cost, *, stock_rules=False):
    """Check all sites against the bound data and sum the frozen Bethe ledger."""
    left = [(s["pred"][0] + s["obs"][0], s["pred"][1] + s["obs"][1]) for s in messages[:-1]]
    right = [(s["future"][0] + s["obs"][0], s["future"][1] + s["obs"][1]) for s in messages[1:]]
    noise = np.stack([arrays["Q"]] * PAST + [arrays["Qp"]] * HORIZON)
    check = transition_checks if stock_rules else joint_transition_checks
    left_eta, left_precision = map(np.stack, zip(*left))
    right_eta, right_precision = map(np.stack, zip(*right))
    checked = jax.device_get(check(left_eta, left_precision, right_eta, right_precision, arrays["theta"], noise))
    if stock_rules and (not checked["proper"].all() or checked["status"].any()):
        raise ArithmeticError("An engine nonlinear rule rejected its proposed cavity")
    def mismatch(actual, reference):
        error = np.max(np.abs(actual - reference) / (1 + np.abs(reference)))
        if not np.isfinite(error):
            raise ArithmeticError("Nonfinite message mismatch")
        return float(error)

    prior = natural(arrays["m0"], arrays["P0"])
    residual = max(mismatch(value, ref) for value, ref in zip(messages[0]["pred"], prior, strict=True))
    for value in messages[-1]["future"]:
        residual = max(residual, mismatch(value, np.zeros_like(value)))
    for key, sites, source, index in (
        ("eta_out", messages[1:], "pred", 0), ("precision_out", messages[1:], "pred", 1),
        ("eta_in", messages[:-1], "future", 0), ("precision_in", messages[:-1], "future", 1),
    ):
        value = np.stack([site[source][index] for site in sites])
        residual = max(residual, mismatch(checked[key], value))
    # Unary-local entropies cancel their share of the edge corrections.
    # Each internal state still contributes one entropy between its two transitions.
    energy = checked["energy"].sum() + gaussian_cost(
        beliefs[0][0] - arrays["m0"], beliefs[0][1], arrays["P0"])
    for index, (mean, covariance) in enumerate(beliefs):
        if 0 < index < len(beliefs) - 1:
            chol = np.linalg.cholesky(covariance)
            energy += .5 * len(mean) * (1 + np.log(2 * np.pi)) + np.log(np.diag(chol)).sum()
        observation = data[f"e{index}"]
        eta, precision = messages[index]["obs"]
        if observation is None:
            residual = max(residual, mismatch(eta, np.zeros_like(eta)),
                           mismatch(precision, np.zeros_like(precision)))
        else:
            kind = "past" if index < PAST else "now" if index == PAST else "goal"
            h = arrays[f"G_{kind}"]
            r = arrays[f"R_{kind}"]
            residual = max(residual, mismatch(eta, h.T @ np.linalg.solve(r, observation - arrays[f"z_{kind}"])),
                           mismatch(precision, h.T @ np.linalg.solve(r, h)))
            energy += gaussian_cost(
                observation - arrays[f"z_{kind}"] - h @ mean,
                h @ covariance @ h.T, arrays[f"R_{kind}"])
    error = abs(energy - evidence_cost) / (1 + abs(evidence_cost))
    if not np.isfinite(error) or error > TOLERANCE or residual > TOLERANCE:
        raise ArithmeticError(f"Directed chain check failed: residual={residual}, ledger error={error}")
    return dict(local_message_residual=float(residual), ledger=float(energy), ledger_error=float(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-rules", action="store_true",
                        help="Use the original compiled D13 transition rule for the local check")
    parser.add_argument("--change", type=int, default=0)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--goal", type=float, default=0.)
    args = parser.parse_args()
    _, arrays = model()
    data = observations(args.change, history=args.history, goal=args.goal)
    start = time.perf_counter()
    messages, energy = solve(arrays, data)
    beliefs = [posterior(site) for site in messages]
    solved = time.perf_counter()
    checks = check_candidate(arrays, data, messages, beliefs, energy, stock_rules=args.stock_rules)
    print(json.dumps(dict(check="engine_local_rules" if args.stock_rules else "joint_gaussians",
                          solve_seconds=solved - start, checked_seconds=time.perf_counter() - start,
                          coolant_kelvin=298.5 + 3.5 * beliefs[PAST][0][3], evidence_cost=energy, **checks)))


if __name__ == "__main__":
    main()
