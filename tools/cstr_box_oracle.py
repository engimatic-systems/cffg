"""Offline correlated-box integration on the frozen Gaussian CSTR models from the paired study.

Run with: uv run --frozen python tools/cstr_box_oracle.py STUDY/run --output NEW_FILE.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import benchmark_cstr_chain as chain
import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp
from scipy.stats import qmc


def action_covariance(arrays, candidate):
    """Assemble the full precision and check its marginals before extracting the action block."""
    sites = candidate["messages"]
    dimension = len(arrays["m0"])
    size = dimension * len(sites)
    precision = np.zeros((size, size))
    precision[:dimension, :dimension] = np.linalg.inv(arrays["P0"])
    for k, site in enumerate(sites):
        current = slice(dimension*k, dimension*(k+1))
        precision[current, current] += site["obs"][1]
        if k == len(sites) - 1:
            break
        following = slice(dimension*(k+1), dimension*(k+2))
        eta, lam = [np.add(site["pred"][i], site["obs"][i]) for i in (0, 1)]
        matrix, _ = chain.linearize(np.linalg.solve(lam, eta), arrays["theta"])
        matrix = np.asarray(matrix)
        weight = np.linalg.inv(arrays["Q"] if k < chain.PAST else arrays["Qp"])
        precision[current, current] += matrix.T @ weight @ matrix
        precision[following, following] += weight
        precision[current, following] -= matrix.T @ weight
        precision[following, current] -= weight @ matrix
    covariance = np.linalg.solve(precision, np.eye(size))
    for k, (_, saved) in enumerate(candidate["beliefs"]):
        stage = slice(dimension*k, dimension*(k+1))
        np.testing.assert_allclose(covariance[stage, stage], saved, rtol=2e-9, atol=2e-10)
    action = dimension * np.arange(chain.PAST, chain.PAST + chain.HORIZON) + (dimension - 1)
    return covariance[np.ix_(action, action)]


def integrate(mean, covariance, order, seed, exponent):
    """Genz conditional transform with one scrambled Sobol point set, in normalized units."""
    center = mean[order]
    chol = np.linalg.cholesky(covariance[np.ix_(order, order)])
    uniforms = qmc.Sobol(len(center), scramble=True, rng=np.random.default_rng(seed)).random_base2(exponent)
    standard = np.zeros_like(uniforms)
    log_weight = np.zeros(len(uniforms))
    for i, coordinate in enumerate(order):
        location = center[i] + standard[:, :i] @ chol[i, :i]
        lower = (-1 - location) / chol[i, i]
        upper = (1 - location) / chol[i, i]
        # Reflect upper tails so the interval mass avoids subtraction of two values near one.
        reflect = lower > 0
        log_left = log_ndtr(np.where(reflect, -upper, lower))
        log_right = log_ndtr(np.where(reflect, -lower, upper))
        log_mass = log_right + np.log(-np.expm1(log_left - log_right))
        quantile = ndtri_exp(np.logaddexp(log_left, np.log(uniforms[:, i]) + log_mass))
        standard[:, i] = np.where(reflect, -quantile, quantile)
        log_weight += log_mass
        if coordinate == 0:
            first_action = location + chol[i, i] * standard[:, i]
    if not np.isfinite(log_weight).all() or not (abs(first_action) <= 1 + 1e-13).all():
        raise ArithmeticError("Unresolved conditional box integration")
    total = logsumexp(log_weight)
    return total - np.log(len(log_weight)), np.dot(np.exp(log_weight - total), first_action)


def estimate(mean, covariance, exponent):
    results = {}
    order = np.arange(len(mean))
    for label, permutation, start in (("forward", order, 98060), ("reverse", order[::-1], 98160)):
        replicas = np.asarray([integrate(mean, covariance, permutation, start+i, exponent) for i in range(12)])
        weights = np.exp(replicas[:, 0] - logsumexp(replicas[:, 0]))
        results[label] = dict(logmass=logsumexp(replicas[:, 0]) - np.log(len(replicas)),
                              first_mean=weights @ replicas[:, 1],
                              replicate_SE_logmass=replicas[:, 0].std(ddof=1)/np.sqrt(len(replicas)),
                              replicate_SE_first_mean=replicas[:, 1].std(ddof=1)/np.sqrt(len(replicas)),
                              replicas=replicas.tolist())
    return results


def audit(study):
    _, arrays = chain.model()
    small = estimate(np.array([0., 2.]), np.linalg.inv([[2., -1.], [-1., 2.]]), 14)
    for result in small.values():
        np.testing.assert_allclose(result["first_mean"], -.31785828753788836, rtol=0, atol=1e-5)
        np.testing.assert_allclose(result["logmass"], -2.630977166506961, rtol=0, atol=1e-5)
    windows = []
    for seed in range(1, 11):
        for tick in (0, 5, 20, 25, 40):
            path = study / f"seed-{seed:02d}/directed/decision-{tick:02d}.json"
            raw = path.read_bytes()
            candidate = json.loads(raw)["candidate"]
            covariance = action_covariance(arrays, candidate)
            correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
            windows.append(dict(seed=seed, tick=tick, adjacent=np.diag(correlation, 1).tolist(),
                                decision_sha256=hashlib.sha256(raw).hexdigest()))
            if seed == 1 and tick == 0:
                chosen_covariance = covariance
                chosen_mean = np.array([belief[0][-1] for belief in
                                        candidate["beliefs"][chain.PAST:chain.PAST+chain.HORIZON]])
    return dict(windows=windows, two_action_control=small, normalized_bounds=[-1, 1],
                chosen=dict(seed=1, tick=0, mean=chosen_mean.tolist(), covariance=chosen_covariance.tolist()),
                numerics=dict(scrambles=12, forward_seed_start=98060, reverse_seed_start=98160,
                              coarse_points_per_scramble=2**14, fine_points_per_scramble=2**17),
                coarse=estimate(chosen_mean, chosen_covariance, 14),
                frozen_reactor=estimate(chosen_mean, chosen_covariance, 17))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.study)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
