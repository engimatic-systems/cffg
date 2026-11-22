"""Two-pass inference for the upstream Taylor chain derived in D13.6."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import numpy as np

from cffg import ef
from cffg.kernel_runtime import parameters
from cffg.runtime_checks import state_is_finite
from cffg.runtime_types import (
    FailureDiagnostic, OuterTerminationReason, RunResult, RunStatus, _replace,
)


@dataclass(frozen=True)
class Natural:
    eta: np.ndarray
    precision: np.ndarray

    def __add__(self, other):
        return Natural(self.eta + other.eta, self.precision + other.precision)

    def moments(self):
        return np.linalg.solve(self.precision, self.eta), _inverse(self.precision)


def _symmetric(matrix):
    return (matrix + matrix.T) / 2


def _inverse(matrix):
    return _symmetric(np.linalg.solve(matrix, np.eye(len(matrix), dtype=matrix.dtype)))


def _natural(mean, covariance):
    precision = _inverse(covariance)
    return Natural(precision @ mean, precision)


class InvalidFactor(ArithmeticError):
    def __init__(self, ref, reason):
        self.factor = ref.group.members[ref.row]
        self.reason = reason
        super().__init__(self.factor, reason)


def _noise(ref, precision):
    if not ef.GAUSSIAN.is_proper_precision(precision):
        raise InvalidFactor(ref, "invalid_noise")
    return _inverse(precision)


@dataclass(frozen=True)
class Affine:
    matrix: np.ndarray
    offset: np.ndarray
    noise: np.ndarray

    def predict(self, mean, covariance):
        return (self.matrix @ mean + self.offset,
                _symmetric(self.noise + self.matrix @ covariance @ self.matrix.T))

    def backward(self, likelihood):
        # I + LQ stays invertible when a partial goal makes L singular.
        eta, precision = likelihood.eta, likelihood.precision
        system = np.eye(len(eta), dtype=eta.dtype) + precision @ self.noise
        return Natural(
            self.matrix.T @ np.linalg.solve(system, eta - precision @ self.offset),
            _symmetric(self.matrix.T @ np.linalg.solve(system, precision) @ self.matrix),
        )


@dataclass(frozen=True)
class FilteredSite:
    prediction: Natural
    observation: Natural
    emission: Affine
    transition: Affine | None


def _observation_update(emission, weights, mean, covariance, value, present):
    matrix, offset = emission.matrix, emission.offset
    if not present:
        return Natural(np.zeros_like(mean), np.zeros_like(covariance)), 0.
    predicted, innovation = emission.predict(mean, covariance)
    chol = np.linalg.cholesky(innovation)
    error = np.linalg.solve(chol, value - predicted)
    cost = .5 * (len(value) * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum()
                 + error @ error)
    return Natural(matrix.T @ (weights @ (value - offset)),
                   _symmetric(matrix.T @ weights @ matrix)), float(cost)


class DirectedChain:
    def __init__(self, arena, plan):
        self.arena = arena
        self.plan = plan
        self.eta_starts = np.asarray(arena.g_eta_starts)
        self.lam_starts = np.asarray(arena.g_lam_starts)
        self.obs_starts = np.asarray(arena.g_obs_starts)
        self.means = {
            member: group.kernel.binding.taylor_linearize
            for group in arena.gauss_groups if group.type == "gaussian_nonlinear"
            for member in group.members
        }
        # A caller can supply a custom State. A nonvacuous exterior message
        # violates the DATA half-edge assumed by the chain derivation.
        exterior = [site.emission.gather[0] for site in plan.sites]
        self.external_eta = np.concatenate([
            np.arange(self.eta_starts[s], self.eta_starts[s + 1]) for s in exterior])
        self.external_lam = np.concatenate([
            np.arange(self.lam_starts[s], self.lam_starts[s + 1]) for s in exterior])

    def linearize(self, member, center, theta):
        return tuple(np.asarray(v) for v in self.means[member](center, theta))

    def admits(self, state):
        devices = getattr(state.g_eta, "devices", lambda: set())()
        return (len(devices) == 1 and next(iter(devices)).platform == "cpu"
                and state_is_finite(state)
                and np.all(np.isin(np.asarray(state.g_mask), (0, 1)))
                and not np.any(np.asarray(state.g_eta)[self.external_eta])
                and not np.any(np.asarray(state.g_lam)[self.external_lam]))

    def candidate(self, state):
        live = {name: np.asarray(value) for name, value in state.params.items()}

        def row(ref):
            return parameters(ref.group, live, ref.row)

        prior = self.plan.prior
        start = row(prior)
        mean, covariance = start["m"], _noise(prior, start["W"])
        prediction = Natural(start["W"] @ mean, start["W"])
        observations, masks = np.asarray(state.g_obs), np.asarray(state.g_mask)
        sites = []
        cost = 0.
        for site in self.plan.sites:
            ref = site.emission
            obs = row(ref)
            emission = Affine(obs["A"], obs["b"], _noise(ref, obs["W"]))
            obs_row = site.obs_row
            value = observations[self.obs_starts[obs_row]:self.obs_starts[obs_row + 1]]
            likelihood, increment = _observation_update(
                emission, obs["W"], mean, covariance, value, masks[obs_row])
            cost += increment
            transition = None
            if site.transition is not None:
                ref = site.transition
                params = row(ref)
                q = _noise(ref, params["W"])
                filtered_mean, filtered_cov = (prediction + likelihood).moments()
                member = ref.group.members[ref.row]
                a, next_mean = self.linearize(member, filtered_mean, params["theta"])
                if not (np.isfinite(next_mean).all() and np.isfinite(a).all()):
                    raise InvalidFactor(ref, "nonfinite_function")
                transition = Affine(a, next_mean - a @ filtered_mean, q)
                mean, covariance = next_mean, _symmetric(q + a @ filtered_cov @ a.T)
            sites.append(FilteredSite(prediction, likelihood, emission, transition))
            if transition is not None:
                prediction = _natural(mean, covariance)

        eta, precision = np.zeros_like(state.g_eta), np.zeros_like(state.g_lam)

        def put(slot, message):
            eta[self.eta_starts[slot]:self.eta_starts[slot + 1]] = message.eta
            precision[self.lam_starts[slot]:self.lam_starts[slot + 1]] = message.precision.ravel()

        future = Natural(np.zeros_like(mean), np.zeros_like(covariance))
        for index in reversed(range(len(sites))):
            site = self.plan.sites[index]
            local = sites[index]
            predicted_by = self.plan.sites[index - 1].transition if index else prior
            put(predicted_by.scatter[0], local.prediction)
            put(site.emission.scatter[1], local.observation)
            cavity = local.prediction + future
            put(site.emission.scatter[0], _natural(*local.emission.predict(*cavity.moments())))
            right = local.observation + future
            if site.transition is not None:
                put(site.transition.scatter[1], future)
                pred_slot, obs_slot, future_slot = site.equality_scatter
                put(pred_slot, right)
                put(obs_slot, cavity)
                put(future_slot, local.prediction + local.observation)
            if index:
                future = sites[index - 1].transition.backward(right)
        device = next(iter(state.g_eta.devices()))
        return _replace(state, g_eta=jax.device_put(eta, device),
                        g_lam=jax.device_put(precision, device)), cost

    def run(self, state, tol):
        """One proposal and an independent ledger/map check at its own cavities."""
        from cffg.directed_chain_certificate import certify

        tolerance = max(float(tol), 64 * np.finfo(state.g_eta.dtype).eps)
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                candidate, evidence = self.candidate(state)
                check = certify(self.arena, candidate, self.linearize, evidence, tolerance)
        except InvalidFactor as error:
            return RunResult(
                state, RunStatus.DIVERGED, 1, (), (), float("inf"),
                inner_status=RunStatus.DIVERGED, inner_residual=float("inf"),
                termination_reason=OuterTerminationReason.INNER_FAILURE, directed_attempts=1,
                failure=FailureDiagnostic(error.reason, error.factor, "directed_chain", 1))
        except (np.linalg.LinAlgError, FloatingPointError):
            check = None
        if check is None or not check.passed:
            # The caller can spend the remaining budget on the coupled solver.
            residual = float("inf") if check is None else check.residual
            return RunResult(
                state, RunStatus.BUDGET_EXHAUSTED, 1, (), (), residual,
                termination_reason=OuterTerminationReason.DIRECTED_CHAIN_UNCERTIFIED,
                directed_attempts=1,
                failure=FailureDiagnostic("uncertified_root", None, "directed_chain", 1))
        return RunResult(
            candidate, RunStatus.CONVERGED, 1, (check.residual,), (check.free_energy,),
            check.residual, outer_residual=check.residual,
            termination_reason=OuterTerminationReason.DIRECTED_CHAIN_CERTIFIED, directed_attempts=1)


def after_directed_attempt(result):
    """Count the spent pass and keep a fallback failure's iteration correct."""
    failure = result.failure
    if failure is not None:
        failure = replace(failure, iteration=failure.iteration + 1)
    return replace(result, iterations=result.iterations + 1, directed_attempts=1, failure=failure)
