"""Guarded adaptive admission using ordinary transport and run-local latches."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.custom_batching import sequential_vmap

from cffg.execution.contracts import ExecutionDelta, GaussianMessage
from cffg.execution.failures import FailureIndex, simultaneous_failure
from cffg.execution.registered import execute_gaussian
from cffg.execution.transport import _gauss_rows, _obs_rows
from cffg.failure_facts import FailureFact, FailureReason, empty_failure, first_failure, local_failure
from cffg.gaussian_readiness import InputReadiness, _has_subnormal
from cffg.kernel_runtime import parameters as kernel_parameters
from cffg.schedule_contracts import SweepResult


class StartupCheck(NamedTuple):
    readiness: tuple[jax.Array, ...]
    failure: FailureFact
    pending: FailureFact

    @property
    def ready(self):
        return jnp.all(jnp.stack([jnp.all(r == InputReadiness.READY) for r in self.readiness]))


class StartupResult(NamedTuple):
    sweep: SweepResult
    admitted: tuple[jax.Array, ...]
    check: StartupCheck


@dataclass(frozen=True)
class StartupProgram:
    eligibility: Callable
    inspect: Callable
    step: Callable


def make_startup_program(arena, make_sweep, *, skip_scheduled=False, propagate=None):
    """Compose a guarded sweep; no energy, convergence or persistent state."""
    groups = tuple(g for g in arena.gauss_groups if g.kernel is not None and g.kernel.binding.iterative)
    if not groups:
        return None
    indices = {g.name: i for i, g in enumerate(groups)}
    failures = FailureIndex(arena)

    def local_metadata(group, eta, precision, params, observation, mask):
        callback = group.kernel.binding.admission
        if callback is None:
            # Undeclared kernels get strict execution, never implicit deferral.
            size = eta.shape[0]
            return jnp.full(size, InputReadiness.READY, jnp.int32), jnp.zeros(size, jnp.int32)
        return callback(eta, precision, params, group.dims,
                        group.observed_role, observation, mask)

    def group_inputs(group, g_eta, g_lam, g_obs, g_mask, params):
        eta, precision = _gauss_rows(arena, g_eta, g_lam, group.gather, group.dims)
        observation = mask = None
        if group.obs_rows is not None:
            observation = _obs_rows(arena, g_obs, group.obs_rows, group.observed_dim)
            mask = g_mask[jnp.asarray(group.obs_rows)]
        return eta, precision, kernel_parameters(group, params), observation, mask

    def eligibility(g_eta, g_lam):
        rows = []
        for group in groups:
            eta, precision = _gauss_rows(arena, g_eta, g_lam, group.scatter, group.dims)
            rows.append(jnp.all(eta == 0, axis=(1, 2)) & jnp.all(precision == 0, axis=(1, 2, 3)))
            rows[-1] &= ~jax.vmap(_has_subnormal)(eta) & ~jax.vmap(_has_subnormal)(precision)
        return tuple(rows)

    def inspect(g_eta, g_lam, g_obs, g_mask, params, eligible, admitted, *, nonfinite_only=False):
        readiness, failure, pending = [], empty_failure(), empty_failure()
        for index, group in enumerate(groups):
            inputs = group_inputs(group, g_eta, g_lam, g_obs, g_mask, params)
            r, reasons = local_metadata(group, *inputs)
            unready = r != InputReadiness.READY
            forbidden = unready & (~eligible[index] | admitted[index])
            reasons = jnp.where((reasons == 0) & forbidden, FailureReason.IMPROPER_CAVITY, reasons)
            if nonfinite_only:
                corrupt = jnp.zeros_like(reasons, dtype=bool)
                for value in jax.tree.leaves(inputs):
                    corrupt |= jnp.any(~jnp.isfinite(value), axis=tuple(range(1, value.ndim)))
                reasons = jnp.where(corrupt, reasons, 0)
            status = jnp.where(reasons != 0, 2, 0).astype(jnp.int32)
            ids = failures.ids(group)
            failure = simultaneous_failure(failure, local_failure(ids, status=status, reasons=reasons))
            pending = simultaneous_failure(pending, local_failure(
                ids, status=unready.astype(jnp.int32)))
            readiness.append(r)
        return StartupCheck(tuple(readiness), failure, pending)

    @sequential_vmap
    def step(messages, g_eta, g_lam, g_obs, g_mask, ef_eta, ef_obs, ef_mask,
             params, eligible, admitted):
        before = inspect(g_eta, g_lam, g_obs, g_mask, params, eligible, admitted)

        def guarded(group, eta, precision, parameters, observation, mask):
            if group.name not in indices or group.kernel.binding.admission is None:
                return execute_gaussian(group, eta, precision, parameters, observation, mask)
            index = indices[group.name]
            r, reasons = local_metadata(group, eta, precision, parameters, observation, mask)
            defer = (r == InputReadiness.PARTIAL) & eligible[index] & ~admitted[index]
            reasons = jnp.where((reasons == 0) & (r != InputReadiness.READY) & ~defer,
                                FailureReason.IMPROPER_CAVITY, reasons)
            owned_eta, owned_precision = _gauss_rows(arena, g_eta, g_lam, group.scatter, group.dims)
            y = jnp.zeros((eta.shape[0], group.dims[0]), eta.dtype) if observation is None else observation
            present = jnp.zeros(eta.shape[0], eta.dtype) if mask is None else mask

            def one(values):
                e, p, par, obs, mk, old_e, old_p, wait, reason = values

                def execute(_):
                    delta = execute_gaussian(
                        group, e[None], p[None], jax.tree.map(lambda v: v[None], par),
                        obs[None] if observation is not None else None,
                        mk[None] if mask is not None else None,
                    )
                    delta = ExecutionDelta(
                        delta.outgoing,
                        jnp.zeros(1, eta.dtype) if delta.residual is None else delta.residual,
                        jnp.zeros(1, jnp.int32) if delta.status is None else delta.status,
                        jnp.zeros(1, jnp.int32) if delta.failure_reason is None else delta.failure_reason,
                    )
                    return jax.tree.map(lambda v: v[0], delta)

                def retain(_):
                    outgoing = tuple(GaussianMessage(old_e[i, :d], old_p[i, :d, :d])
                                     for i, d in enumerate(group.dims))
                    failed = reason != 0
                    return ExecutionDelta(outgoing, jnp.where(failed, jnp.inf, 0.).astype(eta.dtype),
                                          jnp.where(failed, 2, 0).astype(jnp.int32), reason)

                return jax.lax.cond(~wait & (reason == 0), execute, retain, operand=None)

            # Scalar predicates remain real guards even when an outer caller vmaps step.
            return jax.lax.map(one, (eta, precision, parameters, y, present,
                                     owned_eta, owned_precision, defer, reasons))

        raw = make_sweep(arena, skip_scheduled, _gaussian_executor=guarded)
        sweep = raw(messages, g_eta, g_lam, g_obs, g_mask, ef_eta, ef_obs, ef_mask, params, 0., False)
        if propagate is not None:
            m, e, p, ef, fact = propagate(*sweep[:4], g_obs, g_mask, ef_obs, ef_mask, params)
            sweep = (m, e, p, ef, *sweep[4:6], first_failure(sweep[6], fact))
        next_admitted = tuple(a | (r == InputReadiness.READY)
                              for a, r in zip(admitted, before.readiness, strict=True))
        after = inspect(sweep[1], sweep[2], g_obs, g_mask, params, eligible, next_admitted)
        return StartupResult(sweep, next_admitted, after)

    return StartupProgram(eqx.filter_jit(eligibility), eqx.filter_jit(inspect), eqx.filter_jit(step))
