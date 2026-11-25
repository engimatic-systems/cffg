"""Compiled control flow for deterministic scheduled iteration."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg.failure_facts import FailureAt, FailurePhase, empty_failure, empty_failure_at, record_failure
from cffg.iteration import allows_initial_undefined_energy
from cffg.iteration_checks import make_iteration_support
from cffg.iteration_tolerance import energy_within, tolerance
from cffg.iteration_types import IterationOutcome, IterationReason as Reason
from cffg.runtime_types import State, _replace


class _Progress(NamedTuple):
    state: State
    reason: jax.Array
    phase: jax.Array
    count: jax.Array
    energy_count: jax.Array
    deltas: jax.Array
    energies: jax.Array
    previous_fe: jax.Array
    residual: jax.Array
    # Index of the last FE residual's second operand. The two stored
    # energies can differ by more than float32 max when x64 is disabled.
    comparison: jax.Array
    inner_code: jax.Array
    inner_residual: jax.Array
    finite: jax.Array
    zero_support: jax.Array
    failure: FailureAt


_BLOCKS = ("messages", "g_eta", "g_lam", "ef_eta")


def _state_finite(state):
    finite = jnp.asarray(True)
    for leaf in jax.tree.leaves(state):
        if jnp.issubdtype(leaf.dtype, jnp.inexact) and leaf.size:
            finite &= jnp.all(jnp.isfinite(leaf))
    return finite


def _max_delta(before, after):
    delta = jnp.zeros((), dtype=jnp.result_type(*(getattr(before, key) for key in _BLOCKS)))
    for key in _BLOCKS:
        old = getattr(before, key)
        if old.size:
            delta = jnp.maximum(delta, jnp.max(jnp.abs(getattr(after, key) - old)))
    return delta


def _flag(value):
    return None if value < 0 else bool(value)


def _host_outcome(state, p):
    residual = float(p.residual)
    if p.comparison >= 1:
        residual = abs(float(p.energies[p.comparison]) - float(p.energies[p.comparison - 1]))
    return IterationOutcome(
        state, Reason(int(p.reason)), p.deltas[:int(p.count)],
        p.energies[:int(p.energy_count)], residual, int(p.inner_code),
        float(p.inner_residual), _flag(p.finite), _flag(p.zero_support),
        p.failure, FailurePhase(int(p.phase)),
    )


class CompiledIteration:
    """One cached scalar loop, vectorized over independent Colony members."""

    def __init__(self, engine, hooks):
        self.hooks = hooks
        self.step, self.prime, energy, override_energy, _ = hooks
        self.bad_support = make_iteration_support(engine.arena)
        self.energy = override_energy or (lambda s: energy(
            s.messages, s.g_eta, s.g_lam, s.g_obs, s.g_mask,
            s.ef_eta, s.ef_obs, s.ef_mask, s.params,
        ))
        self.is_max = engine.arena.semiring == "max"
        self.objective = engine.objective if self.is_max else None
        self.allow_undefined = allows_initial_undefined_energy(engine.arena)
        controls = ("n_iters", "harden_last", "defer_hard", "damp")
        self.single = jax.jit(self._single, static_argnames=controls)
        self.batch = jax.jit(self._batch, static_argnames=controls)
        self.resume_hard = jax.jit(self._resume, static_argnames=("budget", "batched", "damp"))

    def _initial(self, state, n_iters):
        finite = _state_finite(state)
        bad = self.bad_support(state)
        reason = jnp.select(
            [~finite, bad], [Reason.INITIAL_NONFINITE, Reason.INITIAL_EMPTY_SUPPORT],
            default=Reason.RUNNING,
        ).astype(jnp.int32)
        # The initial support sentinel needs one slot even with a zero budget.
        dtype = jnp.result_type(*(getattr(state, name) for name in _BLOCKS))
        dtype = jnp.promote_types(dtype, jnp.float64 if jax.config.x64_enabled else jnp.float32)
        nan, inf = jnp.asarray(jnp.nan, dtype), jnp.asarray(jnp.inf, dtype)
        return _Progress(
            state=state, reason=reason, phase=jnp.int32(FailurePhase.INITIALIZATION),
            count=jnp.int32(0), energy_count=(finite & bad).astype(jnp.int32),
            deltas=jnp.zeros(n_iters, dtype), energies=jnp.full(max(1, n_iters), inf, dtype),
            previous_fe=nan, residual=inf, comparison=jnp.int32(-1),
            inner_code=jnp.where(finite, 0, 2).astype(jnp.int32),
            inner_residual=jnp.where(finite, 0., inf),
            finite=finite.astype(jnp.int32), zero_support=jnp.where(finite, bad.astype(jnp.int32), -1),
            failure=empty_failure_at(),
        )

    def _primed(self, p):
        s = p.state
        m, e, l, f, *facts = self.prime(
            s.messages, s.g_eta, s.g_lam, s.ef_eta, s.g_obs, s.g_mask,
            s.ef_obs, s.ef_mask, s.params,
        )
        return p._replace(
            state=_replace(s, messages=m, g_eta=e, g_lam=l, ef_eta=f),
            finite=jnp.int32(-1), zero_support=jnp.int32(-1),
            failure=record_failure(p.failure, facts[0] if facts else empty_failure(),
                                   FailurePhase.PRIMING, 0),
        )

    def _advance(self, p, *, damp, harden, tol):
        s = p.state
        m, e, l, f, inner, code, *facts = self.step(
            s.messages, s.g_eta, s.g_lam, s.g_obs, s.g_mask,
            s.ef_eta, s.ef_obs, s.ef_mask, s.params, damp, harden,
        )
        new = _replace(s, messages=m, g_eta=e, g_lam=l, ef_eta=f)
        delta = _max_delta(s, new)
        finite_delta = jnp.isfinite(delta)
        finite = _state_finite(new)
        valid = finite_delta & finite & jnp.isfinite(inner)
        code = jnp.asarray(code, jnp.int32)
        supported_code = (code == 0) | (code == 1)
        check_support = valid & supported_code
        bad = self.bad_support(new)
        reason = jnp.select(
            [~valid, ~supported_code, bad, code == 1],
            [Reason.UPDATE_NONFINITE, Reason.INNER_DIVERGED, Reason.EMPTY_SUPPORT, Reason.INNER_BUDGET],
            default=Reason.RUNNING,
        ).astype(jnp.int32)
        normal = reason == Reason.RUNNING
        empty = reason == Reason.EMPTY_SUPPORT
        inf = jnp.full_like(p.residual, jnp.inf)
        nan = jnp.full_like(p.residual, jnp.nan)
        residual = jnp.where(reason == Reason.INNER_BUDGET, inner, inf)
        record_energy = empty
        energy = inf
        previous = p.previous_fe
        comparison = jnp.int32(-1)
        if self.is_max:
            stationary = normal & (delta <= tol.delta)
            reason = jnp.where(stationary, Reason.MAX_STATIONARY, reason)
            residual = jnp.where(normal, delta, residual)
        else:
            def evaluate(st):
                return jnp.asarray(self.energy(st), dtype=p.energies.dtype)
            fe = jax.lax.cond(normal, evaluate, lambda _: nan, new)
            defined = normal & jnp.isfinite(fe)
            compared = defined & jnp.isfinite(previous)
            invalid = normal & ~defined & (jnp.isfinite(previous) | (not self.allow_undefined))
            difference = jnp.abs(fe - previous)
            stationary = compared & (delta <= tol.delta) & energy_within(fe, previous, tol)
            reason = jnp.where(invalid, Reason.OBJECTIVE_NONFINITE,
                               jnp.where(stationary, Reason.FE_STATIONARY, reason))
            retain = normal & ~invalid & ~compared
            residual = jnp.where(compared, difference, jnp.where(retain, p.residual, residual))
            comparison = jnp.where(compared, p.energy_count, jnp.where(retain, p.comparison, -1))
            previous = jnp.where(defined, fe, previous)
            energy = jnp.where(defined, fe, inf)
            record_energy |= defined

        return p._replace(
            state=new, reason=reason.astype(jnp.int32), count=p.count + 1,
            energy_count=p.energy_count + record_energy.astype(jnp.int32),
            deltas=p.deltas.at[p.count].set(delta),
            energies=p.energies.at[p.energy_count].set(energy),
            previous_fe=previous, residual=residual, comparison=comparison,
            inner_code=code,
            inner_residual=jnp.asarray(inner, p.inner_residual.dtype),
            finite=jnp.where(finite_delta, finite.astype(jnp.int32), -1),
            zero_support=jnp.where(check_support, bad.astype(jnp.int32), -1),
            failure=record_failure(p.failure, facts[0] if facts else empty_failure(),
                                   p.phase, p.count + 1),
        )

    def _phase(self, p, tol, damp, harden, budget):
        stop = p.count + budget
        return jax.lax.while_loop(
            lambda current: (current.reason == Reason.RUNNING) & (current.count < stop),
            partial(self._advance, damp=damp, harden=harden, tol=tol), p,
        )

    def _hard(self, p, tol, damp, budget):
        enter = ((p.reason == Reason.RUNNING) | (p.reason == Reason.FE_STATIONARY)
                 | (p.reason == Reason.MAX_STATIONARY))
        p = p._replace(
            phase=jnp.where(enter, FailurePhase.HARD, p.phase),
            reason=jnp.where(enter, Reason.RUNNING, p.reason),
            previous_fe=jnp.full_like(p.previous_fe, jnp.nan),
        )
        return self._phase(p, tol, damp, True, budget)

    def _finish(self, p):
        budget_end = (p.reason == Reason.RUNNING) | (p.reason == Reason.INNER_BUDGET)
        undefined = (self.allow_undefined and not self.is_max) & budget_end & (p.count > 0) & (p.energy_count == 0)
        return p._replace(
            reason=jnp.where(undefined, Reason.UNDEFINED_LEDGER,
                             jnp.where(p.reason == Reason.RUNNING, Reason.BUDGET, p.reason)),
            residual=jnp.where(undefined, jnp.inf, p.residual),
            comparison=jnp.where(undefined, -1, p.comparison),
        )

    def _phases(self, p, tol, damp, n_iters, harden_last, defer_hard):
        active = p.reason == Reason.RUNNING
        if self.prime is not None and n_iters:
            p = jax.lax.cond(active, self._primed, lambda current: current, p)
        p = p._replace(phase=jnp.where(active, FailurePhase.SOFT, p.phase))
        if n_iters > harden_last:
            p = self._phase(p, tol, damp, False, n_iters - harden_last)
        if defer_hard:
            return p
        if harden_last:
            p = self._hard(p, tol, damp, harden_last)
        return self._finish(p)

    def _resume(self, p, reason, residual, tol, damp, *, budget, batched):
        p = p._replace(reason=reason, residual=residual)
        def finish(current):
            return self._finish(self._hard(current, tol, damp, budget))
        return jax.vmap(finish)(p) if batched else finish(p)

    def _single(self, state, tol, damp, *, n_iters, harden_last, defer_hard=False):
        return self._phases(self._initial(state, n_iters), tol, damp, n_iters, harden_last, defer_hard)

    def _batch(self, states, tol, damp, *, n_iters, harden_last, defer_hard=False):
        p = jax.vmap(partial(self._initial, n_iters=n_iters))(states)
        phases = jax.vmap(partial(self._phases, tol=tol, damp=damp,
                                  n_iters=n_iters, harden_last=harden_last, defer_hard=defer_hard))
        # A vmapped cond selects elementwise. Keep an outer scalar guard so
        # an entirely rejected batch never executes even the forest prime.
        return jax.lax.cond(jnp.any(p.reason == Reason.RUNNING), phases, lambda x: x, p)

    def _certified(self, p, *, batched):
        # NumPy and XLA softmax can break nearly tied rounded probabilities
        # differently. Decode with the public objective once at stationarity,
        # also before a soft-to-hard transition whose eligibility it controls.
        reason, residual = (np.array(x) for x in (p.reason, p.residual))
        indices = np.flatnonzero(reason == Reason.MAX_STATIONARY)
        for i in indices:
            s = jax.tree.map(lambda x: x[i], p.state) if batched else p.state
            if not np.isfinite(self.objective(s)):
                reason.flat[i] = Reason.OBJECTIVE_NONFINITE
                residual.flat[i] = np.inf
        return reason, residual

    def _execute(self, state, tol, damp, n_iters, harden_last, *, batched):
        run = self.batch if batched else self.single
        defer = self.is_max and 0 < harden_last < n_iters and tol.delta != -np.inf
        p = run(state, tol, damp, n_iters=n_iters, harden_last=harden_last, defer_hard=defer)
        if defer:
            reason, residual = self._certified(p, batched=batched)
            p = self.resume_hard(p, reason, residual, tol, damp, budget=harden_last, batched=batched)
        facts = jax.device_get(p._replace(state=None))
        if self.is_max and tol.delta != -np.inf:
            reason, residual = self._certified(p._replace(reason=facts.reason, residual=facts.residual),
                                               batched=batched)
            facts = facts._replace(reason=reason, residual=residual)
        return p.state, facts

    def run_single(self, state, *, n_iters, damp, harden_last, tol):
        state, facts = self._execute(state, tolerance(state.messages.dtype, tol), damp,
                                     n_iters, harden_last, batched=False)
        return _host_outcome(state, facts)

    def run_batch(self, states, *, n_iters, damp, harden_last, tol):
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
        state, facts = self._execute(stacked, tolerance(states[0].messages.dtype, tol), damp,
                                     n_iters, harden_last, batched=True)
        # Observations and parameters are immutable for the entire run.
        updated = {name: tuple(getattr(state, name)) for name in _BLOCKS
                   if getattr(states[0], name).size}
        return [
            _host_outcome(_replace(s, **{name: block[i] for name, block in updated.items()}),
                          jax.tree.map(lambda x: x[i], facts))
            for i, s in enumerate(states)
        ]


def controller_for(engine):
    hooks = (engine._iter_step, engine._prime, engine._bfe,
             vars(engine).get("bfe"), vars(engine).get("objective"))
    current = getattr(engine, "_compiled_iteration", None)
    if current is None or any(a is not b for a, b in zip(current.hooks, hooks, strict=True)):
        current = engine._compiled_iteration = CompiledIteration(engine, hooks)
    return current
