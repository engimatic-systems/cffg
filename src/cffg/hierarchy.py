"""A two-clock categorical hierarchy compiled from two ordinary CFFG templates.

The slow regime selects the fast transition_mixture throughout a period.
Conditional on each regime the child is an exact Markov chain. Its upward
message is log p(observations | regime), computed without the parent prior.
The parent and child carry a *joint* boundary distribution between periods.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import copy
import math
import operator
from threading import RLock

import jax.numpy as jnp
import numpy as np

from cffg.hierarchy_kernel import child_message, initial_cost, parent_message, rollover
from cffg.hierarchy_model import HierarchySpec, canonical_model, compile_hierarchy
from cffg.runtime_types import RunStatus


@dataclass(frozen=True)
class HierarchyLedger:
    """Joint region Bethe terms in nats under the current period posterior."""

    parent: float
    child_prior: float
    child_transitions: float
    child_emissions: float
    child_conditional_negative_entropy: float
    child_endpoint_negative_entropy: float
    regime_separator_entropy: float

    @property
    def total(self):
        return math.fsum(vars(self).values())


@dataclass(frozen=True)
class HierarchyResult:
    status: RunStatus
    child_status: RunStatus | None
    parent_status: RunStatus | None
    parent: np.ndarray | None = None
    child: np.ndarray | None = None
    joint: np.ndarray | None = None
    log_cavity: np.ndarray | None = None
    ledger: HierarchyLedger | None = None
    residual: float | None = None
    period_complete: bool = False

    @property
    def free_energy(self):
        return self.ledger.total if self.ledger is not None else None

    def require_converged(self):
        if self.status is not RunStatus.CONVERGED:
            raise RuntimeError(f"hierarchy inference ended with {self.status.value}")
        return self


@dataclass(frozen=True)
class _Boundary:
    prior: object
    frontier: object
    cavity: object
    cost: object
    child_statistics: object
    parent_statistics: object
    ticks: int = 0
    periods: int = 0


def _count(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class Hierarchy:
    """Deterministic exact elimination on fixed categorical slow and fast levels.

    step() is one observed environment event. preview() solves the same event
    without changing any clocks, messages or statistics. Each nonempty budget
    permits the one certified exact pass on its level. No samples are drawn.
    """

    def __init__(self, spec: HierarchySpec, parent, parent_arrays, child, child_arrays, *, goal_window=None):
        self._lock = RLock()
        self.spec = spec
        self.parent_template, self.child_template = copy.deepcopy((parent, child))
        self._arrays = tuple({k: np.array(v, copy=True) for k, v in a.items()}
                             for a in (parent_arrays, child_arrays))
        p, t, d, b, a = compile_hierarchy(spec, self.parent_template, self._arrays[0],
                                          self.child_template, self._arrays[1])
        dtype = jnp.result_type(p, t, d, b, a)
        self._transition, self._child_transition, self._emission = (jnp.asarray(v, dtype) for v in (t, b, a))
        start = jnp.broadcast_to(d.astype(dtype), (p.size, d.size))
        self._state = _Boundary(p.astype(dtype), start, jnp.zeros_like(p, dtype=dtype), initial_cost(start), jnp.zeros_like(self._child_transition),
                                jnp.zeros_like(self._transition))
        self.structure_hash = spec.structure_hash(parent, child)
        h = sha256(self.structure_hash.encode())
        for value in (p, t, d, b, a):
            data = np.asarray(value)
            h.update(data.dtype.str.encode())
            h.update(data.tobytes())
        self._goals = None
        if goal_window is not None:
            from cffg.hierarchy_goals import GoalWindow
            self._goals = GoalWindow(*goal_window)
            h.update(self._goals.identity.encode())
        self.artifact_hash = h.hexdigest()

    @classmethod
    def from_tables(cls, parent_prior, parent_transition, child_prior, child_transition, emission, *, thickness, goal_window=None):
        """Tables are indexed [next,old], [regime,next,old], and [observed,next]."""
        model = canonical_model(parent_prior, parent_transition, child_prior, child_transition, emission, thickness)
        return cls(*model, goal_window=goal_window)

    @property
    def child_ticks(self):
        return self._state.ticks

    @property
    def parent_periods(self):
        return self._state.periods

    @property
    def child_statistics(self):
        """Sum of the online transition beliefs, one per completed child tick."""
        return np.array(self._state.child_statistics)

    @property
    def parent_statistics(self):
        """Sum of slow transition beliefs, one per completed period."""
        return np.array(self._state.parent_statistics)

    @property
    def period_prior(self):
        """Downward regime message, held fixed until the next slow rollover."""
        return np.exp(np.asarray(self._state.prior))

    @property
    def current_joint(self):
        """Boundary before the next fast transition; includes the scheduled slow rollover."""
        state = self._state
        prior = np.asarray(state.prior)
        cavity = np.asarray(state.cavity)
        weights = prior + (cavity - np.max(cavity[np.isfinite(prior)]))
        mass = weights[:, None] + np.asarray(state.frontier)
        return np.exp(mass - np.logaddexp.reduce(mass, axis=None))

    def preview(self, observation, *, child_budget=1, parent_budget=1, coupled_budget=1):
        with self._lock:
            return self._run(observation, child_budget, parent_budget, coupled_budget, commit=False)

    def step(self, observation, *, child_budget=1, parent_budget=1, coupled_budget=1):
        with self._lock:
            return self._run(observation, child_budget, parent_budget, coupled_budget, commit=True)

    def _run(self, observation, child_budget, parent_budget, coupled_budget, *, commit):
        if observation is None:
            code = -1
        else:
            code = _count(observation, "child observation")
            if code >= self._emission.shape[0]:
                raise ValueError("child observation outside the declared DATA cardinality")
        cb, pb, outer = (_count(b, label) for b, label in
                         ((child_budget, "child budget"), (parent_budget, "parent budget"), (coupled_budget, "coupled budget")))
        ok, budget, diverged, infeasible = (RunStatus.CONVERGED, RunStatus.BUDGET_EXHAUSTED,
                                             RunStatus.DIVERGED, RunStatus.INFEASIBLE)
        if not outer:
            return HierarchyResult(budget, None, None)
        if not cb:
            return HierarchyResult(budget, budget, None)
        state = self._state
        frontier, cost, reverse, cavity, finite = child_message(
            state.frontier, state.cavity, state.cost, self._child_transition, self._emission, code)
        if not bool(finite):
            return HierarchyResult(diverged, diverged, None)
        cavity = np.asarray(cavity)
        if np.isnan(cavity).any() or np.isposinf(cavity).any():
            return HierarchyResult(diverged, diverged, None)
        if np.isneginf(cavity).all():
            return HierarchyResult(infeasible, infeasible, None, log_cavity=cavity)
        if not pb:
            return HierarchyResult(budget, ok, budget, log_cavity=cavity)
        log_z, log_joint, terms, stat = parent_message(state.prior, cavity, frontier, cost, reverse)
        z = float(log_z)
        if z == -math.inf:
            return HierarchyResult(infeasible, ok, infeasible, log_cavity=cavity)
        terms = np.asarray(terms)
        if not math.isfinite(z) or not np.isfinite(terms).all():
            return HierarchyResult(diverged, ok, diverged, log_cavity=cavity)
        ledger = HierarchyLedger(*map(float, terms))
        joint = np.exp(np.asarray(log_joint))
        # Certify the joint ledger against the independently contracted partition
        # and the two normalized boundary marginals against their message product.
        prior = np.asarray(state.prior)
        weights = prior + (cavity - np.max(cavity[np.isfinite(prior)]))
        regime = np.exp(weights - np.logaddexp.reduce(weights))
        prob_residual = max(abs(joint.sum() - 1), np.max(np.abs(joint.sum(axis=1) - regime)))
        energy_residual = abs(ledger.total + z)
        eps = np.finfo(terms.dtype).eps
        # Cancellation in summing the signed Bethe terms scales with their magnitudes.
        energy_scale = max(1., np.abs(terms).sum(), abs(z))
        residual = float(max(energy_residual, prob_residual))
        if not math.isfinite(energy_residual) or not math.isfinite(prob_residual):
            return HierarchyResult(diverged, ok, ok, ledger=ledger, residual=math.nan)
        if energy_residual > 128 * eps * energy_scale or prob_residual > 128 * eps:
            return HierarchyResult(diverged, ok, ok, ledger=ledger, residual=residual)
        ticks = state.ticks + 1
        complete = ticks % self.spec.thickness == 0
        if commit:
            staged = replace(state, ticks=ticks, frontier=frontier, cavity=jnp.asarray(cavity), cost=cost,
                             child_statistics=state.child_statistics + stat)
            if complete:
                prior, frontier, cost, pstat = rollover(log_joint, self._transition)
                staged = replace(staged, prior=prior, frontier=frontier, cavity=jnp.zeros_like(prior), cost=cost,
                                 periods=state.periods+1,
                                 parent_statistics=state.parent_statistics + pstat)
            self._state = staged
        return HierarchyResult(ok, ok, ok, joint.sum(axis=1), joint.sum(axis=0), joint, cavity, ledger, residual, complete)

    @property
    def goal_priors(self):
        """Committed child planning-window Dirichlet counts, keyed by blob name."""
        with self._lock:
            return {} if self._goals is None else {k: v.copy() for k, v in self._goals.priors.items()}

    def plan_goals(self, n_iters, *, data=None, tol=None):
        """Stage the latest converged native child planning window, on its separate GFE ledger."""
        if self._goals is None:
            raise ValueError("no child learned-goal planning window declared")
        with self._lock:
            return self._goals.plan(n_iters, data, tol)

    def finish_episode(self):
        """Roll one staged goal posterior. The environment and regime clocks continue."""
        with self._lock:
            return self._goals is not None and self._goals.finish()

    def checkpoint(self):
        """Return independent arrays suitable for np.savez and pickle-free restoration."""
        from cffg.hierarchy_io import checkpoint
        with self._lock:
            return checkpoint(self)

    def restore(self, values):
        """Validate all fields before installing the saved boundary."""
        from cffg.hierarchy_io import restore
        with self._lock:
            restore(self, values)

    def save_bundle(self, path):
        """Atomically package the hierarchy declaration, native level bundles and checkpoint."""
        from cffg.hierarchy_io import save
        with self._lock:
            save(self, path)

    @classmethod
    def load_bundle(cls, path):
        from cffg.hierarchy_io import load
        return load(cls, path)
