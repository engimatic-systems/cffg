"""Exact tree and synchronous execution on an explicitly declared interval domain."""

from functools import partial
import operator

import jax
import jax.numpy as jnp
import numpy as np

from cffg import intervals as iv
from cffg.engine import Engine
from cffg.interval_execution import IntervalState, edge_belief, ledger, propagate
from cffg.interval_plan import check_boundaries, host_cells
from cffg.readouts.selection import selected_edges
from cffg.runtime_types import FailureDiagnostic, RunResult, RunStatus
from cffg.schedule import Schedule


def _schedule(plan, choice):
    kind = choice.kind if isinstance(choice, Schedule) else choice
    if kind == "auto":
        kind = "two_pass"
    if kind not in ("two_pass", "flood"):
        raise ValueError("interval forests support auto, two_pass, and flood schedules")
    sched = Schedule(
        kind, "exact interval forest", rounds=plan.depth, fingerprint=plan.fingerprint
    )
    if isinstance(choice, Schedule) and choice != sched:
        raise ValueError("interval schedule differs from this forest's schedule")
    return sched


def _uniform(plan, params):
    floats = [x.dtype for x in params.values() if jnp.issubdtype(x.dtype, jnp.floating)]
    dtype = jnp.result_type(*floats) if floats else jnp.asarray(0.0).dtype
    return tuple(
        iv.unit(edge.dimension, edge.capacity, dtype)
        for edge in plan.edges.values()
        for _ in range(2)
    )


def _failure(code, owner, name, iteration):
    status = iv.IntervalStatus(int(code))
    return FailureDiagnostic(
        f"interval_{status.name.lower()}",
        name if owner >= 0 else None,
        "propagation" if owner >= 0 else "ledger",
        iteration,
    )


class IntervalEngine(Engine):
    """Interval-only forest specialization; ordinary models keep their original state layout."""

    def __init__(
        self, arena, schedule="auto", *, mu3_samples=256, mu3_min_ess_fraction=0.1
    ):
        if mu3_samples != 256 or mu3_min_ess_fraction != 0.1:
            raise ValueError("interval forests have no sampled evaluator")
        self.arena = arena
        self.schedule = _schedule(arena, schedule)
        self._mu3_lifecycle = None
        self._pass = jax.jit(
            partial(propagate, arena, synchronous=self.schedule.kind == "flood")
        )
        self._batch_pass = jax.jit(jax.vmap(self._pass))
        self._terms = jax.jit(partial(ledger, arena))
        self._batch_terms = jax.jit(jax.vmap(self._terms))

    @property
    def has_adaptive_gaussian(self):
        return False

    @property
    def objective_metadata(self):
        return {
            node.name: {"objective": "exact_interval", "approximate": False}
            for node in self.arena.nodes
        }

    @property
    def objective_contract(self):
        return {
            "objective": "bethe_free_energy",
            "approximate": False,
            "evidence_bound": None,
            "stationarity": "exact sum-product fixed point on a forest",
        }

    def init_state(self):
        return self.init_from_arena(self.arena)

    def init_from_arena(self, arena):
        if self.arena.fingerprint != arena.fingerprint:
            raise ValueError("different interval forest structure")
        raw = {k: np.asarray(v) for k, v in arena.log_params.items()}
        floats = [v.dtype for v in raw.values() if v.dtype.kind == "f"]
        dtype = (
            jax.dtypes.canonicalize_dtype(np.result_type(*floats))
            if floats
            else jnp.asarray(0.0).dtype
        )
        params = {
            k: jnp.asarray(
                v,
                dtype=dtype
                if v.dtype.kind == "f"
                else jnp.bool_
                if v.dtype.kind == "b"
                else jnp.int32,
            )
            for k, v in raw.items()
        }
        sources = {
            ("chart", name): np.asarray(arena.boundaries[name])
            for name in set(arena.chart.values())
        }
        for port, e in arena.ports.items():
            edge = arena.edges[e]
            sources["port", port] = np.full((edge.dimension, 2 * edge.capacity), np.nan)
        return IntervalState(params, _uniform(arena, params), sources)

    def bind_data(self, state, observations):
        messages = list(state.messages)
        sources = state.sources.copy()
        affected = set()
        for port, value in observations.items():
            if port not in self.arena.ports:
                raise KeyError(f"unknown interval DATA port {port!r}")
            edge = self.arena.edges[self.arena.ports[port]]
            affected.add(self.arena.chart[edge.name])
            slot = edge.slots[1]
            dtype = messages[slot].lower.dtype
            if value is None:
                message = iv.unit(edge.dimension, edge.capacity, dtype)
                sources["port", port] = np.full(
                    (edge.dimension, 2 * edge.capacity), np.nan
                )
            else:
                if not isinstance(value, iv.IntervalMessage) or np.shape(
                    value.lower
                ) != (edge.dimension, edge.capacity):
                    raise ValueError(
                        f"port {port!r} requires IntervalMessage with the declared shape"
                    )
                data = host_cells(*value, dtype=dtype)
                live = np.arange(edge.capacity) < np.asarray(value.count)[:, None]
                sources["port", port] = np.concatenate(
                    [np.where(live, np.asarray(v), np.nan) for v in value[:2]], axis=1
                )
                message = iv.IntervalMessage(*map(jnp.asarray, data))
            messages[slot] = message
        for name in affected:
            controls = [sources["chart", name]]
            controls.extend(
                sources["port", p]
                for p, e in self.arena.ports.items()
                if self.arena.chart[e] == name
            )
            dtype = messages[self.arena.edges[name].slots[0]].lower.dtype
            check_boundaries(np.concatenate(controls, axis=1), dtype)
        return state._replace(messages=tuple(messages), sources=sources)

    def step(self, state, damp=0.0, harden=True):
        if damp != 0:
            raise ValueError("interval messages support only undamped updates")
        updated, code, owner = self._pass(state._replace(sources={}))
        if int(code):
            raise ValueError(
                f"interval update at {self.arena.nodes[int(owner)].name}: {iv.IntervalStatus(int(code)).name}"
            )
        return updated._replace(sources=state.sources)

    def run(
        self, state, n_iters, *, tol=None, damp=0.0, harden_last=0, outer_solver="auto"
    ):
        return self.run_batch(
            [state],
            n_iters,
            tol=tol,
            damp=damp,
            harden_last=harden_last,
            outer_solver=outer_solver,
        )[0]

    def run_batch(
        self, states, n_iters, *, tol=None, damp=0.0, harden_last=0, outer_solver="auto"
    ):
        n_iters = operator.index(n_iters)
        if n_iters < 0 or (tol is not None and (not np.isfinite(tol) or tol < 0)):
            raise ValueError("invalid interval iteration budget or tolerance")
        if harden_last is not None and not 0 <= harden_last <= n_iters:
            raise ValueError("harden_last must be between 0 and n_iters")
        if damp != 0 or outer_solver != "auto":
            raise ValueError(
                "interval forests use undamped sum-product with the declared schedule"
            )
        if n_iters == 0:
            return [
                RunResult(s, RunStatus.BUDGET_EXHAUSTED, 0, (), (), np.inf)
                for s in states
            ]
        if not states:
            return []
        current = jax.tree.map(
            lambda *v: jnp.stack(v), *(s._replace(sources={}) for s in states)
        )
        batch = len(states)
        first_fail = np.zeros(batch, np.int32)
        owner = np.full(batch, -1, np.int32)
        iteration = np.zeros(batch, np.int32)
        exact = self.schedule.kind == "two_pass"
        count = (
            min(n_iters, 1 if exact else self.arena.depth)
            if exact or tol is not None
            else n_iters
        )
        check_messages = not exact and count < self.arena.depth
        stationary = np.zeros(batch, dtype=bool)
        for step in range(count):
            updated, c, o = self._batch_pass(current)
            c = np.asarray(c)
            failed = (first_fail == 0) & (c != 0)
            first_fail[failed] = c[failed]
            owner[failed] = np.asarray(o)[failed]
            iteration[failed] = step + 1
            if check_messages and step + 1 == count:
                per_field = jax.tree.leaves(
                    jax.tree.map(
                        lambda a, b: jnp.all(a == b, axis=tuple(range(1, a.ndim))),
                        current.messages,
                        updated.messages,
                    )
                )
                stationary = np.asarray(jnp.all(jnp.stack(per_field), axis=0))
            current = updated
        _, _, energies, codes, certificates = self._batch_terms(current)
        codes, energies = np.asarray(codes), np.asarray(energies)
        complete = stationary | exact | (count >= self.arena.depth)
        if np.any(complete):
            unresolved = ~np.asarray(certificates) & np.isin(
                codes,
                [
                    iv.IntervalStatus.OK,
                    iv.IntervalStatus.INFEASIBLE,
                    iv.IntervalStatus.IMPROPER,
                ],
            )
            codes = np.where(unresolved, iv.IntervalStatus.NUMERICAL, codes)
        results = []
        for i in range(batch):
            state = jax.tree.map(lambda x: x[i], current)._replace(
                sources=states[i].sources
            )
            failure, trace, residual = None, (), np.inf
            it = count
            if first_fail[i]:
                state = states[i]
                it = int(iteration[i])
                failure = _failure(
                    first_fail[i], owner[i], self.arena.nodes[owner[i]].name, it
                )
                status = RunStatus.DIVERGED
            elif complete[i] and codes[i]:
                status = (
                    RunStatus.INFEASIBLE
                    if codes[i] == iv.IntervalStatus.INFEASIBLE
                    else RunStatus.DIVERGED
                )
                failure = _failure(codes[i], -1, None, it)
                if codes[i] == iv.IntervalStatus.INFEASIBLE:
                    trace = (np.inf,)
                elif codes[i] in (
                    iv.IntervalStatus.CAPACITY,
                    iv.IntervalStatus.NUMERICAL,
                    iv.IntervalStatus.INVALID,
                ):
                    state = states[i]
            elif complete[i]:
                trace, residual = (float(energies[i]),), 0.0
                status = (
                    RunStatus.CONVERGED
                    if exact or tol is not None
                    else RunStatus.BUDGET_EXHAUSTED
                )
            else:
                status = RunStatus.BUDGET_EXHAUSTED
            results.append(
                RunResult(state, status, it, (), trace, residual, failure=failure)
            )
        return results

    def bfe(self, state):
        _, _, value, code, _ = self._terms(state._replace(sources={}))
        if code == iv.IntervalStatus.INFEASIBLE:
            return np.inf
        if code != iv.IntervalStatus.OK:
            raise ValueError(
                f"undefined interval energy: {iv.IntervalStatus(int(code)).name}"
            )
        return float(value)

    objective = bfe

    def interval_ledger(self, state):
        factors, edges, value, code, _ = self._terms(state._replace(sources={}))
        if code != iv.IntervalStatus.OK:
            raise ValueError(
                f"undefined interval ledger: {iv.IntervalStatus(int(code)).name}"
            )
        n = len(self.arena.nodes)
        return {
            "nodes": dict(
                zip(
                    (node.name for node in self.arena.nodes),
                    map(float, factors[:n]),
                    strict=True,
                )
            ),
            "boundaries": dict(
                zip(self.arena.ports, map(float, factors[n:]), strict=True)
            ),
            "edges": dict(zip(self.arena.edges, map(float, edges), strict=True)),
            "total": float(value),
        }

    def interval_marginals(self, state, edges=None):
        names = (
            self.arena.edges
            if edges is None
            else [edges]
            if isinstance(edges, str)
            else edges
        )
        result = {}
        for name in names:
            normalized, _, _, status = edge_belief(state, self.arena.edges[name])
            if status != iv.IntervalStatus.OK:
                raise ValueError(
                    f"undefined interval marginal {name!r}: {iv.IntervalStatus(int(status)).name}"
                )
            result[name] = jax.tree.map(np.asarray, normalized)
        return result

    def marginals(self, state, edges=None):
        selected_edges(self.arena, edges, "categorical", lambda _: 0)
        return {}

    def gaussian_marginals(self, state, edges=None):
        selected_edges(self.arena, edges, "Gaussian", lambda _: 0)
        return {}

    def gaussian_marginal(self, state, edge):
        return self.gaussian_marginals(state, [edge])[edge]

    def ef_marginals(self, state):
        return {}

    def node_beliefs(self, state):
        return {}
