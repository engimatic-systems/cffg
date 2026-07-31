"""Colony execution: many small graphs, one machine (ARCHITECTURE §5b).

An instance is pure data — (structure key, State pytree). Engines are cached
by structure hash, so N instances of one template cost one compilation.
Ordinary iterative plans stack a structure-group's states and run one vmapped
step-loop; mixed-GFE plans share the compiled numerical kernels but retain
per-instance trust controllers until cff-wbhu. Instances couple only through
the host (bind_data / read marginals), never by merging graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiler import compile_template
from cffg.core.ir import Template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine, RunResult, RunStatus, State, _replace


@dataclass
class _Instance:
    key: str  # structure hash
    state: State


class Colony:
    def __init__(self):
        self.engines: dict[str, Engine] = {}
        self.instances: dict[int, _Instance] = {}
        self._next_id = 0
        self.last_results: dict[int, RunResult] = {}

    def spawn(
        self,
        template: Template,
        arrays: dict[str, np.ndarray],
        *,
        observations: dict[str, int | None] | None = None,
    ) -> int:
        """Add an instance; compiles only if this structure is new."""
        key = structure_hash(template)
        if key not in self.engines:
            arena = compile_template(template, arrays=arrays)
            if arena.mu3_groups:
                raise ValueError(
                    "Colony does not accept mu3 evaluators: their per-instance "
                    "explicit PRNG keys and stochastic statuses require "
                    "Engine.prepare_mu3/run"
                )
            self.engines[key] = Engine(arena)
            state = self.engines[key].init_state()
        else:
            # Same structure, this instance's own parameters.
            arena = compile_template(template, arrays=arrays)
            state = _replace(
                self.engines[key].init_state(),
                params={k: jnp.asarray(v) for k, v in arena.log_params.items()},
            )
        if observations:
            state = self.engines[key].bind_data(state, observations)
        iid = self._next_id
        self._next_id += 1
        self.instances[iid] = _Instance(key, state)
        return iid

    def kill(self, iid: int) -> None:
        del self.instances[iid]
        self.last_results.pop(iid, None)

    def bind(self, iid: int, observations: dict[str, int | None]) -> None:
        inst = self.instances[iid]
        inst.state = self.engines[inst.key].bind_data(inst.state, observations)

    def run(
        self, n_iters: int, *, damp: float = 0.0, tol: float | None = None
    ) -> dict[int, RunResult]:
        """Run each structure batch and return one truthful result per instance.

        Rule evaluation remains vmapped.  Termination is tracked per instance
        on the host until cff-wbhu moves this status carry into ``lax.scan``.
        """
        by_key: dict[str, list[int]] = {}
        for iid, inst in self.instances.items():
            by_key.setdefault(inst.key, []).append(iid)

        for key, iids in by_key.items():
            engine = self.engines[key]
            states = [self.instances[i].state for i in iids]
            if engine.arena.gfe_groups or engine.arena.gfe_tm_groups:
                if damp != 0.0:
                    raise ValueError(
                        "damp does not apply to mixed-GFE trust-region "
                        "inference"
                    )
                for iid, state in zip(iids, states):
                    result = engine.run(
                        state,
                        n_iters,
                        tol=tol,
                        harden_last=n_iters,
                    )
                    self.instances[iid].state = result.state
                    self.last_results[iid] = result
                continue
            if len(states) == 1:
                result = engine.run(
                    states[0], n_iters, tol=tol, damp=damp, harden_last=n_iters
                )
                self.instances[iids[0]].state = result.state
                self.last_results[iids[0]] = result
                continue
            batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

            # The batch runs the engine's *scheduled* iteration (cff-d56d) —
            # the same function the sequential path loops — so vmapped and
            # sequential results stay bitwise-comparable. Colored/mixed
            # schedules also prime once (their exact island passes).
            if engine._prime is not None and n_iters >= 1:
                def prime_one(st: State):
                    msgs, g_eta, g_lam, ef_eta = engine._prime(
                        st.messages, st.g_eta, st.g_lam, st.ef_eta,
                        st.g_obs, st.g_mask, st.ef_obs, st.ef_mask, st.params,
                    )
                    return _replace(
                        st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta
                    )

                batched = eqx.filter_vmap(prime_one)(batched)

            def one(st: State):
                msgs, g_eta, g_lam, ef_eta, inner_residual, inner_status = engine._iter_step(
                    st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
                    st.ef_eta, st.ef_obs, st.ef_mask, st.params, damp, True,
                )
                return (
                    _replace(st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta),
                    inner_residual,
                    inner_status,
                )

            step_v = eqx.filter_vmap(one)
            n = len(states)
            active = np.ones(n, dtype=bool)
            statuses = [RunStatus.BUDGET_EXHAUSTED] * n
            deltas: list[list[float]] = [[] for _ in range(n)]
            energies: list[list[float]] = [[] for _ in range(n)]
            residuals = [float("inf")] * n
            inner_statuses = [RunStatus.CONVERGED] * n
            inner_residuals = [0.0] * n
            previous_fe: list[float | None] = [None] * n
            dtype = np.asarray(states[0].messages).dtype
            effective_tol = (
                None if tol is None else max(float(tol), 64.0 * np.finfo(dtype).eps)
            )

            for b, st in enumerate(states):
                if not engine._state_is_finite(st):
                    active[b] = False
                    statuses[b] = RunStatus.DIVERGED
                elif engine._has_zero_support(st):
                    active[b] = False
                    statuses[b] = RunStatus.INFEASIBLE
                    energies[b].append(float("inf"))

            for _ in range(n_iters):
                if not active.any():
                    break
                candidate, inner_r, inner_code = step_v(batched)

                def freeze(old, new):
                    mask = jnp.asarray(active).reshape((n,) + (1,) * (old.ndim - 1))
                    return jnp.where(mask, new, old)

                previous = batched
                batched = jax.tree_util.tree_map(freeze, batched, candidate)
                inner_r_np, inner_code_np = np.asarray(inner_r), np.asarray(inner_code)
                for b in np.flatnonzero(active):
                    old = jax.tree_util.tree_map(lambda x: x[b], previous)
                    st = jax.tree_util.tree_map(lambda x: x[b], batched)
                    delta = max(
                        (
                            float(jnp.max(jnp.abs(getattr(st, f) - getattr(old, f))))
                            for f in ("messages", "g_eta", "g_lam", "ef_eta")
                            if getattr(st, f).size
                        ),
                        default=0.0,
                    )
                    deltas[b].append(delta)
                    inner_residuals[b] = float(inner_r_np[b])
                    code = int(inner_code_np[b])
                    inner_statuses[b] = (
                        RunStatus.CONVERGED if code == 0 else
                        RunStatus.BUDGET_EXHAUSTED if code == 1 else
                        RunStatus.DIVERGED
                    )
                    if (
                        not np.isfinite(delta)
                        or not engine._state_is_finite(st)
                        or not np.isfinite(inner_residuals[b])
                        or inner_statuses[b] is RunStatus.DIVERGED
                    ):
                        statuses[b] = RunStatus.DIVERGED
                        active[b] = False
                        continue
                    if engine._has_zero_support(st):
                        statuses[b] = RunStatus.INFEASIBLE
                        energies[b].append(float("inf"))
                        active[b] = False
                        continue
                    if inner_statuses[b] is RunStatus.BUDGET_EXHAUSTED:
                        residuals[b] = inner_residuals[b]
                        active[b] = False
                        continue
                    if engine.arena.semiring == "max":
                        residuals[b] = delta
                        if effective_tol is not None and delta <= effective_tol:
                            statuses[b] = (
                                RunStatus.CONVERGED
                                if np.isfinite(engine.objective(st))
                                else RunStatus.DIVERGED
                            )
                            active[b] = False
                        continue
                    fe = engine.bfe(st)
                    if np.isfinite(fe):
                        energies[b].append(fe)
                        if previous_fe[b] is not None:
                            residuals[b] = abs(fe - previous_fe[b])
                            if (
                                effective_tol is not None
                                and residuals[b] <= effective_tol
                                and delta <= effective_tol
                            ):
                                statuses[b] = RunStatus.CONVERGED
                                active[b] = False
                        previous_fe[b] = fe
                    elif previous_fe[b] is not None or not (
                        engine.arena.gauss_groups
                        or engine.arena.ef_groups
                        or engine.arena.prec_groups
                    ):
                        statuses[b] = RunStatus.DIVERGED
                        active[b] = False

            for b, iid in enumerate(iids):
                st = jax.tree_util.tree_map(lambda x: x[b], batched)
                if (
                    statuses[b] is RunStatus.BUDGET_EXHAUSTED
                    and not energies[b]
                    and deltas[b]
                    and engine.arena.semiring != "max"
                    and (
                        engine.arena.gauss_groups
                        or engine.arena.ef_groups
                        or engine.arena.prec_groups
                    )
                ):
                    statuses[b] = RunStatus.DIVERGED
                result = RunResult(
                    state=st,
                    status=statuses[b],
                    iterations=len(deltas[b]),
                    message_deltas=tuple(deltas[b]),
                    free_energies=tuple(energies[b]),
                    residual=residuals[b],
                    inner_status=inner_statuses[b],
                    inner_residual=inner_residuals[b],
                )
                self.instances[iid].state = st
                self.last_results[iid] = result
        return {iid: self.last_results[iid] for iid in self.instances}

    def marginals(self, iid: int) -> dict[str, np.ndarray]:
        inst = self.instances[iid]
        return self.engines[inst.key].marginals(inst.state)

    def __len__(self) -> int:
        return len(self.instances)
