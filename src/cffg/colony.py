"""Colony execution: many small graphs, one machine (ARCHITECTURE §5b).

An instance is pure data — (structure key, State pytree). Engines are cached
by structure hash; JAX specializes their programs to numeric shapes and dtypes.
Ordinary iterative plans stack compatible states and run one vmapped
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

from cffg import run_results
from cffg.compiler import compile_template
from cffg.core.ir import Template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine, RunResult, State, _replace
from cffg.iteration import IterationProgress, step_facts
from cffg.runtime_types import RunStatus as RunStatus  # noqa: PLC0414


@dataclass
class _Instance:
    key: str  # structure hash
    state: State


def _batch_signature(state: State) -> tuple:
    """Stack only matching numerical layouts; values do not define a batch."""
    leaves, structure = jax.tree_util.tree_flatten(state)
    return structure, tuple((value.shape, value.dtype) for value in leaves)


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
        by_key: dict[tuple, list[int]] = {}
        for iid, inst in self.instances.items():
            by_key.setdefault((inst.key, _batch_signature(inst.state)), []).append(iid)

        for (key, _signature), iids in by_key.items():
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
            n = len(states)
            policy = engine._iteration_policy(states[0], tol)
            services = engine._iteration_services()
            program = engine._iteration_program()
            progress = [IterationProgress(policy) for _ in states]
            for current, state in zip(progress, states, strict=True):
                current.initial(state, services)
            active = np.asarray([not current.stopped for current in progress])

            def freeze(old, new, *, active=active, n=n):
                mask = jnp.asarray(active).reshape((n,) + (1,) * (old.ndim - 1))
                return jnp.where(mask, new, old)

            batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

            # The batch runs the engine's *scheduled* iteration (cff-d56d) —
            # the same function the sequential path loops — so vmapped and
            # sequential results stay bitwise-comparable. Colored/mixed
            # schedules also prime once (their exact island passes).
            if program.prime is not None and n_iters >= 1 and active.any():
                def prime_one(st: State, *, program=program):
                    msgs, g_eta, g_lam, ef_eta = program.prime(
                        st.messages, st.g_eta, st.g_lam, st.ef_eta,
                        st.g_obs, st.g_mask, st.ef_obs, st.ef_mask, st.params,
                    )
                    return _replace(
                        st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta
                    )

                primed = eqx.filter_vmap(prime_one)(batched)
                # Rejected inputs stay frozen across initial propagation too.
                if active.all():
                    batched = primed
                else:
                    batched = jax.tree_util.tree_map(freeze, batched, primed)

            def one(st: State, *, program=program):
                msgs, g_eta, g_lam, ef_eta, inner_residual, inner_status = program.iter_step(
                    st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
                    st.ef_eta, st.ef_obs, st.ef_mask, st.params, damp, True,
                )
                return (
                    _replace(st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta),
                    inner_residual,
                    inner_status,
                )

            step_v = eqx.filter_vmap(one)

            for _ in range(n_iters):
                if not active.any():
                    break
                candidate, inner_r, inner_code = step_v(batched)

                previous = batched
                batched = jax.tree_util.tree_map(freeze, batched, candidate)
                inner_r_np, inner_code_np = np.asarray(inner_r), np.asarray(inner_code)
                for b in np.flatnonzero(active):
                    old = jax.tree_util.tree_map(lambda x: x[b], previous)
                    state = jax.tree_util.tree_map(lambda x: x[b], batched)
                    facts = step_facts(old, state, inner_r_np[b], inner_code_np[b])
                    progress[b].consume(state, facts, services)
                    active[b] = not progress[b].stopped

            for b, iid in enumerate(iids):
                state = jax.tree_util.tree_map(lambda x: x[b], batched)
                result = run_results.finalize_iteration(progress[b].finish(state))
                self.instances[iid].state = result.state
                self.last_results[iid] = result
        return {iid: self.last_results[iid] for iid in self.instances}

    def marginals(self, iid: int) -> dict[str, np.ndarray]:
        inst = self.instances[iid]
        return self.engines[inst.key].marginals(inst.state)

    def __len__(self) -> int:
        return len(self.instances)
