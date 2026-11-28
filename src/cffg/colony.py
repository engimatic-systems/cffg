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

import jax
import jax.numpy as jnp
import numpy as np

from cffg import run_results
from cffg.compiler import compile_template
from cffg.core.ir import Template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine, RunResult, State, _replace
from cffg.compiled_iteration import controller_for
from cffg.readouts.selection import EdgeSelection
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
    def __init__(self, *, schedule: str = "auto"):
        self.schedule = schedule
        self.engines: dict[str, Engine] = {}
        self.instances: dict[int, _Instance] = {}
        self._next_id = 0
        self.last_results: dict[int, RunResult] = {}

    def spawn(
        self,
        template: Template,
        arrays: dict[str, np.ndarray],
        *,
        observations: dict[str, object] | None = None,
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
            self.engines[key] = Engine(arena, schedule=self.schedule)
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

    def bind(self, iid: int, observations: dict[str, object]) -> None:
        """Apply Engine.bind_data's partial update to this instance only."""
        inst = self.instances[iid]
        inst.state = self.engines[inst.key].bind_data(inst.state, observations)

    def run(
        self, n_iters: int, *, damp: float = 0.0, tol: float | None = None,
        outer_solver: str = "auto",
    ) -> dict[int, RunResult]:
        """Run each structure batch and return one truthful result per instance.

        Ordinary iterative rule evaluation remains vmapped. Exact schedules
        reuse Engine's per-instance propagation and certification. Termination
        is tracked independently for each member. Mixed GFE uses the host trust
        controller by default. Select ``outer_solver="compiled_trust_region"``
        to batch deterministic fixed-support phases across compatible members.
        """
        if n_iters < 0:
            raise ValueError("n_iters must be non-negative")
        if outer_solver not in ("auto", "trust_region", "compiled_trust_region"):
            raise ValueError("Colony outer_solver must be 'auto', 'trust_region', or 'compiled_trust_region'")
        by_key: dict[tuple, list[int]] = {}
        for iid, inst in self.instances.items():
            by_key.setdefault((inst.key, _batch_signature(inst.state)), []).append(iid)

        for (key, _signature), iids in by_key.items():
            engine = self.engines[key]
            states = [self.instances[i].state for i in iids]
            if (engine.arena.gfe_groups or engine.arena.gfe_tm_groups
                    or engine.arena.hybrid_gfe_groups or engine.has_adaptive_gaussian):
                if damp != 0.0:
                    raise ValueError(
                        "damp does not apply to mixed-GFE trust-region "
                        "inference"
                    )
                batch_trust = (
                    outer_solver == "compiled_trust_region"
                    and not (engine.has_adaptive_gaussian or engine.arena.mu3_groups)
                    and not engine._can_run_exact(n_iters, damp, n_iters)
                )
                if batch_trust:
                    results = engine._run_outer_batch(
                        states, n_iters, tol=tol,
                        harden_last=n_iters if len(engine.arena.delta_pairs) else 0,
                    )
                else:
                    results = [engine.run(state, n_iters, tol=tol, harden_last=n_iters,
                                          outer_solver=outer_solver) for state in states]
                for iid, result in zip(iids, results, strict=True):
                    self.instances[iid].state = result.state
                    self.last_results[iid] = result
                continue
            if len(states) == 1 or engine._can_run_exact(n_iters, damp, n_iters):
                for iid, state in zip(iids, states, strict=True):
                    result = engine.run(
                        state, n_iters, tol=tol, damp=damp, harden_last=n_iters
                    )
                    self.instances[iid].state = result.state
                    self.last_results[iid] = result
                continue
            outcomes = controller_for(engine).run_batch(
                states, n_iters=n_iters, damp=damp, tol=tol,
                harden_last=n_iters if len(engine.arena.delta_pairs) else 0,
            )
            for iid, outcome in zip(iids, outcomes, strict=True):
                result = run_results.finalize_iteration(outcome, factor_names=engine._failure_names)
                self.instances[iid].state = result.state
                self.last_results[iid] = result
        return {iid: self.last_results[iid] for iid in self.instances}

    def marginals(self, iid: int, edges: EdgeSelection = None) -> dict[str, np.ndarray]:
        inst = self.instances[iid]
        return self.engines[inst.key].marginals(inst.state, edges=edges)

    def gaussian_marginals(
        self, iid: int, edges: EdgeSelection = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        inst = self.instances[iid]
        return self.engines[inst.key].gaussian_marginals(inst.state, edges=edges)

    def __len__(self) -> int:
        return len(self.instances)
