"""Colony execution: many small graphs, one machine (ARCHITECTURE §5b).

An instance is pure data — (structure key, State pytree). Engines are cached
by graph declarations and executable bindings. JAX specializes their programs
to numeric shapes and dtypes.
Ordinary iterative plans stack compatible states and run one vmapped
step-loop; mixed-GFE plans share the compiled numerical kernels but retain
per-instance trust controllers until cff-wbhu. Instances couple only through
the host (bind_data / read marginals), never by merging graphs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from cffg import run_results
from cffg.compiler import _CompilationPlan, _prepare_compilation
from cffg.core.ir import Template
from cffg.core.plates import expanded_blob_refs
from cffg.core.serialize import structure_hash
from cffg.engine import Engine, RunResult, State
from cffg.compiled_iteration import controller_for
from cffg.readouts.selection import EdgeSelection
from cffg.resolution.contracts import ResolvedCompileData
from cffg.runtime_types import RunStatus as RunStatus  # noqa: PLC0414


@dataclass
class _Instance:
    key: str  # structure hash
    state: State


def _batch_signature(state: State) -> tuple:
    """Stack only matching numerical layouts; values do not define a batch."""
    leaves, structure = jax.tree_util.tree_flatten(state)
    return structure, tuple((value.shape, value.dtype) for value in leaves)


def _declarations(template):
    # Serialization canonicalizes some declarations and erases source data.
    # Compare the actual graph declarations before trusting a hash bucket.
    return replace(
        template,
        blobs={key: replace(ref, path="", init=()) for key, ref in template.blobs.items()},
        plates={key: replace(spec, fragment=_declarations(spec.fragment))
                for key, spec in template.plates.items()},
    )


def _providers():
    from cffg.core.ir import FACTOR_TYPES
    from cffg.kernels import KERNELS
    from cffg.learn import FAMILIES
    from cffg.mean_functions import MEAN_IMPLEMENTATIONS

    # Strong references make replacement detectable even when a callback is
    # re-registered with the same advertised implementation and version.
    return tuple((name, provider) for registry in
                 (FACTOR_TYPES, KERNELS, FAMILIES, MEAN_IMPLEMENTATIONS)
                 for name, provider in registry.items())


@dataclass(frozen=True)
class _SpawnStructure:
    key: str
    declarations: Template
    providers: tuple
    expanded: Template
    plan: _CompilationPlan
    resolved: ResolvedCompileData
    x64: bool

    def matches(self, declarations, providers):
        return (
            len(self.providers) == len(providers)
            and all(a == b and x is y for (a, x), (b, y) in zip(self.providers, providers))
            and self.x64 == jax.config.x64_enabled
            and declarations == self.declarations
        )

    def resolve(self, incoming, arrays):
        template = (replace(self.expanded, blobs=expanded_blob_refs(incoming))
                    if incoming.plates else incoming)
        return self.plan.resolve(template, arrays, None, reuse=self.resolved)


class Colony:
    def __init__(self):
        self.engines: dict[str, Engine] = {}
        self.instances: dict[int, _Instance] = {}
        self._spawn_structures: dict[str, list[_SpawnStructure]] = {}
        self._next_id = 0
        self.last_results: dict[int, RunResult] = {}

    def spawn(
        self,
        template: Template,
        arrays: dict[str, np.ndarray],
        *,
        observations: dict[str, object] | None = None,
    ) -> int:
        """Validate current parameters and reuse the matching structural plan."""
        fingerprint = structure_hash(template)
        declarations, providers = _declarations(template), _providers()
        bucket = self._spawn_structures.get(fingerprint, ())
        matching = [
            entry for entry in bucket
            if entry.key in self.engines and entry.matches(declarations, providers)
        ]
        cached = matching[0] if matching else None
        new = None
        if cached is None:
            expanded, plan = _prepare_compilation(template, "sum")
            data = plan.resolve(expanded, arrays, None)
            arena = plan.assemble(data)
            if arena.mu3_groups:
                raise ValueError(
                    "Colony does not accept mu3 evaluators: their per-instance "
                    "explicit PRNG keys and stochastic statuses require "
                    "Engine.prepare_mu3/run"
                )
            key = fingerprint if not bucket else f"{fingerprint}:{len(bucket)}"
            new = _SpawnStructure(
                key, deepcopy(declarations), providers, deepcopy(expanded),
                plan, data, jax.config.x64_enabled,
            )
            engine = Engine(arena)
        else:
            data = cached.resolve(template, arrays)
            compatible = next(
                (entry for entry in matching if data.learn_plans == entry.resolved.learn_plans), None
            )
            if compatible is None:
                # Learning schemas are part of the executable even when a
                # custom packer admits another valid numeric source shape.
                key = f"{fingerprint}:{len(bucket)}"
                new = replace(cached, key=key, resolved=data, x64=jax.config.x64_enabled)
                engine = Engine(cached.plan.assemble(data))
            else:
                key, engine = compatible.key, self.engines[compatible.key]

        params = {name: jnp.asarray(value) for name, value in data.log_params.items()}
        state = engine._state_lifecycle.init_state(params)
        if observations:
            state = engine.bind_data(state, observations)
        if new is not None:
            self.engines[key] = engine
            self._spawn_structures.setdefault(fingerprint, []).append(new)
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
