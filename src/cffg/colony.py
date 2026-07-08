"""Colony execution: many small graphs, one machine (ARCHITECTURE §5b).

An instance is pure data — (structure key, State pytree). Engines are cached
by structure hash, so N instances of one template cost one compilation; each
tick stacks a structure-group's states and runs one vmapped step-loop per
unique structure. Instances couple only through the host (bind_data / read
marginals), never by merging graphs.
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
from cffg.engine import Engine, State, _replace


@dataclass
class _Instance:
    key: str  # structure hash
    state: State


class Colony:
    def __init__(self):
        self.engines: dict[str, Engine] = {}
        self.instances: dict[int, _Instance] = {}
        self._next_id = 0

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
            self.engines[key] = Engine(compile_template(template, arrays=arrays))
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

    def bind(self, iid: int, observations: dict[str, int | None]) -> None:
        inst = self.instances[iid]
        inst.state = self.engines[inst.key].bind_data(inst.state, observations)

    def run(self, n_iters: int, *, damp: float = 0.0) -> None:
        """One vmapped step-loop per unique structure present."""
        by_key: dict[str, list[int]] = {}
        for iid, inst in self.instances.items():
            by_key.setdefault(inst.key, []).append(iid)

        for key, iids in by_key.items():
            engine = self.engines[key]
            states = [self.instances[i].state for i in iids]
            if len(states) == 1:
                st, _ = engine.run(states[0], n_iters, damp=damp)
                self.instances[iids[0]].state = st
                continue
            batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

            def one(st: State) -> State:
                msgs, g_eta, g_lam = engine._step(
                    st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
                    st.params, damp, True,
                )
                return _replace(st, messages=msgs, g_eta=g_eta, g_lam=g_lam)

            step_v = eqx.filter_vmap(one)
            for _ in range(n_iters):
                batched = step_v(batched)
            for b, iid in enumerate(iids):
                self.instances[iid].state = jax.tree_util.tree_map(
                    lambda x: x[b], batched
                )

    def marginals(self, iid: int) -> dict[str, np.ndarray]:
        inst = self.instances[iid]
        return self.engines[inst.key].marginals(inst.state)

    def __len__(self) -> int:
        return len(self.instances)
