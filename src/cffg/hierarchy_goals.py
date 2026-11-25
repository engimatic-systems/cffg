"""Episode cadence for an optional native learned-goal planning window."""

from __future__ import annotations

import copy
from hashlib import sha256

import numpy as np

from cffg.compiler import compile_template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine
from cffg.runtime_types import RunStatus


class GoalWindow:
    def __init__(self, template, arrays):
        self.template = copy.deepcopy(template)
        self.arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
        if self.template.learnables or set(self.arrays) != set(self.template.blobs):
            raise ValueError("child goal window requires explicit fixed arrays and in-graph goal hyperpriors")
        self.engine = Engine(compile_template(self.template, arrays=self.arrays))
        names = sorted({name for group in self.engine.arena.gfe_groups for name in group.goal_prior_blobs})
        if not names or self.engine.arena.mu3_groups:
            raise ValueError("child goal window requires deterministic learned-goal inference")
        self.priors = {k: self.arrays[k].copy() for k in names}
        self.pending = None
        self.episodes = 0
        digest = sha256(structure_hash(self.template).encode())
        for k, v in sorted(self.arrays.items()):
            digest.update(repr((k, v.shape, v.dtype.str)).encode())
            digest.update(v.tobytes())
        self.identity = digest.hexdigest()

    def plan(self, n_iters, data, tol):
        state = self.engine.with_goal_priors(self.engine.init_state(), self.priors)
        if data:
            state = self.engine.bind_data(state, data)
        if tol is None:
            tol = np.sqrt(np.finfo(state.messages.dtype).eps)
        result = self.engine.run(state, n_iters, tol=tol, harden_last=0)
        if result.status is RunStatus.CONVERGED:
            self.pending = self.engine.goal_prior_updates(result.state)
        return result

    def finish(self):
        if self.pending is None:
            return False
        self.priors, self.pending = self.pending, None
        self.episodes += 1
        return True
