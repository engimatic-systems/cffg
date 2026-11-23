"""Matched two-command frozen coolant surrogate in PCGym actuator units.

Run with uv run python examples/bounded_coolant.py. Dimensionless dynamics
are x1=u0, x2=u1-x1. Each arm executes its first proposal, then receives the
same saturated terminal feedback law. This is a tracking fixture, not CSTR
kinetics. Both arms share deterministic observations, preview and zero noise.
"""

import json
from math import log, pi
from time import perf_counter

import jax
import jax.numpy as jnp

from cffg.bounded_actions import BoundedGaussianPlan, BoundStatus


def main():
    plan = BoundedGaussianPlan((2,), (-1., -1.), (1., 1.))
    # Three normalized unit-variance residuals u0, u1-2, u1-u0-2.
    mean = jnp.array([0., 2.])
    cov = jnp.array([[2., 1.], [1., 2.]]) / 3
    baseline = log(6*pi) / 2
    def bounded(m):
        out = plan.solve(m, (cov,), baseline)
        return out.mean, out.command, -out.log_mass, out.status
    def clipped(m):
        return m, jnp.clip(m[0], -1., 1.), jnp.array(0.), jnp.array(0)
    for name, fn in (("clipped", clipped), ("bounded", bounded)):
        solve = jax.jit(fn)
        jax.block_until_ready(solve(mean))
        start = perf_counter()
        proposal, applied, penalty, status = jax.block_until_ready(solve(mean))
        elapsed = perf_counter() - start
        if status != BoundStatus.SUCCESS:
            raise RuntimeError(f"{name} failed with status {status}")
        x1 = float(applied)
        final_command = min(1., max(-1., 2 + x1))
        print(json.dumps(dict(arm=name, proposal_K=(298.5 + 3.5*proposal).tolist(),
                              applied_K=298.5 + 3.5*x1, box_penalty=float(penalty),
                              terminal_tracking_error=abs(final_command-x1-2),
                              warm_decision_seconds=elapsed)))


if __name__ == "__main__":
    main()
