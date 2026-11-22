"""Run the matched P4/H2 frozen-window actuator ablation on the retained CSTR model."""

import json
from time import perf_counter

import jax
import numpy as np

import cstr_runtime_fixture as reactor
from cffg.bounded_actions import BoundStatus
from cffg.bounded_chain import BoundedChainRegion, BoundedCoordinate


def declaration(horizon):
    template, arrays = reactor.model(horizon=horizon)
    actions = tuple(BoundedCoordinate(f"s{i}c", 3, -1., 1.)
                    for i in range(reactor.PAST, reactor.PAST + horizon))
    return BoundedChainRegion(template, actions), arrays


def compare():
    region, arrays = declaration(2)
    engine = region.compile(arrays=arrays)
    data = reactor.observations(horizon=2)
    state = reactor.Binder(engine).bind(engine.init_state(), data)
    start = perf_counter()
    result = engine.run(state, reactor.BUDGET, tol=reactor.TOLERANCE)
    result.require_converged()
    shared_seconds = perf_counter() - start
    arms = {}
    for name in ("clipped", "bounded"):
        start = perf_counter()
        if name == "clipped":
            marginals = engine.gaussian_marginals(result.state)
            proposal = np.array([marginals[a.edge][0][a.coordinate] for a in region.actions])
            command = np.clip(proposal[0], -1., 1.)
            objective = result.free_energies[-1]
        else:
            decision = region.evaluate(engine, result)
            if decision.bounded.status != BoundStatus.SUCCESS:
                raise ArithmeticError(f"bounded solve failed: {decision.bounded.status}")
            proposal = np.asarray(decision.bounded.mean[:2])
            command = float(decision.bounded.command)
            objective = float(decision.bounded.free_energy)
        elapsed = perf_counter() - start
        initial = np.r_[data[f"e{reactor.PAST}"], 0., command]
        successor = np.asarray(reactor.cstr_rk4(initial, arrays["theta"], reactor.CONFIGURATION))
        goal = data[f"e{reactor.PAST + 1}"][0]
        arms[name] = dict(proposal_K=(298.5 + 3.5*proposal).tolist(), applied_K=298.5 + 3.5*command,
                          objective=objective, next_concentration=.85 + .15*successor[0],
                          absolute_concentration_tracking_error=.15*abs(successor[0]-goal),
                          decision_readout_seconds=elapsed)
    return dict(history=4, horizon=2, shared_solve_seconds=shared_seconds,
                reason=result.termination_reason.value, log_mass=float(decision.bounded.log_mass),
                action_covariance=decision.gaussian_covariance[:2, :2].tolist(), arms=arms)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    print(json.dumps(compare(), indent=2))
