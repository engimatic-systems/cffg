"""Paired-study adapter for the external frozen CSTR physics controller.

Run the seed-0 comparison with `uv run python tools/cstr_host_comparator.py check`.
The online controller and offline replay use the same external act implementation.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter
from types import FunctionType

if __package__:
    from . import cstr_reactor_pilot as pilot
else:
    import cstr_reactor_pilot as pilot


def _frozen(experiment_physics, seed):
    agent = experiment_physics.make_agent(seed)
    agent.training = False
    agent.frozen_theta = agent.plant.prior_mean
    agent.refresh_model()
    agent.reset_episode()
    return agent


def create(experiment, seed):
    """Construct once per episode, after pilot.load_runtime or standalone."""
    experiment = Path(experiment).resolve()
    if not hasattr(pilot, "fixture"):
        # The pilot imports the fixture and benchmark as sibling scripts.
        tools_dir = str(Path(pilot.__file__).resolve().parent)
        if tools_dir not in sys.path:
            sys.path.append(tools_dir)
        pilot.load_runtime(experiment)
    cstr_dir = str(experiment / "cstr")
    if cstr_dir not in sys.path:
        sys.path.insert(0, cstr_dir)
    import experiment_physics
    from common import physics_agent

    if (Path(experiment_physics.__file__).resolve() != experiment / "cstr/experiment_physics.py"
            or Path(physics_agent.__file__).resolve() != experiment / "common/physics_agent.py"):
        raise RuntimeError("Host controller modules came from a different experiment")
    agent = _frozen(experiment_physics, seed)
    if type(agent) is not physics_agent.PhysicsController:
        raise TypeError("Expected the external PhysicsController")
    return HostComparator(agent)


class HostComparator:
    def __init__(self, agent):
        import jax

        self._agent = agent
        self._prior = None
        self._result = self._means = self._solved_at = None
        # Observe the existing solve and pre-clip readout once each. Keep both
        # hooks instance-local so a reference controller can run untouched.
        run = agent.plan_engine.run
        act = agent.act.__func__
        edge_means = act.__globals__["edge_means"]

        def run_and_record(*args, **kwargs):
            result = run(*args, **kwargs)
            jax.block_until_ready(result.state)
            self._result = result
            self._solved_at = perf_counter()
            return result

        def read_and_record(*args, **kwargs):
            self._means = edge_means(*args, **kwargs)
            self._solved_at = perf_counter()
            return self._means

        agent.plan_engine.run = run_and_record
        self._act = FunctionType(act.__code__, {**act.__globals__, "edge_means": read_and_record},
                                 act.__name__, act.__defaults__, act.__closure__).__get__(agent)
        jax.block_until_ready(agent.plan_base)

    @staticmethod
    def next_prior(host_record, applied_normalized, fallback=False):
        """Advance using what was applied on the source trajectory, in normalized units."""
        np = pilot.np
        plan = host_record["candidate"]["plan_normalized"]
        if (not fallback and host_record["error"] is None and plan is not None
                and applied_normalized == host_record["clipped_normalized"]):
            return np.asarray(plan, dtype=float)
        # act shifts away the first row. A held or counterfactual action
        # invalidates the rollout behind the entire unused tail.
        return np.full(np.shape(host_record["shifted_plan_normalized"]), applied_normalized, dtype=float)

    def decide(self, params, arrays, history, observation, tick):
        """Pilot decision contract. The host constructs its own frozen model."""
        record, command, self._prior = self._decision(
            params, history, observation, tick, self._prior, pilot.DEADLINE)
        return record, command

    def replay(self, params, record, prior_plan_normalized):
        """Re-solve a recorded window with an explicit unshifted prior and no deadline.

        Pass None for a cold plan. To reconstruct a different trajectory's warm
        start, advance its prefix with next_prior and its actual actions and holds.
        """
        np = pilot.np
        schedule = np.array(params["SP"]["Ca"], dtype=float)
        schedule[record["preview_indices"]] = record["preview_Ca"]
        params = {**params, "SP": {**params["SP"], "Ca": schedule}}
        replay, _, _ = self._decision(params, record["history"], record["observation"],
                                      record["tick"], prior_plan_normalized, float("inf"))
        if (not np.array_equal(replay["preview_indices"], record["preview_indices"])
                or not np.array_equal(replay["preview_Ca"], record["preview_Ca"])
                or replay["masks"] != record["masks"]
                or any(not np.array_equal(value, record["data"][key]) for key, value in replay["data"].items())):
            raise ValueError("Archived host replay window disagrees with reconstructed evidence")
        return replay

    def _decision(self, params, history, observation, tick, prior, deadline):
        start = perf_counter()
        np = pilot.np
        agent = self._agent
        observation = np.asarray(observation)
        inputs = pilot.window(params, history, observation, tick)
        past = [(np.asarray(obs[:agent.spec.n_x], dtype=float), np.array([u], dtype=float))
                for obs, u in inputs["history"]]
        last = past[-1][1] if past else np.zeros(agent.spec.n_u)
        shape = (agent.cfg.horizon, agent.spec.n_u)
        if prior is None:
            shifted = np.tile(last, (agent.cfg.horizon, 1))
        else:
            prior = np.asarray(prior, dtype=float)
            if prior.shape != shape or not np.isfinite(prior).all() or np.any(np.abs(prior) > 1):
                raise ValueError(f"Expected a finite applied-range host plan of shape {shape}")
            shifted = np.vstack([prior[1:], prior[-1:]])
        inputs.update(prior_plan_normalized=prior, shifted_plan_normalized=shifted)

        # These are all of act's trajectory inputs in evaluation mode. Reset
        # each even for nonconsecutive replay windows on the same instance.
        agent.t, agent.hist, agent.last_u, agent.plan_u = tick, past.copy(), last, prior
        agent.spec = replace(agent.spec, schedule=[np.asarray(params["SP"]["Ca"], dtype=float)])
        self._result = self._means = self._solved_at = None
        error = checks = raw = kelvin = clipped = None
        candidate = dict(plan_normalized=None, posterior_means=None, free_energies=())
        solve_start = perf_counter()
        try:
            original_action = self._act(observation)
            if self._result.status.value != "converged":
                raise ArithmeticError(f"Host solve returned {self._result.status.value}")
            raw = float(self._means[0, -1])
            kelvin = float(agent._u_raw(raw).item())
            if not np.isfinite(self._means).all():
                raise ArithmeticError("Nonfinite host posterior readout")
            clipped = float(original_action.item())
            candidate["plan_normalized"] = agent.plan_u
            checks = dict(host_status=self._result.status.value, iterations=self._result.iterations,
                          residual=self._result.residual, termination_reason=self._result.termination_reason.value,
                          coolant_kelvin=kelvin)
        except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = self._result
        candidate["posterior_means"] = self._means
        status = None if result is None else result.status.value
        if result is not None:
            candidate.update(free_energies=result.free_energies, iterations=result.iterations,
                             residual=result.residual, termination_reason=result.termination_reason.value,
                             inner_status=result.inner_status.value, inner_residual=result.inner_residual,
                             failure=asdict(result.failure) if result.failure is not None else None)
        record = dict(**inputs, candidate=candidate, checks=checks, error=error, host_status=status,
                      status="verification_failed" if error else "checked", proposed_kelvin=kelvin,
                      proposed_normalized=raw, clipped_normalized=clipped)
        ready = perf_counter()
        late = ready - start > deadline
        fallback = error is not None or late

        def select(hold):
            command = last.copy() if hold else original_action
            u = float(command.item())
            plan = self.next_prior(record, u, hold)
            if hold:
                agent.hist = [*past, (observation[:agent.spec.n_x].astype(float), command)]
                agent.last_u, agent.plan_u, agent.t = command, plan, tick + 1
            return command, u, plan

        command, u, next_plan = select(fallback)
        applied = perf_counter()
        if not fallback and applied - start > deadline:
            late = fallback = True
            command, u, next_plan = select(True)
            applied = perf_counter()
        solved = self._solved_at
        record.update(deadline_miss=late or applied - start > deadline, fallback=fallback,
                      applied_normalized=u, applied_kelvin=298.5 + 3.5 * u,
                      solve_s=None if solved is None else solved - solve_start, through_clip_s=ready - start,
                      observation_to_action_s=applied - start, applied_at=applied, checked_at=None if error else ready)
        return record, command, next_plan


def self_check(experiment):
    """Compare real closed-loop seed-0 outputs, including actions hidden by clipping."""
    from unittest.mock import patch

    host = create(experiment, 0)
    import experiment_physics
    from common.graphs import edge_means
    from pcgym import make_env

    np = pilot.np
    params = experiment_physics.ENV_PARAMS
    reference = _frozen(experiment_physics, 0)
    run = reference.plan_engine.run
    reference_result = None

    def observe_reference(*args, **kwargs):
        nonlocal reference_result
        reference_result = run(*args, **kwargs)
        return reference_result

    reference.plan_engine.run = observe_reference
    env = make_env(params)
    np.random.seed(0)
    obs, _ = env.reset(seed=0)
    history, windows = [], {}
    clips = 0
    reward_sum = 0.
    # Disable only the wall-clock rejection for the comparison with bare act.
    # Exercise the real late-hold path separately below.
    with patch.object(pilot, "DEADLINE", float("inf")):
        for tick in range(params["N"] - 1):
            prior = reference.plan_u
            original = reference.act(obs)
            record, cmd = host.decide(params, None, history, obs, tick)
            assert record["host_status"] == reference_result.status.value == "converged"
            assert record["checks"] and not record["fallback"]
            assert np.array_equal(record["prior_plan_normalized"], prior)
            assert np.array_equal(cmd, original)
            assert np.array_equal(record["candidate"]["plan_normalized"], reference.plan_u)
            assert np.array_equal(record["candidate"]["free_energies"], reference_result.free_energies)
            first_mean = edge_means(reference.plan_engine, reference_result.state, [f"s{reference.cfg.past}"])
            raw = first_mean[0, -1]
            assert record["proposed_normalized"] == raw
            assert record["proposed_kelvin"] == reference._u_raw(raw).item()
            clips += int(abs(raw) > 1)
            if tick in pilot.REPLAY_TICKS:
                windows[tick] = record
            history.append((obs.copy(), original.item()))
            obs, reward, *_ = env.step(original)
            reward_sum += float(np.asarray(reward).item())
    assert clips
    for record in reversed(windows.values()):
        replay = host.replay(params, record, record["prior_plan_normalized"])
        assert replay["proposed_normalized"] == record["proposed_normalized"]
        for key in ("posterior_means", "free_energies", "plan_normalized"):
            assert np.array_equal(replay["candidate"][key], record["candidate"][key])

    late_window = windows[5]
    with patch.object(pilot, "DEADLINE", 0.):
        held, cmd = host.decide(params, None, late_window["history"], late_window["observation"], 5)
    last = late_window["history"][-1][1]
    assert held["error"] is None and held["deadline_miss"] and held["fallback"] and cmd.item() == last
    after, _ = host.decide(params, None, [*late_window["history"], (late_window["observation"], last)], obs, 6)
    assert np.all(after["shifted_plan_normalized"] == last)
    assert after["data"]["e3"][-1] == last
    different = 0.25 if held["clipped_normalized"] != 0.25 else -0.25
    assert np.all(host.next_prior(held, different) == different)
    return dict(seed=0, matched_actions=len(history), matched_unclipped_proposals=len(history),
                saturated_proposals=clips, exact_replays=len(windows), return_sum=reward_sum,
                status_counts=reference.stats["status_counts"], late_hold_checked=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check"])
    parser.add_argument("--experiment", type=Path, default=pilot.EXPERIMENT)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("The seed-0 comparison requires Python assertions")
    print(json.dumps(self_check(args.experiment)))
