"""Run the cpc-7tbv changing-reactor development pilot in the pinned PCGym environment."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from pilot import DEADLINE, EXPERIMENT, HERE, ROOT, configure, diagnostic, jsonable, parameters, save


SEEDS = (6401, 6402, 6403, 6404, 6405)
CONDITIONS = ("stationary", "switching")
ARMS = ("frozen", "learned", "informed")
START, END, STEPS = 178, 373, 599
NOMINAL, CHANGED = 50000., 48150.
TAPE_SEED = 6400
LEADS = (15, 60)
ORIGINS = sorted(set(range(8, STEPS - max(LEADS) + 1, 24)) | {START-1, START, START+5, END-1, END, END+5})


def schedule(condition):
    import numpy as np
    coefficients = np.full(STEPS, NOMINAL)
    if condition == "switching":
        coefficients[START:END] = CHANGED
    elif condition != "stationary":
        raise ValueError(condition)
    return coefficients


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def signatures():
    files = [HERE / n for n in ("continual_run.py", "continual_estimator.py", "continual_plant.py",
                                  "cffg_adapter.py", "forecast.py", "pilot.py")]
    files.extend([HERE.parent / "cstr_runtime_fixture.py", HERE.parent / "cstr_reactor_pilot.py", ROOT / "uv.lock"])
    return {str(f.relative_to(ROOT)): digest(f) for f in files}


def verify_protocol(out):
    plan = read(out / "study.json")
    if plan["source_sha256"] != signatures():
        raise ValueError("Numerical study sources differ from the frozen protocol")
    tree = subprocess.check_output(["git", "rev-parse", "HEAD:src/cffg"], cwd=ROOT, text=True).strip()
    if tree != plan["library_tree"]:
        raise ValueError("CFFG source tree differs from the frozen protocol")


def protocol(runtime):
    from continual_estimator import method_config
    return dict(ticket="cpc-7tbv", split="development", study_revision=runtime["head"],
                library_tree=runtime["library_tree"], source_sha256=signatures(), runtime=runtime,
                tape_seed=TAPE_SEED, seeds=SEEDS, steps=STEPS, arms=ARMS, conditions=CONDITIONS,
                ua=dict(nominal=NOMINAL, changed=CHANGED, changed_interval=[START, END]),
                method=method_config(), forecast_leads=LEADS, forecast_origins=ORIGINS,
                warm_deadline_s=DEADLINE, planner="P4/H15, fixed latent-bias estimator, shared priors and goals",
                information="Each controller sees noisy concentration and temperature, applied coolant and setpoints. "
                            "Only the informed comparator receives the coefficient for this transition.",
                prediction="All one-step observations predicted before the plant step with the applied action. "
                           "Tape rollouts use known exogenous commands and hold the current estimated UA fixed.",
                recovery="First of five consecutive pre-action estimates within 1% of current true UA.",
                scoring=dict(before=[0, START], changed=[START, END], returned=[END, STEPS],
                             transition_window=25, settled_changed=[START+25, END]),
                decision_rule=dict(min_median_paired_changed_rmse_reduction_pct=1., minimum_wins_of_five=4,
                                   comparators=["informed vs frozen", "learned vs frozen"],
                                   max_stationary_learned_to_frozen_mean_rmse_ratio=1.02,
                                   min_settled_ua_mae_reduction_pct=50.,
                                   require_lower_learned_changed_tape_T_rmse=True, max_holds=0,
                                   purpose="Engineering screen for a larger control study on new seeds"))


def write_row(stream, obj):
    stream.write(json.dumps(obj, default=jsonable, allow_nan=False) + "\n")


class ReactorPlanner:
    def __init__(self, arrays, cfg):
        from cffg_adapter import Adapter
        self.adapter = Adapter(arrays, cfg)
        self.theta = arrays["theta"].copy()
        self.ua = NOMINAL

    def install(self, value):
        import equinox as eqx
        import numpy as np
        from cffg import set_params
        if self.ua != value:
            a = self.adapter
            theta = self.theta.copy()
            theta[2] = np.log(value)
            params = set_params(a.engine, a.base.params, {"tr1": {"parameters": theta}})
            a.base = eqx.tree_at(lambda state: state.params, a.base, params)
            self.ua = value

    def solve(self, data):
        return self.adapter.solve(data)


def model_check(arrays, fixture):
    import numpy as np
    from cffg_adapter import Adapter
    from continual_estimator import predict
    planner = ReactorPlanner(arrays, fixture.CONFIGURATION)
    data = fixture.observations()
    def action():
        result = planner.solve(data)
        assert result["status"] == "ok"
        return result["action"]
    original = action()
    example = 51500.
    planner.install(example)
    changed = action()
    theta = arrays["theta"].copy()
    theta[2] = np.log(example)
    rebuilt = Adapter({**arrays, "theta": theta}, fixture.CONFIGURATION).solve(data)
    planner.install(NOMINAL)
    restored = action()
    if not (rebuilt["status"] == "ok" and abs(changed-rebuilt["action"]) < 1e-10
            and abs(changed-original) > 1e-4 and abs(restored-original) < 1e-10):
        raise ValueError("Parameter delivery to planner failed")
    augmented = np.r_[data["e4"], 0., .25]
    direct = np.asarray(fixture.cstr_rk4(augmented, theta, fixture.CONFIGURATION))[:2]
    forecast_error = float(max(abs(direct-predict(data["e4"], .25, example))))
    if forecast_error > 1e-9:
        raise ValueError("Parameter delivery to forecast failed")
    return dict(original=original, changed=changed, rebuilt=rebuilt["action"], restored=restored,
                forecast_error=forecast_error, probe_ua=example)


def tape(make_env, params, condition, out):
    import numpy as np
    from continual_estimator import GridReference, Learner, predict
    from continual_plant import make_plant
    from forecast import _commands
    coefficients = schedule(condition)
    controls = _commands(STEPS)
    plant = make_plant(make_env, params, coefficients, TAPE_SEED)
    learner, grid = Learner(), GridReference()
    forecasts, states = [], []
    lo, half = np.array([.7, 300.]), np.array([.15, 25.])
    try:
        with (out / f"tape-{condition}.jsonl").open("x") as stream:
            obs = plant.obs.copy()
            for tick, command in enumerate(controls):
                estimates = dict(frozen=NOMINAL, learned=float(learner.ua), grid=float(grid.ua),
                                 informed=float(coefficients[tick]))
                predictions = {k: predict(obs, command, value) for k, value in estimates.items()}
                if tick in ORIGINS:
                    for k, value in estimates.items():
                        state = obs.copy()
                        for h, u in enumerate(controls[tick:tick+max(LEADS)], 1):
                            state = predict(state, float(u), value)
                            if h in LEADS:
                                forecasts.append(dict(tick=tick, lead=h, method=k, ua=value,
                                                      prediction=lo+(state+1)*half))
                next_obs, truth, reward, done = plant.step(float(command))
                if done != (tick == STEPS-1):
                    raise ValueError("Invalid tape termination")
                updates = dict(learned=learner.update(obs, command, next_obs),
                               grid=grid.update(obs, command, next_obs))
                write_row(stream, dict(tick=tick, obs=obs, next_obs=next_obs, truth=truth, reward=reward,
                                       target=params["SP"]["Ca"][tick+1], true_ua=coefficients[tick],
                                       applied=command, estimates=estimates, predictions=predictions, updates=updates))
                states.append(truth.copy())
                obs = next_obs
    finally:
        plant.close()
    for row in forecasts:
        row["truth"] = states[row["tick"]+row["lead"]-1]
    save(out / f"tape-forecasts-{condition}.json", forecasts)


def episode(make_env, params, arrays, fixture, window, seed, condition, arm, out, *, motion_sd=None):
    import numpy as np
    from continual_estimator import Learner, predict, warmup
    from continual_plant import make_plant
    directory = out / f"seed-{seed}"
    directory.mkdir(exist_ok=True)
    stem = directory / f"{condition}-{arm}"
    coefficients = schedule(condition)
    setup_start = perf_counter()
    planner = ReactorPlanner(arrays, fixture.CONFIGURATION)
    warmup()
    for ua, hist in ((51100., True), (NOMINAL, False)):
        planner.install(ua)
        if planner.solve(fixture.observations(history=hist, goal=.19))["status"] != "ok":
            raise ValueError("Controller warmup failed")
    if arm in ("candidate", "original"):
        learner = Learner(motion_sd=motion_sd)
    elif arm in ARMS and motion_sd is None:
        learner = Learner() if arm == "learned" else None
    else:
        raise ValueError(f"Unsupported learner configuration: {arm}, {motion_sd}")
    setup_s = perf_counter()-setup_start
    plant = make_plant(make_env, params, coefficients, seed)
    history = []
    total = 0.
    holds = 0
    start = perf_counter()
    try:
        obs = plant.obs.copy()
        last = 0.
        with stem.with_suffix(".jsonl").open("x") as stream:
            for tick in range(STEPS):
                decision_start = perf_counter()
                update = None
                if learner is not None and history:
                    previous, applied = history[-1]
                    update = learner.update(previous, applied, obs)
                ua = float(learner.ua) if learner is not None else NOMINAL
                if arm == "informed":
                    ua = float(coefficients[tick])
                after_update = perf_counter()
                planner.install(ua)
                inputs = window(params, history, obs, tick)
                proposal = planner.solve(inputs["data"])
                status, raw = proposal["status"], proposal.get("action")
                if status == "ok" and not np.isfinite(raw):
                    status, raw = "nonfinite", None
                proposed = float(np.clip(raw, -1., 1.)) if status == "ok" else last
                decision_s = perf_counter() - decision_start
                late = decision_s > DEADLINE
                held = late or status != "ok"
                applied = last if held else proposed
                before_forecast = perf_counter()
                prediction = predict(obs, applied, ua)
                forecast_s = perf_counter() - before_forecast
                next_obs, truth, reward, done = plant.step(applied)
                if done != (tick == STEPS-1):
                    raise ValueError("Unexpected episode termination")
                total += reward
                holds += held
                write_row(stream, dict(tick=tick, obs=obs, next_obs=next_obs, truth=truth,
                          target=params["SP"]["Ca"][tick+1], true_ua=coefficients[tick],
                          ua=ua, prediction=prediction, applied=applied, raw_action=raw,
                          status=status, held=held, late=late, decision_s=decision_s,
                          update_s=after_update-decision_start, forecast_s=forecast_s,
                          update=diagnostic(update), reward=reward, residual=proposal.get("residual")))
                history.append((obs.copy(), applied))
                obs, last = next_obs, applied
        terminal_update = learner.update(history[-1][0], last, obs) if learner is not None else None
    finally:
        plant.close()
    save(stem.with_suffix(".json"), dict(seed=seed, condition=condition, arm=arm, steps=STEPS,
         total_return=total, holds=holds, warm_setup_s=setup_s, episode_s=perf_counter()-start,
         terminal_update=diagnostic(terminal_update), trace_sha256=digest(stem.with_suffix(".jsonl")),
         motion_sd=learner.motion_sd if learner is not None else None))


def run_all(out):
    verify_protocol(out)
    qualification = read(out / "qualified.json")
    if qualification["study_sha256"] != digest(out/"study.json"):
        raise ValueError("Qualification protocol mismatch")
    started = perf_counter()
    for i, seed in enumerate(SEEDS):
        for j, condition in enumerate(CONDITIONS):
            shift = (i+j) % len(ARMS)
            for arm in ARMS[shift:]+ARMS[:shift]:
                stem = out / f"seed-{seed}" / f"{condition}-{arm}"
                if stem.with_suffix(".json").exists():
                    entry = read(stem.with_suffix(".json"))
                    if entry["trace_sha256"] != digest(stem.with_suffix(".jsonl")):
                        raise ValueError(f"Invalid completed episode {stem}")
                    continue
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(out),
                                "--seed", str(seed), "--condition", condition, "--arm", arm], check=True)
                print(f"Completed {seed} {condition} {arm}", flush=True)
    save(out / "completed.json", dict(episodes=len(SEEDS)*len(CONDITIONS)*len(ARMS),
         elapsed_this_launch_s=perf_counter()-started, study_sha256=digest(out/"study.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if args.phase == "run":
        return run_all(out)
    if args.phase == "worker":
        verify_protocol(out)
        if None in (args.seed, args.condition, args.arm):
            parser.error("worker requires seed, condition and arm")
    # Establish CPU, float64 and package provenance before any numerical imports.
    from types import SimpleNamespace
    configure(SimpleNamespace(arm="cffg"))
    import cstr_reactor_pilot as reactor
    import cstr_runtime_fixture as fixture
    make_env, base, runtime = reactor.load_runtime(EXPERIMENT)
    params = parameters(base, True)
    _, arrays = fixture.model()
    if args.phase == "worker":
        return episode(make_env, params, arrays, fixture, reactor.window, args.seed,
                       args.condition, args.arm, out)
    from continual_estimator import warmup
    from continual_plant import qualification
    out.mkdir(parents=True, exist_ok=True)
    if (out / "study.json").exists():
        raise FileExistsError("Prepare needs a fresh study directory")
    save(out / "study.json", protocol(runtime))
    warmup()
    checks = qualification(make_env, params, schedule("switching"))
    if not checks["passed"]:
        raise ValueError("Plant qualification failed")
    actuator_check = model_check(arrays, fixture)
    for condition in CONDITIONS:
        tape(make_env, params, condition, out)
    save(out / "qualified.json", dict(study_sha256=digest(out/"study.json"), plant=checks,
                                      parameter_delivery=actuator_check))


if __name__ == "__main__":
    main()
