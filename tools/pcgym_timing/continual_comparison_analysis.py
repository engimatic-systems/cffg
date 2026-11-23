"""Audit the four closed-loop reactor planners and compare complete paired seeds.

Run after measurement with `uv run python tools/pcgym_timing/continual_comparison_analysis.py
--output ROOT`. Both the primary comparison and its bootstrap use one RMSE per seed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from continual_analysis import CONDITIONS, EPS, N, WINDOWS, control, latency, require
from one_percent_analysis import bootstrap, digest, paired_rows, read, recovery_guard, settings, tape_kernel
from pilot import parameters, save


LIBRARIES = ("cffg", "rxinfer", "numpyro", "gtsam")
SEEDS = tuple(range(6601, 6611))
MOTION = {"frozen": None, "original": .012, "candidate": .003, "informed": None}
N_EPISODES = len(LIBRARIES) * len(SEEDS) * len(CONDITIONS) * len(MOTION)
BOOTSTRAP_SEED, DRAWS = 40871, 10000
FIRST_CHANGE = WINDOWS["during"][0]


def metadata(root, hashes):
    def load(name):
        hashes[name] = digest(root / name)
        return read(root / name)

    plan = load("protocol.json")
    ph = hashes["protocol.json"]
    required = dict(seeds=list(SEEDS), conditions=list(CONDITIONS), motion_sd=MOTION, steps=N,
                    decision_deadline_s=1., stationary_budget_pct=1.,
                    schedule=dict(nominal=50000., changed=48150., changed_interval=[178, 373]),
                    scoring=dict(before=[0, 178], changed=[178, 373], returned=[373, N],
                                 settled_changed=[203, 373], transition_window=25))
    for k, v in required.items():
        require(plan[k] == v, f"Protocol differs at {k}")
    require(tuple(plan["libraries"]) == LIBRARIES and tuple(plan["arms"]) == tuple(MOTION)
            and plan["starting_method"]["initial_ua"] == plan["schedule"]["nominal"]
            and plan["starting_method"]["learner"]["motion_log_sd"] == MOTION["original"],
            "Protocol library, regime or estimator mismatch")
    qualified, opened, completed = [load(name + ".json") for name in ("qualified", "opened", "completed")]
    require(all(x["protocol_sha256"] == ph for x in (qualified, opened, completed))
            and opened["qualified_sha256"] == completed["qualified_sha256"] == hashes["qualified.json"]
            and completed["episodes"] == N_EPISODES and completed["decisions"] == N_EPISODES * N
            and opened["opened"] == completed["opened"]
            and datetime.fromisoformat(completed["finished"]) >= datetime.fromisoformat(completed["opened"]),
            "Qualification, opening and completion markers disagree")
    require(set(qualified["evidence"]) == set(LIBRARIES), "Missing library qualification")
    workers = {}
    for lib in LIBRARIES:
        name = f"qualification/{lib}.json"
        evidence = load(name)
        require(evidence["library"] == lib and evidence["protocol_sha256"] == ph
                and hashes[name] == qualified["evidence"][lib], f"Qualification mismatch for {lib}")
        check = evidence["checks"]
        require(len(check["affine"]) == 3 and all(r["passed"] for r in check["affine"])
                and abs(check["updated"] - check["fresh"]) <= 1e-8
                and abs(check["nominal"] - check["restored"]) <= 1e-8
                and abs(check["nominal"] - check["updated"]) >= 1e-4, f"Failed oracle for {lib}")
        if lib == "cffg":
            p = check["plant"]
            require(p["passed"] and p["switch_ticks"] == [178, 373] and p["continuous_state_time_and_integrator"]
                    and p["learner_observation"] == ["Ca_norm", "T_norm"], "Invalid hidden plant qualification")
        worker = load(f"workers/{lib}.json")
        blocks = worker["blocks"]
        require(worker["library"] == lib and [b["seed"] for b in blocks] == list(SEEDS),
                f"Worker manifest has wrong identity or seeds for {lib}")
        for b in blocks:
            require(b["library"] == lib and b["protocol_sha256"] == ph
                    and b == load(f"workers/{lib}-{b['seed']}.json"), f"Worker block mismatch for {lib}/{b['seed']}")
            require(b["runtime"]["library_tree"] == plan["frozen"]["library_tree"],
                    f"Worker used a different CFFG source tree at {lib}/{b['seed']}")
            require(all(np.isfinite(b[k]) and b[k] >= 0 for k in ("adapter_setup_s", "first_warmup_s", "process_wall_s")),
                    f"Invalid worker timing for {lib}/{b['seed']}")
        workers[lib] = blocks
    measured = sum(b["process_wall_s"] for blocks in workers.values() for b in blocks)
    require(np.isclose(measured, completed["worker_s"], rtol=1e-13, atol=1e-10), "Process time total mismatch")
    return plan, workers


def replay_forecasts(rows, terminal_update, trace, arm, initial_ua):
    """Replay each one-step forecast from the preceding measurement and the command actually applied."""
    import jax
    with jax.enable_x64(True):
        means, (_, slope, conditional) = jax.device_get(tape_kernel()(
            trace["obs"], trace["applied"], np.log(trace["ua"] / initial_ua)))
    predictions = np.array([r["prediction"] for r in rows])
    error = float(np.max(abs(means - predictions)))
    require(error <= 1e-11, f"One-step {arm} forecast used different inputs (max error {error})")
    if MOTION[arm] is not None:
        updates = [r["update"] for r in rows[1:]] + [terminal_update]
        for field, replay in (("temperature_log_ua_derivative", slope),
                              ("conditional_temperature_variance", conditional)):
            require(np.allclose([u[field] for u in updates], replay, rtol=1e-11, atol=1e-12),
                    f"Forecast likelihood mismatch at {arm}/{field}")
    return error


def check_native(lib, rows):
    """Check the status witness recorded by each frozen native solver."""
    for r in rows:
        native = r["native"]
        require(isinstance(native, dict), f"Missing {lib} diagnostics at tick {r['tick']}")
        if r["status"] != "ok":
            continue
        if lib == "cffg":
            ok = native["residual"] == r["residual"] and np.isfinite(r["residual"])
        elif lib == "rxinfer":
            ok = (native["slices_completed"] == 20 and native["passes_per_slice"] == 1
                  and np.isfinite(native["min_cholesky_diagonal"]) and native["min_cholesky_diagonal"] > 0)
        elif lib == "numpyro":
            ok = native["hessian_positive"] is True and 0 <= native["newton_decrement"] <= 1e-6
        else:
            ok = native["termination"] == "native LM cost tolerance" and np.isfinite(native["final_error"])
        require(ok, f"Invalid {lib} success diagnostics at tick {r['tick']}")


def held_commands(rows, lib, seed, condition, arm, trace_path):
    events = []
    for r in rows:
        if not r["held"]:
            continue
        if r["status"] == "ok":
            kind = "deadline_only"
        else:
            kind = "failure_and_deadline" if r["late"] else "failure_only"
        event = {k: r[k] for k in ("tick", "status", "late", "native", "decision_s", "update_s",
                                    "forecast_s", "applied")}
        events.append(dict(event, library=lib, arm=arm, condition=condition, seed=seed, kind=kind,
                           trace=trace_path, plant_time_min=r["tick"] * (26 / 60),
                           applied_coolant_K=295 + (r["applied"] + 1) * (302 - 295) / 2))
    return events


def audit_episode(root, lib, seed, condition, arm, plan, ph, targets, hashes):
    stem = f"{lib}/seed-{seed}/{condition}-{arm}"
    path = root / (stem + ".jsonl")
    side = read(path.with_suffix(".json"))
    q = MOTION[arm]
    require((side["library"], side["protocol_sha256"], side["motion_sd"]) == (lib, ph, q),
            f"Wrong library, protocol or estimator at {stem}")
    require(all(np.isfinite(side[k]) and side[k] >= 0 for k in ("episode_s", "warm_setup_s")),
            f"Invalid episode timing at {stem}")
    record, trace, times = control(path, seed, condition, arm, settings(plan, q),
                                   learner_arm=arm if q is not None else "learned")
    require(np.array_equal(targets, trace["target"]), f"Wrong next-state setpoint schedule at {stem}")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    statuses = Counter(r["status"] for r in rows)
    require(all(isinstance(s, str) and s for s in statuses), f"Invalid status at {stem}")
    check_native(lib, rows)
    if "native_status_counts" in side:
        require(statuses == side["native_status_counts"], f"Status counts disagree at {stem}")
    audit = record["audit"]
    audit["max_forecast_replay_error"] = replay_forecasts(
        rows, side["terminal_update"], trace, arm, plan["starting_method"]["initial_ua"])
    audit["status_counts"] = dict(statuses)
    record["motion_sd"] = q
    trace["hold_events"] = held_commands(rows, lib, seed, condition, arm, stem + ".jsonl")
    hashes[stem + ".jsonl"] = audit["trace_sha256"]
    hashes[stem + ".json"] = digest(path.with_suffix(".json"))
    return record, trace, times


def matched_history(a, b, decisions, observations):
    """Verify common histories through the first command that a hold actually changes."""
    unequal = np.flatnonzero(a["applied"][:decisions] != b["applied"][:decisions])
    held = np.flatnonzero(a["held"][:decisions] | b["held"][:decisions])
    split = int(unequal[0]) if unequal.size else None
    if split is not None:
        require(a["held"][split] or b["held"][split], f"Unexplained action difference at matched tick {split}")
    obs_end = observations if split is None else min(observations, split + 1)
    ua_end = min(decisions, obs_end)
    require(np.array_equal(a["obs"][:obs_end], b["obs"][:obs_end])
            and np.array_equal(a["ua"][:ua_end], b["ua"][:ua_end]),
            "Observation or pre-action UA differs before a permissible divergence")
    return dict(decisions_checked=decisions if split is None else split, observations_checked=obs_end,
                first_hold=int(held[0]) if held.size else None, first_divergent_held_action=split)


def histories(traces):
    stationary, changing = traces["stationary"], traces["switching"]
    results = {"stationary_informed_frozen": matched_history(stationary["informed"], stationary["frozen"], N, N),
               "switching_informed_frozen_before_change": matched_history(
                   changing["informed"], changing["frozen"], FIRST_CHANGE, FIRST_CHANGE + 1)}
    for arm in MOTION:
        # The oracle knows the change when choosing action 178. Others first observe x179.
        stop = FIRST_CHANGE if arm == "informed" else FIRST_CHANGE + 1
        results[arm + "_before_observing_change"] = matched_history(stationary[arm], changing[arm],
                                                                    stop, FIRST_CHANGE + 1)
    return results


def arm_summary(rows):
    def values(arm, field):
        return np.asarray([r[arm][field] for r in rows.values()])

    summary = {}
    for arm in MOTION:
        stationary = values(arm, "stationary_tracking_rmse")
        changed = values(arm, "changed_tracking_rmse")
        entry = dict(stationary_tracking_rmse=stationary.mean(), changed_tracking_rmse=changed.mean(),
                     stationary_penalty_pct=100 * (stationary.mean()/values("frozen", "stationary_tracking_rmse").mean()-1),
                     changed_gain_pct=np.median(values(arm, "changed_gain_pct")),
                     changed_wins=int((changed < values("frozen", "changed_tracking_rmse")).sum()),
                     mean_changed_return_delta=values(arm, "changed_return_delta").mean(),
                     settled_ua_mae=values(arm, "settled_ua_mae").mean(), median_recovery_delay={}, unrecovered={})
        for stage in ("during", "after"):
            delays = [r[arm]["recovery_delay"][stage] for r in rows.values()]
            entry["unrecovered"][stage] = delays.count(None)
            entry["median_recovery_delay"][stage] = None if None in delays else float(np.median(delays))
        summary[arm] = entry
    return summary


def engineering_references(rows, summary, totals):
    adjusted, original, frozen = (summary[a] for a in ("candidate", "original", "frozen"))
    ratio = adjusted["stationary_tracking_rmse"] / frozen["stationary_tracking_rmse"]
    reduction = 100 * (1 - adjusted["settled_ua_mae"] / frozen["settled_ua_mae"])
    gain, previous = adjusted["changed_gain_pct"], original["changed_gain_pct"]
    gates = dict(stationary_penalty=dict(passed=ratio <= 1.01, candidate_over_frozen=ratio, maximum_ratio=1.01),
                 settled_parameter_error=dict(passed=reduction >= 50, reduction_pct=reduction, required_pct=50),
                 recovery=recovery_guard(rows), reliability=dict(passed=totals["holds"] == 0, **totals),
                 retention=dict(passed=gain >= .9*previous, candidate_gain_pct=gain, original_gain_pct=previous,
                                retained_gain_pct=100*gain/previous if previous > 0 else None))
    for arm, name in (("candidate", "candidate_control_benefit"), ("informed", "informed_control_headroom")):
        s = summary[arm]
        gates[name] = dict(passed=s["changed_gain_pct"] >= 1 and 5*s["changed_wins"] >= 4*len(rows),
                           median_reduction_pct=s["changed_gain_pct"], lower_seeds=s["changed_wins"], seeds=len(rows))
    return gates


def summarize_library(episodes, clocks, blocks):
    rows = paired_rows(episodes)
    summary = arm_summary(rows)
    records = [entry for conditions in episodes.values() for arms in conditions.values() for entry in arms.values()]
    totals = {k: sum(r["audit"][k] for r in records) for k in ("failures", "holds", "misses")}
    regimes = {}
    for condition in (*CONDITIONS, "both"):
        sources = CONDITIONS if condition == "both" else (condition,)
        regimes[condition] = {arm: {
            clock: latency(np.concatenate([arr for c in sources for arr in clocks[c, arm][clock]]))
            for clock in ("decision_s", "update_s", "forecast_s")} for arm in MOTION}
    worker_cost = {k: latency([b[k] for b in blocks]) for k in ("adapter_setup_s", "first_warmup_s", "process_wall_s")}
    return dict(episodes=episodes, per_seed=rows, summary=summary, gates=engineering_references(rows, summary, totals),
                within_library_intervals=bootstrap(rows), pooled_latency_s=regimes, worker_cost_s=worker_cost,
                totals=totals, status_counts=dict(sum((Counter(r["audit"]["status_counts"]) for r in records), Counter())))


def between_libraries(libraries, seeds):
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(len(seeds), size=(DRAWS, len(seeds)))
    observations = {}
    for lib, data in libraries.items():
        rows = [data["per_seed"][str(s)] for s in seeds]
        changed = np.array([r["candidate"]["changed_tracking_rmse"] for r in rows])
        baseline = np.array([r["frozen"]["changed_tracking_rmse"] for r in rows])
        observations[lib] = dict(changed_rmse=changed, changed_learning_benefit=baseline-changed,
                                 changed_gain_pct_points=np.array([r["candidate"]["changed_gain_pct"] for r in rows]),
                                 stationary_rmse=np.array([r["candidate"]["stationary_tracking_rmse"] for r in rows]))
    result = {}
    for peer, metrics in observations.items():
        if peer == "cffg":
            continue
        comparisons = result[peer] = {}
        for key, right in metrics.items():
            left = observations["cffg"][key]
            paired = left - right
            mean, reference = float(paired.mean()), float(right.mean())
            comparisons[key] = dict(mean=mean, median=np.median(paired), min=paired.min(), max=paired.max(),
                                    n=len(seeds), ci95=np.quantile(paired[indices].mean(axis=1), [.025,.975]).tolist(),
                                    per_seed=dict(zip(map(str, seeds), paired.tolist(), strict=True)),
                                    cffg_mean=left.mean(), peer_mean=reference,
                                    relative_to_peer_pct=100*mean/abs(reference) if reference else None)
    return result


def analyze(root):
    hashes = {}
    plan, workers = metadata(root, hashes)
    ph = hashes["protocol.json"]
    base = dict(N=60, tsim=26., SP={"Ca": np.repeat([.85, .90, .87], 20).tolist()})
    targets = np.array(parameters(base, True)["SP"]["Ca"][1:])
    libraries, history_audit = {}, {}
    hold_events = []
    noise_reference = {}
    paired_noise = {str(s): 0. for s in SEEDS}
    max_forecast = max_reward = max_sensor = 0.
    breaks = 0
    for lib in LIBRARIES:
        folder = root / lib
        require({p.name for p in folder.glob("seed-*") if p.is_dir()} == {f"seed-{s}" for s in SEEDS},
                f"Incomplete or unexpected seed set under {folder}")
        episodes = {}
        history_audit[lib] = {}
        clocks = defaultdict(lambda: defaultdict(list))
        for seed in SEEDS:
            directory = folder / f"seed-{seed}"
            files = {p.name for p in directory.iterdir() if p.suffix in (".json", ".jsonl")}
            expected = {f"{c}-{a}.{ext}" for c in CONDITIONS for a in MOTION for ext in ("json", "jsonl")}
            require(files == expected, f"Unexpected episode file set in {directory}")
            sample = str(seed)
            results = episodes[sample] = {}
            traces = {}
            for condition in CONDITIONS:
                entries = results[condition] = {}
                traces[condition] = {}
                for arm in MOTION:
                    record, trace, times = audit_episode(root, lib, seed, condition, arm, plan, ph, targets, hashes)
                    hold_events.extend(trace["hold_events"])
                    traces[condition][arm] = trace
                    entries[arm] = record
                    noise = trace["noise"]
                    reference = noise_reference.setdefault(seed, noise)
                    difference = float(np.max(abs(noise - reference)))
                    require(difference <= 64*EPS, f"Unmatched noise in {lib}/{seed}/{condition}/{arm}")
                    paired_noise[sample] = max(paired_noise[sample], difference)
                    a = record["audit"]
                    max_forecast = max(max_forecast, a["max_forecast_replay_error"])
                    max_reward = max(max_reward, a["max_reward_replay_error"])
                    max_sensor = max(max_sensor, a["max_noise_replay_error"])
                    for clock, seconds in times.items():
                        clocks[condition, arm][clock].append(seconds)
            checks = history_audit[lib][sample] = histories(traces)
            breaks += sum(v["first_divergent_held_action"] is not None for v in checks.values())
        libraries[lib] = summarize_library(episodes, clocks, workers[lib])
    require(len(hold_events) == sum(s["totals"]["holds"] for s in libraries.values()),
            "Held command records disagree with audited episode totals")
    audit = dict(episodes=N_EPISODES, decisions=N_EPISODES*N, max_reward_replay_error=max_reward,
                 max_forecast_replay_error=max_forecast, max_sensor_noise_replay_error=max_sensor,
                 max_paired_noise_difference=max(paired_noise.values()), per_seed_noise_difference=paired_noise,
                 matched_histories=history_audit, history_comparisons_diverged_on_hold=breaks,
                 paired_seed_interpretation="Each seed supplies the same exogenous noise and schedule. "
                 "A held command can separate previously identical trajectories. Its realized outcomes "
                 "and all later outcomes remain in the paired seed comparison.")
    return dict(schema_version=1, protocol_sha256=ph, seeds=list(SEEDS), libraries=libraries, hold_events=hold_events,
                between_libraries=between_libraries(libraries, SEEDS), audit=audit, raw_sha256=hashes,
                bootstrap=dict(draws=DRAWS, seed=BOOTSTRAP_SEED, level=.95, unit="whole paired seed",
                               method="Pointwise descriptive percentile interval for the mean paired difference"),
                primary_metric="changed_rmse", primary_unit="mol/L",
                effect_direction="CFFG minus each peer. Lower candidate RMSE indicates better control. "
                "Learning benefit is own frozen RMSE minus own candidate RMSE. Positive benefit favors learning.",
                relative_effect="100 times the mean paired difference divided by the absolute peer mean.",
                engineering_references="The stationary limit is 1% versus the same planner with frozen UA. "
                "Other gates describe the previous engineering screen. Between-library comparisons "
                "report effect sizes and intervals without a practical equivalence threshold.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed comparison archive")
    root = parser.parse_args().output.expanduser().resolve()
    result = analyze(root)
    save(root / "analysis.json", result)
    print(f"Wrote {root / 'analysis.json'} from {result['audit']['decisions']} audited decisions")


if __name__ == "__main__":
    main()
