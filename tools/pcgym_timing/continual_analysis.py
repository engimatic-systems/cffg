"""Audit saved changing-UA development trajectories and write analysis.json and report.txt."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from pilot import save
from summarize import physical_observations


SEEDS = tuple(range(6401, 6406))
CONDITIONS = ("stationary", "switching")
ARMS = ("frozen", "learned", "informed")
METHODS = (*ARMS, "grid")
N = 599
WINDOWS = {
    "full": (0, N), "before": (0, 178), "during": (178, 373), "after": (373, N),
    "changed_first25": (178, 203), "changed_plateau": (203, 373),
    "returned_first25": (373, 398), "returned_plateau": (398, N),
}
EPS = np.finfo(np.float64).eps


def require(condition, message):
    if not condition:
        raise ValueError(message)


def series(values, shape, field):
    array = np.asarray(values, dtype=float)
    require(array.shape == shape and np.isfinite(array).all(), f"Invalid {field} (shape {array.shape})")
    return array


def rmse(error):
    return np.sqrt(np.mean(np.square(error), axis=0))


def species(values):
    return dict(zip(("Ca", "T"), map(float, values), strict=True))


def latency(samples):
    return dict(n=len(samples), mean=float(np.mean(samples)), median=float(np.median(samples)),
                p95=float(np.quantile(samples, .95)), p99=float(np.quantile(samples, .99)),
                max=float(np.max(samples)))


def recovery(estimate, truth):
    results = {}
    for name in ("during", "after"):
        start, stop = WINDOWS[name]
        within = abs(estimate[start:stop] / truth[start:stop] - 1) <= .01
        runs = np.flatnonzero(np.convolve(within.astype(int), np.ones(5, dtype=int), "valid") == 5)
        delay = int(runs[0]) if len(runs) else None
        results[name] = dict(first_tick=None if delay is None else start + delay, delay=delay,
                             confirmation_delay=None if delay is None else delay + 4)
    return results


def read_series(path, seed, condition):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require([r["tick"] for r in rows] == list(range(N)), f"Missing, duplicate or unordered ticks in {path}")

    def column(key, width=None):
        shape = (N,) if width is None else (N, width)
        return series([r[key] for r in rows], shape, f"{path.name} {key}")

    observed = physical_observations(column("next_obs", 2))
    normalized_obs = column("obs", 2)
    previous = physical_observations(normalized_obs)
    truth = column("truth", 2)
    true_ua = column("true_ua")
    required_ua = np.full(N, 50000.)
    if condition == "switching":
        required_ua[178:373] = 48150.
    require(np.array_equal(true_ua, required_ua), f"Incorrect UA change schedule in {path}")
    require(np.allclose(previous[0], [.8, 330.], atol=1e-12, rtol=0)
            and np.array_equal(previous[1:], observed[:-1]), f"Discontinuous observation history in {path}")
    target, applied, reward = column("target"), column("applied"), column("reward")
    expected = -1000 * (truth[:, 0] - target) ** 2
    reward_error = float(np.max(abs(reward - expected)))
    require(np.allclose(reward, expected, atol=1e-12, rtol=1e-12), f"Reward mismatch in {path}")
    require(np.max(abs(applied)) <= 1, f"Applied coolant out of bounds in {path}")
    noise = observed / truth - 1
    replay = .001 * np.random.RandomState(seed).normal(size=(N, 2))
    replay_error = float(np.max(abs(noise - replay)))
    require(replay_error <= 32 * EPS, f"Exogenous noise disagrees with seed {seed} in {path}: {replay_error}")
    audit = dict(ticks=N, max_reward_replay_error=reward_error, max_noise_replay_error=replay_error)
    return rows, dict(observed=observed, obs=normalized_obs, truth=truth, true_ua=true_ua, target=target,
                      applied=applied, reward=reward, noise=noise), audit


def physical_prediction(predictions):
    # The model uses the fixture's float64 bounds. PCGym decodes its sensors
    # using float32 Box bounds in physical_observations().
    normalized = series(predictions, (N, 2), "prediction")
    return np.array([.7, 300.]) + (normalized + 1) * np.array([.15, 25.])


def metrics(common, prediction, ua):
    true_ua = common["true_ua"]
    ua_error = ua - true_ua
    measures = {}
    for name, (start, stop) in WINDOWS.items():
        i = slice(start, stop)
        measures[name] = dict(
            ticks=stop - start, return_sum=float(sum(common["reward"][i])),
            tracking_rmse=float(rmse(common["truth"][i, 0] - common["target"][i])),
            observation_prediction_rmse=species(rmse(prediction[i] - common["observed"][i])),
            truth_prediction_rmse=species(rmse(prediction[i] - common["truth"][i])),
            mean_ua=float(np.mean(ua[i])), ua_mae=float(np.mean(abs(ua_error[i]))),
            ua_mare_pct=float(100*np.mean(abs(ua_error[i] / true_ua[i]))),
        )
    return dict(windows=measures, recovery=recovery(ua, true_ua))


def check_ua(method, ua, truth):
    require(np.all(ua > 0), f"Nonpositive {method} UA estimate")
    if method == "frozen":
        require(np.all(ua == 50000), "Frozen arm changed its UA")
    if method == "informed":
        require(np.array_equal(ua, truth), "Informed arm missed the pre-action coefficient")


def audit_updates(updates, estimates, observed, predictions, method, settings):
    """Pair each update with the transition it scores, including the terminal one."""
    def col(key):
        return series([u[key] for u in updates], (N,), method + " " + key)

    require([u["n_updates"] for u in updates] == list(range(1, N+1)), f"{method} update count")
    before, after = col("ua_before"), col("ua_after")
    require(np.array_equal(before, estimates) and np.array_equal(after[:-1], estimates[1:])
            and np.isclose(before[0], settings["initial_ua"], atol=1e-9, rtol=0),
            f"Broken {method} pre-action update sequence")
    lower, upper = settings["ua_bounds"]
    require(np.all((after >= lower - 1e-9) & (after <= upper + 1e-9)), f"{method} UA outside prior bounds")
    expected, innovation = col("temperature_prediction"), col("temperature_innovation")
    require(np.allclose(expected + innovation, observed, atol=1e-12, rtol=0), "Update used a different observation")
    total = col("temperature_variance")
    require(np.all(total > 0), "Invalid predictive variance")
    log_score = col("log_predictive_density")
    result = dict(updates=N, initial_ua=before[0], final_ua=after[-1])
    if method == "learned":
        cfg = settings["learner"]
        variance = col("log_ua_variance")
        prior = np.r_[cfg["initial_log_sd"]**2, variance[:-1]] + cfg["motion_log_sd"]**2
        derivative = col("temperature_log_ua_derivative")
        noise = col("conditional_temperature_variance")
        require(np.all(noise > 0) and np.all(variance > 0), "Invalid EKF variance")
        require(np.allclose(expected, predictions, rtol=1e-12, atol=1e-12), "Learner prediction/innovation mismatch")
        require(np.allclose(total, noise + derivative**2 * prior, rtol=1e-12, atol=0), "EKF prior variance mismatch")
        gain = prior * derivative / total
        replay = np.clip(before * np.exp(gain * innovation), lower, upper)
        require(np.allclose(after, replay, atol=1e-8, rtol=0), "Replayed scalar UA update disagrees")
        require(np.allclose(variance, (1-gain*derivative)**2 * prior + gain**2 * noise,
                            rtol=1e-12, atol=0), "EKF posterior variance mismatch")
        expected_score = -.5 * (np.log(2 * np.pi * total) + innovation**2 / total)
        require(np.allclose(log_score, expected_score, rtol=1e-12, atol=1e-12), "EKF log score mismatch")
        result["max_update_replay_ua_error"] = float(np.max(abs(replay - after)))
    return result


def control(path, seed, condition, arm, settings):
    rows, common, audit = read_series(path, seed, condition)
    ua = series([r["ua"] for r in rows], (N,), "ua")
    check_ua(arm, ua, common["true_ua"])
    sidecar = json.loads(path.with_suffix(".json").read_text())
    require(sidecar["trace_sha256"] == sha256(path.read_bytes()).hexdigest()
            and (sidecar["seed"], sidecar["arm"], sidecar["condition"], sidecar["steps"]) ==
            (seed, arm, condition, N), f"Episode provenance mismatch in {path}")
    if arm == "learned":
        require(rows[0]["update"] is None, "Spurious first-tick update")
        updates = [r["update"] for r in rows[1:]] + [sidecar["terminal_update"]]
        audit["learning"] = audit_updates(updates, ua, [r["next_obs"][1] for r in rows],
                                          [r["prediction"][1] for r in rows], arm, settings)
    else:
        require(all(r["update"] is None for r in rows) and sidecar["terminal_update"] is None,
                f"Unexpected parameter update in {path}")
    audit["trace_sha256"] = sidecar["trace_sha256"]
    result = metrics(common, physical_prediction([r["prediction"] for r in rows]), ua)
    times = {}
    for key in ("decision_s", "update_s", "forecast_s"):
        times[key] = series([r[key] for r in rows], (N,), key)
        require(np.all(times[key] >= 0), f"Negative {key} in {path}")
    require(np.all(times["update_s"] <= times["decision_s"]), f"Update outside decision clock in {path}")
    failures = holds = misses = clipped = 0
    last = 0.
    for r in rows:
        ok, late = r["status"] == "ok", r["decision_s"] > 1.
        held = late or not ok
        require(r["held"] is held and r["late"] is late, f"Fallback/deadline flag mismatch in {path} tick {r['tick']}")
        raw = r["raw_action"]
        if ok:
            require(raw is not None and np.isfinite(raw), f"Invalid successful action in {path}")
        applied = last if held else np.clip(raw, -1, 1)
        require(applied == r["applied"], f"Applied command mismatch in {path} tick {r['tick']}")
        last = applied
        failures += int(not ok)
        holds += held
        misses += late
        clipped += int(not held and abs(raw) > 1)
    audit.update(failures=failures, holds=holds, misses=misses, clipped=clipped)
    require(holds == sidecar["holds"] and np.isclose(sum(common["reward"]), sidecar["total_return"],
            rtol=1e-12, atol=1e-12), f"Saved hold/return mismatch in {path}")
    result.update(audit=audit, warm_setup_s=sidecar["warm_setup_s"], episode_s=sidecar["episode_s"], latency_s={k: latency(v) for k, v in times.items()})
    common.update(ua=ua, held=np.array([r["held"] for r in rows]))
    return result, common, times


def tape(path, condition, settings):
    rows, common, audit = read_series(path, 6400, condition)
    estimates, predictions, results = {}, {}, {}
    for method in METHODS:
        ua = series([r["estimates"][method] for r in rows], (N,), method)
        check_ua(method, ua, common["true_ua"])
        predictions[method] = physical_prediction([r["predictions"][method] for r in rows])
        estimates[method] = ua
        results[method] = metrics(common, predictions[method], ua)
    for method in ("learned", "grid"):
        updates = [r["updates"][method] for r in rows]
        audit[method] = audit_updates(updates, estimates[method], [r["next_obs"][1] for r in rows],
                                     [r["predictions"][method][1] for r in rows], method, settings)
    contrasts = {}
    for name, (start, stop) in WINDOWS.items():
        i = slice(start, stop)
        effects = predictions["informed"][i] - predictions["frozen"][i]
        noise = common["observed"][i] - common["truth"][i]
        score = {m: float(np.mean([r["updates"][m]["log_predictive_density"] for r in rows[i]]))
                 for m in ("learned", "grid")}
        sensitivity = np.array([r["updates"]["learned"]["temperature_log_ua_derivative"]
                                / np.sqrt(r["updates"]["learned"]["conditional_temperature_variance"])
                                for r in rows[i]])
        contrasts[name] = dict(
            mean_log_predictive_density_normalized_T=score,
            median_local_change_in_conditional_temperature_sd=float(np.median(abs(sensitivity * np.log(48150/50000)))),
            informed_frozen_prediction_rms=species(rmse(effects)), sensor_noise_rms=species(rmse(noise)),
            learned_grid_ua_rms=float(rmse(estimates["learned"][i] - estimates["grid"][i])),
            learned_grid_prediction_rms=species(rmse(predictions["learned"][i] - predictions["grid"][i])),
        )
    common["estimates"] = estimates
    return dict(audit=audit, methods=results, contrasts=contrasts), common


def forecast_scores(path, common, scheduled_origins):
    records = json.loads(path.read_text())
    groups = defaultdict(dict)
    for r in records:
        tick, lead, method = r["tick"], r["lead"], r["method"]
        require(lead in (15, 60) and method in METHODS and isinstance(tick, int) and 0 <= tick <= N-lead,
                f"Invalid forecast key {(tick, lead, method)} in {path}")
        require(r["ua"] == common["estimates"][method][tick], f"Forecast UA differs from origin estimate at {tick}")
        truth = series(r["truth"], (2,), "forecast truth")
        require(np.allclose(truth, common["truth"][tick+lead-1], atol=1e-10, rtol=0),
                f"Incorrect forecast target at {(tick, lead, method)}")
        require(tick not in groups[lead, method], f"Duplicate forecast at {(tick, lead, method)}")
        groups[lead, method][tick] = series(r["prediction"], (2,), "forecast") - truth
    result = {}
    for lead in (15, 60):
        origins = sorted(groups[lead, "frozen"])
        require(origins == scheduled_origins and all(sorted(groups[lead, m]) == origins for m in METHODS),
                f"Unmatched {lead}-step forecast origins in {path}")
        crosses = np.array([np.any(common["true_ua"][t:t+lead] != common["true_ua"][t]) for t in origins])
        result[str(lead)] = dict(origins=origins, n=len(origins), methods={}, changes_within_horizon={},
                                 n_crossing=int(crosses.sum()))
        for method in METHODS:
            errors = np.asarray([groups[lead, method][t] for t in origins])
            result[str(lead)]["methods"][method] = species(rmse(errors))
            result[str(lead)]["changes_within_horizon"][method] = {
                label: species(rmse(errors[subset])) if subset.any() else None
                for label, subset in (("crossing", crosses), ("unchanged", ~crosses))}
    return result


def paired_comparisons(episodes, condition):
    contrasts = {}
    for arm in ("learned", "informed"):
        contrasts[arm] = {}
        for period in WINDOWS:
            by_seed = {}
            for seed in SEEDS:
                results = episodes[str(seed)][condition]
                trial, base = (results[a]["windows"][period] for a in (arm, "frozen"))
                by_seed[str(seed)] = dict(
                    tracking_rmse_delta=trial["tracking_rmse"] - base["tracking_rmse"],
                    tracking_reduction_pct=100 * (1 - trial["tracking_rmse"] / base["tracking_rmse"]),
                    return_delta=trial["return_sum"] - base["return_sum"],
                    return_gain_pct=100 * (trial["return_sum"]-base["return_sum"]) / abs(base["return_sum"]),
                    ua_mae_delta=trial["ua_mae"] - base["ua_mae"])
            records = list(by_seed.values())
            contrasts[arm][period] = dict(
                seeds=by_seed, lower_tracking_seeds=sum(r["tracking_rmse_delta"] < 0 for r in records),
                median_tracking_reduction_pct=float(np.median([r["tracking_reduction_pct"] for r in records])),
                median_return_gain_pct=float(np.median([r["return_gain_pct"] for r in records])),
                mean_return_delta=float(np.mean([r["return_delta"] for r in records])),
                mean_tracking_rmse_delta=float(np.mean([r["tracking_rmse_delta"] for r in records])))
    return contrasts


def gates(episodes, contrasts, tapes):
    checks = {}
    for arm in ("learned", "informed"):
        row = contrasts["switching"][arm]["during"]
        reduction, wins = row["median_tracking_reduction_pct"], row["lower_tracking_seeds"]
        checks[arm+"_control_headroom" if arm == "informed" else "learned_control_benefit"] = dict(
            passed=reduction >= 1. and wins >= 4, median_reduction_pct=reduction, lower_seeds=wins,
            required_median_pct=1., required_lower_seeds=4)
    def average(condition, arm, period, key):
        return float(np.mean([episodes[str(s)][condition][arm]["windows"][period][key] for s in SEEDS]))
    ratio = average("stationary", "learned", "full", "tracking_rmse") / average(
        "stationary", "frozen", "full", "tracking_rmse")
    checks["stationary_penalty"] = dict(passed=ratio <= 1.02, learned_over_frozen=ratio, maximum_ratio=1.02)
    settled = {a: average("switching", a, "changed_plateau", "ua_mae") for a in ("learned", "frozen")}
    reduction = 100 * (1 - settled["learned"] / settled["frozen"])
    checks["settled_parameter_error"] = dict(passed=reduction >= 50., reduction_pct=reduction,
                                              required_pct=50., mean_ua_mae=settled)
    tape_t = {a: tapes["switching"]["methods"][a]["windows"]["during"]["observation_prediction_rmse"]["T"]
              for a in ("learned", "frozen")}
    checks["changed_tape_temperature"] = dict(passed=tape_t["learned"] < tape_t["frozen"], rmse_K=tape_t)
    totals = {k: sum(episodes[str(s)][c][a]["audit"][k] for s in SEEDS for c in CONDITIONS for a in ARMS)
              for k in ("holds", "failures", "misses")}
    checks["reliability"] = dict(passed=all(v == 0 for v in totals.values()), **totals)
    return checks


def arm_means(episodes):
    summary = {}
    for c in CONDITIONS:
        arms = summary[c] = {}
        for a in ARMS:
            windows = arms[a] = {}
            for name in WINDOWS:
                rows = [episodes[str(s)][c][a]["windows"][name] for s in SEEDS]
                record = {k: float(np.mean([r[k] for r in rows])) for k in (
                    "return_sum", "tracking_rmse", "mean_ua", "ua_mae", "ua_mare_pct")}
                for k in ("observation_prediction_rmse", "truth_prediction_rmse"):
                    record[k] = {v: float(np.mean([r[k][v] for r in rows])) for v in ("Ca", "T")}
                windows[name] = record
    return summary


def same_history(first, second, stop, fields):
    """A held command can change every successor; check up to the first hold."""
    holds = np.flatnonzero(first["held"][:stop] | second["held"][:stop])
    end = int(holds[0]) if len(holds) else stop
    for k in fields:
        require(np.array_equal(first[k][:end], second[k][:end]),
                f"Different {k} through decision {end-1} on matched history")
    return dict(ticks_checked=end, first_hold=int(holds[0]) if len(holds) else None)


def analyze(root):
    study = json.loads((root / "study.json").read_text())
    qualified = json.loads((root/"qualified.json").read_text())
    complete = json.loads((root/"completed.json").read_text())
    study_hash = sha256((root/"study.json").read_bytes()).hexdigest()
    require(qualified["study_sha256"] == study_hash == complete["study_sha256"]
            and qualified["plant"]["passed"] and complete["episodes"] == 30, "Incomplete or mismatched study")
    require(study["split"] == "development" and study["tape_seed"] == 6400 and study["steps"] == N
            and set(study["seeds"]) == set(SEEDS) and set(study["arms"]) == set(ARMS)
            and set(study["conditions"]) == set(CONDITIONS) and study["forecast_leads"] == [15,60]
            and study["warm_deadline_s"] == 1. and study["ua"] ==
            dict(nominal=50000., changed=48150., changed_interval=[178,373]),
            "Saved study disagrees with the development design")
    scoring = study["scoring"]
    require(scoring["before"] == [0,178] and scoring["changed"] == [178,373]
            and scoring["returned"] == [373,N] and scoring["settled_changed"] == [203,373]
            and scoring["transition_window"] == 25, "Scoring windows differ from study.json")
    rule = study["decision_rule"]
    require(rule["min_median_paired_changed_rmse_reduction_pct"] == 1. and rule["minimum_wins_of_five"] == 4
            and rule["max_stationary_learned_to_frozen_mean_rmse_ratio"] == 1.02
            and rule["min_settled_ua_mae_reduction_pct"] == 50.
            and rule["require_lower_learned_changed_tape_T_rmse"] and rule["max_holds"] == 0,
            "Engineering thresholds differ from study.json")
    plant = qualified["plant"]
    require(plant["continuous_state_time_and_integrator"] and plant["switch_ticks"] == [178,373]
            and plant["learner_observation"] == ["Ca_norm", "T_norm"] and plant["raw_observation_width"] == 4,
            "Missing hidden-channel and continuity qualification")
    episodes, tapes, paired_noise, forecasts, histories = {}, {}, {}, {}, {}
    pooled = defaultdict(lambda: defaultdict(list))
    targets = None
    tape_common = {}
    max_replay = max_reward = 0.
    for c in CONDITIONS:
        tapes[c], tape_common[c] = tape(root/f"tape-{c}.jsonl", c, study["method"])
        forecasts[c] = forecast_scores(root/f"tape-forecasts-{c}.json", tape_common[c], study["forecast_origins"])
        max_replay = max(max_replay, tapes[c]["audit"]["max_noise_replay_error"])
        max_reward = max(max_reward, tapes[c]["audit"]["max_reward_replay_error"])
    def match(a, b, what):
        delta = float(np.max(abs(a - b)))
        require(delta <= 64*EPS, f"Unmatched {what}, maximum difference {delta}")
        return delta
    paired_noise["tape_6400"] = match(tape_common["stationary"]["noise"], tape_common["switching"]["noise"], "tape noise")
    match(tape_common["stationary"]["applied"], tape_common["switching"]["applied"], "tape commands")
    for name in ("learned", "grid"):
        require(np.array_equal(tape_common["stationary"]["estimates"][name][:179],
                               tape_common["switching"]["estimates"][name][:179]),
                f"{name} tape estimate anticipates the first changed transition")
    require(np.array_equal(tape_common["stationary"]["obs"][:179], tape_common["switching"]["obs"][:179]),
            "Tape observations differ before the first affected state")
    for c in CONDITIONS:
        if targets is None:
            targets = tape_common[c]["target"]
        match(targets, tape_common[c]["target"], "tape targets")
    for seed in SEEDS:
        results = episodes[str(seed)] = {}
        noises, trajectories = [], {}
        for c in CONDITIONS:
            arms = results[c] = {}
            for a in ARMS:
                path = root / f"seed-{seed}" / f"{c}-{a}.jsonl"
                record, common, times = control(path, seed, c, a, study["method"])
                arms[a] = record
                noises.append(common["noise"])
                trajectories[c, a] = common
                match(targets, common["target"], "control targets")
                max_replay = max(max_replay, record["audit"]["max_noise_replay_error"])
                max_reward = max(max_reward, record["audit"]["max_reward_replay_error"])
                for k, v in times.items():
                    pooled[c, a][k].append(v)
        tests = histories[str(seed)] = {}
        tests["stationary_informed_frozen"] = same_history(
            trajectories["stationary", "informed"], trajectories["stationary", "frozen"], N, ("applied",))
        for a in ("frozen", "learned"):
            tests[a+"_before_observing_change"] = same_history(
                trajectories["stationary", a], trajectories["switching", a], 179, ("obs", "ua", "applied"))
        paired_noise[str(seed)] = max(match(noises[0], draws, f"seed {seed} noise") for draws in noises)
    contrasts = {c: paired_comparisons(episodes, c) for c in CONDITIONS}
    checks = gates(episodes, contrasts, tapes)
    return dict(study=study, qualification=qualified, windows=WINDOWS, tapes=tapes, forecasts=forecasts, episodes=episodes,
                control_means=arm_means(episodes), paired=contrasts, gates=checks, proceed=all(v["passed"] for v in checks.values()),
                pooled_latency_s={c: {a: {k: latency(np.concatenate(v)) for k,v in pooled[c,a].items()}
                                     for a in ARMS} for c in CONDITIONS},
                audit=dict(episodes=len(SEEDS)*len(CONDITIONS)*len(ARMS), transitions_per_episode=N,
                           control_transitions=N*len(SEEDS)*len(CONDITIONS)*len(ARMS),
                           tape_transitions=N*2, max_noise_replay_error=max_replay,
                           max_reward_replay_error=max_reward, paired_noise_max_abs=paired_noise,
                           matched_histories=histories))


def report(d):
    audit, checks = d["audit"], d["gates"]
    lines = ["Changing-reactor development pilot", "",
             ("GO" if d["proceed"] else "NO-GO") + " for a larger control study under the preregistered screen.",
             "Five paired development seeds (6401–6405) and one shared-tape seed (6400). "
             "The coefficient equals 50000 initially, drops to 48150 on switching transitions [178,373), "
             "and returns to 50000. The thresholds are engineering decisions. Generalization to new seeds remains open.", ""]
    for arm, key in (("Informed", "informed_control_headroom"), ("Learned", "learned_control_benefit")):
        g = checks[key]
        lines.append(f"{status(g)}  {arm} median paired changed-period concentration RMSE reduction "
                     f"{g['median_reduction_pct']:+.3f}% (required ≥1%), lower on {g['lower_seeds']}/5 seeds (required ≥4).")
    g = checks["stationary_penalty"]
    lines.append(f"{status(g)}  Stationary learned/frozen ratio of mean full-episode RMSE "
                 f"{g['learned_over_frozen']:.6f} (maximum 1.02).")
    g = checks["settled_parameter_error"]
    ua = g["mean_ua_mae"]
    lines.append(f"{status(g)}  Settled changed UA MAE fell {g['reduction_pct']:.2f}% "
                 f"(required ≥50%). Learned {ua['learned']:.1f}, frozen {ua['frozen']:.1f}.")
    g = checks["changed_tape_temperature"]
    t = g["rmse_K"]
    lines.append(f"{status(g)}  Changed-period shared-tape next sensor T RMSE in K, "
                 f"learned {t['learned']:.5f} and frozen {t['frozen']:.5f} (learned must be lower).")
    g = checks["reliability"]
    lines.append(f"{status(g)}  {g['holds']} holds, {g['failures']} solve failures, {g['misses']} deadline misses.")
    if checks["settled_parameter_error"]["passed"] and checks["changed_tape_temperature"]["passed"]:
        lines.append("The lower settled coefficient error and improved predictions on shared "
                     "commands support parameter identification at this noise level. The decision on a larger "
                     "control trial also depends on tracking benefit and the stationary cost of adaptation.")
    lines.extend(["", "Trace audit",
        f"{audit['episodes']} complete episodes of {N} transitions, {audit['control_transitions']} control transitions "
        f"and {audit['tape_transitions']} tape transitions. Saved trace hashes and per-episode returns agree.",
        f"Maximum reward discrepancy {audit['max_reward_replay_error']:.3g} against "
        "−1000 × (next true concentration − target)².",
        f"Maximum multiplicative noise discrepancy {audit['max_noise_replay_error']:.3g} against the seeded "
        f"draws and {max(audit['paired_noise_max_abs'].values()):.3g} across conditions and arms.",
        "The stationary informed and frozen commands and the first 179 stationary/switching learned and frozen "
        "observations, pre-action coefficients and commands were checked for exact equality on every seed up to "
        "any first held decision. Tick 178 precedes observation of the first affected successor.",
        "Both tapes used the same exogenous coolant and noise. Every closed-loop action was checked against "
        "clipping to [−1,1] and previous-action fallback. Each controller saw two measured state channels.",
        "Every learned post-transition update links to the next pre-action estimate, including at both switches. "
        "All 599 scalar Kalman updates per trajectory were replayed from saved innovations, slopes and variances. "
        "Grid update counts and posterior-to-next-estimate links were checked on both tapes.",
        "", "Closed-loop control", "The arms drive different trajectories with matched exogenous noise. "
        "Tracking uses next true concentration. Returns use the simulator reward. "
        "Tables report arithmetic means of the five per-seed scores.",
        "Windows are half-open in transition ticks. " + ", ".join(
            f"{k} [{a},{b})" for k,(a,b) in WINDOWS.items()) + ".",
        "condition    arm       window              RMSE Ca     return       UA MAE    UA MARE %   sensor RMSE Ca      T K"])
    for c in CONDITIONS:
        for a in ARMS:
            for w in WINDOWS:
                r = d["control_means"][c][a][w]
                sensor = r["observation_prediction_rmse"]
                lines.append(f"{c:12} {a:9} {w:17} {r['tracking_rmse']:10.6f} "
                             f"{r['return_sum']:10.4f} {r['ua_mae']:12.3f} {r['ua_mare_pct']:10.4f} "
                             f"{sensor['Ca']:16.7f} {sensor['T']:8.4f}")
    lines.extend(["", "Paired comparisons against frozen",
        "Positive return delta and positive tracking reduction favor the named arm. The return gain divides "
        "each paired return difference by the absolute frozen return before taking the median.",
        "condition    arm       window             mean return Δ  median gain %   mean RMSE Δ    median reduction %    lower / 5"])
    for c in CONDITIONS:
        for a in ("informed", "learned"):
            for w,r in d["paired"][c][a].items():
                lines.append(f"{c:12} {a:9} {w:17} {r['mean_return_delta']:+13.5f} "
                             f"{r['median_return_gain_pct']:+14.4f} {r['mean_tracking_rmse_delta']:+14.7f} "
                             f"{r['median_tracking_reduction_pct']:+20.4f} {r['lower_tracking_seeds']:8}")
    lines.extend(["", "Switching recovery",
        "Delay to the first tick of five consecutive pre-action estimates within 1% of true UA. "
        "Confirmation takes four further transitions. Censored means the five-tick run was absent.",
        "source         method       changed delay     returned delay"])
    def recovery_line(source, method, data):
        fields = [str(data[w]["delay"]) if data[w]["delay"] is not None else "censored" for w in ("during","after")]
        lines.append(f"{source:14} {method:10} {fields[0]:>14} {fields[1]:>18}")
    for s in SEEDS:
        for a in ARMS:
            recovery_line(str(s), a, d["episodes"][str(s)]["switching"][a]["recovery"])
    for a in METHODS:
        recovery_line("tape 6400", a, d["tapes"]["switching"]["methods"][a]["recovery"])
    lines.extend(["", "Identification on the shared tape",
        "Each method observed the same two saved channels and commands. Each prediction preceded its observation. "
        "The scalar learner fits log UA from temperature innovations. The grid is a finite-grid reference "
        "with local diffusion and occasional restarts. UA 48150 lies between its grid points. "
        "These prediction and parameter comparisons isolate estimation from feedback control.",
        "condition    method     window               mean UA      UA MAE   MARE %     sensor Ca       T K       true Ca       T K"])
    for c in CONDITIONS:
        for a in METHODS:
            for w,r in d["tapes"][c]["methods"][a]["windows"].items():
                sensor, truth = r["observation_prediction_rmse"], r["truth_prediction_rmse"]
                lines.append(f"{c:12} {a:9} {w:17} {r['mean_ua']:11.2f} {r['ua_mae']:10.2f} "
                             f"{r['ua_mare_pct']:8.3f} {sensor['Ca']:13.7f} {sensor['T']:9.4f} "
                             f"{truth['Ca']:13.7f} {truth['T']:9.4f}")
    lines.extend(["", "Tape prediction separation and reference agreement",
        "T separation is the RMS difference between informed and frozen predictions; noise is the measured sensor "
        "error RMS. Local signal is |Δlog UA × temperature slope| divided by conditional noise SD, evaluated at "
        "the learned point. Scores are mean one-step log predictive densities of normalized T. The grid score "
        "integrates the discrete prior; the grid RMSE table uses the model evaluated at the grid's UA mean.",
        "condition    window               T separation K   T noise K    local signal (SD)   learned-grid UA RMS   log score (learned/grid)"])
    for c in CONDITIONS:
        for w,r in d["tapes"][c]["contrasts"].items():
            score = r["mean_log_predictive_density_normalized_T"]
            lines.append(f"{c:12} {w:17} {r['informed_frozen_prediction_rms']['T']:14.4f} "
                         f"{r['sensor_noise_rms']['T']:11.4f} {r['median_local_change_in_conditional_temperature_sd']:20.3f} "
                         f"{r['learned_grid_ua_rms']:21.3f} {score['learned']:12.4f} / {score['grid']:.4f}")
    lines.extend(["", "Shared-tape open-loop endpoint RMSE",
        "Each forecast holds the coefficient available at its origin fixed and follows the known coolant sequence. "
        "A horizon may cross a hidden change. Errors compare endpoint means to physical simulator truth.",
        "condition     lead  origins   method          Ca RMSE        T RMSE K"])
    for c in CONDITIONS:
        for lead, row in d["forecasts"][c].items():
            for a, score in row["methods"].items():
                lines.append(f"{c:12} {lead:>4} {row['n']:8}   {a:9} {score['Ca']:13.6f} {score['T']:15.4f}")
    lines.extend(["", "Warm latency, milliseconds pooled over 2,995 decisions per condition and arm",
        "Decision time includes the preceding update, parameter delivery, window, certified solve and clipping. "
        "Forecast time starts after command selection. The one-second deadline uses decision time alone.",
        "condition    arm       clock           mean     median        p95        p99        max"])
    for c in CONDITIONS:
        for a in ARMS:
            for k,timing in d["pooled_latency_s"][c][a].items():
                values = " ".join(f"{1000*timing[q]:10.4f}" for q in ("mean","median","p95","p99","max"))
                lines.append(f"{c:12} {a:9} {k:12} {values}")
    lines.extend(["", "The experiment uses benchmark-local physical parameter estimation and public CFFG "
                  "parameter replacement for the planning model. The learner approximates the previous-state "
                  "uncertainty from two noisy sensors. It omits latent-bias uncertainty and correlations between "
                  "successive innovations. Each condition has a single shared development tape.",
                  "analysis.json contains every seed-level score, pre/post-switch contrast, setup time, "
                  "update audit, timing quantile, exact gate and protocol."])
    return "\n".join(lines) + "\n"


def status(gate):
    return "PASS" if gate["passed"] else "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed study directory")
    root = parser.parse_args().output.expanduser().resolve()
    data = analyze(root)
    rendered = report(data)
    save(root / "analysis.json", data)
    temporary = root / "report.txt.tmp"
    temporary.write_text(rendered)
    temporary.replace(root / "report.txt")
    print(rendered.splitlines()[2])


if __name__ == "__main__":
    main()
