"""Score the frozen learner design on paired development or fresh validation traces.

Run with --output ROOT after validation. Development scoring never reads validation files.
"""
import argparse
from collections import defaultdict
from datetime import datetime
from functools import cache
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from continual_analysis import (
    CONDITIONS, EPS, N, WINDOWS, audit_updates, check_ua, control, latency, metrics,
    physical_prediction, read_series, require, same_history, series,
)
from one_percent_run import ARMS, CANDIDATES, DEVELOPMENT_SEEDS, FRESH_SEEDS, ORIGINAL
from pilot import HERE, save


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def settings(study, q):
    base = study["starting_method"]
    return {**base, "learner": {**base["learner"], "motion_log_sd": q}}


def check_design(root, archive, label, q):
    study = read(root/"protocol.json")
    require(Path(study["archive"]).resolve() == archive.resolve(), "Wrong development archive")
    require(study["candidates"] == CANDIDATES and label in CANDIDATES and CANDIDATES[label] == q,
            "Candidate differs from registered motion SD")
    require(study["original_motion_sd"] == study["starting_method"]["learner"]["motion_log_sd"] == ORIGINAL
            and study["primary_stationary_limit_pct"] == 1 and study["sensitivity_limits_pct"] == [2, 3]
            and tuple(study["development_seeds"]) == tuple(DEVELOPMENT_SEEDS)
            and tuple(study["validation_seeds"]) == tuple(FRESH_SEEDS)
            and study["development_tape"] == 6400 and study["validation_tape"] == 6500
            and tuple(study["controls"]) == ARMS and tuple(study["conditions"]) == CONDITIONS
            and study["decision_deadline_s"] == 1, "Study design mismatch")
    require(study["schedule"] == {"nominal": 50000., "changed": 48150., "changed_interval": [178,373]}
            and study["scoring"] == dict(before=[0,178], changed=[178,373], returned=[373,N],
                                         transition_window=25, settled_changed=[203,373]), "Unexpected study windows")
    for name, expected in study["archive_sha256"].items():
        require(digest(archive/name) == expected, f"Archived input changed: {name}")
    require(read(archive/"study.json")["method"] == study["starting_method"], "Retained method differs")
    return study


@cache
def tape_kernel():
    # The runner normally inserts tools for the public reactor fixture. Standalone
    # offline scoring needs the same imports, without initializing a simulator.
    import jax
    if str(HERE.parent) not in sys.path:
        sys.path.append(str(HERE.parent))
    from continual_estimator import _likelihood_model, _mean

    def predict(obs, applied, z):
        return _mean(obs, applied, z), _likelihood_model(obs, applied, z)
    return jax.jit(jax.vmap(predict))


def audit_tape(path, seed, study, q, archive_tape=None):
    import jax
    rows, common, audit = read_series(path, seed, "switching")
    command = common["applied"]
    tick = np.arange(N)
    scheduled = .65 * np.sin(2*np.pi*tick/73) + .20 * np.cos(2*np.pi*tick/19)
    require(np.allclose(command, scheduled, rtol=0, atol=8*EPS), "Tape commands differ from the exogenous schedule")
    require(all(set(r["estimates"]) == set(ARMS) and set(r["predictions"]) == set(ARMS)
                and set(r["updates"]) == {"original", "candidate"} for r in rows), "Unexpected tape streams")
    methods = {}
    for arm in ARMS:
        ua = series([r["estimates"][arm] for r in rows], (N,), arm)
        check_ua(arm, ua, common["true_ua"])
        predictions = series([r["predictions"][arm] for r in rows], (N,2), arm+" prediction")
        # Recompute the observation forecast and local EKF likelihood using only
        # the measurements, estimate and applied command available before each step.
        with jax.enable_x64(True):
            predicted, (_, slope, noise) = jax.device_get(tape_kernel()(
                common["obs"], command, np.log(ua / study["starting_method"]["initial_ua"])))
        require(np.allclose(predictions, predicted, atol=1e-11, rtol=0), f"{arm} tape forecast uses wrong inputs")
        audit[arm] = dict(max_forecast_replay_error=float(np.max(abs(predictions - predicted))))
        if arm in ("original", "candidate"):
            sd = ORIGINAL if arm == "original" else q
            updates = [r["updates"][arm] for r in rows]
            for field, expected in (("temperature_log_ua_derivative", slope), ("conditional_temperature_variance", noise)):
                require(np.allclose([u[field] for u in updates], expected, atol=1e-12, rtol=1e-11),
                        f"{arm} temperature likelihood differs at {field}")
            audit[arm].update(audit_updates(updates, ua, [r["next_obs"][1] for r in rows],
                                            predictions[:,1], "learned", settings(study, sd)))
        methods[arm] = metrics(common, physical_prediction(predictions), ua)
    if archive_tape is not None:
        old, _, _ = read_series(archive_tape, 6400, "switching")
        for current, previous in zip(rows, old, strict=True):
            require(all(current[k] == previous[k] for k in (
                "tick", "obs", "next_obs", "truth", "target", "true_ua", "applied", "reward")),
                f"Development replay substituted a tape transition at {current['tick']}")
            for arm, old_arm in (("original","learned"), ("frozen","frozen"), ("informed","informed")):
                require(abs(current["estimates"][arm]-previous["estimates"][old_arm]) < 1e-7 and
                        np.allclose(current["predictions"][arm], previous["predictions"][old_arm], atol=1e-12, rtol=0),
                        f"Retained tape {arm} differs")
    return dict(seed=seed, sha256=digest(path), audit=audit, methods=methods), common["target"]


def collect(folder, archive, study, q, seeds, dev, targets):
    episodes, histories, errors = {}, {}, {}
    times = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        records = episodes[str(seed)] = {}
        trajectories = {}
        for condition in CONDITIONS:
            arms = records[condition] = {}
            for arm in ARMS:
                retained = dev and arm != "candidate"
                saved_arm = "learned" if retained and arm == "original" else arm
                base = archive if retained else folder
                path = base/f"seed-{seed}"/f"{condition}-{saved_arm}.jsonl"
                sd = {"original": ORIGINAL, "candidate": q}.get(arm)
                if not retained:
                    sidecar = read(path.with_suffix(".json"))
                    require("motion_sd" in sidecar and sidecar["motion_sd"] == sd, f"Wrong motion SD in {path}")
                record, trace, clocks = control(path, seed, condition, saved_arm, settings(study, sd),
                                                 learner_arm=saved_arm if sd is not None else "learned")
                require(np.array_equal(targets, trace["target"]), f"Unmatched target schedule in {path}")
                record["motion_sd"] = sd
                arms[arm] = record
                trajectories[condition, arm] = trace
                for key, values in clocks.items():
                    times[condition,arm][key].append(values)
        common = trajectories["stationary", "frozen"]
        error = max(float(np.max(abs(t["noise"] - common["noise"]))) for t in trajectories.values())
        require(error <= 64*EPS, f"Unmatched exogenous noise within seed {seed}")
        errors[str(seed)] = error
        tests = histories[str(seed)] = {}
        tests["stationary_informed_frozen"] = same_history(common, trajectories["stationary","informed"], N, ("applied",))
        for arm in ("frozen", "original", "candidate"):
            tests[arm+"_before_observing_change"] = same_history(
                trajectories["stationary",arm], trajectories["switching",arm], 179, ("obs","ua","applied"))
    clocks = {c: {a: {k: latency(np.concatenate(v)) for k,v in times[c,a].items()} for a in ARMS} for c in CONDITIONS}
    return episodes, clocks, dict(max_paired_noise_abs=errors, matched_histories=histories,
                                 episodes=len(seeds)*len(ARMS)*len(CONDITIONS), transitions=N*len(seeds)*8)


def paired_rows(episodes):
    rows = {}
    for seed, records in episodes.items():
        arms = rows[seed] = {}
        for arm in ARMS:
            stationary = records["stationary"][arm]["windows"]["full"]
            switching = records["switching"][arm]
            changed = switching["windows"]["during"]
            frozen_changed = records["switching"]["frozen"]["windows"]["during"]
            frozen_stationary = records["stationary"]["frozen"]["windows"]["full"]
            arms[arm] = dict(stationary_tracking_rmse=stationary["tracking_rmse"],
                             changed_tracking_rmse=changed["tracking_rmse"],
                             stationary_penalty_pct=100*(stationary["tracking_rmse"]/frozen_stationary["tracking_rmse"]-1),
                             changed_gain_pct=100*(1-changed["tracking_rmse"]/frozen_changed["tracking_rmse"]),
                             changed_return=changed["return_sum"],
                             changed_return_delta=changed["return_sum"]-frozen_changed["return_sum"],
                             settled_ua_mae=switching["windows"]["changed_plateau"]["ua_mae"],
                             recovery_delay={s: switching["recovery"][s]["delay"] for s in ("during","after")})
    return rows


def summarize(rows, tape):
    result = {}
    for arm in ARMS:
        records = [r[arm] for r in rows.values()]
        frozen = [r["frozen"] for r in rows.values()]
        def mean(key):
            return float(np.mean([r[key] for r in records]))
        result[arm] = dict(
            stationary_tracking_rmse=mean("stationary_tracking_rmse"), changed_tracking_rmse=mean("changed_tracking_rmse"),
            stationary_penalty_pct=100*(mean("stationary_tracking_rmse")/np.mean([f["stationary_tracking_rmse"] for f in frozen])-1),
            changed_gain_pct=float(np.median([r["changed_gain_pct"] for r in records])),
            changed_wins=sum(r["changed_tracking_rmse"] < f["changed_tracking_rmse"] for r,f in zip(records,frozen,strict=True)),
            mean_changed_return_delta=mean("changed_return_delta"), settled_ua_mae=mean("settled_ua_mae"),
            changed_tape_T_rmse_K=tape["methods"][arm]["windows"]["during"]["observation_prediction_rmse"]["T"],
            median_recovery_delay={}, unrecovered={})
        for stage in ("during","after"):
            delay = [r["recovery_delay"][stage] for r in records]
            result[arm]["unrecovered"][stage] = delay.count(None)
            # Median of the reached episodes can hide censoring. Leave it undefined.
            result[arm]["median_recovery_delay"][stage] = None if None in delay else float(np.median(delay))
    return result


def recovery_guard(rows):
    sides = {}
    for stage in ("during","after"):
        start, stop = WINDOWS[stage]
        diffs, missing_original, missing_candidate = {}, 0, 0
        for seed, arms in rows.items():
            c = arms["candidate"]["recovery_delay"][stage]
            o = arms["original"]["recovery_delay"][stage]
            missing_original += o is None
            missing_candidate += c is None
            # After censoring, a five-tick streak can first start in the last
            # four observations. This gives an upper bound on c minus o.
            diffs[seed] = None if c is None else c - (stop-start-4 if o is None else o)
        median = None if missing_candidate else float(np.median(list(diffs.values())))
        sides[stage] = dict(passed=median is not None and median <= 5, median_delta=median,
                            upper_bound=bool(missing_original), per_seed_delta=diffs,
                            candidate_unrecovered=missing_candidate, original_unrecovered=missing_original)
    return dict(passed=all(v["passed"] for v in sides.values()), max_median_delay_ticks=5, switches=sides)


def score_gates(summary, rows, totals):
    gates = {}
    for arm, name in (("candidate","candidate_control_benefit"), ("informed","informed_control_headroom")):
        r = summary[arm]
        gates[name] = dict(passed=r["changed_gain_pct"] >= 1 and 5*r["changed_wins"] >= 4*len(rows),
                           median_reduction_pct=r["changed_gain_pct"], lower_seeds=r["changed_wins"],
                           seeds=len(rows), required_median_pct=1, required_fraction=.8)
    candidate, original, frozen = (summary[a] for a in ("candidate","original","frozen"))
    # Compare the ratio of mean per-seed RMSE, as in the preregistered pilot.
    ratio = candidate["stationary_tracking_rmse"]/frozen["stationary_tracking_rmse"]
    gates["stationary_penalty"] = dict(passed=ratio <= 1.01, candidate_over_frozen=ratio, maximum_ratio=1.01)
    ua_reduction = 100*(1-candidate["settled_ua_mae"]/frozen["settled_ua_mae"])
    gates["settled_parameter_error"] = dict(passed=ua_reduction >= 50, reduction_pct=ua_reduction, required_pct=50)
    gates["changed_tape_temperature"] = dict(passed=candidate["changed_tape_T_rmse_K"] < frozen["changed_tape_T_rmse_K"],
                                             rmse_K={a:r["changed_tape_T_rmse_K"] for a,r in summary.items()})
    gates["reliability"] = dict(passed=all(v == 0 for v in totals.values()), **totals)
    gates["recovery"] = recovery_guard(rows)
    gain, previous = candidate["changed_gain_pct"], original["changed_gain_pct"]
    gates["retention"] = dict(passed=gain >= .9*previous, candidate_gain_pct=gain, original_gain_pct=previous,
                              retained_gain_pct=100*gain/previous if previous > 0 else None,
                              minimum_gain_pct=.9*previous)
    return gates


def bootstrap(rows):
    """Descriptive paired percentile intervals. Each draw resamples complete seed blocks."""
    n, draws, seed = len(rows), 10000, 40871
    samples = np.random.default_rng(seed).integers(n, size=(draws,n))
    def vector(arm, field):
        return np.array([a[arm][field] for a in rows.values()])[samples]
    def bounds(values):
        return np.quantile(values, [.025,.975]).tolist()
    result = {}
    baseline = vector("frozen","stationary_tracking_rmse").mean(axis=1)
    for arm in ("candidate", "original", "informed"):
        stationary = vector(arm,"stationary_tracking_rmse").mean(axis=1)
        changed = np.median(vector(arm,"changed_gain_pct"), axis=1)
        result[arm] = dict(stationary_penalty_pct=bounds(100*(stationary/baseline-1)), changed_gain_pct=bounds(changed))
    return dict(method="Paired whole-seed percentile interval, descriptive only", draws=draws, seed=seed,
                level=.95, arms=result)


def score(root, archive, label, q, *, dev):
    study = check_design(root, archive, label, q)
    folder, seeds, tape_seed = ((root/"dev"/label, DEVELOPMENT_SEEDS, 6400) if dev else
                               (root/"validation", FRESH_SEEDS, 6500))
    tape, targets = audit_tape(folder/"tape-switching.jsonl", tape_seed, study, q,
                               archive/"tape-switching.jsonl" if dev else None)
    episodes, clocks, audit = collect(folder, archive, study, q, seeds, dev, targets)
    if dev:
        archived = read(archive/"analysis.json")["episodes"]
        for s in seeds:
            for c in CONDITIONS:
                for a in ("original","frozen","informed"):
                    old = archived[str(s)][c]["learned" if a == "original" else a]
                    require(old["audit"]["trace_sha256"] == episodes[str(s)][c][a]["audit"]["trace_sha256"],
                            f"Retained comparator differs from archived audit: {s}/{c}/{a}")
    rows = paired_rows(episodes)
    summary = summarize(rows, tape)
    totals = {k: sum(record["audit"][k] for conditions in episodes.values()
                     for arms in conditions.values() for record in arms.values())
              for k in ("failures","holds","misses")}
    gates = score_gates(summary, rows, totals)
    failed = sum(not v["passed"] for v in gates.values())
    other_pass = all(v["passed"] for k,v in gates.items() if k != "stationary_penalty")
    ratio = gates["stationary_penalty"]["candidate_over_frozen"]
    stationary_thresholds = {str(t): bool(ratio <= 1+t/100) for t in (1,2,3)}
    thresholds = {k: v and other_pass for k,v in stationary_thresholds.items()}
    require(thresholds["1"] == (failed == 0), "Primary gate disagrees with sensitivity evaluation")
    audit["max_noise_replay_error"] = max(record["audit"]["max_noise_replay_error"]
        for conditions in episodes.values() for arms in conditions.values() for record in arms.values())
    return dict(label=label, q=q, split="development" if dev else "validation", seeds=list(seeds),
                eligible=not failed, proceed=not failed, failed_gate_count=failed, gates=gates,
                stationary_penalty_pct=summary["candidate"]["stationary_penalty_pct"],
                changed_gain_pct=summary["candidate"]["changed_gain_pct"],
                retained_gain_pct=gates["retention"]["retained_gain_pct"],
                recovery_delay_delta={k:v["median_delta"] for k,v in gates["recovery"]["switches"].items()},
                thresholds=thresholds, stationary_thresholds=stationary_thresholds,
                per_seed=rows, summary=summary, episodes=episodes, tape=tape, audit=audit,
                pooled_latency_s=clocks)


def development(root, archive, label, q):
    """Read one candidate's development data and retained comparators. No output mutation."""
    return score(Path(root), Path(archive), label, q, dev=True)


def verify_freeze(root, archive, label, q):
    frozen = read(root/"freeze.json")
    plan = read(root/"protocol.json")
    require(frozen["protocol_sha256"] == digest(root/"protocol.json") and frozen["study"] == plan
            and frozen["development_sha256"] == digest(root/"development.json"), "Mismatched selection hashes")
    for name in ("one_percent_analysis.py", "continual_analysis.py"):
        require(frozen["analysis_sha256"][name] == digest(HERE/name), f"Frozen scoring source changed: {name}")
    development_record = read(root/"development.json")
    candidates = development_record["candidates"]
    require(len(candidates) == len(CANDIDATES) and {r["label"]:r["q"] for r in candidates} == CANDIDATES,
            "Development set differs from the preregistered alternatives")
    for r in candidates:
        count = sum(not gate["passed"] for gate in r["gates"].values())
        require(r["failed_gate_count"] == count and r["proceed"] == (count == 0), "Development gate count mismatch")
    ordered = sorted(candidates, key=lambda r:(r["failed_gate_count"],max(0,r["stationary_penalty_pct"]-1),-r["q"]))
    require(development_record["ranking"] == [r["label"] for r in ordered] and
            frozen["selected"] == {"label": label, "motion_sd": q} and (label,q) == (ordered[0]["label"],ordered[0]["q"])
            and frozen["development_passed"] == ordered[0]["proceed"], "Frozen selection disagrees with ranking")
    opened = read(root/"validation"/"opened.json")
    completed = read(root/"validation-completed.json")
    require(opened["freeze_sha256"] == digest(root/"freeze.json") and completed["episode_count"] == 80
            and datetime.fromisoformat(frozen["frozen"]) <= datetime.fromisoformat(opened["opened"])
            <= datetime.fromisoformat(completed["finished"]), "Validation timing or freeze identity mismatch")
    require(Path(plan["archive"]).resolve() == archive.resolve(), "Wrong comparator archive")
    return dict(freeze_sha256=opened["freeze_sha256"], protocol_sha256=frozen["protocol_sha256"],
                development_sha256=frozen["development_sha256"], selected=frozen["selected"],
                frozen=frozen["frozen"], opened=opened["opened"], finished=completed["finished"])


def validation(root, archive, label, q):
    """Audit selection and complete fresh evidence, then write validation.json."""
    root, archive = Path(root), Path(archive)
    provenance = verify_freeze(root, archive, label, q)
    result = score(root, archive, label, q, dev=False)
    result.update(freeze=provenance, paired_intervals=bootstrap(result["per_seed"]))
    save(root/"validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output.expanduser().resolve()
    selected = read(root/"freeze.json")["selected"]
    archive = Path(read(root/"protocol.json")["archive"])
    record = validation(root, archive, selected["label"], selected["motion_sd"])
    print(f"{record['label']} validation: {'GO' if record['proceed'] else 'HOLD'}, "
          f"{record['failed_gate_count']} failed primary gates")


if __name__ == "__main__":
    main()
