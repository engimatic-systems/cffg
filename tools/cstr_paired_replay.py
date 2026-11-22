"""Diagnose saved PCGym seed pairs after the timed study has finished.

    uv run --frozen python tools/cstr_paired_replay.py STUDY --output NEW_DIRECTORY

STUDY contains run/seed-NN/{directed,host}, or is the run directory itself.
summary.json links to the complete common-window and stock measurements. Deltas
are directed minus host. Offline solves measure proposals; deadline and applied
action comparisons use the original timed decisions.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import traceback

import cstr_reactor_pilot as pilot


ARMS = ("directed", "host")
WINDOW = ("tick", "observation", "history", "preview_indices", "preview_Ca", "data", "masks")
SELF_TOL = 1e-8
HOST_MODULES = ("agent", "graphs", "physics_exp", "setpoint_exp")


def read(path):
    return json.loads(path.read_bytes())


def decision_path(directory, tick):
    return directory / f"decision-{tick:02}.json"


def action(record, *, timed):
    keys = ("proposed_kelvin", "proposed_normalized", "clipped_normalized", "status", "error")
    result = {key: record[key] for key in keys}
    result["checked"] = record["error"] is None and record["checks"] is not None
    for key in ("deadline_miss", "fallback", "applied_normalized", "applied_kelvin"):
        result[key] = record[key] if timed else None
    return result


def difference(left, right, key):
    values = (left[key], right[key])
    if all(isinstance(value, (float, int)) and math.isfinite(value) for value in values):
        return values[0] - values[1]
    return None


@dataclass
class Episode:
    seed: int
    arm: str
    directory: Path
    manifest: dict
    summary: dict
    actions: list
    first: dict
    stock_ticks: set

    def read(self, tick):
        return read(decision_path(self.directory, tick))


def inventory(directory, seed, arm):
    """Validate the observed prefix, including the *applied* action in every past row."""
    saved = read(directory / "episode.json") if (directory / "episode.json").exists() else {}
    manifest = read(directory / "manifest.json") if (directory / "manifest.json").exists() else {}
    if manifest and (manifest["seed"], manifest["arm"]) != (seed, arm):
        raise ValueError(f"Manifest identity mismatch at {directory}")
    actions, history, first = [], [], dict(saved.get("first", {}))
    if not saved:
        for path in directory.glob("first-*.json"):
            event = read(path)
            first[path.stem.removeprefix("first-")] = event.get("decision", event)["tick"]
    stock_ticks = set(pilot.REPLAY_TICKS) | set(first.values())
    paths = sorted(directory.glob("decision-*.json"))
    if saved and len(paths) != saved["decisions"]:
        raise ValueError(f"Episode decision count mismatch at {directory}")
    if paths and not manifest:
        raise ValueError(f"Missing manifest for decisions in {directory}")
    next_obs = None
    for tick, path in enumerate(paths):
        record = read(path)
        if path != decision_path(directory, tick) or (record["seed"], record["tick"]) != (seed, tick):
            raise ValueError(f"Noncontiguous or misidentified trajectory at {path}")
        if record["history"] != history or (tick and record["observation"] != next_obs):
            raise ValueError(f"Decision does not match the observed trajectory at {path}")
        act = action(record, timed=True)
        expected = (history[-1][1] if history else 0.) if act["fallback"] else act["clipped_normalized"]
        if act["applied_normalized"] != expected:
            raise ValueError(f"Recorded action inconsistent with the hold/clip policy at {path}")
        for label, flag in (("miss", act["deadline_miss"]), ("verification-failure", not act["checked"]),
                            ("failure", act["fallback"])):
            if flag:
                first.setdefault(label, tick)
                stock_ticks.add(tick)
        actions.append(act)
        history = (history + [[record["observation"], act["applied_normalized"]]])[-pilot.fixture.PAST:]
        step = directory / f"step-{tick:02}.json"
        next_obs = read(step)["next_observation"] if step.exists() else None
    return Episode(seed, arm, directory, manifest, saved, actions, first, stock_ticks)


def select(episodes):
    """Rank actual paired decisions, with independent denominators for raw and applied."""
    selection = dict(largest_raw=None, largest_applied=None, comparable_raw=0, comparable_applied=0,
                     first_miss={}, first_failed_check={}, stock_failures=[])
    wanted = defaultdict(list)
    for seed, pair in sorted(episodes.items()):
        for arm, episode in pair.items():
            for event, label in (("first_miss", "miss"), ("first_failed_check", "verification-failure")):
                if label in episode.first and arm not in selection[event]:
                    tick = episode.first[label]
                    selection[event][arm] = dict(seed=seed, tick=tick)
                    wanted[seed, tick].append(f"{arm}_{event}")
        for tick, (left, right) in enumerate(zip(pair["directed"].actions, pair["host"].actions)):
            for kind, key in (("raw", "proposed_kelvin"), ("applied", "applied_kelvin")):
                delta = difference(left, right, key)
                if delta is None:
                    continue
                selection[f"comparable_{kind}"] += 1
                label = f"largest_{kind}"
                best = selection[label]
                if best is None or abs(delta) > abs(best["delta_K"]):
                    selection[label] = dict(seed=seed, tick=tick, delta_K=delta, directed=left, host=right)
    for event in ("first_miss", "first_failed_check"):
        for arm in ARMS:
            selection[event].setdefault(arm, None)
    for label in ("largest_raw", "largest_applied"):
        row = selection[label]
        if row is not None:
            wanted[row["seed"], row["tick"]].append(label)
    return selection, wanted


def stock_summary(report):
    """Keep failed components visible even when a stock run or audit stopped early."""
    same = report.get("same_state", {})
    posterior = [(value, edge, moment) for edge, errors in report.get("posterior_errors", {}).items()
                 for moment, value in errors.items()]
    worst = max(posterior, default=None, key=lambda row: float(row[0]))
    reasons = []
    if report.get("missing_decision"):
        reasons.append("missing_decision")
    else:
        if not same.get("certified"):
            reasons.append("same_state_failed" if same else "same_state_unavailable")
            for key in ("root", "host_root", "assembly_error", "precision_asymmetry", "FE_roundtrip_drift",
                        "raw_FE_change", "canonical_FE_change", "host_FE_change"):
                if key in same and not float(same[key]) <= same["tol"]:
                    reasons.append(key)
            reasons.extend(key for key in ("finite", "feasible", "proper_reference") if same.get(key) is False)
            reasons.extend(key for key in ("status", "host_status") if same.get(key, 0) != 0)
            reasons.extend(key for key, fact in same.get("facts", {}).items() if fact["reason"] != 0)
            if same.get("proper_marginals") != same.get("marginal_count"):
                reasons.append("improper_marginals")
        if report.get("stock_status") != "converged" or report.get("stock_failure"):
            reasons.append("stock_solve_failed_or_unavailable")
        if "directed_residual" not in report:
            reasons.append("local_or_stock_comparison_unavailable")
        if "FE_error" in report and not float(report["FE_error"]) <= pilot.fixture.TOLERANCE:
            reasons.append("FE_parity_failed")
        if any(not float(row[0]) <= pilot.PARITY for row in posterior):
            reasons.append("posterior_parity_failed")
    return dict(historical_parity=report["historical_parity"], same_state_certified=same.get("certified"),
                stock_status=report.get("stock_status"), stock_reason=report.get("stock_reason"),
                stock_kelvin=report.get("stock_kelvin"), stock_FE=report.get("stock_FE"),
                directed_residual=report.get("directed_residual"), coolant_error_K=report.get("coolant_error_K"),
                FE_error=report.get("FE_error"), worst_whitened_posterior=worst,
                root=same.get("root"), FE_successor_change=same.get("canonical_FE_change"),
                stock_return_local_check_passed=True if "stock_return_local_check" in report else
                False if "stock_return_local_check_error" in report else None,
                reasons=reasons)


def stock_trajectory(episode, arrays, runtime, selected, out):
    folder = out / "stock" / f"seed-{episode.seed:02}"
    folder.mkdir(parents=True)
    ticks = episode.stock_ticks | {t for s, t in selected if s == episode.seed}
    comparison = pilot.stock_replays(folder, arrays, ticks, len(episode.actions),
                                     decisions_dir=episode.directory, stock_runtime=runtime)
    rows = {}
    for tick in sorted(ticks):
        path = folder / f"stock-{tick:02}.json"
        report = read(path)
        rows[tick] = dict(tick=tick, decision_sha256=report.get("decision_sha256"),
                          raw=str(path.relative_to(out)), **stock_summary(report))
    return dict(seed=episode.seed, ticks=list(rows.values()),
                negative_controls_raw=str((folder / "negative-controls.json").relative_to(out)),
                negative_controls_passed=comparison["negative_controls_passed"],
                all_historical_parity=comparison["all_historical_parity"]), rows


def trajectory_difference(pair, tick):
    """Compare the *two timed windows* at the selected tick in normalized units."""
    if any(tick >= len(ep.actions) for ep in pair.values()):
        return None
    left, right = (pair[arm].read(tick) for arm in ARMS)
    np = pilot.np
    past = [(np.asarray(a[:2]) - b[:2], u - v)
            for (a, u), (b, v) in zip(left["history"], right["history"], strict=True)]
    obs_delta = np.subtract(left["observation"][:2], right["observation"][:2])
    return dict(same_bound_data=left["data"] == right["data"],
                same_preview=left["preview_Ca"] == right["preview_Ca"],
                observation_max_abs=float(np.max(np.abs(obs_delta))),
                history_observation_max_abs=max((float(np.max(np.abs(x))) for x, _ in past), default=None),
                history_action_max_abs=max((abs(u) for _, u in past), default=None))


def equal_tree(actual, saved):
    """Compare deterministic status, structure and numeric readouts, including posteriors."""
    np = pilot.np
    if isinstance(saved, dict):
        return isinstance(actual, dict) and actual.keys() == saved.keys() and all(
            equal_tree(actual[k], v) for k, v in saved.items())
    if isinstance(saved, (list, tuple)):
        if not isinstance(actual, (list, tuple, np.ndarray)) or len(actual) != len(saved):
            return False
        return all(equal_tree(x, y) for x, y in zip(actual, saved))
    if isinstance(saved, (int, float)) and not isinstance(saved, bool):
        return isinstance(actual, (float, int, np.number)) and math.isclose(
            actual, saved, abs_tol=SELF_TOL, rel_tol=SELF_TOL)
    return actual == saved


def stock_window(runtime, arrays, record, sha, path):
    report = dict(tick=record["tick"], decision_sha256=sha, historical_parity=False)
    try:
        pilot.replay(*runtime, arrays, record, report)
    except Exception:
        report["replay_error"] = traceback.format_exc()
    pilot.save(path, report)
    return stock_summary(pilot.serial(report))


def common_window(host, episode, record, prior, params, arrays):
    """Both programs consume the archived window and the same causal setpoint schedule."""
    np = pilot.np
    observation = np.asarray(record["observation"])
    history = [(np.asarray(obs), u) for obs, u in record["history"]]
    directed, _ = pilot.decide(params, arrays, history, observation, record["tick"])
    comparator = host.replay(params, record, prior)
    for name, proposal in zip(ARMS, (directed, comparator)):
        for key in WINDOW:
            if pilot.serial(proposal[key]) != record[key]:
                raise ValueError(f"{name} {key} differs from {episode.directory}, tick {record['tick']}")
    return directed, comparator


def self_check(source, record, replayed):
    fields = ["candidate", "checks", "status", "error", "proposed_normalized", "proposed_kelvin",
              "clipped_normalized"]
    if source == "host":
        fields.extend(["host_status", "prior_plan_normalized", "shifted_plan_normalized"])
    mismatches = [k for k in fields if not equal_tree(pilot.serial(replayed[k]), record[k])]
    last = record["history"][-1][1] if record["history"] else 0.
    applied = last if record["fallback"] else replayed["clipped_normalized"]
    if not equal_tree(applied, record["applied_normalized"]):
        mismatches.append("applied_with_recorded_fallback")
    return dict(passed=not mismatches, mismatches=mismatches, tolerance=SELF_TOL,
                applied_with_recorded_fallback=applied)


def offline(record):
    """Discard only the counterfactual wall-clock gate and process-local timestamps."""
    return {key: value for key, value in record.items()
            if key not in {"deadline_miss", "fallback", "applied_normalized", "applied_kelvin",
                           "checked_at", "applied_at"}}


def replay_episode(host, episode, pair, wanted, stock_rows, runtime, params, arrays, out):
    seeds_ticks = {t: reasons for (s, t), reasons in wanted.items()
                   if s == episode.seed and t < len(episode.actions)}
    prior = None
    prefix = []
    plan_mismatches = []
    windows = []
    for tick in range(max(seeds_ticks, default=-1) + 1):
        record = episode.read(tick)
        sha = pilot.sha(decision_path(episode.directory, tick))
        act = episode.actions[tick]
        if episode.arm == "host":
            # The host archive supplies the real incoming plan. Check the causal
            # transition from the previous archived proposal, even on held ticks.
            if not equal_tree(pilot.serial(prior), record["prior_plan_normalized"]):
                plan_mismatches.append(tick)
            prior = record["prior_plan_normalized"]
        if tick in seeds_ticks:
            folder = out / "windows" / f"seed-{episode.seed:02}" / f"tick-{tick:02}" / episode.arm
            folder.mkdir(parents=True)
            directed, comparator = common_window(host, episode, record, prior, params, arrays)
            own = directed if episode.arm == "directed" else comparator
            sanity = self_check(episode.arm, record, own)
            if plan_mismatches:
                sanity.update(passed=False, inconsistent_prior_at_ticks=list(plan_mismatches))
            raw = folder / "common.json"
            pilot.save(raw, dict(seed=episode.seed, tick=tick, source=episode.arm, decision_sha256=sha,
                                 actual=record, prior_plan_normalized=prior, prefix=prefix,
                                 self_replay=sanity, directed=offline(directed), host=offline(comparator)))
            if episode.arm == "directed":
                stock = stock_rows[tick]
                stock_raw = stock["raw"]
            else:
                path = folder / "stock.json"
                # This certificate is for the directed candidate on the host window.
                # Bind its digest to that candidate, not to the host's timed decision.
                stock = stock_window(runtime, arrays, directed, pilot.sha(raw), path)
                stock_raw = str(path.relative_to(out))
            left, right = action(directed, timed=False), action(comparator, timed=False)
            clipped_delta = difference(left, right, "clipped_normalized")
            actual = {arm: ep.actions[tick] if tick < len(ep.actions) else None for arm, ep in pair.items()}
            windows.append(dict(seed=episode.seed, tick=tick, source=episode.arm,
                                reasons=seeds_ticks[tick], decision_sha256=sha,
                                raw=str(raw.relative_to(out)), stock_raw=stock_raw, self_replay=sanity,
                                actual=actual, actual_window_difference=trajectory_difference(pair, tick),
                                directed=left, host=right,
                                directed_local_FE=directed["candidate"].get("evidence_cost"),
                                directed_local_residual=(directed["checks"] or {}).get("local_message_residual"),
                                host_solver_status=comparator["host_status"],
                                raw_delta_K=difference(left, right, "proposed_kelvin"),
                                clipped_delta_K=3.5 * clipped_delta if clipped_delta is not None else None,
                                historical_parity=stock["historical_parity"],
                                same_state_certified=stock["same_state_certified"],
                                stock_findings=stock["reasons"], stock_FE=stock["stock_FE"],
                                stock_coolant_error_K=stock["coolant_error_K"]))
        elif episode.arm == "directed":
            comparator = host.replay(params, record, prior)

        # For a directed input the host was never run live. Advance its previous
        # proposal using what the plant actually received, including holds.
        previous = record if episode.arm == "host" else comparator
        prior = host.next_prior(previous, act["applied_normalized"], fallback=act["fallback"])
        prefix.append(dict(tick=tick, decision_sha256=sha, applied_normalized=act["applied_normalized"],
                           fallback=act["fallback"], host_status=previous["host_status"],
                           next_prior_plan_normalized=prior))
    return windows


def compare_histories(windows):
    """Measure each controller's change between inputs, including the host's incoming plan."""
    grouped = defaultdict(dict)
    for window in windows:
        grouped[window["seed"], window["tick"]][window["source"]] = window
    rows = []
    for (seed, tick), sources in sorted(grouped.items()):
        if not all(arm in sources for arm in ARMS):
            continue
        d, h = sources["directed"], sources["host"]
        left, right = (d["actual"][arm] for arm in ARMS)
        rows.append(dict(seed=seed, tick=tick, actual_raw_delta_K=difference(left, right, "proposed_kelvin"),
                         actual_applied_delta_K=difference(left, right, "applied_kelvin"),
                         controller_delta_on_directed_input_K=d["raw_delta_K"],
                         controller_delta_on_host_input_K=h["raw_delta_K"],
                         directed_change_between_inputs_K=difference(d["directed"], h["directed"], "proposed_kelvin"),
                         host_change_between_inputs_K=difference(d["host"], h["host"], "proposed_kelvin"),
                         both_self_replays_passed=d["self_replay"]["passed"] and h["self_replay"]["passed"]))
    return rows


def source_hashes(runtime):
    """Qualify relative filenames by repository so the two uv.lock files stay distinct."""
    roots = {"checkout": Path(runtime["reference_path"]).parents[1], "experiment": Path(runtime["experiment"])}
    hashes = {}
    for filename, digest in runtime["file_sha256"].items():
        path = Path(filename)
        for role, root in roots.items():
            if path.is_relative_to(root):
                hashes[f"{role}/{path.relative_to(root)}"] = digest
                break
    return hashes


def validate_runtime(info, study, episodes):
    sources = source_hashes(info)
    study_sources = {f"checkout/{p}": v for p, v in study.get("runtime_sources", {}).items()}
    for name, expected in study_sources.items():
        if sources.get(name) != expected:
            raise ValueError(f"Replay source disagrees with study.json at {name}")
    matches = []
    for pair in episodes.values():
        for ep in pair.values():
            if not ep.manifest:
                continue
            recorded = ep.manifest["runtime"]
            if study and recorded["head"] != study["revision"]:
                raise ValueError(f"Arm HEAD disagrees with study revision at {ep.directory}")
            for key in ("library_tree", "python_version"):
                if recorded[key] != info[key]:
                    raise ValueError(f"{key} differs at {ep.directory}")
            if recorded["pcgym_source"]["vcs_info"]["commit_id"] != info["pcgym_source"]["vcs_info"]["commit_id"]:
                raise ValueError(f"PCGym revision differs at {ep.directory}")
            versions = {name: module["version"] for name, module in recorded["modules"].items()}
            if versions != {name: module["version"] for name, module in info["modules"].items()}:
                raise ValueError(f"Package versions differ at {ep.directory}")
            measured = source_hashes(recorded)
            required = set(study_sources)
            if study:
                required.update(key for key in sources if key.startswith("experiment/"))
            if ep.arm == "directed":
                required.discard("checkout/tools/cstr_host_comparator.py")
                required.difference_update(f"experiment/common/{m}.py" for m in HOST_MODULES)
            for name in sorted(required | measured.keys()):
                if name not in measured or measured[name] != sources.get(name):
                    raise ValueError(f"Source hash differs or is missing at {ep.directory}, {name}")
            matches.append(dict(seed=ep.seed, arm=ep.arm, measured_head=recorded["head"],
                                verified_source_count=len(measured)))
    info.update(study_revision=study.get("revision"), role_relative_sha256=sources, verified_manifests=matches)


def run(study, out, experiment):
    run_dir = study / "run" if (study / "run").is_dir() else study
    if (run_dir / "study.json").exists() and not (run_dir / "processes.json").is_file():
        raise ValueError("Post-study replay requires the launcher's final processes.json")
    seeds = sorted(int(p.name[5:]) for p in run_dir.glob("seed-*") if p.is_dir() and p.name[5:].isdigit())
    if not seeds:
        raise ValueError(f"No seed directories in {run_dir}")
    metadata = read(run_dir / "study.json") if (run_dir / "study.json").exists() else {}
    if metadata:
        seeds = sorted(set(seeds) | {int(s) for s in metadata["order_by_seed"]})
    out.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    _, params, info = pilot.load_runtime(experiment)
    from cstr_host_comparator import create

    sources = [Path(__file__), pilot.ROOT / "tools/cstr_host_comparator.py", pilot.ROOT / "tools/cstr_paired_study.py"]
    sources.extend(experiment / "common" / f"{module}.py" for module in HOST_MODULES)
    info["file_sha256"].update({str(path): pilot.sha(path) for path in sources})
    engine, _, arrays = pilot.fixture.make_engine()
    runtime = engine, pilot.fixture.Binder(engine), engine.init_state()
    episodes = {seed: {arm: inventory(run_dir / f"seed-{seed:02}" / arm, seed, arm) for arm in ARMS}
                for seed in seeds}
    validate_runtime(info, metadata, episodes)
    pilot.save(out / "runtime.json", info)
    for pair in episodes.values():
        for ep in pair.values():
            if not ep.manifest:
                continue
            expected = dict(parameters=params, configuration=pilot.fixture.CONFIGURATION)
            if ep.arm == "directed":
                expected["model"] = arrays
            for key, value in expected.items():
                if ep.manifest[key] != pilot.serial(value):
                    raise ValueError(f"{key} disagrees with the replay runtime at {ep.directory}")
    selection, wanted = select(episodes)
    stock, stock_rows = [], {}
    for seed, pair in episodes.items():
        print(f"Stock replay, seed {seed}", flush=True)
        entry, rows = stock_trajectory(pair["directed"], arrays, runtime, wanted, out)
        stock.append(entry)
        stock_rows[seed] = rows
        for tick, row in rows.items():
            if not row["historical_parity"]:
                selection["stock_failures"].append(dict(seed=seed, tick=tick, reasons=row["reasons"]))
                if row["decision_sha256"]:
                    wanted[seed, tick].append("directed_stock_failure")
    windows = []
    for seed, pair in episodes.items():
        if any(s == seed for s, t in wanted):
            print(f"Common-input replay, seed {seed}", flush=True)
            host = create(experiment, seed)
            from experiment_physics import CONFIG

            host_manifest = pair["host"].manifest
            if host_manifest and host_manifest["model"]["configuration"] != pilot.serial(asdict(CONFIG)):
                raise ValueError(f"Host model configuration differs for seed {seed}")
            for episode in pair.values():
                windows.extend(replay_episode(host, episode, pair, wanted, stock_rows[seed],
                                              runtime, params, arrays, out))
    coverage = [dict(seed=ep.seed, arm=ep.arm, decisions=len(ep.actions), complete=ep.summary.get("complete"),
                     first=ep.first, directory=str(ep.directory),
                     manifest_sha256=pilot.sha(ep.directory / "manifest.json") if ep.manifest else None)
                for pair in episodes.values() for ep in pair.values()]
    passed = bool(windows) and all(row["self_replay"]["passed"] for row in windows)
    pilot.save(out / "summary.json", dict(
        study=str(run_dir), runtime="runtime.json", coverage=coverage, selection=selection, stock=stock,
        windows=windows, comparisons=compare_histories(windows),
        all_self_replays_passed=passed, offline_wall_s=perf_counter() - start,
        method=dict(
            selection="Absolute raw and applied kelvin differences on overlapping actual ticks, earliest "
                      "seed/tick wins ties. Missing raw proposals have no raw comparison.",
            inputs="Each selected seed/tick uses each available trajectory as its own common window. "
                   "Within-controller changes subtract the host-trajectory result from the directed-trajectory "
                   "result, including changes in observations, applied history and the host warm plan.",
            actions="Signed deltas are directed minus host. Offline actions are raw and clipped proposals. "
                    "Applied actions and deadlines are observed only in timed trajectories.",
            host="Host inputs use their archived incoming plan, checked against previous decisions. "
                 "Directed inputs reconstruct the host plan from tick zero using each actual applied action.",
            stock="The archived directed candidate and independent stock run use identical evidence. "
                  "On a host input, stock audits the replayed directed candidate and its decision_sha256 "
                  "identifies common.json, which holds that candidate. Host FE and posterior "
                  "readouts belong to the host's own linearized model. The separate local recheck of the "
                  "returned stock state is diagnostic and has no historical acceptance threshold.",
            timing="Offline sequential process, shared compiled engine and JIT across seeds, fresh stock "
                   "initial state for each solve. Offline wall time includes setup and artifact I/O.")))
    print(out / "summary.json", flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, default=pilot.EXPERIMENT)
    args = parser.parse_args()
    return run(args.study.expanduser().resolve(), args.output.expanduser().resolve(),
               args.experiment.expanduser().resolve())


if __name__ == "__main__":
    sys.exit(main())
