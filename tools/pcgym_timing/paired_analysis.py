"""Audit the paired PCGym archives and summarize uncertainty across whole seeds.

Run with `uv run python tools/pcgym_timing/paired_analysis.py ROOT`.
The outputs are ROOT/summary.json and ROOT/paired-seeds.json.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from pilot import ARMS, latency, rng_digest, save
from summarize import audit_episode, persistence_errors, physical_observations, read


SEEDS = tuple(range(1001, 1031))
LENGTHS = {"short": 59, "long": 599}
PEERS = tuple(a for a in ARMS if a != "cffg")
LEADS = (1, 5, 15, 60)
VARIABLES = ("Ca", "T")
ORIGINS = np.linspace(4, 599 - max(LEADS), 12, dtype=int).tolist()
PARTICLES = 512
TIE = 1e-8
DRAWS = 20_000
BOOTSTRAP_SEED = 20261122
CONTROL_COSTS = ("process_s", "previous_stage_to_episode_s", "adapter_setup_s",
                 "warmup_s", "episode_s", "first_decision_s")
FORECAST_COSTS = ("filter_total_s", "propagation_compile_s", "draws_total_s",
                  "propagation_total_s", "batch_wall_s")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all(), "Nonfinite values in distribution")
    return dict(mean=values.mean(), median=np.median(values), p95=np.quantile(values, .95),
                min=values.min(), max=values.max())


def interval(values):
    return np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None]


def relative(delta, peer):
    return 100 * (delta / abs(peer)) if peer else None


def totals(records, key):
    counts = Counter()
    for record in records:
        counts.update(record[key])
    return dict(counts)


def grid(calculate):
    return {str(h): {v: calculate(str(h), v) for v in VARIABLES} for h in LEADS}


def canonical_digest(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(content.encode()).hexdigest()


def worker_folder(folder, arm, phase, seed):
    directory = folder / f"{arm}-{phase}"
    manifest = directory / "manifest.json"
    if manifest.exists():
        info = read(manifest)
        require((info["arm"], info["phase"], info["seed"]) == (arm, phase, seed),
                f"Worker identity disagrees with its directory at {directory}")
    return directory


def control_episode(folder, seed, steps):
    result, noise = audit_episode(folder, steps)
    require(result["seed"] == seed and read(folder / "process.json")["code"] == 0,
            f"Wrong seed or failed control process in {folder}")
    rng = np.random.RandomState(seed)
    rng.normal(size=(steps, 2))  # Includes the final observation, beyond the last decision.
    require(result["rng_digest"] == rng_digest(rng), f"Unexpected final sensor RNG in {folder}")
    rows = [json.loads(line) for line in (folder / "ticks.jsonl").read_text().splitlines()]
    times = {k: np.array([r[k] for r in rows]) for k in ("decision_s", "solve_s")}
    for values in times.values():
        require(np.isfinite(values).all() and (values >= 0).all(), f"Invalid tick time in {folder}")
    errors = np.array([r["next_true_state"][0] - r["target"] for r in rows])
    reward = sum(r["reward"] for r in rows)
    require(np.isfinite(errors).all() and np.array_equal(errors, [r["error_Ca"] for r in rows])
            and np.isclose(result["rmse_Ca"], np.sqrt(np.mean(errors**2)), rtol=1e-13, atol=0)
            and np.isfinite(reward) and abs(reward - result["return_sum"]) < TIE / 100,
            f"Return or tracking error disagrees with tick data in {folder}")
    late = times["decision_s"] > 1.
    require(np.array_equal(late, [r["late"] for r in rows]) and result["missed"] == late.sum()
            and result["failures"] == steps - result["status_counts"].get("ok", 0),
            f"Failure or deadline count disagrees with tick data in {folder}")
    require(all(np.isfinite(result[k]) and result[k] >= 0 for k in CONTROL_COSTS),
            f"Invalid stage timing in {folder}")
    result.update(return_sum=reward, first_decision_s=times["decision_s"][0],
                  latency_s=latency(times["decision_s"]), solve_s=latency(times["solve_s"]))
    native = [r["native"] for r in rows]
    result["native_counts"] = dict(Counter(r.get("status", "no_status") for r in native))
    reasons = [r.get("termination") or r.get("stop_reason") for r in native]
    result["native_termination_counts"] = dict(Counter(reason for reason in reasons if reason))
    diagnostics = {}
    for key in ("residual", "slices_completed", "min_cholesky_diagonal", "iterations",
                "newton_decrement", "hessian_positive", "final_error"):
        values = np.array([r[key] for r in native if key in r], dtype=float)
        if values.size:
            finite = values[np.isfinite(values)]
            diagnostics[key] = dict(reported=len(values), finite=len(finite),
                                    min=finite.min() if finite.size else None,
                                    max=finite.max() if finite.size else None)
    result["native_diagnostics"] = diagnostics
    return result, noise, times


def control_seed(folder, seed, pooled_times):
    control, audit = {}, {}
    for phase, steps in LENGTHS.items():
        arms, noises = {}, []
        for arm in ARMS:
            entry, noise, times = control_episode(worker_folder(folder, arm, phase, seed), seed, steps)
            arms[arm] = entry
            noises.append(noise)
            pooled_times[phase][arm].append(times)
        difference = np.max(abs(np.asarray(noises) - noises[0]))
        require(difference <= 32 * np.finfo(float).eps, f"Unmatched noise in {folder}, {phase}")
        audit[phase] = dict(max_paired_noise_difference=difference, rng_digest=arms["cffg"]["rng_digest"],
                            max_noise_replay_error=max(r["max_noise_replay_error"] for r in arms.values()))
        paired = {}
        for peer in PEERS:
            value = arms[peer]["return_sum"]
            delta = arms["cffg"]["return_sum"] - value
            paired[peer] = dict(delta=delta, relative_percent=relative(delta, value))
        control[phase] = dict(arms=arms, paired=paired)
    return control, audit


def forecast_truth(folder, seed):
    truth = read(folder / "forecast-truth.json")
    checksum = canonical_digest({k: v for k, v in truth.items() if k != "sha256"})
    require(truth["sha256"] == checksum and truth["protocol"] == "pcgym-cstr-open-loop-v1"
            and truth["split"] == "held-out" and truth["truth_seed"] == 100000 + seed
            and truth["draw_seed"] == 200000 + seed and truth["origin_ticks"] == ORIGINS
            and truth["horizons"] == list(LEADS) and truth["steps"] == 599 and truth["resets"] == 1
            and truth["observation_noise_percentage"] == .001, f"Wrong forecast manifest in {folder}")
    physical = np.asarray(truth["truth_physical"])
    observed = physical_observations(truth["observations_normalized"])
    commands = np.asarray(truth["controls"])
    require(physical.shape == observed.shape == (600, 2) and commands.shape == (599,)
            and np.isfinite(commands).all() and (physical > 0).all(), f"Invalid truth arrays in {folder}")
    noise = observed / physical - 1
    rng = np.random.RandomState(truth["truth_seed"])
    expected = np.vstack(([0., 0.], .001 * rng.normal(size=(599, 2))))
    noise_error = np.max(abs(noise - expected))
    require(noise_error <= 16 * np.finfo(float).eps, f"Forecast noise replay mismatch in {folder}")
    draw_hashes = {}
    for tick in ORIGINS:
        rng = np.random.Generator(np.random.PCG64([truth["draw_seed"], tick]))
        draws = rng.standard_normal((max(LEADS) + 1, PARTICLES, 3))
        draw_hashes[tick] = sha256(draws.astype("<f8", copy=False)).hexdigest()
    audit = dict(truth_sha256=checksum, truth_seed=truth["truth_seed"], draw_seed=truth["draw_seed"],
                 fixture_sha256=truth["fixture_sha256"], commands_sha256=canonical_digest(truth["controls"]),
                 truth_trajectory_sha256=canonical_digest(truth["truth_physical"]),
                 max_truth_noise_replay_error=noise_error, n_draw_hashes_checked=0)
    return truth, draw_hashes, audit


def point_metrics(scores, ticks, h, v, intervals):
    rows = [scores[t][h][v] for t in ticks]
    n = len(rows)
    errors = np.array([abs(r["error"]) for r in rows])
    scale = np.max(errors, initial=0.) or 1.
    scaled = errors / scale
    cell = dict(n=n, rmse=scale * np.sqrt(np.mean(scaled**2)) if n else None,
                mae=scale * np.mean(scaled) if n else None)
    if intervals:
        cell.update(coverage90=np.mean([r["covered90"] for r in rows]) if n else None,
                    mean_width90=np.mean([r["width90"] for r in rows]) if n else None)
    return cell


def forecast_arm(folder, truth, draw_hashes):
    result = read(folder / "result.json")
    process = read(folder / "process.json")
    rows = result["origins"]
    require(result["particles"] == PARTICLES and result["dataset_sha256"] == truth["sha256"]
            and [r["tick"] for r in rows] == ORIGINS, f"Forecast inputs disagree in {folder}")
    counts = Counter(r["status"] for r in rows)
    passed = counts["ok"]
    n = len(rows)
    require((result["n_origins"], result["n_scored"], result["n_failed"]) == (n, passed, n - passed)
            and result["complete"] == (n == passed) and result["status"] == ("ok" if n == passed else "failed")
            and process["code"] == int(n != passed), f"Inconsistent forecast outcomes in {folder}")
    failures, scores = [], {}
    hashes_checked = 0
    for r in rows:
        tick = r["tick"]
        if "draws_sha256" in r:
            require(r["draws_sha256"] == draw_hashes[tick], f"Unmatched draws in {folder}, tick {tick}")
            hashes_checked += 1
        if r["status"] != "ok":
            failures.append(dict(tick=tick, status=r["status"], error=r["error"]))
            continue
        require("draws_sha256" in r and r["native_status"] == "ok" and r["filter_proper"]
                and r["propagation_finite"], f"Invalid successful forecast in {folder}, tick {tick}")
        scores[tick] = r["scores"]
        for h in LEADS:
            for j, v in enumerate(VARIABLES):
                cell = r["scores"][str(h)][v]
                actual = truth["truth_physical"][tick + h][j]
                lo, hi = cell["lower90"], cell["upper90"]
                require(np.isfinite([lo, hi, cell["mean"], actual, cell["error"], cell["width90"]]).all()
                        and actual == cell["truth"] and cell["error"] == cell["mean"] - actual
                        and hi >= lo and cell["width90"] == hi - lo and cell["covered90"] == (lo <= actual <= hi),
                        f"Score disagrees with forecast or truth in {folder}, tick {tick}, lead {h}, {v}")
    timings = {k: result["timings"][k] for k in FORECAST_COSTS}
    costs = (*timings.values(), process["wall_s"], result["adapter_setup_s"])
    require(all(np.isfinite(x) and x >= 0 for x in costs),
            f"Invalid forecast timing in {folder}")
    entry = dict(n_origins=n, n_scored=passed, n_failed=n - passed, status_counts=dict(counts),
                 native_counts=dict(Counter(r["native_status"] or "no_status" for r in rows)),
                 failed_origins=failures, process_s=process["wall_s"], adapter_setup_s=result["adapter_setup_s"],
                 timings=timings, leads=grid(lambda h, v: point_metrics(scores, scores, h, v, True)))
    return entry, scores, hashes_checked


def matched_cell(left, right, ticks, h, v):
    a = point_metrics(left, ticks, h, v, False)
    b = point_metrics(right, ticks, h, v, False)
    cell = dict(n_origins=len(ticks))
    for metric in ("rmse", "mae"):
        delta = a[metric] - b[metric] if ticks else None
        cell.update({f"cffg_{metric}": a[metric], f"peer_{metric}": b[metric],
                     f"{metric}_delta": delta, f"{metric}_relative_percent": relative(delta, b[metric])})
    return cell


def forecast_seed(folder, seed):
    truth, draw_hashes, audit = forecast_truth(folder, seed)
    arms, scores = {}, {}
    for arm in ARMS:
        directory = worker_folder(folder, arm, "forecast", seed)
        arms[arm], scores[arm], checked = forecast_arm(directory, truth, draw_hashes)
        audit["n_draw_hashes_checked"] += checked

    observed = physical_observations(truth["observations_normalized"])
    actual = np.asarray(truth["truth_physical"])
    baseline = {tick: {str(h): {v: {"error": observed[tick, j] - actual[tick + h, j]}
                                for j, v in enumerate(VARIABLES)} for h in LEADS} for tick in ORIGINS}
    scores["persistence"] = baseline
    baseline_metrics = grid(lambda h, v: point_metrics(baseline, ORIGINS, h, v, False))
    persistence = persistence_errors(truth)
    for h, metrics in baseline_metrics.items():
        for v, cell in metrics.items():
            require(np.isclose(cell["rmse"], persistence[h][v], rtol=1e-13, atol=0),
                    f"Persistence score mismatch at {folder}, lead {h}, {v}")
            cell["rmse"] = persistence[h][v]

    paired = {}
    left = scores["cffg"]
    for peer in (*PEERS, "persistence"):
        right = scores[peer]
        ticks = sorted(left.keys() & right.keys())
        paired[peer] = grid(lambda h, v: matched_cell(left, right, ticks, h, v))
    return dict(arms=arms, paired=paired, baseline=baseline_metrics), audit


def paired_returns(records, indices):
    delta = np.array([r["delta"] for r in records])
    percent = [r["relative_percent"] for r in records if r["relative_percent"] is not None]
    samples = delta[indices]
    return dict(n=len(delta), **distribution(delta), ci95=interval(samples.mean(axis=1)),
                median_ci95=interval(np.median(samples, axis=1)), positive=int(np.sum(delta > TIE)),
                negative=int(np.sum(delta < -TIE)), ties=int(np.sum(abs(delta) <= TIE)),
                relative_percent_mean=float(np.mean(percent)) if percent else None,
                relative_percent_n=len(percent), relative_percent=distribution(percent) if percent else None)


def summarize_control(seeds, pooled_times, indices):
    summary = {}
    for phase in LENGTHS:
        paired = {peer: paired_returns([s["control"][phase]["paired"][peer] for s in seeds], indices)
                  for peer in PEERS}
        arms = {}
        for arm in ARMS:
            records = [s["control"][phase]["arms"][arm] for s in seeds]
            metrics = ["return_sum", "rmse_Ca", *CONTROL_COSTS]
            if phase == "long":
                metrics.extend(("rmse_Ca_first100", "rmse_Ca_last100"))
            entry = {k: distribution([r[k] for r in records]) for k in metrics}
            entry.update({k: sum(r[k] for r in records) for k in ("missed", "failures", "holds")})
            for key in ("status_counts", "native_counts", "native_termination_counts"):
                entry[key] = totals(records, key)
            entry["native_diagnostics"] = {}
            diagnostics = [r["native_diagnostics"] for r in records]
            for key in sorted(set().union(*diagnostics)):
                ranges = [d[key] for d in diagnostics if key in d]
                entry["native_diagnostics"][key] = dict(
                    reported=sum(d["reported"] for d in ranges), finite=sum(d["finite"] for d in ranges),
                    min=min((d["min"] for d in ranges if d["finite"]), default=None),
                    max=max((d["max"] for d in ranges if d["finite"]), default=None))
            times = pooled_times[phase][arm]
            entry.update(decisions=sum(r["n"] for r in records),
                         process_total_s=sum(r["process_s"] for r in records),
                         pooled_rmse_Ca=np.sqrt(np.mean([r["rmse_Ca"]**2 for r in records])),
                         latency_s=latency(np.concatenate([r["decision_s"] for r in times])),
                         solve_s=latency(np.concatenate([r["solve_s"] for r in times])))
            arms[arm] = entry
        summary[phase] = dict(paired=paired, arms=arms)
    return summary


def pooled_metric(cells, key, counts, *, rms=False, weights=None, denominators=None):
    """Pool seed metrics by origin count, resampling all origins of each seed together."""
    values = np.array([c[key] if count else 0. for c, count in zip(cells, counts, strict=True)])
    scale = np.max(values, initial=0.) or 1.
    scaled = values / scale
    sums = counts * (scaled**2 if rms else scaled)
    mean = sums.sum() / counts.sum()
    estimate = scale * (np.sqrt(mean) if rms else mean)
    if weights is None:
        return estimate
    numerator = np.einsum("ij,j->i", weights, sums)
    valid = denominators > 0
    averages = numerator[valid] / denominators[valid]
    return estimate, scale * (np.sqrt(averages) if rms else averages)


def pooled_cell(cells):
    counts = np.array([c["n"] for c in cells])
    cell = {"n": int(counts.sum())}
    for key in cells[0]:
        if key != "n":
            cell[key] = pooled_metric(cells, key, counts, rms=key == "rmse") if cell["n"] else None
    return cell


def paired_forecasts(cells, weights):
    counts = np.array([c["n_origins"] for c in cells])
    denominators = weights @ counts
    cell = dict(n_origins=int(counts.sum()), n_seeds=int(np.count_nonzero(counts)),
                n_bootstrap_nonempty=int(np.count_nonzero(denominators)))
    for key in ("rmse", "mae"):
        delta, percent, ci, left, right = None, None, [None, None], None, None
        if cell["n_origins"]:
            kwargs = dict(rms=key == "rmse", weights=weights, denominators=denominators)
            left, left_boot = pooled_metric(cells, f"cffg_{key}", counts, **kwargs)
            right, right_boot = pooled_metric(cells, f"peer_{key}", counts, **kwargs)
            delta = left - right
            percent = relative(delta, right)
            ci = interval(left_boot - right_boot)
        cell.update({f"cffg_{key}": left, f"peer_{key}": right, f"{key}_delta": delta,
                     f"{key}_delta_ci95": ci, f"{key}_relative_percent": percent})
    return cell


def summarize_forecasts(seeds, weights):
    records = [s["forecast"] for s in seeds]
    arms = {}
    for arm in ARMS:
        rows = [r["arms"][arm] for r in records]
        entry = {key: sum(r[key] for r in rows) for key in ("n_origins", "n_scored", "n_failed")}
        entry.update(status_counts=totals(rows, "status_counts"), native_counts=totals(rows, "native_counts"),
                     leads=grid(lambda h, v: pooled_cell([r["leads"][h][v] for r in rows])),
                     process_total_s=sum(r["process_s"] for r in rows),
                     process_s=distribution([r["process_s"] for r in rows]),
                     adapter_setup_s=distribution([r["adapter_setup_s"] for r in rows]),
                     timings={k: distribution([r["timings"][k] for r in rows]) for k in FORECAST_COSTS})
        arms[arm] = entry
    paired = {peer: grid(lambda h, v: paired_forecasts([r["paired"][peer][h][v] for r in records], weights))
              for peer in (*PEERS, "persistence")}
    baseline = grid(lambda h, v: pooled_cell([r["baseline"][h][v] for r in records]))
    return dict(arms=arms, paired=paired, baseline=baseline)


def analyze(root):
    folders = {p.name for p in root.glob("seed-*") if p.is_dir()}
    expected = {f"seed-{s}" for s in SEEDS}
    require(folders == expected, f"Seed directories differ: missing {sorted(expected - folders)}, "
            f"unexpected {sorted(folders - expected)}")
    pooled_times = {p: {a: [] for a in ARMS} for p in LENGTHS}
    seeds = []
    for seed in SEEDS:
        folder = root / f"seed-{seed}"
        control, control_audit = control_seed(folder, seed, pooled_times)
        forecast, audit = forecast_seed(folder, seed)
        audit["control"] = control_audit
        seeds.append(dict(seed=seed, control=control, forecast=forecast, audit=audit))
    audits = [s["audit"] for s in seeds]
    for key in ("fixture_sha256", "commands_sha256"):
        require(len({a[key] for a in audits}) == 1, f"Forecast {key} differs across seeds")
    controls = [a["control"][p] for a in audits for p in LENGTHS]
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    n = len(seeds)
    indices = rng.integers(n, size=(DRAWS, n))
    # The count matrix represents precisely the same resampled seed IDs as indices.
    weights = np.bincount((indices + n * np.arange(DRAWS)[:, None]).ravel(),
                          minlength=n * DRAWS).reshape(DRAWS, n)
    audit = dict(seeds=list(SEEDS), max_noise_replay_error=max(a["max_noise_replay_error"] for a in controls),
                 max_paired_noise_difference=max(a["max_paired_noise_difference"] for a in controls),
                 max_truth_noise_replay_error=max(a["max_truth_noise_replay_error"] for a in audits),
                 distinct_truth_trajectories=len({a["truth_trajectory_sha256"] for a in audits}),
                 n_draw_hashes_checked=sum(a["n_draw_hashes_checked"] for a in audits))
    summary = dict(schema_version=1, n_seeds=n, paired_seed_file="paired-seeds.json", audit=audit,
                   primary="long", secondary="short", numerical_tie_abs_return=TIE,
                   bootstrap=dict(draws=DRAWS, seed=BOOTSTRAP_SEED, rng="NumPy PCG64", unit="paired seed",
                                  interval="pointwise percentile 95%, linear quantiles"),
                   relative_percent="100 * (CFFG - comparator) / abs(comparator), undefined at zero",
                   forecast_denominators="Arm metrics pool all scored origins. Paired estimates use origins "
                   "scored by both methods. Persistence alone uses all origins. Bootstrap replicates with "
                   "zero matched origins have undefined error and are counted separately.",
                   control=summarize_control(seeds, pooled_times, indices),
                   forecast=summarize_forecasts(seeds, weights))
    return summary, dict(schema_version=1, seeds=seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("~/agent-share/pcgym-paired-30"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    summary, seeds = analyze(root)
    save(root / summary["paired_seed_file"], seeds)
    save(root / "summary.json", summary)
    print(f"Wrote {root / 'summary.json'} and {root / summary['paired_seed_file']}")


if __name__ == "__main__":
    main()
