"""Score post-observation extraction beliefs, pre-observation predictions and applied control.

--stage development writes development/analysis.json. --stage stopped and heldout
write analysis.json. Only heldout opens evaluation traces and requires study seals.
"""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from plant import DT, FLOW_HIGH, FLOW_LOW, HIDDEN, NOMINAL_FLOW, NOMINAL_KLA, OUTLETS
from study import ARMS, CONDITIONS, DEADLINE, DEVELOPMENT, ENDURANCE, HELDOUT, LIBRARIES, MODES, PRACTICAL, save, sha as digest

BOOT_SEED = 89021
BOOT_DRAWS = 10000
Z90 = NormalDist().inv_cdf(.95)
CHI2_8_90 = 13.361566136511728


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def array_hash(value):
    return sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).digest()


def average(values):
    return float(np.mean(values)) if len(values) else None


def pctl(values, percent):
    return float(np.percentile(values, percent)) if len(values) else None


def strict_average(values):
    return average(values) if np.isfinite(values).all() else None


def density(residual, covariance):
    """Return per-tick Gaussian log score and squared Mahalanobis distance; invalid ticks stay NaN."""
    n, dim = residual.shape
    log_score, distance = np.full(n, np.nan), np.full(n, np.nan)
    finite = np.isfinite(residual).all(axis=1) & np.isfinite(covariance).all(axis=(1, 2))
    rows = np.flatnonzero(finite)
    matrix = covariance[rows]
    scale = np.max(np.abs(matrix), axis=(1, 2))
    skew = np.max(np.abs(matrix - matrix.swapaxes(1, 2)), axis=(1, 2))
    rows = rows[skew <= 1e-10 * scale]

    def calculate(indices):
        if not len(indices):
            return
        try:
            chol = np.linalg.cholesky(covariance[indices])
        except np.linalg.LinAlgError:
            if len(indices) > 1:
                mid = len(indices) // 2
                calculate(indices[:mid])
                calculate(indices[mid:])
            return
        whitened = np.linalg.solve(chol, residual[indices, :, None])[:, :, 0]
        sq = np.sum(whitened**2, axis=1)
        score = -.5 * (dim * np.log(2 * np.pi) + sq) - np.log(chol.diagonal(axis1=1, axis2=2)).sum(axis=1)
        valid = np.isfinite(score) & np.isfinite(sq)
        log_score[indices[valid]], distance[indices[valid]] = score[valid], sq[valid]

    calculate(rows)
    return log_score, distance


def load_trace(path, seed, expected_steps, condition, mode):
    with np.load(path, allow_pickle=False) as src:
        a = {key: src[key] for key in src.files}
    n = expected_steps
    shapes = {"x": (n+1, 10), "obs": (n, 2), "kla": (n,), "phase": (n,), "m": (n, 11), "P": (n, 11, 11),
              "pred": (n, 2), "pred_cov": (n, 2, 2), "u": (n, 2), "target": (n,), "failed": (n,),
              "late": (n,), "latency": (n,)}
    for name, shape in shapes.items():
        require(name in a and a[name].shape == shape, f"{path}: {name} must have shape {shape}")
    require(a["seed"].shape == () and a["seed"].item() == seed, f"{path}: seed mismatch")
    for name in ("x", "obs", "kla", "phase", "m", "u", "target", "latency"):
        require(np.isfinite(a[name]).all(), f"{path}: {name} must include a finite value on every tick")
    require((a["kla"] > 0).all() and (a["latency"] >= 0).all(), f"{path}: invalid Kla or latency")
    require(np.equal(a["phase"], np.floor(a["phase"])).all(), f"{path}: phase labels must be integers")
    require(condition != "stationary" or not np.any(a["phase"]), f"{path}: stationary phase is nonzero")
    require(condition != "changing" or np.any(a["phase"]), f"{path}: changing tape has no changed ticks")
    require(np.array_equal(a["phase"] != 0, a["kla"] != NOMINAL_KLA), f"{path}: Kla and phase disagree")
    require(a["event"].shape == (n,) and a["event"].dtype.kind == "U", f"{path}: invalid event labels")
    a["meta"] = json.loads(str(a["meta"].item())) if "meta" in a else {}
    flags = ["failed", "late", "filter_failed", "planner_failed"]
    if mode == "control":
        flags.append("held")
    else:
        a.setdefault("held", np.zeros(n, dtype=bool))
    for name in flags + ["held"]:
        require(name in a and a[name].shape == (n,) and a[name].dtype.kind == "b", f"{path}: invalid {name}")
    require(not (mode == "tape" and a["held"].any()), f"{path}: tape cannot hold a commanded action")
    reset = a.get("reset_failed", np.asarray(False))
    require(reset.shape == () and reset.dtype.kind == "b", f"{path}: reset_failed must be a scalar boolean")
    a["reset_failed"] = reset.item()
    failed = a["filter_failed"] | a["planner_failed"]
    failed[0] |= a["reset_failed"]
    require(np.array_equal(a["failed"], failed), f"{path}: inconsistent failure flags")
    if mode == "control":
        require(np.array_equal(a["late"], a["latency"] > DEADLINE), f"{path}: deadline flag mismatch")
        held = a["late"] | a["planner_failed"] | np.r_[a["reset_failed"], a["filter_failed"][:-1]]
        require(np.array_equal(held, a["held"]), f"{path}: held flag mismatch")
        previous = np.vstack([NOMINAL_FLOW, a["u"][:-1]])
        require(np.array_equal(a["u"][held], previous[held]), f"{path}: held command was not the previous applied flow")
    for name in ("t_filter", "t_plan"):
        if name in a:
            require(a[name].shape == (n,), f"{path}: invalid {name}")
    return a


def score_arrays(a):
    x, m = a["x"][1:], a["m"]
    error = m[:, HIDDEN] - x[:, HIDDEN]
    cov = a["P"][:, HIDDEN, :][:, :, HIDDEN]
    joint, dist = density(error, cov)
    var = cov.diagonal(axis1=1, axis2=2)
    valid = np.isfinite(var) & (var > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        width = 2 * Z90 * np.sqrt(np.where(valid, var, np.nan))
        diag_score = -.5 * (np.log(2 * np.pi * var) + error**2 / var).sum(axis=1)
        est = np.exp(m[:, 10])
    require(np.isfinite(est).all(), "Nonfinite exponentiated posterior log Kla")
    diag_score[~valid.all(axis=1) | ~np.isfinite(diag_score)] = np.nan
    outlet, _ = density(a["obs"] - a["pred"], a["pred_cov"])
    normalized = (a["u"] - FLOW_LOW) / (FLOW_HIGH - FLOW_LOW)
    moves = np.diff(a["u"], axis=0, prepend=NOMINAL_FLOW[None, :]) / (FLOW_HIGH - FLOW_LOW)
    return dict(error2=error**2, width=width, covered=valid & (np.abs(error) <= width / 2),
                joint=joint, joint_hit=dist <= CHI2_8_90, diag=diag_score, outlet=outlet, est=est,
                kla_error=est-a["kla"], log_kla_error=m[:, 10]-np.log(a["kla"]),
                tracking=x[:, 8]-a["target"], effort=np.sum(((a["u"] - NOMINAL_FLOW) / [15., 8.])**2, axis=1),
                movement=np.mean(moves**2, axis=1),
                saturation=np.isclose(normalized, 0, atol=1e-10, rtol=0) | np.isclose(normalized, 1, atol=1e-10, rtol=0),
                flow_violation=((normalized < -1e-10) | (normalized > 1+1e-10)).any(axis=1),
                state_outside=((x < 0) | (x > 1)).any(axis=1))


def metrics(a, scores, indices):
    v = {key: arr[indices] for key, arr in scores.items()}
    n = len(v["joint"])
    if not n:
        return None
    full, diag, outlet = v["joint"], v["diag"], v["outlet"]
    diff = full - diag
    t = a["latency"][indices]
    finite_width = np.isfinite(v["width"])
    stats = dict(n=n, hidden_rmse=float(np.sqrt(np.mean(v["error2"]))),
                 hidden_coverage90=float(v["covered"].mean()), hidden_width90=strict_average(v["width"]),
                 hidden_rmse_by_state=np.sqrt(v["error2"].mean(axis=0)).tolist(),
                 hidden_coverage90_by_state=v["covered"].mean(axis=0).tolist(),
                 hidden_width90_by_state=[strict_average(width) for width in v["width"].T],
                 hidden_invalid_marginals=int((~finite_width).sum()), joint_coverage90=float(v["joint_hit"].mean()),
                 kla_rmse=float(np.sqrt(np.mean(v["kla_error"]**2))),
                 kla_mae=float(np.abs(v["kla_error"]).mean()), log_kla_rmse=float(np.sqrt(np.mean(v["log_kla_error"]**2))),
                 tracking_rmse=float(np.sqrt(np.mean(v["tracking"]**2))), effort=average(v["effort"]),
                 flow_movement=average(v["movement"]),
                 saturation_fraction=v["saturation"].mean(axis=0).tolist(),
                 flow_violation_ticks=int(v["flow_violation"].sum()), state_outside_unit_box_ticks=int(v["state_outside"].sum()),
                 latency_p50_s=pctl(t, 50), latency_p95_s=pctl(t, 95))
    for key, values in (("joint_log_score", full), ("diag_log_score", diag),
                        ("full_minus_diag_log_score", diff), ("outlet_log_score", outlet)):
        finite = np.isfinite(values)
        stats[key] = strict_average(values)
        stats[key+"_valid_mean"] = average(values[finite])
        stats[key+"_undefined"] = int((~finite).sum())
    for name in ("failed", "late", "filter_failed", "planner_failed", "held"):
        stats["holds" if name == "held" else name] = int(a[name][indices].sum())
    return stats


def segments(phase):
    bounds = [0, *(np.flatnonzero(phase[1:] != phase[:-1]) + 1).tolist(), len(phase)]
    return [(start, stop, phase[start].item()) for start, stop in zip(bounds[:-1], bounds[1:])]


def recovery(a, scores):
    """Score each excursion and ensuing return, starting the clock at its first endpoint tick."""
    output = []
    for start, stop, changed in segments(a["phase"] != 0):
        if not changed and start == 0:
            continue
        kla = a["kla"][start:stop]
        # A regime can ramp into a plateau. Only ticks following arrival count.
        endpoint = kla[np.argmax(np.abs(kla - NOMINAL_KLA))] if changed else NOMINAL_KLA
        at_end = np.isclose(kla, endpoint, rtol=1e-12, atol=0)
        reached = np.flatnonzero(at_end)
        require(len(reached) and np.all(np.diff(reached) == 1), "Noncontiguous recovery endpoint")
        arrival, end = start + int(reached[0]), start + int(reached[-1]) + 1
        ok = (np.abs(scores["est"][arrival:end] - endpoint) <= .02 * endpoint) & ~a["filter_failed"][arrival:end]
        windows = np.lib.stride_tricks.sliding_window_view(ok, 5).all(axis=1) if len(ok) >= 5 else []
        hits = np.flatnonzero(windows)
        delay = int(hits[0]) if len(hits) else None
        output.append(dict(start=start, stop=stop, arrival=arrival, endpoint=float(endpoint), changed=bool(changed),
                           delay_ticks=delay, delay_hours=None if delay is None else delay * DT,
                           event=str(a["event"][start]), confirmation_tick=None if delay is None else arrival+delay+4,
                           censored=delay is None, followup_ticks=end-arrival))
    return output


def check_pairing(a, references, seed, condition, mode):
    # Closed-loop actions differ. The additive sensor noise paired by seed is recoverable.
    noise = a["obs"] - a["x"][1:, OUTLETS]
    key = (seed, "exogenous")
    shared = (array_hash(a["x"][0]), array_hash(a["target"]), array_hash(a["y0"]) if "y0" in a else None)
    if key in references:
        prior_shared, prior_noise = references[key]
        require(prior_shared == shared, f"Initial truth or targets differ for seed {seed}")
        require(np.allclose(noise, prior_noise, rtol=0, atol=1e-13), f"Sensor noise differs for seed {seed}")
    else:
        references[key] = (shared, noise)
    key = (seed, condition, "schedule")
    now = (array_hash(a["kla"]), array_hash(a["phase"]), sha256(a["event"].tobytes()).digest())
    require(references.setdefault(key, now) == now, f"Regime schedule differs for {seed}/{condition}")
    if mode == "tape":
        for key, values in (((seed, "probing"), (a["u"],)),
                            ((seed, condition, "tape"), (a["x"], a["obs"]))):
            now = tuple(array_hash(x) for x in values)
            require(references.setdefault(key, now) == now, f"Common probing tape differs for {seed}/{condition}")


def score_run(path, seed, condition, mode, arm, library, steps, refs):
    a = load_trace(path, seed, steps, condition, mode)
    check_pairing(a, refs, seed, condition, mode)
    s = score_arrays(a)
    changed = a["phase"] != 0
    phases = [dict(start=i, stop=j, phase=event, changed=bool(changed[i:j].any()), metrics=metrics(a, s, slice(i, j)))
              for i, j, event in segments(a["event"])]
    flags = {name: np.flatnonzero(a[name]).tolist() for name in ("failed", "late", "held", "filter_failed", "planner_failed")}
    flags["reset_failed"] = [0] if a["reset_failed"] else []
    timing = dict(first_s=float(a["latency"][0]))
    for name in ("t_filter", "t_plan"):
        if name in a:
            finite = a[name][np.isfinite(a[name])]
            timing[name] = dict(valid=len(finite), p50_s=pctl(finite, 50), p95_s=pctl(finite, 95))
    if "rss_samples" in a:
        rss = a["rss_samples"]
        require(rss.ndim == 2 and rss.shape[1] == 2 and np.isfinite(rss).all(), f"{path}: invalid RSS samples")
        timing["rss_samples"] = rss.tolist()
    row = dict(seed=seed, condition=condition, mode=mode, arm=arm, library=library, n=steps,
               overall=metrics(a, s, slice(None)), changed=metrics(a, s, changed),
               first600=metrics(a, s, slice(600)), after600=metrics(a, s, slice(600, None)),
               phases=phases, by_event={event: metrics(a, s, a["event"] == event) for event in np.unique(a["event"])},
               recovery=recovery(a, s), reset_failed=a["reset_failed"], events=flags, timing=timing, meta=a["meta"],
               diagnostics=json.loads(a["diagnostics"].item()) if "diagnostics" in a else [])
    return row, a["latency"], changed


def summarize_metrics(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    sums = {"n", "failed", "late", "holds", "filter_failed", "planner_failed", "hidden_invalid_marginals"}
    out = {}
    for name in values[0]:
        items = [v[name] for v in values]
        if name in sums or name.endswith(("_ticks", "_undefined")):
            out[name] = sum(items)
        elif name == "saturation_fraction" or name.endswith("_by_state"):
            out[name] = [None if None in column else average(column) for column in zip(*items, strict=True)]
        else:
            out[name] = None if None in items else average(items)
    return out


def groups(rows, timings):
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[tuple(r[k] for k in ("condition", "mode", "arm", "library"))].append(r)
    result = []
    for key, members in by_arm.items():
        condition, mode, arm, library = key
        group = dict(condition=condition, mode=mode, arm=arm, library=library, n_seeds=len(members),
                     reset_failures=sum(r["reset_failed"] for r in members))
        for scope in ("overall", "changed", "first600", "after600"):
            group[scope] = summarize_metrics([r[scope] for r in members])
            if group[scope] is None:
                continue
            samples = []
            for r in members:
                t, changed = timings[(r["seed"], *key)]
                index = {"changed": changed, "first600": slice(600), "after600": slice(600, None), "overall": slice(None)}
                samples.append(t[index[scope]])
            pooled = np.concatenate(samples)
            group[scope].update(latency_p50_s=pctl(pooled, 50), latency_p95_s=pctl(pooled, 95))
        labels = sorted(set().union(*(r["by_event"] for r in members)))
        group["by_event"] = {label: summarize_metrics([r["by_event"].get(label) for r in members]) for label in labels}
        recoveries = [e for r in members for e in r["recovery"]]
        group["recovery"] = {label: dict(episodes=len(events), censored=sum(e["censored"] for e in events),
                                         recovered_median_ticks=pctl([e["delay_ticks"] for e in events if not e["censored"]], 50))
                             for label in sorted({e["event"] for e in recoveries})
                             if (events := [e for e in recoveries if e["event"] == label])}
        result.append(group)
    return result


def interval(pairs):
    values = [p["delta"] for p in pairs]
    ci = None
    if len(values) >= 2:
        rng = np.random.default_rng(BOOT_SEED)
        draws = rng.choice(values, size=(BOOT_DRAWS, len(values)), replace=True)
        ci = np.percentile(draws.mean(axis=1), [2.5, 97.5]).tolist()
    return dict(n_seeds=len(values), per_seed=pairs, mean_difference=average(values), ci95=ci)


def comparisons(rows, seeds):
    by = {(r["condition"], r["mode"], r["arm"], r["library"], r["seed"]): r for r in rows}
    result = []
    for condition, mode in [(c, m) for c in CONDITIONS for m in MODES]:
        scope = "changed" if condition == "changing" else "overall"
        metrics_to_pair = (("hidden_rmse", "kla_rmse", "joint_log_score", "outlet_log_score", "full_minus_diag_log_score")
                           if mode == "tape" else ("tracking_rmse", "effort", "holds", "latency_p95_s"))
        for arm in ARMS:
            for metric in metrics_to_pair:
                pairs = []
                for seed in seeds:
                    vals = {}
                    for lib in LIBRARIES:
                        run = by.get((condition, mode, arm, lib, seed))
                        if run and run[scope] and run[scope][metric] is not None:
                            vals[lib] = run[scope][metric]
                    if len(vals) == 2:
                        pairs.append(dict(seed=seed, **vals, delta=vals["cffg"]-vals["rxinfer"]))
                result.append(dict(condition=condition, mode=mode, arm=arm, metric=metric, scope=scope, **interval(pairs)))
    return result


def within_library(rows):
    by = {(r["seed"], r["condition"], r["mode"], r["arm"], r["library"]): r for r in rows}
    contrasts = defaultdict(list)
    for r in rows:
        seed, cond, mode, arm, lib = (r[k] for k in ("seed", "condition", "mode", "arm", "library"))
        scope = "changed" if cond == "changing" else "overall"
        measured = r[scope]
        baseline = by.get((seed, cond, mode, "frozen", lib))
        if arm != "frozen" and baseline:
            metric = "hidden_rmse" if mode == "tape" else "tracking_rmse"
            current, fixed = measured[metric], baseline[scope][metric]
            contrasts[cond, mode, arm, lib, arm+"_minus_frozen", metric].append(
                dict(seed=seed, value=current, reference=fixed, delta=current-fixed))
        joint = measured["full_minus_diag_log_score"]
        if joint is not None:
            contrasts[cond, mode, arm, lib, "full_minus_diagonal", "joint_log_score"].append(
                dict(seed=seed, value=measured["joint_log_score"], reference=measured["diag_log_score"], delta=joint))
        if "step1" in r["by_event"] and "step2" in r["by_event"]:
            metric = "hidden_rmse" if mode == "tape" else "tracking_rmse"
            first = r["by_event"]["step1"][metric]
            second = r["by_event"]["step2"][metric]
            contrasts[cond, mode, arm, lib, "second_minus_first_abrupt_visit", metric].append(
                dict(seed=seed, value=second, reference=first, delta=second-first))
    return [dict(condition=c, mode=m, arm=a, library=lib, contrast=k, metric=metric, **interval(pairs))
            for (c, m, a, lib, k, metric), pairs in contrasts.items()]


def heldout_seal(root, seeds, conditions):
    frozen, completed = read(root/"freeze.json"), read(root/"completed.json")
    require(frozen["protocol_sha256"] == digest(root/"protocol.json") and frozen["pilot_sha256"] == digest(root/"pilot.json"),
            "Freeze does not bind protocol and pilot")
    require(completed["freeze_sha256"] == digest(root/"freeze.json"), "Completion does not bind freeze")
    require(read(root/"pilot.json")["status"] == "go", "Held-out scores require a green pilot")
    require(completed["runtime_sha256"] == frozen["runtime_sha256"] and
            completed["protocol_sha256"] == frozen["protocol_sha256"], "Completion runtime or protocol differs")
    require(frozen["source_sha256"]["tools/extraction/analysis.py"] == digest(Path(__file__)), "Frozen scorer changed")
    require(frozen["seeds"] == completed["seeds"] == seeds and frozen["steps"] == completed["steps"],
            "Evaluation schedule differs from freeze")
    require(tuple(conditions) == CONDITIONS, "Evaluation needs both frozen conditions")
    require(len(seeds) > 0 and set(seeds) <= set(HELDOUT), "Unexpected held-out seed")
    return completed["runs"], frozen["steps"], frozen["runtime_sha256"]


def run_paths(seeds, conditions):
    for seed in seeds:
        for condition in conditions:
            for mode in MODES:
                for arm in ARMS:
                    for lib in LIBRARIES:
                        yield seed, condition, mode, arm, lib


def inventory(root, split, specs, steps, seals=None, runtime_sha=None):
    rows, missing, hashes, timings, refs = [], [], {}, {}, {}
    files = {str(Path(split)/f"seed-{seed}"/cond/f"{mode}-{arm}-{lib}.npz"): (seed, cond, mode, arm, lib)
             for seed, cond, mode, arm, lib in specs}
    if seals is not None:
        require(set(files) == set(seals), "Sealed trace inventory differs from scheduled evaluation")
    protocol_sha = digest(root/"protocol.json") if (root/"protocol.json").is_file() else None
    for relative, spec in files.items():
        path = root / relative
        sidecar = path.with_suffix(".json")
        if not path.is_file() or not sidecar.is_file():
            missing.append(relative)
            continue
        checksum = digest(path)
        side = read(sidecar)
        require(side.pop("sha256") == checksum, f"Sidecar digest differs for {relative}")
        if seals is not None:
            require(seals[relative] == checksum, f"Sealed digest differs for {relative}")
        seed, cond, mode, arm, lib = spec
        require([side.get(k) for k in ("stage", "seed", "steps", "condition", "mode", "arm", "library")] ==
                [split, seed, steps, cond, mode, arm, lib], f"Sidecar identity differs for {relative}")
        row, latency, changed = score_run(path, seed, cond, mode, arm, lib, steps, refs)
        require(row["meta"] == side, f"Trace metadata and sidecar differ for {relative}")
        require(side["protocol_sha256"] == protocol_sha, f"Protocol digest differs for {relative}")
        if runtime_sha is not None:
            require(side["runtime_sha256"] == runtime_sha, f"Runtime digest differs for {relative}")
        hashes[relative] = checksum
        rows.append(dict(path=relative, **row))
        timings[spec] = (latency, changed)
    if seals is not None:
        require(not missing, f"Missing sealed traces: {missing}")
    return (dict(expected_runs=len(files), scored_runs=len(rows), expected_ticks=len(files)*steps,
                 scored_ticks=sum(r["n"] for r in rows), missing=missing, sha256=hashes),
            rows, groups(rows, timings))


def analyze(root, stage, *, seeds=None, conditions=CONDITIONS, decision_path=None):
    require(stage in ("development", "stopped", "heldout"), "Unknown analysis stage")
    root = Path(root)
    is_eval = stage == "heldout"
    if stage == "stopped":
        require(not (root/"freeze.json").exists() and not (root/"completed.json").exists(),
                "Stopped output cannot replace a frozen evaluation")
    if is_eval and decision_path is not None:
        require(digest(decision_path) == digest(root/"pilot.json"), "Decision differs from the frozen pilot")
    protocol = read(root/"protocol.json") if (root/"protocol.json").exists() else {}
    seeds = list(seeds if seeds is not None else (read(root/"freeze.json")["seeds"] if is_eval else DEVELOPMENT))
    require(len(seeds) and len(set(seeds)) == len(seeds) and all(type(seed) is int for seed in seeds), "Invalid seed list")
    require(len(conditions) and len(set(conditions)) == len(conditions) and set(conditions) <= set(CONDITIONS),
            "Unknown or repeated condition")
    if not is_eval:
        require(set(seeds) <= set(DEVELOPMENT), "Development may only use designated development seeds")
    seals, steps, runtime_sha = heldout_seal(root, seeds, conditions) if is_eval else (None, protocol.get("short_steps", 600), None)
    split = "heldout" if is_eval else "development"
    inv, rows, grouped = inventory(root, split, run_paths(seeds, conditions), steps, seals, runtime_sha)
    end_specs = [(ENDURANCE, "changing", "control", "adaptive", lib) for lib in LIBRARIES]
    end_inv, endurance, _ = inventory(root, "endurance", end_specs, protocol.get("long_steps", 6000))
    paired = comparisons(rows, seeds)
    base = next(p for p in paired if (p["arm"], p["condition"], p["mode"], p["metric"]) ==
                ("adaptive", "changing", "tape", "hidden_rmse"))
    primary = dict(**base, comparison="cffg_minus_rxinfer", margin=PRACTICAL, draws=BOOT_DRAWS, bootstrap_seed=BOOT_SEED)
    if stage == "stopped" or len(base["per_seed"]) != len(seeds):
        primary.update(mean_difference=None, ci95=None)
    decision_file = decision_path or root/"pilot.json"
    decision = read(decision_file) if decision_file.is_file() else {}
    if stage == "stopped":
        require(decision.get("status") == "no_go", "Stopped analysis requires a recorded no_go decision")
    return dict(schema_version=1, stage=stage, status=("stopped" if stage == "stopped" else "partial" if inv["missing"] else "complete"),
                seeds=seeds, steps=steps, conditions=list(conditions),
                primary=primary, inventory=inv, runs=rows, groups=grouped, comparisons=paired, within_library=within_library(rows),
                endurance_inventory=end_inv, endurance=endurance, decision=decision,
                scoring=dict(coverage=.90, hidden_indices=HIDDEN.tolist(), outlet_indices=OUTLETS.tolist(),
                             primary="Eight post-observation hidden states, all nonnominal transitions, one RMSE per seed.",
                             invalid="Finite carried means score on all ticks. Invalid covariances miss coverage; undefined densities invalidate the all-tick mean. Valid-only means include explicit denominators via undefined counts.",
                             aggregation="Equal seed means within group, counts summed, overall/changed/first600 latencies pooled by tick.",
                             effort="Mean squared deviation from the nominal L,G flows in action-prior units, summed across pumps.",
                             recovery="First tick of five consecutive valid Kla estimates within 2% of the endpoint. Censored at endpoint departure. Delay in model hours is delay_ticks x 0.05."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Study root")
    p.add_argument("--stage", choices=("development", "stopped", "heldout"), required=True)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    p.add_argument("--decision", type=Path, help="Pilot decision JSON (default ROOT/pilot.json)")
    args = p.parse_args()
    result = analyze(args.output, args.stage, seeds=args.seeds, conditions=args.conditions, decision_path=args.decision)
    path = args.output / ("development/analysis.json" if args.stage == "development" else "analysis.json")
    save(path, result)
    print(f"{path} ({result['status']}, {result['inventory']['scored_runs']} scored runs)")


if __name__ == "__main__":
    main()
