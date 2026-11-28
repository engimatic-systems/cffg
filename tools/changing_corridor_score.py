"""Score a completed frozen changing-corridor study. Run with the external study root."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import digamma, entr, rel_entr


SEEDS = tuple(range(19701, 19707))
ARMS = ("changing", "stationary", "no_epistemics")
WORLDS = ("change", "sham")
TARGET = (0, 2, 0)  # mode, applied action, source position
COLUMNS = list(product(range(2), range(3), range(2)))
TARGET_INDEX = COLUMNS.index(TARGET)
UNCHANGED = np.arange(len(COLUMNS)) != TARGET_INDEX
GATES = {
    "target_trace_gain": 0.002,
    "target_d1_gain_nats": 0.005,
    "probe_visit_fraction_advantage": 0.03,
    "true_target_kl_advantage_nats": 0.03,
    "goal_occupancy": 0.50,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_bytes(path, digest):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, f"SHA-256 mismatch for {path}")
    return data


def law(protocol, key):
    values = np.asarray(protocol[key], dtype=float)
    shape = (3, 2, 2) if key == "mode_transition" else (2, 3, 2, 2)
    require(values.shape == shape, f"{key} shape must be {shape}")
    require(np.isfinite(values).all() and (values >= 0).all()
            and np.allclose(values.sum(axis=-2), 1.0, rtol=0, atol=4 * np.finfo(values.dtype).eps),
            f"{key} must contain categorical probability columns")
    return values


def validate_protocol(protocol):
    require(protocol["version"] in (1, 2), "Unknown warmup policy version")
    for key, expected in (("seeds", list(SEEDS)), ("arms", list(ARMS)), ("worlds", list(WORLDS)),
                          ("observed_column", list(TARGET)), ("warmup", 128), ("followup", 48),
                          ("saturation_window", 8), ("saturation_max_d1", 0.08),
                          ("saturation_min_visits", 16)):
        require(protocol[key] == expected, f"Unexpected frozen {key}: {protocol[key]}")
    old = law(protocol, "old_position")
    new = law(protocol, "new_position")
    law(protocol, "mode_transition")
    differs = np.any(old != new, axis=-2).reshape(-1)
    require(np.array_equal(differs, ~UNCHANGED), "Only the declared target column must change")
    return {world: (new if world == "change" else old).swapaxes(-1, -2).reshape(12, 2)
            for world in WORLDS}


def concentrations(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    require(values.shape == (len(rows), 6, 2, 2), f"Invalid {key} shape")
    columns = values.swapaxes(-1, -2).reshape(len(rows), 12, 2)
    require(np.isfinite(columns).all() and (columns > 0).all(), f"Invalid {key} concentrations")
    return columns


def uncertainty(rows):
    alpha = concentrations(rows, "alpha")
    total = alpha.sum(axis=-1, keepdims=True)
    trace = (1 - np.square(alpha / total).sum(axis=-1)) / (total[..., 0] + 1)
    forecast = concentrations(rows, "forecast")
    total = forecast.sum(axis=-1, keepdims=True)
    mean = forecast / total
    d1 = (entr(mean) + mean * digamma(forecast + 1)).sum(axis=-1) - digamma(total[..., 0] + 1)
    return trace, d1, mean


def solver(receipt, planning):
    require(receipt["status"] == receipt["inner_status"] == "converged"
            and np.isfinite(receipt["free_energy"]), "Uncertified solve in an applied row")
    if planning:
        require(receipt["reason"] == "root_converged" and np.isfinite(receipt["outer_residual"]),
                "Uncertified planning root in an applied row")


def validate_rows(rows, warmup, version):
    previous = None
    for t, row in enumerate(rows):
        require(type(row["t"]) is int and row["t"] == t, "Noncontiguous applied row indices")
        for key, size in (("x", 2), ("m", 2), ("u", 3), ("x_next", 2), ("m_next", 2)):
            require(type(row[key]) is int and 0 <= row[key] < size, f"Invalid {key} at {t}")
        if previous is not None:
            require(previous["x_next"] == row["x"] and previous["m_next"] == row["m"],
                    f"Broken state trajectory at {t}")
            require(previous["after"] == row["alpha"], f"Broken committed posterior at {t}")
        solver(row["filter"], planning=False)
        if t < warmup:
            warm_action = 2 * (t % 2) if version == 1 else (2 if row["m"] == 1 or row["x"] == 0 else 0)
            require(row["plan"] is None and row["u"] == warm_action, "Unexpected warmup plan or control")
        else:
            solver(row["plan"], planning=True)
        for key in ("plan_s", "plan_peak_rss_bytes", "observe_s", "update_s", "update_peak_rss_bytes"):
            require((row[key] is None) == (t < warmup and key.startswith("plan_")),
                    f"Missing or unexpected phase measurement {key} at {t}")
        previous = row
    if rows:
        concentrations(rows, "after")


def compute_cost(rows):
    costs = {}
    for key in ("plan_s", "observe_s", "update_s", "plan_peak_rss_bytes", "update_peak_rss_bytes"):
        samples = [row[key] for row in rows if row[key] is not None]
        require(all(isinstance(v, (float, int)) and not isinstance(v, bool)
                    and np.isfinite(v) and v >= 0 for v in samples), f"Invalid {key}")
        if key.endswith("_s"):
            costs[key] = sum(samples) if samples else None
        else:
            costs[key] = max(samples, default=None)
    return costs


def saturation(rows, warmup, window, d1):
    if len(rows) < warmup:
        return None
    exposures = sum((r["m"], r["u"], r["x"]) == TARGET for r in rows[:warmup])
    peak = float(d1[warmup - window:warmup, TARGET_INDEX].max())
    return {"warm_exposures": exposures, "last_window_max_d1": peak,
            "passed": exposures >= 16 and peak <= 0.08}


def mean_metrics(records):
    means = {}
    for key in records[0]:
        values = [record[key] for record in records]
        means[key] = None if None in values else float(np.mean(values))
    return means


def score_cell(rows, warmup, window, truth, trace, d1, mean):
    before = slice(warmup - window, warmup)
    after = slice(warmup, warmup + 16)
    metrics, columns = {}, {}
    for label, array in (("trace", trace), ("d1", d1)):
        pre, post = array[before].mean(axis=0), array[after].mean(axis=0)
        for period, value in (("warm", pre), ("post", post), ("gain", post - pre)):
            columns[f"{label}_{period}"] = value.tolist()
            metrics[f"target_{label}_{period}"] = float(value[TARGET_INDEX])
            metrics[f"unchanged_{label}_{period}"] = float(value[UNCHANGED].mean())

    kl = rel_entr(truth, mean[warmup:]).sum(axis=-1)
    for label, data in (("first24", kl[:24]), ("all48", kl)):
        columns[f"kl_{label}"] = data.mean(axis=0).tolist()
        metrics[f"target_kl_{label}"] = float(data[:, TARGET_INDEX].mean())
        metrics[f"unchanged_kl_{label}"] = float(data[:, UNCHANGED].mean())
    # The known mode transition law is shared by agent and world for every action.
    metrics["mode_kl"] = 0.0
    in_range = kl[:, TARGET_INDEX] <= 0.1
    delays = np.flatnonzero(in_range[:-2] & in_range[1:-1] & in_range[2:])
    delay = int(delays[0]) if len(delays) else None
    post_rows = rows[warmup:]
    visits = sum((r["m"], r["u"], r["x"]) == TARGET for r in post_rows)
    opportunities = sum((r["m"], r["x"]) == (TARGET[0], TARGET[2]) for r in post_rows)
    metrics.update(probe_visits=visits, probe_opportunities=opportunities,
                   visit_fraction=visits / len(post_rows),
                   visits_per_opportunity=visits / opportunities if opportunities else None,
                   goal_occupancy=sum(r["x_next"] == 1 for r in post_rows) / len(post_rows))
    return {"metrics": metrics, "columns": columns,
            "recovery": {"delay": delay, "right_censored": delay is None,
                         "followup_steps": len(post_rows)}}


def cell_key(seed, arm, world):
    return f"{seed}/{arm}-{world}"


def validate_models(root, manifest, protocol_hash):
    require(manifest["protocol_sha256"] == protocol_hash, "Models bound to a different protocol")
    require(set(manifest["arms"]) == set(ARMS), "Model inventory disagrees with frozen arms")
    for arm in ARMS:
        require(set(manifest["arms"][arm]) == {"plan", "filter"}, f"Wrong model inventory for {arm}")
        for role in ("plan", "filter"):
            model = manifest["arms"][arm][role]
            path = (root / model["file"]).resolve()
            require(path.is_relative_to(root), f"Model file outside study root: {path}")
            verified_bytes(path, model["sha256"])


def read_cell(root, key, identity, seal, models, protocol, truth):
    spec = seal["cells"][key]
    base = root / "cells" / key
    receipt = json.loads(verified_bytes(base.with_suffix(".json"), spec["json"]))
    rows = [json.loads(line) for line in verified_bytes(base.with_suffix(".jsonl"), spec["jsonl"]).splitlines()]
    seed, arm, world = identity
    for field, expected in (("seed", seed), ("arm", arm), ("world", world),
                            ("protocol_sha256", seal["protocol_sha256"]), ("models_sha256", seal["models_sha256"]),
                            ("plan_hash", models["arms"][arm]["plan"]["structure_hash"]),
                            ("filter_hash", models["arms"][arm]["filter"]["structure_hash"]),
                            ("source_commit", seal["source_commit"]), ("completed_steps", len(rows))):
        require(receipt[field] == expected, f"Wrong {field} in receipt")
    status = receipt["status"]
    require(status in ("complete", "unsaturated", "failed"), f"Unknown cell status {status}")
    warmup, window = protocol["warmup"], protocol["saturation_window"]
    require(len(rows) <= warmup + protocol["followup"], "Trace exceeds budget")
    validate_rows(rows, warmup, protocol["version"])
    sat = None
    if rows:
        require([rows[0]["x"], rows[0]["m"]] == protocol["initial_state"], "Wrong initial state")
        trace, d1, mean = uncertainty(rows)
        sat = saturation(rows, warmup, window, d1)
    result = {"receipt": receipt, "digests": spec, "saturation": sat,
              "applied_compute": compute_cost(rows), "metrics": None}
    if status == "complete":
        require(len(rows) == warmup + protocol["followup"], "Incomplete applied trace marked complete")
        require(sat["passed"], "Unsaturated warmup marked complete")
        result.update(score_cell(rows, warmup, window, truth, trace, d1, mean))
    elif status == "unsaturated":
        require(len(rows) == warmup and not sat["passed"], "Inconsistent unsaturated receipt")
    elif len(rows) > warmup:
        require(sat["passed"], "Post-switch rows after an unsaturated warmup")
    warm_path = [(r["m"], r["x"], r["u"], r["m_next"], r["x_next"]) for r in rows[:warmup]]
    first_action = rows[warmup]["u"] if len(rows) > warmup else None
    return result, warm_path, first_action


def seed_comparisons(cells, seed, world):
    changing, stationary, no_epistemics = [cells[cell_key(seed, arm, world)]["metrics"] for arm in ARMS]
    return {
        "target_trace_gain": changing and changing["target_trace_gain"],
        "target_d1_gain_nats": changing and changing["target_d1_gain"],
        "probe_visit_fraction_advantage": (changing["visit_fraction"] - no_epistemics["visit_fraction"]
                                          if changing and no_epistemics else None),
        "true_target_kl_advantage_nats": (stationary["target_kl_first24"] - changing["target_kl_first24"]
                                          if stationary and changing else None),
        "goal_occupancy": changing and changing["goal_occupancy"],
    }


def group_summary(cells, arm, world):
    group = [cells[cell_key(seed, arm, world)] for seed in SEEDS]
    delays = [cell["recovery"]["delay"] for cell in group]
    peaks = {k: max(cell["applied_compute"][k] for cell in group)
             for k in ("plan_peak_rss_bytes", "update_peak_rss_bytes")}
    return {
        "means": mean_metrics([cell["metrics"] for cell in group]),
        "mean_applied_seconds": mean_metrics([{k: v for k, v in cell["applied_compute"].items()
                                               if k.endswith("_s")} for cell in group]),
        "pooled_peak_rss_bytes": peaks,
        "recovery_censored_count": delays.count(None),
        "mean_recovery_delay": None if None in delays else float(np.mean(delays)),
    }


def score(root):
    root = root.resolve()
    seal_bytes = (root / "complete.json").read_bytes()
    seal = json.loads(seal_bytes)
    protocol = json.loads(verified_bytes(root / "protocol.json", seal["protocol_sha256"]))
    truths = validate_protocol(protocol)
    freeze = json.loads(verified_bytes(root / "freeze.json", seal["freeze_sha256"]))
    require(isinstance(seal["source_commit"], str) and seal["source_commit"], "Missing source commit")
    for field in ("source_commit", "protocol_sha256", "models_sha256"):
        require(freeze[field] == seal[field], f"Completion and freeze disagree on {field}")
    qualification = json.loads(verified_bytes(root / "qualification.json", freeze["qualification_sha256"]))
    require(qualification["status"] == "passed" and
            all(qualification[k] == seal[k] for k in ("models_sha256", "protocol_sha256")),
            "Study lacks matching passed development qualification")
    thresholds = protocol["gate_thresholds"]
    require(thresholds == GATES, "Unexpected frozen gate thresholds")
    models = json.loads(verified_bytes(root / "models.json", seal["models_sha256"]))
    validate_models(root, models, seal["protocol_sha256"])
    identities = list(product(SEEDS, ARMS, WORLDS))
    keys = {cell_key(*identity) for identity in identities}
    require(set(seal["cells"]) == keys, "Completion manifest disagrees with fixed cell grid")
    cell_dir = root / "cells"
    expected = {f"{key}.{extension}" for key in keys for extension in ("json", "jsonl")}
    actual = {p.relative_to(cell_dir).as_posix() for p in cell_dir.rglob("*")
              if p.is_file() and p.suffix in (".json", ".jsonl")}
    require(actual == expected, f"Cell file inventory differs: {sorted(actual ^ expected)}")

    cells, warm_paths, first_actions = {}, {}, {}
    for seed, arm, world in identities:
        key = cell_key(seed, arm, world)
        try:
            cell, warm, action = read_cell(root, key, (seed, arm, world), seal, models, protocol, truths[world])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid cell {key}: {exc}") from exc
        cells[key] = cell
        reference = warm_paths.get(seed, [])
        common = min(len(warm), len(reference))
        require(warm[:common] == reference[:common], f"Mismatched paired warmup at {key}")
        if len(warm) > len(reference):
            warm_paths[seed] = warm
        if action is not None:
            prior = first_actions.setdefault((seed, arm), action)
            require(prior == action, f"Mismatched first post-switch action at {key}")

    complete = all(cell["metrics"] is not None for cell in cells.values())
    per_seed = {str(seed): {world: seed_comparisons(cells, seed, world) for world in WORLDS} for seed in SEEDS}
    world_means = {world: mean_metrics([per_seed[str(seed)][world] for seed in SEEDS])
                   for world in WORLDS} if complete else None
    gates = {}
    for name, threshold in thresholds.items():
        estimate = world_means["change"][name] if complete else None
        verdict = ("pass" if estimate >= threshold else "fail") if complete else "unscored"
        gates[name] = {"minimum": threshold, "estimate": estimate, "status": verdict}
    return {
        "complete_grid": complete, "source_commit": seal["source_commit"],
        "protocol_version": protocol["version"], "freeze_sha256": seal["freeze_sha256"],
        "complete_sha256": hashlib.sha256(seal_bytes).hexdigest(),
        "protocol_sha256": seal["protocol_sha256"], "models_sha256": seal["models_sha256"],
        "models": models["arms"], "column_order_mode_action_source": COLUMNS,
        "pairing": {"common_warm_prefix_matches": True, "observed_first_actions_match_between_worlds": True,
                    "all_cells_applied_first_post_action": all(c["receipt"]["completed_steps"] > protocol["warmup"]
                                                               for c in cells.values())},
        "cells": cells, "per_seed_comparisons": per_seed,
        "equal_seed_mean_comparisons": world_means,
        "arm_world": ({arm: {world: group_summary(cells, arm, world) for world in WORLDS}
                       for arm in ARMS} if complete else None),
        "change_world_gates": gates,
    }


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def render(report):
    lines = ["Changing-corridor held-out study", f"Source commit {report['source_commit']}",
             f"Protocol SHA-256 {report['protocol_sha256']}", f"Model inventory SHA-256 {report['models_sha256']}",
             f"Completion manifest SHA-256 {report['complete_sha256']}",
             "The target is the position transition from x=0 under action 2 in mode 0.",
             "The sham world uses the original law for that same target column.", "",
             "Change-world gates (minimum thresholds on six equally weighted seeds)",
             "metric | estimate | minimum | outcome"]
    for name, gate in report["change_world_gates"].items():
        lines.append(f"{name} | {fmt(gate['estimate'])} | {gate['minimum']} | {gate['status']}")
    lines += ["", "Paired estimates by seed and world", "seed | world | " + " | ".join(GATES)]
    for seed, comparisons in report["per_seed_comparisons"].items():
        for world, values in comparisons.items():
            lines.append(f"{seed} | {world} | " + " | ".join(fmt(values[k]) for k in GATES))
    if report["complete_grid"]:
        for world, values in report["equal_seed_mean_comparisons"].items():
            lines.append(f"mean | {world} | " + " | ".join(fmt(values[k]) for k in GATES))
    else:
        lines += ["", "The frozen grid contains unsuccessful cells. All aggregate estimates are unscored."]

    lines += ["", "Applied trajectories", "cell | status | applied steps | reason | failure"]
    for key, cell in report["cells"].items():
        r = cell["receipt"]
        failure = json.dumps(r["failure"], sort_keys=True) if r.get("failure") is not None else ""
        lines.append(f"{key} | {r['status']} | {r['completed_steps']} | {r['reason']} | {failure}")

    if report["complete_grid"]:
        tables = (
            ("Committed posterior trace and forecast D1 (first 16 post minus last 8 warm)",
             ("target_trace_gain", "unchanged_trace_gain", "target_d1_gain", "unchanged_d1_gain")),
            ("Probe visitation and goal occupancy (all 48 post-switch steps)",
             ("probe_visits", "probe_opportunities", "visit_fraction", "visits_per_opportunity", "goal_occupancy")),
            ("Exact conditional KL in nats (unchanged is the unweighted mean of 11 position columns)",
             ("target_kl_first24", "unchanged_kl_first24", "target_kl_all48", "unchanged_kl_all48", "mode_kl")),
        )
        for title, fields in tables:
            lines += ["", title, "arm/world | " + " | ".join(fields)]
            for arm, worlds in report["arm_world"].items():
                for world, group in worlds.items():
                    lines.append(f"{arm}/{world} | " + " | ".join(fmt(group["means"][k]) for k in fields))
        lines += ["", "Prediction recovery (zero-based start of three consecutive post-switch target KLs <= 0.1 nat)",
                  "arm/world | mean delay | right-censored seeds"]
        for arm, worlds in report["arm_world"].items():
            for world, group in worlds.items():
                lines.append(f"{arm}/{world} | {fmt(group['mean_recovery_delay'])} | {group['recovery_censored_count']}")
        lines += ["", "Compute across applied steps, including warmup. Updates occur inside observe.",
                  "Times are mean cell totals. RSS is the largest sampled whole-process resident size in the phase.",
                  "arm/world | planning s | observe s | update s | planning peak bytes | update peak bytes"]
        for arm, worlds in report["arm_world"].items():
            for world, group in worlds.items():
                times = group["mean_applied_seconds"]
                peaks = group["pooled_peak_rss_bytes"]
                values = [times["plan_s"], times["observe_s"], times["update_s"],
                          peaks["plan_peak_rss_bytes"], peaks["update_peak_rss_bytes"]]
                lines.append(f"{arm}/{world} | " + " | ".join(map(fmt, values)))
    lines += ["", "An absent denominator or a censored seed leaves its respective mean undefined (NA).",
              "score.json contains individual cell diagnostics, per-column values, saturation and raw trace digests."]
    return "\n".join(lines) + "\n"


def write_report(root):
    root = root.resolve()
    report = score(root)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    summary = render(report)
    (root / "score.json").write_text(payload)
    (root / "score.txt").write_text(summary)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        write_report(args.root)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"Unable to score {args.root}: {exc}\n")
    print(f"Wrote {args.root / 'score.json'} and {args.root / 'score.txt'}")


if __name__ == "__main__":
    main()
