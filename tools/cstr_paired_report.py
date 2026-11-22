"""Audit saved PCGym CSTR seed pairs and write an offline HTML report.

    uv run --frozen python tools/cstr_paired_report.py [STUDY_ROOT]

Reads STUDY_ROOT/run and STUDY_ROOT/replay/summary.json. The seed-0 pilot manifest
fixes the experiment parameters. Only results from the ten held-out pairs enter
the estimates. This command does not import controller code or run simulations.
"""
import argparse
import hashlib
import html
import json
import os
import shlex
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import numpy as np


SEEDS = tuple(range(1, 11))
ARMS = ("directed", "host")
ROOT = Path.home() / "agent-share/cffg-reactor-paired"
PILOT = ROOT.parent / "cffg-reactor-pilot/seed-0/manifest.json"
STEPS = 59
SWITCHES = (20, 40)
COUNTS = {
    "state_box_violations": "True state box excursions (steps)",
    "input_excursions": "Noisy Ca or T outside input box (decisions)",
    "applied_actuator_violations": "Applied actuator violations (decisions)",
    "clips": "Clipped proposals (decisions)",
    "applied_saturations": "Applied at a bound (decisions)",
    "misses": "Deadline misses (decisions)",
    "verification_failures": "Failed checks or solves (decisions)",
    "holds": "Fallback holds (decisions)",
}
OUTCOMES = {"return_sum": "Native episode return", "Ca_RMSE": "True Ca RMSE",
            "Ca_MAE": "True Ca MAE", **COUNTS,
            "clip_excess_K": "Total proposed excess beyond bounds (K)",
            "max_clip_excess_K": "Largest proposed excess beyond a bound (K)"}
LATENCY = {"launch_to_first_checked_s": "Launch to first checked proposal",
           "launch_to_first_applied_s": "Launch to first command ready",
           "launch_to_first_plant_call_s": "Launch to first plant call",
           "cold_solve_s": "First solve", "cold_through_clip_s": "First decision through clip",
           "cold_observation_to_action_s": "First decision to command ready",
           "cold_plant_call_s": "First decision to plant call",
           "warm_clip_median_s": "Repeated through clip, median",
           "warm_clip_p95_s": "Repeated through clip, p95",
           "warm_median_s": "Repeated to command ready, median",
           "warm_p95_s": "Repeated to command ready, p95",
           "warm_call_median_s": "Repeated to plant call, median",
           "warm_call_p95_s": "Repeated to plant call, p95"}


def read(path):
    try:
        return json.loads(path.read_bytes())
    except (ValueError, OSError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def require(condition, location, message):
    if not condition:
        raise ValueError(f"{location}: {message}")


def close(actual, expected, location, name, atol=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.allclose(a, b, rtol=1e-12, atol=atol),
            location, f"{name} differs from reconstructed value")


def indexed_files(directory, kind, length):
    expected = [directory / f"{kind}-{t:02}.json" for t in range(length)]
    require(sorted(directory.glob(f"{kind}-*.json")) == expected, directory,
            f"expected {length} consecutive {kind} files")
    return expected


def phase(tick):
    return "Switch window" if any(abs(tick + 1 - i) <= 2 for i in SWITCHES) else "Other decisions"


@dataclass
class Episode:
    directory: Path
    complete: bool = False
    decisions: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    noise: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    first: dict = field(default_factory=dict)
    exit_code: int = 0
    startup: dict = field(default_factory=dict)


def check_window(r, history, observation, schedule, tick, path):
    close(r["observation"], observation, path, "observation")
    require(r["history"] == history[-4:], path, "history disagrees with prior applied commands")
    indices = [min(tick + j, STEPS) for j in range(1, 16)]
    goals = [schedule[i] for i in indices]
    require(r["preview_indices"] == indices and r["preview_Ca"] == goals, path, "preview mismatch")
    past = [obs[:2] + [u] for obs, u in history[-4:]]
    rows = ([None] * (4 - len(past)) + past + [observation[:2]]
            + [[2 * (goal - .7) / (1 - .7) - 1, 0.] for goal in goals])
    masks = {f"e{i}": row is not None for i, row in enumerate(rows)}
    require(r["masks"] == masks and r["data"].keys() == masks.keys(), path, "window masks or keys")
    for i, expected in enumerate(rows):
        key = f"e{i}"
        if expected is None:
            require(r["data"][key] is None, path, f"expected masked {key}")
        else:
            close(r["data"][key], expected, path, key)


def check_action(r, last, path):
    raw, clipped, applied = (r[k] for k in ("proposed_normalized", "clipped_normalized",
                                             "applied_normalized"))
    failed = r["error"] is not None
    require(r["status"] == ("verification_failed" if failed else "checked") and (failed or r["checks"]),
            path, "decision check status")
    if clipped is not None or not failed:
        close(raw, (r["proposed_kelvin"] - 298.5) / 3.5, path, "proposed coolant")
        close(clipped, np.clip(raw, -1, 1), path, "clipped coolant")
        if (r.get("checks") or {}).get("coolant_kelvin") is not None:
            close(r["checks"]["coolant_kelvin"], r["proposed_kelvin"], path, "checked coolant")
    elapsed = r["observation_to_action_s"]
    require(0 <= r["through_clip_s"] <= elapsed <= r["observation_to_plant_call_s"],
            path, "inconsistent decision clocks")
    if r["solve_s"] is not None:
        require(0 <= r["solve_s"] <= r["through_clip_s"], path, "inconsistent solve clock")
    require(r["deadline_miss"] == (elapsed > 1), path, "deadline status")
    require(r["fallback"] == (failed or r["deadline_miss"]), path, "fallback status")
    close(applied, last if r["fallback"] else clipped, path, "applied command")
    close(r["applied_kelvin"], 298.5 + 3.5 * applied, path, "applied kelvin")


def latency(rows, timings, path):
    values = {key: timings.get(key) for key in LATENCY}
    checked = next((r["tick"] for r in rows if r["error"] is None), None)
    require(timings.get("first_checked_tick") == checked, path, "first checked tick")
    require((timings["launch_to_first_checked_s"] is None) == (checked is None),
            path, "launch to first checked proposal")
    if not rows:
        return values
    first = rows[0]
    for key, source in (("cold_solve_s", "solve_s"), ("cold_observation_to_action_s", "observation_to_action_s")):
        require(values[key] == first[source], path, key)
    values.update(cold_through_clip_s=first["through_clip_s"], cold_plant_call_s=first["observation_to_plant_call_s"])
    launch_ready = timings["launch_to_first_applied_s"]
    require(launch_ready >= first["observation_to_action_s"], path, "launch boundary")
    close(timings["launch_to_first_plant_call_s"] - launch_ready,
          first["observation_to_plant_call_s"] - first["observation_to_action_s"], path, "launch to call", 1e-8)
    if checked == 0:
        close(timings["launch_to_first_checked_s"] - launch_ready,
              first["through_clip_s"] - first["observation_to_action_s"], path, "launch to check", 1e-8)
    if len(rows) > 1:
        for source, prefix in (("through_clip_s", "warm_clip"), ("observation_to_action_s", "warm"),
                               ("observation_to_plant_call_s", "warm_call")):
            samples = [r[source] for r in rows[1:]]
            for name, pct in (("median", 50), ("p95", 95)):
                key = f"{prefix}_{name}_s"
                values[key] = float(np.percentile(samples, pct))
                if prefix == "warm":
                    close(values[key], timings[key], path, key)
    return values


def audit_episode(directory, seed, arm, manifest, code):
    e = Episode(directory, exit_code=code)
    e.startup = {k: manifest["runtime"][k] for k in ("numerical_import_s", "simulator_import_s")}
    e.startup["setup_s"] = manifest.get("setup_s")
    path = directory / "episode.json"
    summary = read(path) if path.exists() or not code else None
    params = manifest["parameters"]
    schedule = params["SP"]["Ca"]
    # PCGym normalizes observations with Gym's float32 Box endpoints and span.
    lo, hi = [np.asarray(params["o_space"][k], dtype=np.float32) for k in ("low", "high")]
    observation = (2 * (np.array(params["x0"]) - lo) / (hi - lo) - 1).tolist()
    history, errors = [], []
    counts = Counter()
    n = summary["decisions"] if summary else len(list(directory.glob("decision-*.json")))
    scored = summary["scored_steps"] if summary else len(list(directory.glob("step-*.json")))
    require(0 <= scored <= n <= STEPS and n - scored <= 1, path, "episode lengths")
    decisions = indexed_files(directory, "decision", n)
    # The pilot persists a plant output before checking its validity.
    steps = sorted(directory.glob("step-*.json"))
    require(scored <= len(steps) <= n, path, "step coverage")
    indexed_files(directory, "step", len(steps))
    if summary and n > scored:
        require("plant-failure" in summary["first"], path, "unscored plant decision without a failure record")
    for tick, decision_path in enumerate(decisions):
        r = read(decision_path)
        require(r["tick"] == tick and r["seed"] == seed, decision_path, "tick or seed")
        check_window(r, history, observation, schedule, tick, decision_path)
        check_action(r, history[-1][1] if history else 0., decision_path)
        u, raw, clipped = (r[k] for k in ("applied_normalized", "proposed_normalized", "clipped_normalized"))
        flags = dict(misses=r["deadline_miss"], holds=r["fallback"], verification_failures=r["error"] is not None,
                     clips=clipped is not None and clipped != raw, applied_saturations=abs(u) == 1.,
                     applied_actuator_violations=abs(u) > 1., input_excursions=max(map(abs, observation[:2])) > 1)
        counts.update(flags)
        for name, label in (("misses", "miss"), ("holds", "failure"), ("verification_failures", "verification-failure"),
                            ("input_excursions", "input-box-excursion")):
            if flags[name]:
                e.first.setdefault(label, tick)
        history.append([r["observation"], u])
        e.decisions.append({k: v for k, v in r.items() if k not in ("candidate", "data", "masks", "history")})
        if tick >= scored:
            break
        step_path = steps[tick]
        s = read(step_path)
        require(s["tick"] == tick and s["reward_setpoint_index"] == tick + 1
                and s["reward_setpoint_Ca"] == schedule[tick + 1], step_path, "reward setpoint")
        state, obs = (np.asarray(s[k]) for k in ("next_true_state", "next_observation"))
        require(state.shape == obs.shape == (3,) and np.isfinite([state, obs]).all(), step_path, "plant output")
        observation = obs.tolist()
        noisy_physical = (obs + 1) * (hi - lo) / 2 + lo
        close(state[2], schedule[tick], step_path, "plant setpoint channel")
        close(noisy_physical[2], state[2], step_path, "observed setpoint channel")
        e.noise.append((noisy_physical[:2] - state[:2]) / (.001 * state[:2]))
        err = state[0] - schedule[tick + 1]
        close(s["Ca_error"], err, step_path, "tracking error")
        close(s["reward"], -params["r_scale"]["Ca"] * err**2, step_path, "reward")
        violation = bool(np.any(state[:2] < params["o_space"]["low"][:2]) or
                         np.any(state[:2] > params["o_space"]["high"][:2]))
        require(s["state_box_violation"] == violation, step_path, "state box")
        counts["state_box_violations"] += violation
        if violation:
            e.first.setdefault("state-box-excursion", tick)
        require(tick + 1 == scored or not (s["terminated"] or s["truncated"]), step_path, "early terminal flag")
        errors.append(err)
        e.steps.append(s)
    terminal = bool(scored == STEPS and e.steps[-1]["terminated"] and not e.steps[-1]["truncated"])
    if summary:
        require(summary["complete"] == terminal and (not terminal or code == 0), path, "completion status")
        e.complete = terminal
        for name in COUNTS.keys() - {"input_excursions"}:
            require(summary["counts"][name] == counts[name], path, name)
    if summary and "plant-failure" in summary["first"]:
        tick = summary["first"]["plant-failure"]
        require(n and tick == n - 1 and not terminal, path, "plant failure position")
        e.first["plant-failure"] = tick
        e.first.setdefault("failure", tick)
    if summary:
        require(e.first == summary["first"], path, "first-event index")
    for label, tick in (e.first.items() if summary else []):
        evidence = read(directory / f"first-{label}.json")
        require(evidence.get("decision", evidence) == read(decisions[tick]), path, f"first {label} decision")
        if "step" in evidence:
            require(evidence["step"] == read(steps[tick]), path, f"first {label} step")
    excess = [3.5 * abs(r["proposed_normalized"] - r["clipped_normalized"]) for r in e.decisions
              if r["clipped_normalized"] is not None]
    e.metrics = {k: counts[k] for k in COUNTS}
    e.metrics.update(clip_excess_K=sum(excess), max_clip_excess_K=max(excess, default=0.))
    e.metrics.update(return_sum=sum(s["reward"] for s in e.steps),
                     Ca_RMSE=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                     Ca_MAE=float(np.mean(np.abs(errors))) if errors else None)
    for key in (("return_sum", "Ca_RMSE", "Ca_MAE") if summary else ()):
        if e.metrics[key] is None:
            require(summary[key] is None, path, key)
        else:
            close(summary[key], e.metrics[key], path, key)
    if scored:
        # PCGym uses the global NumPy RandomState stream reset to the held-out seed.
        expected_noise = np.random.RandomState(seed).standard_normal((scored, 2))
        close(e.noise, expected_noise, path, "standardized observation noise vs seed", 1e-9)
    if summary:
        e.startup["env_build_reset_s"] = summary["timings"]["env_build_reset_s"]
        e.metrics.update(latency(e.decisions, summary["timings"], path))
    return e


def source_hashes(info):
    repo = Path(info["reference_path"]).parent.parent
    experiment = Path(info["experiment"])
    hashes = {}
    for filename, digest in info["file_sha256"].items():
        path = Path(filename)
        if path.is_relative_to(experiment):
            key = ("experiment", str(path.relative_to(experiment)))
        elif path.is_relative_to(repo):
            key = ("cffg", str(path.relative_to(repo)))
        else:
            key = ("external", filename)
        hashes[key] = digest
    return hashes


def load_study(run, pilot):
    study = read(run / "study.json")
    protocol = study["protocol"]
    require(protocol == pilot["protocol"] and protocol["pilot_seed"] == 0
            and protocol["held_out_paired_seeds"] == list(SEEDS), run, "held-out pilot protocol")
    order = {str(s): list(ARMS if s % 2 else reversed(ARMS)) for s in SEEDS}
    require(study["order_by_seed"] == order, run, "seed roster or alternating order")
    processes = read(run / "processes.json")
    planned = [(s, arm) for s in SEEDS for arm in order[str(s)]]
    require([(p["seed"], p["arm"]) for p in processes] == planned, run, "process roster or order")
    expected_dirs = {f"seed-{s:02d}" for s in SEEDS}
    require({d.name for d in run.glob("seed-*")} <= expected_dirs, run, "unexpected seed directory")
    episodes, runtime = {}, None
    arm_hashes, models = {}, {}
    for process in processes:
        seed, arm, code = (process[k] for k in ("seed", "arm", "exit_code"))
        directory = run / f"seed-{seed:02d}" / arm
        path = directory / "manifest.json"
        if code and not path.exists() and not (directory / "episode.json").exists():
            require(not any(directory.glob("decision-*.json")) and not any(directory.glob("step-*.json")),
                    directory, "raw trajectory without a manifest")
            episodes[seed, arm] = Episode(directory, exit_code=code)
            continue
        m = read(path)
        require((m["seed"], m["arm"], m["order"]) == (seed, arm, order[str(seed)].index(arm)), path, "arm identity")
        require(m["protocol"] == protocol and m["parameters"] == pilot["parameters"]
                and m["configuration"] == pilot["configuration"], path, "pilot configuration")
        require(m["model"] == models.setdefault(arm, m["model"]), path, "model changed within arm")
        if arm == "directed":
            require(m["model"] == pilot["model"], path, "directed model")
        info = m["runtime"]
        require(info["head"] == study["revision"] and m["launch_boundary"] == "before uv", path,
                "runtime revision or launch boundary")
        hashes = source_hashes(info)
        for source, digest in study["runtime_sources"].items():
            if arm == "directed" and source == "tools/cstr_host_comparator.py":
                continue
            require(hashes.get(("cffg", source)) == digest, path, f"runtime hash of {source}")
        common = {k: info[k] for k in ("head", "hardware", "os", "devices", "environment", "pcgym_source",
                                        "modules", "library_tree", "python_version")}
        require(arm_hashes.setdefault(arm, hashes) == hashes, path, "source hashes changed within arm")
        for previous in arm_hashes.values():
            require(all(hashes[k] == previous[k] for k in hashes.keys() & previous.keys()),
                    path, "shared runtime source has a different hash")
        if runtime is None:
            runtime = common
        require(common == runtime, path, "runtime or hardware differs across arms")
        try:
            episodes[seed, arm] = audit_episode(directory, seed, arm, m, code)
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"{directory}: invalid episode evidence: {exc}") from exc
    noise_pairs = []
    for seed in SEEDS:
        d, h = (episodes[seed, arm] for arm in ARMS)
        n = min(len(d.noise), len(h.noise))
        if d.decisions and h.decisions:
            require(d.decisions[0]["observation"] == h.decisions[0]["observation"], run, f"seed {seed} reset")
        if n:
            close(d.noise[:n], h.noise[:n], run, f"seed {seed} paired noise", 1e-9)
            noise_pairs.extend(np.abs(np.asarray(d.noise[:n]) - h.noise[:n]).ravel())
    return study, episodes, runtime, arm_hashes, noise_pairs


def fmt(value):
    if value is None:
        return "—"
    return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)


def esc(value):
    return html.escape(fmt(value))


def table(headers, rows, caption):
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = "".join('<tr>' + "".join(
        f'<th scope="row">{v}</th>' if i == 0 else f'<td>{v}</td>' for i, v in enumerate(row)
    ) + '</tr>' for row in rows)
    return (f'<div class="scroll"><table><caption>{esc(caption)}</caption><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def link(path, root, label):
    return f'<a href="{quote(os.path.relpath(path.resolve(), root.resolve()))}">{esc(label)}</a>'


def estimate(pairs, draws):
    if not pairs:
        return [None] * 5 + ["0 / 0 / 0"]
    d, h = np.array(pairs).T
    delta = d - h
    # Reuse the same seed indices for every metric with the same eligible roster.
    n = len(pairs)
    if n not in draws:
        draws[n] = np.random.default_rng(9806).integers(n, size=(10000, n))
    low, high = np.percentile(delta[draws[n]].mean(axis=1), [2.5, 97.5])
    signs = " / ".join(str(np.count_nonzero(mask)) for mask in (delta > 0, delta == 0, delta < 0))
    return [d.mean(), h.mean(), delta.mean(), low, high, signs]


def outcome_tables(episodes, metrics, draws, timing=False):
    aggregate, details, estimates = [], [], {}
    for key, label in metrics.items():
        pairs, rows = [], []
        for seed in SEEDS:
            d, h = (episodes[seed, arm] for arm in ARMS)
            a, b = d.metrics.get(key), h.metrics.get(key)
            usable = a is not None and b is not None and (timing or (d.complete and h.complete))
            if usable:
                pairs.append((a, b))
            rows.append([seed, fmt(a), fmt(b), fmt(a - b if usable else None)])
        result = estimate(pairs, draws)
        estimates[key] = result
        aggregate.append([esc(label), len(pairs), *map(fmt, result)])
        details.append("<details><summary>" + esc(label) + "</summary>" +
                       table(["Seed", "Directed", "Host", "Directed − host"], rows, label) + "</details>")
    summary = table(["Measure", "Pairs", "Directed mean", "Host mean", "Mean Δ", "95% low", "95% high", "+ / tie / −"],
                    aggregate, "Paired means in seconds" if timing else "Paired episode outcomes")
    return summary + "".join(details), estimates


def reward_profile(episodes, draws, total):
    complete = [s for s in SEEDS if all(episodes[s, arm].complete for arm in ARMS)]
    if not complete:
        return ""
    rows, contributions = [], []
    for name, span in (("First scored step", slice(0, 1)), ("Steps 2–5", slice(1, 5)),
                       ("First five combined", slice(0, 5)), ("Steps 6–59", slice(5, None))):
        pairs = [[sum(step["reward"] for step in episodes[s, arm].steps[span]) for arm in ARMS] for s in complete]
        d, h, delta, low, high, _ = estimate(pairs, draws)
        share = None if total == 0 else delta / total * 100
        contributions.append(f"{delta:+.6f}" + ("" if share is None else f" ({share:.1f}% of the total gap)"))
        rows.append([name, len(pairs), *map(fmt, (d, h, delta, low, high, share))])
    first_holds = [sum(episodes[s, arm].decisions[0]["fallback"] for s in complete) for arm in ARMS]
    misses = [sum(episodes[s, arm].metrics["misses"] for s in complete) for arm in ARMS]
    return (f"<h3>When the return difference accumulated</h3><p>The first scored step contributed "
            f"{contributions[0]} to the mean paired difference. The first five together contributed "
            f"{contributions[2]}. On the first decision, directed held the initial command on "
            f"{first_holds[0]}/{len(complete)} seeds and host on {first_holds[1]}/{len(complete)}. "
            f"Across these complete pairs, directed missed {misses[0]} deadlines and host missed {misses[1]}. "
            "These are observed rewards under the one-second deadline. The timed trajectories alone provide no "
            "full-episode return for a host whose first proposal reached the plant in time.</p>" +
            table(["Scored steps","Pairs","Directed mean reward","Host mean reward","Mean Δ",
                   "95% low","95% high","Share of full Δ (%)"], rows,
                  "Contributions to native return (the first-five row includes the first step and steps 2–5)"))


def first_step_intervention(episodes, params, root, runtime):
    path = root / "first-step-intervention.json"
    if not path.exists():
        return ""
    study = read(path)
    require(runtime and all(study[k] == runtime[k] for k in ("library_tree", "pcgym_source")), path,
            "one-step simulator provenance")
    seeds = study["seeds"]
    require(sorted(row["seed"] for row in seeds) == list(SEEDS), path, "intervention seed roster")
    lo, hi = [np.asarray(params["o_space"][k][:2], np.float32) for k in ("low", "high")]
    variants = ("directed", "host", "host_checked_proposal")
    commands = {k: [] for k in variants}
    rewards = {k: [] for k in variants}
    reversals = 0
    for row in seeds:
        seed = row["seed"]
        host = episodes[seed, "host"]
        decision = host.decisions[0]
        require(row["host_held"] == decision["fallback"] and row["host_was_late"] == decision["deadline_miss"]
                and decision["error"] is None and decision["clipped_normalized"] is not None,
                path, f"host proposal and hold at seed {seed}")
        for name in variants:
            result = row["outcomes"][name]
            state = np.asarray(result["true_state"])
            obs = np.asarray(result["observation"])
            if name in ARMS:
                e = episodes[seed, name]
                u = e.decisions[0]["applied_kelvin"]
                for key, actual in (("next_true_state", state), ("next_observation", obs), ("reward", result["reward"])):
                    close(actual, e.steps[0][key], path, f"{name} {key} at seed {seed}")
            else:
                u = 298.5 + 3.5 * decision["clipped_normalized"]
                noise = ((obs[:2] + 1) * (hi - lo) / 2 + lo - state[:2]) / (.001 * state[:2])
                close(noise, host.noise[0], path, f"intervention noise at seed {seed}", 1e-9)
            close(result["command_kelvin"], u, path, f"{name} command")
            close(result["reward"], -params["r_scale"]["Ca"] * (state[0] - params["SP"]["Ca"][1])**2,
                  path, f"{name} native reward")
            commands[name].append(u)
            rewards[name].append(result["reward"])
        actual_gap = rewards["directed"][-1] - rewards["host"][-1]
        changed_gap = rewards["directed"][-1] - rewards["host_checked_proposal"][-1]
        reversals += int(actual_gap * changed_gap < 0)
    labels = ("Timed directed command", "Timed host command", "Host checked proposal applied")
    rows = [[label, f"{np.mean(commands[k]):.4f}", f"{np.mean(rewards[k]):.5f}"]
            for label, k in zip(labels, variants)]
    return ("<h3>Post-hoc first-step intervention</h3><p>PCGym was reset on each of the ten original seeds. "
            "Applying each timed first command reproduced its saved true state, observation and reward. "
            "A third run changed only the first host command, applying its checked, clipped proposal. "
            f"The first-step reward ranking reversed on {reversals}/{len(seeds)} seeds. Later steps were "
            f"unobserved under this intervention. {link(path, root, 'One-step intervention evidence')}.</p>" +
            table(["First plant input","Mean command K","Mean first reward"], rows, "One simulated step per command and seed"))


def launch_table(episodes):
    rows = []
    for (seed, arm), e in episodes.items():
        rows.append([seed, arm, *[fmt(e.startup.get(k)) for k in ("numerical_import_s", "simulator_import_s",
                     "setup_s", "env_build_reset_s")], fmt(e.metrics.get("launch_to_first_checked_s")),
                     fmt(e.metrics.get("cold_observation_to_action_s")),
                     esc(e.decisions[0]["deadline_miss"] if e.decisions else None)])
    first = episodes[1, "directed"]
    lead = ""
    if first.decisions and first.startup:
        lead = (f"<p>The first invocation from the measurement checkout was seed 1, directed. Numerical "
                f"imports took {fmt(first.startup['numerical_import_s'])} s and simulator imports took "
                f"{fmt(first.startup['simulator_import_s'])} s. Launch to the first checked proposal took "
                f"{fmt(first.metrics.get('launch_to_first_checked_s'))} s. Its first decision "
                f"{'missed' if first.decisions[0]['deadline_miss'] else 'met'} the one-second deadline. "
                "The estimates include this invocation. The import timers measure elapsed time within each boundary.</p>")
    return lead + table(["Seed","Controller","Numeric imports","Simulator imports","Controller setup","Build/reset",
                        "Launch to checked","Cold to command","Cold miss"], rows, "Startup components by process (seconds)")


def coverage(episodes, root):
    rows = []
    for (seed, arm), e in sorted(episodes.items()):
        directory = e.directory
        paths = [(directory / "manifest.json", "Manifest"), (directory / "episode.json", "Episode"),
                 (directory.parent / f"{arm}.log", "Process log")]
        if e.decisions:
            paths += [(directory / "decision-00.json", "First decision")]
        if e.steps:
            paths += [(directory / "step-00.json", "First step")]
        events = [link(directory / f"decision-{tick:02}.json", root, f"{label} at {tick}")
                  for label, tick in sorted(e.first.items())]
        rows.append([seed, arm, "Complete" if e.complete else "Incomplete", e.exit_code,
                     len(e.decisions) if e.metrics else "—", len(e.steps) if e.metrics else "—",
                     "; ".join(events) or "—", " · ".join(link(p, root, label) for p, label in paths if p.exists())])
    return table(["Seed","Controller","Episode","Exit","Decisions","Scored","First events (tick)","Evidence"],
                 rows, "All twenty scheduled processes")


def clip_bound(att):
    raw = att["proposed_normalized"]
    clipped = att["clipped_normalized"]
    return clipped if clipped is not None and raw != clipped else None


def clipping(episodes):
    phases = ("Switch window", "Other decisions")
    totals = []
    for arm in ARMS:
        for period in phases:
            rows = [r for s in SEEDS for r in episodes[s, arm].decisions if phase(r["tick"]) == period]
            proposals = [r for r in rows if r["clipped_normalized"] is not None]
            clipped = [r for r in proposals if clip_bound(r) is not None]
            exceedance = [abs(r["proposed_normalized"] - r["clipped_normalized"]) * 3.5 for r in clipped]
            totals.append([arm, period, len(rows), len(proposals), sum(clip_bound(r) == -1 for r in clipped),
                           sum(clip_bound(r) == 1 for r in clipped), len(clipped), fmt(sum(exceedance)),
                           fmt(np.mean(exceedance) if exceedance else None), fmt(max(exceedance, default=None)),
                           sum(abs(r["applied_normalized"]) == 1 for r in rows)])
    aligned = []
    for period in phases:
        matched = [(d, h) for seed in SEEDS for d, h in zip(*(episodes[seed, arm].decisions for arm in ARMS))
                   if phase(d["tick"]) == period]
        proposed = [(d, h) for d, h in matched if all(r["clipped_normalized"] is not None for r in (d, h))]
        aligned.append([period, len(matched), len(proposed),
                        *[sum(clip_bound(d) == clip_bound(h) == bound for d, h in proposed) for bound in (-1, 1)],
                        *[sum(d["applied_normalized"] == h["applied_normalized"] == bound
                              for d, h in matched) for bound in (-1, 1)]])
    return (table(["Controller","Phase","Decisions","Proposals","Clip low","Clip high","All clips",
                   "Sum excess K","Mean per clip K","Max excess K","Applied at bound"], totals,
                  "Proposal clipping and applied saturation on each controller’s own trajectory") +
            table(["Phase","Aligned ticks","Both proposals","Both clip low","Both clip high",
                   "Both apply low","Both apply high"], aligned, "Aligned by seed and tick on separate trajectories"))



def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stock_result(path, seed, tick, source, digest, expected, root):
    raw = read(path)
    require(raw["tick"] == tick and raw.get("decision_sha256") == digest, path, "stock input digest")
    same = raw.get("same_state", {})
    post = [float(v) for edge in raw.get("posterior_errors", {}).values() for v in edge.values()]
    passed = bool(same.get("certified") and raw.get("stock_status") == "converged" and not raw.get("stock_failure")
                  and raw.get("directed_residual") is not None and float(raw["directed_residual"]) <= 1e-6
                  and float(raw.get("FE_error", np.inf)) <= 1e-6 and post and all(v <= .001 for v in post))
    require(raw["historical_parity"] == passed == expected, path, "stock parity evidence")
    messages = raw.get("message_errors", {})
    max_errors = [max((float(x) for pair in messages.get(role, []) for x in pair.values()), default=None)
                  for role in ("pred", "future", "obs")]
    local = raw.get("stock_return_local_check")
    local_error = raw.get("stock_return_local_check_error", "").strip().splitlines()
    recheck = (f"Passed ({fmt(local['local_message_residual'])})" if local else
               local_error[-1] if local_error else "Unavailable")
    return [link(path, root, f"{seed} / {tick} / {source}"), "Pass" if passed else "Fail",
            esc(same.get("certified")), esc(raw.get("stock_status")), *map(fmt, [
                same.get("root"), same.get("canonical_FE_change"), raw.get("stock_reported_root"),
                raw.get("FE_error"), max(post, default=None), raw.get("coolant_error_K"), *max_errors]),
            esc(recheck)]


def replay_tables(replay, episodes, root, runtime, hashes):
    summary_path = replay / "summary.json"
    if not summary_path.exists():
        return "<p>Offline replay summary unavailable. Identical-input clipping and solver parity are unmeasured.</p>"
    summary = read(summary_path)
    replay_runtime = read(replay / summary["runtime"])
    if runtime:
        for key in ("library_tree", "pcgym_source"):
            require(replay_runtime[key] == runtime[key], replay, f"offline runtime {key}")
        for name, module in runtime["modules"].items():
            require(replay_runtime["modules"][name]["version"] == module["version"], replay,
                    f"offline version of {name}")
        measured = {key: digest for arm in hashes.values() for key, digest in arm.items()}
        offline = source_hashes(replay_runtime)
        for key, digest in measured.items():
            if key == ("cffg", "tools/cstr_paired_study.py") and key not in offline:
                continue
            require(offline.get(key) == digest, replay, f"offline source hash for {key}")
    roster = {(s, arm) for s in SEEDS for arm in ARMS}
    coverage_rows = summary["coverage"]
    require(len(coverage_rows) == len(roster) and {(r["seed"], r["arm"]) for r in coverage_rows} == roster,
            summary_path, "replay seed coverage")
    for r in coverage_rows:
        e = episodes[r["seed"], r["arm"]]
        manifest = e.directory / "manifest.json"
        require(r["manifest_sha256"] == (sha(manifest) if manifest.exists() else None)
                and r["decisions"] == len(e.decisions), summary_path, "replay manifest or decision count")
    selection = summary["selection"]
    selected = set()
    extrema = []
    for kind, field in (("raw", "proposed_kelvin"), ("applied", "applied_kelvin")):
        differences = [(seed, d["tick"], d[field] - h[field]) for seed in SEEDS
                       for d, h in zip(*(episodes[seed, arm].decisions for arm in ARMS))
                       if all(isinstance(r[field], (int, float)) and np.isfinite(r[field]) for r in (d, h))]
        best = max(differences, key=lambda x: abs(x[2]), default=None)
        recorded = selection[f"largest_{kind}"]
        require(len(differences) == selection[f"comparable_{kind}"] and
                best == ((recorded["seed"], recorded["tick"], recorded["delta_K"]) if recorded else None),
                summary_path, f"largest observed {kind} difference")
        if best:
            selected.add(best[:2])
            extrema.append([kind.title(), len(differences), *map(fmt, best)])
        else:
            extrema.append([kind.title(), 0, "—", "—", "—"])
    for event, label in (("first_miss", "miss"), ("first_failed_check", "verification-failure")):
        for arm in ARMS:
            first = next(((s, episodes[s, arm].first[label]) for s in SEEDS if label in episodes[s, arm].first), None)
            record = selection[event][arm]
            require(first == ((record["seed"], record["tick"]) if record else None), summary_path, event)
            if first:
                selected.add(first)
    diagnostics, controls = [], []
    seeds = [item["seed"] for item in summary["stock"]]
    require(sorted(seeds) == list(SEEDS), summary_path, "stock replay seed roster")
    parity_count = 0
    stock_count = 0
    local_rejections = 0
    for entry in summary["stock"]:
        seed = entry["seed"]
        e = episodes[seed, "directed"]
        ticks = [r["tick"] for r in entry["ticks"]]
        fixed = {0, 5, 20, 25, 40} | set(e.first.values())
        require(len(ticks) == len(set(ticks)) and fixed <= set(ticks), summary_path, "stock tick coverage")
        for item in entry["ticks"]:
            tick = item["tick"]
            decision = e.directory / f"decision-{tick:02}.json"
            digest = sha(decision) if decision.exists() else None
            require(digest == item["decision_sha256"], summary_path, "directed decision digest")
            diagnostics.append(stock_result(replay / item["raw"], seed, tick, "directed", digest,
                                            item["historical_parity"], root))
            stock_count += 1
            parity_count += item["historical_parity"]
            local_rejections += item.get("stock_return_local_check_passed") is False
            if digest and not item["historical_parity"]:
                selected.add((seed, tick))
        control = replay / entry["negative_controls_raw"]
        data = read(control)
        require(data["passed"] == entry["negative_controls_passed"], control, "negative control verdict")
        controls.append(link(control, root, f"Seed {seed} {'passed' if data['passed'] else 'failed'}"))
    windows, inputs, same_clips = [], [], Counter()
    expected_windows = {(s,t,arm) for s,t in selected for arm in ARMS if t < len(episodes[s, arm].decisions)}
    seen = set()
    for item in summary["windows"]:
        seed, tick, source = (item[k] for k in ("seed", "tick", "source"))
        identity = (seed, tick, source)
        require(identity in expected_windows and identity not in seen, summary_path, f"unexpected window {identity}")
        seen.add(identity)
        common_path = replay / item["raw"]
        common = read(common_path)
        decision = episodes[seed, source].directory / f"decision-{tick:02}.json"
        digest = sha(decision)
        require(item["decision_sha256"] == common["decision_sha256"] == digest and common["actual"] == read(decision),
                common_path, "source decision or digest")
        passed = common["self_replay"]["passed"]
        require(common["self_replay"] == item["self_replay"], common_path, "self-replay verdict")
        for arm in ARMS:
            action, raw = item[arm], common[arm]
            require(all(action[k] == raw[k] for k in ("proposed_normalized", "proposed_kelvin",
                        "clipped_normalized", "status", "error")), common_path, f"{arm} offline readout")
            require(action["checked"] == (raw["error"] is None and raw["checks"] is not None),
                    common_path, f"{arm} checked status")
            for key in ("history", "observation", "masks", "data", "preview_indices", "preview_Ca"):
                require(raw[key] == common["actual"][key], common_path, f"{arm} common {key}")
        d, h = (item[arm] for arm in ARMS)
        usable = passed and all(a["checked"] and a["clipped_normalized"] is not None for a in (d, h))
        bound = clip_bound(d) if usable and clip_bound(d) == clip_bound(h) else None
        if usable:
            same_clips["comparable"] += 1
            same_clips[bound] += 1
            close(item["raw_delta_K"], d["proposed_kelvin"] - h["proposed_kelvin"], common_path, "raw delta")
            close(item["clipped_delta_K"], 3.5 * (d["clipped_normalized"] - h["clipped_normalized"]),
                  common_path, "clipped delta")
        stock_path = replay / item["stock_raw"]
        if source == "host":
            diagnostics.append(stock_result(stock_path, seed, tick, source, sha(common_path), item["historical_parity"], root))
        else:
            require(read(stock_path)["historical_parity"] == item["historical_parity"], stock_path, "window verdict")
        comparison = item.get("actual_window_difference") or {}
        inputs.append([link(common_path, root, f"{seed} / {tick} / {source}"),
                       esc(comparison.get("same_bound_data")), fmt(comparison.get("observation_max_abs")),
                       fmt(comparison.get("history_observation_max_abs")), fmt(comparison.get("history_action_max_abs")),
                       esc(item.get("host_solver_status")), fmt(item.get("directed_local_residual")),
                       fmt(item.get("directed_local_FE")), fmt(item.get("stock_FE")),
                       esc(", ".join(item.get("stock_findings", [])) or "—")])
        windows.append([link(common_path, root, f"{seed} / {tick} / {source}"), esc(", ".join(item["reasons"])),
                        "Pass" if passed else "Fail", fmt(d["proposed_kelvin"]), fmt(h["proposed_kelvin"]),
                        fmt(item["raw_delta_K"]), fmt(item["clipped_delta_K"]),
                        {1: "High", -1: "Low"}.get(bound, "Neither" if usable else "Unavailable"),
                        link(stock_path, root, "Pass" if item["historical_parity"] else "Fail")])
    require(seen == expected_windows, summary_path, "selected common-input windows missing")
    comparisons = []
    for row in summary.get("comparisons", []):
        keys = ("actual_raw_delta_K", "actual_applied_delta_K", "controller_delta_on_directed_input_K",
                "controller_delta_on_host_input_K", "directed_change_between_inputs_K", "host_change_between_inputs_K",
                "both_self_replays_passed")
        comparisons.append([f"{row['seed']} / {row['tick']}", *[esc(row[k]) for k in keys]])
    return (f"<p>{link(summary_path, root, 'Replay summary')} · "
            f"{link(replay / summary['runtime'], root, 'Replay runtime')}. "
            "Selection ranks absolute differences on actual trajectories, breaking ties by earliest seed and tick. "
            "The table gives signed directed minus host differences.</p>" +
            table(["Command","Comparable ticks","Seed","Tick","Δ K"], extrema, "Largest observed command differences") +
            f"<p>Both checked proposals clipped to 295 K on {same_clips[-1]} and to 302 K on {same_clips[1]} "
            f"of {same_clips['comparable']} comparable selected windows. There were {len(windows)} replayed windows. "
            "Each seed and tick can contribute both source rows, even if their inputs coincide. Each row uses the source trajectory’s observation, applied history and preview. On host inputs, replay "
            "uses the archived host plan. On directed inputs, it reconstructs the host plan from the applied prefix. "
            "Eligibility requires a passing source self-replay and two checked proposals. Selection favors extreme "
            "differences and failures. Offline proposals have no observed plant response or deadline.</p>" +
            table(["Seed / tick / source","Selected for","Self-replay","Directed K","Host K","Raw Δ K",
                   "After clip Δ K","Both clip to","Stock parity"], windows, "Controller proposals on identical inputs") +
            table(["Seed / tick / source","Timed bound inputs identical","Max current Ca/T gap",
                   "Max past Ca/T gap","Max past action gap","Host solve","Local directed residual",
                   "Directed FE","Stock FE","Stock findings"], inputs,
                  "Timed input gaps in normalized units and offline solver diagnostics") +
            table(["Seed / tick","Actual raw Δ","Actual applied Δ","Both on directed input Δ",
                   "Both on host input Δ","Directed change across inputs","Host change across inputs",
                   "Both self-replays passed"], comparisons, "Paired trajectory and common-input contrasts (K)") +
            "<p>Within-controller changes subtract the proposal on the host trajectory input from the proposal "
            "on the directed trajectory input. The observations, histories and host warm plans may differ together.</p>" +
            f"<h3>Stock engine on directed candidates</h3><p>On directed trajectories, {parity_count} of {stock_count} "
            f"stock comparisons passed historical parity. The returned stock state failed the separate local recheck "
            f"on {local_rejections} of those {stock_count} comparisons. The historical verdict combines the local check, "
            "same-state stock certificate, convergence, posterior error ≤ 0.001 and scaled free-energy error ≤ 0.000001. "
            "Each message column gives the largest relative natural-parameter or precision error across sites. "
            "The separate local recheck evaluates the returned stock state. Each row links to the full residuals, "
            "message directions, diagnostics and termination reason.</p>" +
            table(["Seed / tick / source","Historical","Same state","Stock status","Candidate root","Successor FE change",
                   "Stock root","Scaled FE error","Max whitened posterior error","Coolant error K","Forward message",
                   "Backward message","Observation message","Stock local recheck"], diagnostics, "Offline stock comparisons") +
            "<p>Tick 5 negative controls (stale history, preview and action belief) · " + " · ".join(controls) + "</p>")



def return_chart(episodes, root):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 3), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    for i, (arm, color, marker) in enumerate(zip(ARMS, ("#12616a", "#a54520"), ("o", "s"))):
        xs = [s for s in SEEDS if episodes[s, arm].complete]
        ys = [episodes[s, arm].metrics["return_sum"] for s in xs]
        ax.scatter(np.asarray(xs) + (i - .5) * .16, ys, color=color, marker=marker, label=arm.title(), zorder=3)
    for seed in SEEDS:
        pair = [episodes[seed, arm] for arm in ARMS]
        if all(e.complete for e in pair):
            ax.plot([seed - .08, seed + .08], [e.metrics["return_sum"] for e in pair], color="#858585", lw=1)
    ax.set(xlabel="Held-out seed", ylabel="Native episode return", xticks=SEEDS, xlim=(.5, 10.5))
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncols=2)
    figure.savefig(root / "paired-returns.png", format="png", dpi=150)


def write_report(root, pilot_path, replay):
    pilot = read(pilot_path)
    run = root / "run"
    study, episodes, runtime, hashes, noise = load_study(run, pilot)
    replay_section = replay_tables(replay, episodes, root, runtime, hashes)
    draws = {}
    outcomes, estimates = outcome_tables(episodes, OUTCOMES, draws)
    timings, timing_estimates = outcome_tables(episodes, LATENCY, draws, timing=True)
    complete = sum(e.complete for e in episodes.values())
    pairs = sum(all(episodes[s, arm].complete for arm in ARMS) for s in SEEDS)
    _, _, delta, low, high, _ = estimates["return_sum"]
    if pairs:
        direction = "higher" if delta > 0 else "lower" if delta < 0 else "equal"
        conclusion = (f"The directed controller’s mean native return was {direction} across {pairs} complete pairs. "
                      f"The mean directed minus host difference was {fmt(delta)} "
                      f"(95% paired seed bootstrap interval {fmt(low)} to {fmt(high)}). "
                      f"Positive / tied / negative seed differences: {estimates['return_sum'][-1]}.")
    else:
        conclusion = "No complete seed pairs were available to estimate the mean difference in native return."
    warm_d, warm_h = timing_estimates["warm_median_s"][:2]
    warm_text = ""
    if warm_d is not None:
        warm_text = (f"<p>For repeated decisions, the paired mean of the within-seed median time to a ready "
                     f"command was {warm_d*1000:.3f} ms for directed and {warm_h*1000:.3f} ms for host.</p>")
    env = study["protocol"]
    provenance = ("No arm manifest available." if runtime is None else
                  f"{esc(runtime['hardware'])}, {esc(runtime['os'])}, {esc(', '.join(runtime['devices']))}. "
                  f"PCGym revision <code>{esc(runtime['pcgym_source']['vcs_info']['commit_id'])}</code>.")
    switch_text = ("The descriptive cut was fixed before examining held-out returns. Switch windows include "
                   "scored steps within two steps of 20 or 40, inclusive (decision ticks 17–21 and 37–41). "
                   "Other decisions form the comparison group. Low and high clips count checked proposals "
                   "strictly outside 295–302 K, including late proposals. Excess is the absolute distance "
                   "in kelvin from the clipped proposal. Applied saturation also counts commands exactly "
                   "on a bound and held commands. Counts pool available decisions, including recorded prefixes.")
    command = "uv run --frozen python tools/cstr_paired_report.py " + shlex.quote(str(root))
    if pilot_path != PILOT:
        command += " --pilot-manifest " + shlex.quote(str(pilot_path))
    if replay != root / "replay":
        command += " --replay " + shlex.quote(str(replay))
    study_cmd = "uv run --frozen python tools/cstr_paired_study.py run --output NEW_ROOT/run"
    replay_cmd = "uv run --frozen python tools/cstr_paired_replay.py NEW_ROOT --output NEW_ROOT/replay"
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paired reactor control study</title><style>
*{{box-sizing:border-box}}body{{margin:0;color:#172523;background:#fbfcfa;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:86rem;margin:auto;padding:2rem clamp(1rem,4vw,4rem) 4rem}}h1,h2,h3{{line-height:1.2}}
h1{{font-size:clamp(2rem,5vw,3.5rem);margin-bottom:.4em}}h2{{border-top:1px solid #aebfba;padding-top:1em;margin-top:1.8em}}
p{{max-width:85ch}}a{{color:#075c62}}nav a{{display:inline-block;margin:.2em 1em .2em 0}}
.scroll{{overflow-x:auto;margin:1rem 0}}table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:.35em .7em;text-align:right;border-bottom:1px solid #d7deda}}
thead th{{background:#dcebe7}}th:first-child,caption{{text-align:left}}caption{{font-weight:650;padding:.4em 0}}
summary{{cursor:pointer}}details{{margin:.35em 0}}pre{{overflow-x:auto;padding:1em;background:#e9efec}}
img{{max-width:100%;height:auto}}figure{{margin:1rem 0}}code{{overflow-wrap:anywhere}}
</style></head><body><main><h1>Paired reactor control</h1>
<p>{complete} of 20 episodes completed · {pairs} complete pairs · held-out seeds 1–10</p>
<p>{conclusion}</p><nav aria-label="Report sections">
<a href="#outcomes">Outcomes</a><a href="#latency">Latency</a><a href="#actions">Clipping</a>
<a href="#replay">Offline replay</a><a href="#coverage">Episode evidence</a><a href="#method">Scope and reproduction</a></nav>
<section id="outcomes"><h2>Closed-loop outcomes</h2>
<p>Higher native return and lower tracking error indicate better tracking. Return is the sum of native rewards.
Each reward is −1000 times the squared error in the next true Ca against its reward-index setpoint.
RMSE is root mean squared error and MAE is mean absolute error. Box excursions count individual
steps outside either physical Ca or temperature bound. Input excursions use the noisy normalized observations.</p>
<p>Δ is directed minus host. The last column counts positive, exactly zero and negative paired differences.
Resampling uses 10,000 draws of complete seed pairs with replacement (NumPy default_rng, seed 9806).
Intervals are the 2.5th and 97.5th percentiles of the paired mean. Outcomes share draws. Each expandable
table includes individual seeds and observed prefixes. Only completed pairs contribute outcome differences
and means. A missing value is shown as a dash.</p>
<figure><a href="paired-returns.png"><img src="paired-returns.png" width="800" height="300"
alt="Native return by controller and held-out seed. Exact values follow in the native episode return table."></a>
<figcaption>Completed episodes by seed. A line connects the two controllers where both finished.</figcaption></figure>
{outcomes}{reward_profile(episodes, draws, delta)}
{first_step_intervention(episodes, pilot["parameters"], root, runtime)}</section><section id="latency"><h2>Latency boundaries</h2>
<p>All times are wall seconds. Launch starts before uv and includes interpreter startup, imports,
controller construction and plant reset. First checked can occur after tick 0, and a checked proposal can be late.
Command ready includes readout, clipping and hold selection. Plant call ends immediately before env.step.
Cold solve is the recorded first solve boundary for each implementation. Through clip starts with assembling
the observed window and ends at the recorded ready time, including a failed attempt where applicable.
Repeated times are within-episode median and p95 over ticks after tick 0. Means and intervals give each seed
equal weight and use pairs with both recorded boundaries. {esc(env['compute'])}.
Directed decisions start a new solve each tick. The host reuses its shifted action plan.</p>{warm_text}{launch_table(episodes)}{timings}</section>
<section id="actions"><h2>Clipping by controller and phase</h2><p>{switch_text}</p>{clipping(episodes)}
<p>Aligned counts describe the same seed and decision time. Once applied actions differ, the two trajectories
can have different observations and applied histories. These counts do not identify the effect of clipping on return.</p></section>
<section id="replay"><h2>Offline replay</h2>{replay_section}</section>
<section id="coverage"><h2>Episode evidence</h2>
<p>The audit reconstructed each available scored transition, applied history, observation, input mask, preview,
reward, error, bound flag and episode total. It checked the applied command against the recorded check and
deadline. It recovered standardized noise as (denormalized next observation − next true state) divided by
(0.001 × next true state), using the simulator’s float32 observation box. {len(noise)//2} matched next observations
were compared across controllers. Maximum absolute component difference was {fmt(max(noise, default=None))}.
Available samples were also checked against the reset seed’s NumPy noise stream (absolute tolerance 1e−9).</p>
{coverage(episodes, root)}</section><section id="method"><h2>Scope and reproduction</h2>
<p>{esc(env['environment'])}. {esc(env['schedule'])}. {esc(env['observation'])}. Both controllers had
{esc(env['knowledge'])}. Directed controller · {esc(env['directed'])}. Host controller · {esc(env['comparator'])}.
{esc(env['budget'])}. Actuator limits were applied at readout.</p>
<p>The experiment covers one simulated reactor, one known model and one setpoint schedule.
The intervals describe seed variability for this configuration. They are marginal for each measure.
Latency reflects this machine and software. Offline solver checks concern selected saved windows.
Observed differences combine controller design, approximation, computation and closed-loop feedback.</p>
<p>{provenance} Study revision <code>{esc(study['revision'])}</code>.
{link(run/'study.json', root, 'Study protocol and source hashes')} · {link(run/'processes.json', root, 'Process outcomes')}
· {link(pilot_path, root, 'Seed-0 pilot manifest')}. Seed 0 supplies the protocol and stays outside the estimates.
Each arm manifest records runtime, package versions, hardware, source hashes and command arguments.</p>
<p>From the study checkout, reproduce in a fresh directory or regenerate this report from its saved files.</p>
<pre><code>{esc(study_cmd)}
{esc(replay_cmd)}
{esc(command)}</code></pre></section></main></body></html>"""
    with TemporaryDirectory(dir=root) as staging:
        stage = Path(staging)
        return_chart(episodes, stage)
        (stage / "index.html").write_text(content, encoding="utf-8")
        for name in ("paired-returns.png", "index.html"):
            (stage / name).replace(root / name)
    print(f"{root / 'index.html'} ({complete}/20 complete episodes, {pairs}/10 complete pairs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=ROOT, help="artifact root containing run/study.json")
    parser.add_argument("--pilot-manifest", type=Path, default=PILOT, help="frozen seed-0 pilot manifest")
    parser.add_argument("--replay", type=Path, help="offline replay directory (default ROOT/replay)")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    replay = (args.replay or root / "replay").expanduser().resolve()
    if args.replay and not (replay / "summary.json").is_file():
        parser.error(f"missing replay summary in {replay}")
    try:
        write_report(root, args.pilot_manifest.expanduser().resolve(), replay)
    except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
        parser.exit(1, f"Report generation failed: {exc}\n")


if __name__ == "__main__":
    main()
