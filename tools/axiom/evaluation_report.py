"""Audit released AXIOM campaign evidence and write summary.json and index.html offline."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

import numpy as np

import gameworld_replay
import gameworld_replay_check
import reference_protocol as reference


HERE = Path(__file__).resolve().parent
STAGES = {
    "development": ("Development pilot", "pilot_passed.json"),
    "pre_evaluation": ("Engineering qualification", "pre_evaluation_passed.json"),
    "evaluation": ("Primary evaluation", "completed.json"),
}
# Published tools_sha256 in reference-v1/protocol.json, archived before the campaign.
REFERENCE_FILES = {
    "reference_protocol.json": "8ae51014f094c8d8ff2ef4069ea6d6ec83598474cec0526d88b9c85a6a318370",
    "reference_protocol.py": "2d3669b3232330ec105535a153d22df8a5b047553d5fe2431ca5947d4f2577cf",
    "gameworld_replay.py": "c08629e1ea0456ad68529510d2fa0bc6ba6a7a811c0dfa0aac06c9b77b3960ac",
    "gameworld_replay_check.py": "a905ba0b9eb378bc773d8a4c8e37e9ced259bfeb9d585dffab3f4aa0aa574cdf",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_json(path):
    data = path.read_bytes()
    value = json.loads(data)
    require(isinstance(value, dict), f"{path.name} needs a JSON object")
    return value, digest(data)


def inventory(study, protocol):
    expected = {s: list(reference.seed_pairs(protocol, s)) for s in STAGES}
    registered = dict(arm="untuned", dtype="float32", precision="compensated", horizon=32,
                      candidates=512, draws=3, reduction_cadence=500)
    for key, value in registered.items():
        require(study[key] == value, f"study {key} differs from registered arm")
    require(study["stages"] == expected, "study run list differs from frozen seeds, steps or game order")
    require(study["protocol_sha256"] == REFERENCE_FILES["reference_protocol.json"], "protocol hash mismatch")
    require(study["reference_sha256"] == "ef5141f70e4ef6792a40e7ade7086109fc9e04b68afed19764294f9b53282caf",
            "archived reference manifest mismatch")
    files = study["source_files"]
    require(isinstance(files, dict) and files and all(is_digest(v) for v in files.values()),
            "invalid source inventory")
    source = digest(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    require(study["source_sha256"] == source, "source inventory digest mismatch")
    for name in ("evaluation_report.py", *REFERENCE_FILES):
        key = f"tools/axiom/{name}"
        if name != "gameworld_replay_check.py" or key in files:
            require(files.get(key) == reference.sha(HERE / name), f"scoring source differs from study: {key}")
    return expected


def step_stream(path, sealed, ident):
    """Read a byte-bounded snapshot. A live writer may have an unfinished last line."""
    rewards, decision, observation = [], [], []
    actions = Counter()
    endings = reductions = proposals = accepted = spread = 0
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = os.fstat(stream.fileno()).st_size
        while remaining:
            line = stream.readline(remaining)
            require(line, "step stream truncated during read")
            remaining -= len(line)
            if not sealed and not remaining and not line.endswith(b"\n"):
                break
            hashed.update(line)
            row = json.loads(line)
            tick = len(rewards)
            require(type(row["tick"]) is int and row["tick"] == tick < ident["steps"], f"invalid tick at {tick}")
            action = row["action"]
            require(type(action) is int and 0 <= action < len(gameworld_replay.ACTIONS[ident["game"]]),
                    f"invalid action at {tick}")
            require(is_digest(row["frame_sha256"]), f"invalid frame digest at {tick}")
            require(type(row["terminated"]) is bool and type(row["truncated"]) is bool, f"invalid flags at {tick}")
            require(finite_number(row["reward"]), f"invalid reward at {tick}")
            for key in ("observation_s", "decision_s"):
                require(finite_number(row[key]) and row[key] >= 0, f"invalid {key} at {tick}")
            require(row["reduction_performed"] is ((tick + 1) % 500 == 0), f"reduction clock mismatch at {tick}")
            require(row["valid"] is True, f"invalid observation at {tick}")
            native = row["game"]
            fields = dict(tick=tick, applied=action, episode=endings, reward=row["reward"], event="step",
                          rgb_sha256=row["frame_sha256"], terminated=row["terminated"], truncated=row["truncated"])
            require(all(native[k] == v for k, v in fields.items()), f"native step disagrees at {tick}")
            done = row["terminated"] or row["truncated"]
            require((row["reset"] is not None) == done, f"terminal/reset mismatch at {tick}")
            if done:
                reset = row["reset"]
                require(reset["tick"] == tick+1 and reset["episode"] == endings+1 and reset["event"] == "reset"
                        and is_digest(reset["rgb_sha256"]), f"reset receipt disagrees at {tick}")
            if row["reduction_performed"]:
                reduction = row["reduction"]
                require(reduction["step"] == tick+1 and reduction["status"] == "ok", f"invalid reduction at {tick}")
                reductions += 1
                proposals += reduction["pairs"]
                accepted += reduction["accepted"]
            spread += row["decision"]["total_score_range"] > 0
            rewards.append(row["reward"])
            decision.append(row["decision_s"])
            observation.append(row["observation_s"])
            actions[action] += 1
            endings += done
    measures = dict(observed_steps=len(rewards), observed_return=math.fsum(rewards), episode_endings=endings,
                    reductions=reductions, proposed_pairs=proposals, accepted_merges=accepted,
                    decisions_with_score_spread=spread,
                    action_counts=dict(sorted(actions.items())), steps_sha256=hashed.hexdigest(),
                    first_decision_s=decision[0] if decision else None,
                    first_observation_s=observation[0] if observation else None,
                    subsequent_decision_median_s=statistics.median(decision[1:]) if len(decision) > 1 else None,
                    subsequent_observation_median_s=statistics.median(observation[1:]) if len(observation) > 1 else None)
    return measures, rewards


def read_run(root, stage, ident, source):
    name = f"{ident['game']}-{ident['model_seed']}"
    folder = root / stage / name
    row = {**ident, "path": f"{stage}/{name}", "status": "missing", "observed_steps": 0}
    result_file, tape = folder / "result.json", folder / "steps.jsonl"
    try:
        require(not any(p.is_symlink() for p in (root / stage, folder, result_file, tape)),
                "stage, run and input files must be local archive paths")
        result = None
        if result_file.exists():
            result, row["result_sha256"] = read_json(result_file)
            for key in ("game", "game_seed", "model_seed"):
                require(type(result[key]) is type(ident[key]) and result[key] == ident[key], f"{key} mismatch")
            if result.get("status") == "failed":
                row["applied_before_failure"] = result.get("applied_before_failure", {})
            require(result["stage"] == stage, "result stage mismatch")
            require(result["source_sha256"] == source, "source digest mismatch")
            require(result["status"] in ("complete", "failed"), "unknown terminal status")
        if not tape.exists():
            if result is None:
                return row, []
            if (result["status"] == "failed" and result["steps"] == 0 and result["return_sum"] == 0
                    and result["steps_sha256"] == digest(b"")):
                row.update(status="failed", reason=result.get("reason", "worker failed before first record"))
                return row, []
        metrics, rewards = step_stream(tape, result is not None, ident)
        row.update(metrics)
        if result is None:
            row["status"] = "in_progress"
            return row, []
        require(result["steps_sha256"] == metrics["steps_sha256"], "step tape digest mismatch")
        require(type(result["steps"]) is int and result["steps"] == len(rewards), "step count mismatch")
        require(finite_number(result["return_sum"]) and result["return_sum"] == metrics["observed_return"],
                "raw return mismatch")
        if result["status"] == "failed":
            row.update(status="failed", reason=result.get("error") or result.get("reason") or "worker failed")
            return row, []
        require(len(rewards) == ident["steps"], "incomplete result marked complete")
        row["process_s"] = result["process_s"]
        require(finite_number(row["process_s"]) and row["process_s"] >= 0, "invalid worker elapsed time")
        row["initialization_s"] = result["initialization_s"]
        row["resumed_from"] = result["resumed_from"]
        require(finite_number(row["initialization_s"]) and 0 <= row["initialization_s"] <= row["process_s"],
                "invalid initialization time")
        require(type(row["resumed_from"]) is int and 0 <= row["resumed_from"] <= ident["steps"],
                "invalid worker restart tick")
        if stage == "pre_evaluation":
            row["resume_matched_steps"] = result["resume_matched_steps"]
            require(result["resume_matched_steps"] == len(rewards) - 500, "incomplete continuation comparison")
        score = gameworld_replay.reward_scores(rewards, 1000)
        row.update(status="complete", last_1000_mean=score["rolling_mean"][-1],
                   trailing_1000=score["rolling_mean"])
        return row, rewards
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        row.update(status="invalid", reason=str(exc))
        return row, []


def verify_marker(path, stage, rows, source):
    require(not path.is_symlink(), f"linked {path.name}")
    marker, fingerprint = read_json(path)
    require(marker["stage"] == stage, f"{path.name} stage mismatch")
    require(marker["source_sha256"] == source, f"{path.name} source mismatch")
    require(all(row["status"] == "complete" for row in rows), f"{path.name} covers unfinished or invalid runs")
    hashes = {f"{r['game']}-{r['model_seed']}": r["result_sha256"] for r in rows}
    require(marker["runs"] == hashes, f"{path.name} result inventory/digest mismatch")
    return fingerprint


def audit(root):
    for name, expected in REFERENCE_FILES.items():
        require(reference.sha(HERE / name) == expected, f"frozen scoring input changed: {name}")
    gameworld_replay_check.scoring_check()
    protocol = json.loads((HERE / "reference_protocol.json").read_bytes())
    expected = {s: list(reference.seed_pairs(protocol, s)) for s in STAGES}
    summary = {"status": "in_progress", "stages": {}, "primary": None, "margin": protocol["scoring"]["margin"],
               "study_sha256": None, "source_sha256": None, "reporter_sha256": reference.sha(Path(__file__)),
               "reference_sha256": REFERENCE_FILES, "historical": protocol["table1"], "games": protocol["games"]}
    study_path = root / "study.json"
    released = False
    if study_path.exists():
        try:
            require(not study_path.is_symlink(), "linked study manifest")
            study, summary["study_sha256"] = read_json(study_path)
            expected = inventory(study, protocol)
            summary["source_sha256"] = study["source_sha256"]
            summary["revision"] = study["revision"]
            summary["reference_manifest_sha256"] = study["reference_sha256"]
            released = True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            summary.update(status="invalid", error=str(exc))
    records = []
    for stage, (_, receipt) in STAGES.items():
        entry = {"planned": len(expected[stage]), "steps_per_run": expected[stage][0]["steps"],
                 "status": "locked", "counts": {}, "runs": [], "receipt": None}
        summary["stages"][stage] = entry
        if not released:
            continue
        rows = entry["runs"]
        for ident in expected[stage]:
            row, rewards = read_run(root, stage, ident, study["source_sha256"])
            rows.append(row)
            if stage == "evaluation" and row["status"] == "complete":
                records.append({**ident, "arm": "untuned", "condition": "base", "status": "ok", "rewards": rewards})
        counts = Counter(row["status"] for row in rows)
        entry["counts"] = dict(counts)
        released = False
        if counts["invalid"] or counts["failed"]:
            entry["status"] = summary["status"] = "invalid" if counts["invalid"] else "failed"
        elif counts["complete"] == len(rows):
            entry["status"] = "awaiting_receipt"
        else:
            entry["status"] = "in_progress"
        marker = root / receipt
        if marker.exists():
            try:
                fingerprint = verify_marker(marker, stage, rows, study["source_sha256"])
            except (OSError, KeyError, ValueError, TypeError) as exc:
                entry.update(status="invalid", error=str(exc))
                summary["status"] = "invalid"
            else:
                entry.update(status="complete", receipt=receipt, receipt_sha256=fingerprint)
                released = True
    if released:
        primary = reference.score_primary(protocol, records)
        require(primary["scored"] == len(expected["evaluation"]) and primary["macro"] is not None,
                "frozen primary scorer rejected the archive")
        for game, metrics in primary["games"].items():
            curves = np.array([r["trailing_1000"] for r in summary["stages"]["evaluation"]["runs"]
                               if r["game"] == game])
            metrics["trailing_1000"] = {"first_ending_action": 1000, "mean": curves.mean(axis=0).tolist(),
                                        "sample_sd": curves.std(axis=0, ddof=1).tolist()}
        summary.update(status="complete", primary=primary)
    return summary


def table(headers, rows):
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def number(value):
    return "—" if value is None else f"{value:,.3f}".rstrip("0").rstrip(".")


def interval(values):
    return f"[{number(values[0])}, {number(values[1])}]"


STATUS = {"locked": "Locked", "in_progress": "In progress", "awaiting_receipt": "Awaiting signed receipt",
          "complete": "Complete", "failed": "Failed", "invalid": "Invalid evidence", "missing": "Awaiting run"}
VERDICT = {"within_margin": "Within margin", "above_margin": "Above margin", "below_margin": "Below margin",
           "inconclusive": "Inconclusive"}


def stage_section(stage, entry):
    title = STAGES[stage][0]
    if not entry["runs"]:
        return ""
    rows, timings = [], []
    for run in entry["runs"]:
        folder = run["path"]
        links = []
        if "steps_sha256" in run:
            links.append(f'<a href="{folder}/steps.jsonl">Steps</a>')
        if "result_sha256" in run:
            links.append(f'<a href="{folder}/result.json">Result</a>')
        state = STATUS[run["status"]]
        if "reason" in run:
            state += " · " + escape(str(run["reason"]))
        if run["observed_steps"]:
            timings.append((run["game"], str(run["model_seed"]), number(run.get("resumed_from")),
                            number(run.get("process_s")), number(run.get("initialization_s")),
                            *[number(run[k]) for k in ("first_decision_s", "subsequent_decision_median_s",
                                                      "first_observation_s", "subsequent_observation_median_s")]))
        rows.append((run["game"], str(run["model_seed"]), str(run["game_seed"]), state,
                     f'{run["observed_steps"]:,} / {run["steps"]:,}', number(run.get("observed_return")),
                     number(run.get("last_1000_mean")), " · ".join(links) or "—"))
    contents = table(("Game", "Model seed", "Game seed", "Status", "Actions", "Reward to date",
                      "Last 1,000 mean", "Evidence"), rows)
    timing_table = ""
    if timings:
        timing_table = "<h3>Decision and observation times</h3><p>Seconds per real action. Subsequent medians " \
                       "cover actions after the first and can include later compilation. Worker wall and initialization cover the final process attempt. Initialization includes the simulator and actor and ends before any checkpoint restore.</p>" + table(
            ("Game", "Seed", "Resume tick", "Worker wall (s)", "Initialization (s)", "First decision", "Subsequent median", "First update", "Subsequent median"),
            timings)
    telemetry = [r for r in entry["runs"] if "reductions" in r]
    activity = ""
    if telemetry:
        activity = (f"<p>{sum(r['episode_endings'] for r in telemetry):,} episode endings. "
                    f"{sum(r['reductions'] for r in telemetry):,} scheduled reductions, "
                    f"{sum(r['proposed_pairs'] for r in telemetry):,} proposed pairs and "
                    f"{sum(r['accepted_merges'] for r in telemetry):,} accepted merges. "
                    f"Candidate plans had distinct scores at "
                    f"{sum(r['decisions_with_score_spread'] for r in telemetry):,} decisions.</p>")
    return f'<details><summary>{title} · {len(rows)} assigned</summary>{activity}{contents}{timing_table}</details>'


def primary_section(primary):
    if primary is None:
        return ""
    macro = primary["macro"]
    body = ("The equal-game mean difference was <strong>" + number(macro["delta"]) + "</strong> raw reward units "
            f'with a simultaneous interval of {interval(macro["ci"])}. The population comparison is '
            f'<strong>{VERDICT[macro["verdict"]].lower()}</strong> against the ±10 margin. ')
    inside = sum(g["verdict"] == "within_margin" for g in primary["games"].values())
    body += f"{inside} of 10 per-game intervals are wholly within that margin."
    rows = []
    printed = []
    for game, data in primary["games"].items():
        historical = data["published_mean_sd"]
        rows.append((game, number(data["mean"]), number(data["sample_sd"]), number(historical[0]),
                     number(data["delta"]), interval(data["ci"]), VERDICT[data["verdict"]]))
        point = data["printed_target"]
        printed.append((game, number(point["delta"]), interval(point["ci"]), VERDICT[point["verdict"]]))
    point = macro["printed_target"]
    printed.append(("Equal-game mean", number(point["delta"]), interval(point["ci"]), VERDICT[point["verdict"]]))
    estimates = table(("Game", "Candidate mean", "Sample SD", "Published mean", "Difference",
                       "Population interval", "Result"), rows)
    targets = table(("Game", "Difference", "95% interval", "Result"), printed)
    return f'''<section><h2>Return against published AXIOM</h2><p>{body}</p>{estimates}
<p>Positive differences favor the candidate. The population intervals account for uncertainty in both ten-seed
means and rounding of the published values. Their approximate simultaneous coverage assumes independent,
approximately normal seed totals within each game. The equal-game interval averages the game endpoints.</p>
<details><summary>Comparison to the printed means as fixed targets</summary>
<p>These 95% intervals use variation across the ten new seeds. The equal-game estimate first averages
the ten games for each seed. Historical sampling uncertainty enters the separate population comparison above.</p>
{targets}</details></section>'''


CSS = """
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#192e35;background:#f5f6f2}
body{margin:0}main{background:white;max-width:68rem;margin:auto;padding:clamp(1rem,4vw,3rem)}
h1,h2{font-family:Georgia,serif;color:#14353d}h1{font-size:clamp(2rem,5vw,3rem);margin:.4em 0}
h2{margin:1.5em 0 .5em}p{line-height:1.55;max-width:80ch}.lead{font-size:1.15rem}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{padding:.55rem .7rem;border-bottom:1px solid #d4dfdf;text-align:right;font-variant-numeric:tabular-nums}
thead{background:#eaf0ee}th:first-child,td:first-child{text-align:left}summary{cursor:pointer;font-weight:600}
details{padding:.7rem 0}a{color:#086274}footer{margin-top:2rem;color:#506267}
"""


def render(summary, summary_sha):
    stages = summary["stages"]
    if summary["primary"]:
        intro = ("All 100 primary runs are complete and their raw rewards have been checked. "
                 "The comparison below uses ten fresh model seeds in each of ten games.")
    elif "error" in summary:
        intro = "The study manifest failed verification. " + escape(summary["error"])
    elif summary["study_sha256"] is None:
        intro = "Awaiting the study manifest. The planned campaign begins with one 10,000-action Explode pilot."
    else:
        current = next(k for k, v in stages.items() if v["status"] not in ("locked", "complete"))
        e = stages[current]
        done = e["counts"].get("complete", 0)
        steps = sum(r["observed_steps"] for r in e["runs"])
        issues = e["counts"].get("invalid", 0) + e["counts"].get("failed", 0)
        intro = (f'{STAGES[current][0]} · {STATUS[e["status"]]}. {done} of {e["planned"]} runs complete, '
                 f'{steps:,} of {e["planned"] * e["steps_per_run"]:,} real actions recorded.')
        if issues:
            intro += f" {issues} runs failed or contain invalid evidence."
        intro += " The primary comparison requires 100 verified returns and the final receipt."
    stage_rows = []
    for key, entry in stages.items():
        done = entry["counts"].get("complete", 0) if entry["runs"] else "—"
        status = STATUS[entry["status"]]
        if entry["receipt"]:
            status = f'<a href="{entry["receipt"]}">{status}</a>'
        if "error" in entry:
            status += " · " + escape(entry["error"])
        stage_rows.append((STAGES[key][0], f'{entry["steps_per_run"]:,}', str(entry["planned"]), str(done), status))
    historical = summary["historical"]
    arms = {"untuned": "AXIOM", "fixed_r": "Fixed radius", "no_bmr": "No reduction", "no_ig": "No novelty",
            "BBF": "BBF", "DreamerV3": "DreamerV3"}
    old = [(game, *[f"{historical[k][i][0]} ± {historical[k][i][1]}" for k in arms])
           for i, game in enumerate(summary["games"])]
    receipts = '<a href="study.json">Study manifest</a> · ' if summary["study_sha256"] else ""
    runs_html = "".join(stage_section(k, entry) for k, entry in stages.items())
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AXIOM on Gameworld</title>
<style>{CSS}</style></head><body><main><header><h1>AXIOM on Gameworld</h1><p class="lead">{intro}</p></header>
{primary_section(summary["primary"])}
<section><h2>Campaign progress</h2>
{table(("Stage", "Actions per run", "Assigned", "Verified runs", "Stage status"), stage_rows)}
<p>The untuned float32 agent uses compensated updates and evaluates 512 action strings over 32 imagined steps, with three draws per string. The qualification tests engineering behavior and exact continuation. Primary return sums every actual
post-action reward across episode resets. The trailing mean covers the latest 1,000 real actions.</p>
{runs_html}</section>
<section><h2>Published benchmarks</h2><p>Table 1 gives the historical mean ± standard deviation of total reward
over ten seeds. BBF, DreamerV3 and the AXIOM variants are published comparators. The campaign evaluates
the untuned candidate. The historical seed histories are unavailable for pairing.
<a href="https://arxiv.org/html/2505.24784v1#S3.T1">AXIOM paper, Table 1</a></p>
{table(("Game", *arms.values()), old)}</section>
<footer><p>{receipts}<a href="summary.json">Machine-readable summary</a> · SHA-256 <code>{summary_sha}</code></p>
<p>The summary includes per-run totals and full trailing means for verified runs. A completed primary includes
per-game curve means and sample standard deviations across the ten seeds.</p></footer></main></body></html>'''


def publish(root):
    directory = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        summary = audit(root)
        data = (json.dumps(summary, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        outputs = {"index.html": render(summary, digest(data)).encode(), "summary.json": data}
        with tempfile.TemporaryDirectory(dir=root) as staging:
            for name, content in outputs.items():
                (Path(staging) / name).write_bytes(content)
            for name in ("summary.json", "index.html"):
                (Path(staging) / name).replace(root / name)
        return summary["status"]
    finally:
        os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("root", type=Path, nargs="?", help="existing campaign archive")
    group.add_argument("--output", type=Path, help="existing campaign archive")
    args = parser.parse_args()
    status = publish(args.root or args.output)
    print(f"{status} · summary.json and index.html")
    return int(status in ("failed", "invalid"))


if __name__ == "__main__":
    raise SystemExit(main())
