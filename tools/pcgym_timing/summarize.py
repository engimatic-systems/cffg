"""Audit the pilot's saved outputs and render its timing and forecast report."""
import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np

from pilot import ARMS, save


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    cells = lambda tag, seq: "".join(f"<{tag}>{v}</{tag}>" for v in seq)
    return ("<table><thead><tr>" + cells("th", headers) + "</tr></thead><tbody>"
            + "".join("<tr>" + cells("td", row) + "</tr>" for row in rows) + "</tbody></table>")


def audit_episode(directory, expected):
    result = read(directory / "result.json")
    rows = [json.loads(line) for line in (directory / "ticks.jsonl").read_text().splitlines()]
    if (len(rows) != result["n"] or len(rows) != expected
            or [r["tick"] for r in rows] != list(range(expected)) or not result["complete"]):
        raise ValueError(f"Incomplete episode in {directory}")
    last = 0.
    for row in rows:
        held = row["status"] != "ok" or row["decision_s"] > 1.
        applied = last if held else float(np.clip(row["action"], -1, 1))
        if applied != row["applied"] or held != row["held"]:
            raise ValueError(f"Incorrect action or fallback at {directory}, tick {row['tick']}")
        last = applied
    if not np.isclose(sum(r["reward"] for r in rows), result["return_sum"]):
        raise ValueError(f"Return mismatch in {directory}")
    if sum(r["held"] for r in rows) != result["holds"]:
        raise ValueError(f"Hold count mismatch in {directory}")
    result["process_s"] = read(directory / "process.json")["wall_s"]
    result["status_counts"] = dict(Counter(r["status"] for r in rows))
    # PCGym's observation noise is multiplicative in physical units.
    states = np.vstack(([.8, 330.], [r["next_true_state"] for r in rows[:-1]]))
    physical_obs = [.7, 300.] + (np.array([r["obs"] for r in rows])+1)*[.15, 25.]
    return result, physical_obs / states - 1


def summarize(root, campaign):
    runs = root / campaign
    results = {}
    noise = {"short": [], "long": []}
    digests = set()
    truth = read(root / "forecast-truth.json")
    for arm in ARMS:
        directory = lambda phase: runs / f"{arm}-{phase}"
        q = read(directory("oracle") / "result.json")
        f = read(directory("forecast") / "result.json")
        if f["dataset_sha256"] != truth["sha256"]:
            raise ValueError(f"Wrong forecast origin data for {arm}")
        digests.add(f["dataset_sha256"])
        entry = results[arm] = dict(qualified=q["complete"], oracle=q["cases"], forecast=f,
                                   forecast_process_s=read(directory("forecast") / "process.json")["wall_s"])
        for phase, steps in (("short", 59), ("long", 599)):
            entry[phase], draws = audit_episode(directory(phase), steps)
            noise[phase].append(draws)
    noise_error = {}
    for phase, draws in noise.items():
        noise_error[phase] = float(np.max(abs(np.asarray(draws) - draws[0])))
        if noise_error[phase] > 1e-10 or len({r[phase]["rng_digest"] for r in results.values()}) != 1:
            raise ValueError(f"Unmatched external noise stream in {phase} episodes")
    if len(digests) != 1:
        raise ValueError("Mismatched forecast truth")
    projected = {phase: 30*sum(r[phase]["process_s"] for r in results.values())/3600
                 for phase in ("short", "long")}
    return dict(arms=results, noise_relative_difference=noise_error, hours_per_30_seeds=projected)


def report(root, campaign, summary):
    names = dict(cffg="CFFG", rxinfer="RxInfer", numpyro="NumPyro", gtsam="GTSAM")
    body = ["<h1>Four-library reactor timing pilot</h1>",
            "<p>One development seed per library on the Apple M2 Max. The 59- and 599-decision episodes "
            "use the same physical decision interval. Each native controller receives four prior measurements "
            "and applied actions, the current observation, and fifteen future setpoints. Late or failed "
            "decisions hold the previously applied action. The initial hold is 298.5 K.</p>"]
    arms = summary["arms"]
    timing_rows, control_rows, forecast_rows = [], [], []
    for key, r in arms.items():
        name = names[key]
        short, long = r["short"], r["long"]
        timing_rows.append([name, f'{short["process_s"]:.2f}', f'{long["process_s"]:.2f}',
                            f'{long["episode_s"]:.2f}', f'{long["warmup_s"]:.2f}',
                            *(f'{long["latency_s"][s]*1000:.2f}' for s in ("median","p95","max"))])
        for length, row in (("59",short), ("599",long)):
            link = f'{campaign}/{key}-' + ("short" if length=="59" else "long") + "/result.json"
            control_rows.append([f'<a href="{escape(link)}">{name}</a>', length,
                                 f'{row["return_sum"]:.4f}', f'{row["rmse_Ca"]:.5f}',
                                 row["missed"], row["failures"], row["holds"]])
        f = r["forecast"]
        h = f["summary"]["60"]
        metric = lambda value: "—" if value is None else f"{value:.5g}"
        forecast_rows.append([name, f'{f["n_scored"]}/{f["n_origins"]}', f'{r["forecast_process_s"]:.2f}',
                              metric(h["Ca"]["rmse"]), metric(h["T"]["rmse"]),
                              metric(h["Ca"]["coverage90"]), metric(h["T"]["coverage90"])])
    body.append("<h2>Wall time</h2>" + table(
        ["Library", "59 steps, process s", "599 steps, process s", "599 steps, episode s",
         "Synthetic warm-up s (599)", "Decision median ms", "p95 ms", "Maximum ms"], timing_rows))
    p = summary["hours_per_30_seeds"]
    body.append(f"<p>At these process times, 30 seeds need <strong>{p['short']:.2f} hours short and "
                f"{p['long']:.2f} hours long</strong>. One pass at both lengths totals "
                f"{sum(p.values()):.2f} hours. A second deadline condition at the same cost would give "
                f"{2*sum(p.values()):.2f} hours of control episodes. Forecasts and offline sampling are additional. "
                "Fresh processes include imports, native backend setup, two synthetic warm-up solves, "
                "simulation, and logging. Each decision clock includes binding, solve, status readout, and clipping.</p>")
    body.append("<h2>Control and fallback</h2>" + table(
        ["Library", "Decisions", "PCGym return", "Concentration RMSE", "Late", "Failed", "Held"], control_rows))
    body.append("<p>Development returns describe these trajectories only. Late and failed counts can overlap. "
                "A common plant random stream supplies sensor noise across arms. Each arm observes its own trajectory.</p>")
    body.append("<h2>60-step forecasts</h2>" + table(
        ["Library", "Origins scored", "Batch process s", "Concentration RMSE", "Temperature RMSE, K",
         "Concentration coverage, 90% interval", "Temperature coverage, 90% interval"], forecast_rows))
    body.append("<p>Each library estimates the initial posterior with future goals masked. Every arm then "
                "uses the same controlled RK4 propagator and 512 matched draws over twelve origins from "
                "a separate open-loop simulator trajectory. The JSON contains errors, coverage and width "
                "at leads 1, 5, 15 and 60. The twelve origins are too few for a calibration ranking.</p>")
    oracle_rows = []
    for key, r in arms.items():
        oracle_rows.extend([[names[key], c["case"], "passed" if c["passed"] else "failed",
                             f'{c["mean_whitened"]:.3g}', f'{c["cov_whitened"]:.3g}',
                             f'{c["absolute_errors"]["next_action_cov"]:.3g}'] for c in r["oracle"]])
    body.append("<h2>Independent Gaussian checks</h2>" + table(
        ["Library","Window","Result","Scaled mean error","Scaled covariance error","Lag covariance error"], oracle_rows))
    body.append('<p><a href="summary.json">Full measurements and numerical diagnostics</a> · '
                '<a href="forecast-truth.json">Open-loop truth and controls</a></p>')
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Four-library reactor timing pilot</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#172a37;margin:3rem auto;max-width:1100px;padding:0 1.2rem}
h1,h2{line-height:1.15}h2{margin-top:2em}p{max-width:85ch}table{border-collapse:collapse;display:block;overflow:auto}
th,td{border-bottom:1px solid #c7d4d8;padding:.45em .65em;text-align:right;font-variant-numeric:tabular-nums}
th{background:#eaf0f0;font-size:.87rem}th:first-child,td:first-child{text-align:left}a{color:#125d74}
</style>""" + "".join(body) + "</html>"
    (root / "index.html").write_text(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--campaign", default="measured")
    args = parser.parse_args()
    summary = summarize(args.output, args.campaign)
    save(args.output / "summary.json", summary)
    report(args.output, args.campaign, summary)
