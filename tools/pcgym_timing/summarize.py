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


def physical_observations(normalized):
    # PCGym subtracts its Gym Box bounds in float32 before scaling its float64 state.
    low = np.array([.7, 300.], dtype=np.float32)
    high = np.array([1., 350.], dtype=np.float32)
    return low + (np.asarray(normalized) + 1) * (high - low) / 2


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
    if expected == 599:
        for key, subset in (("first100", rows[:100]), ("last100", rows[-100:])):
            result[f"rmse_Ca_{key}"] = float(np.sqrt(np.mean([r["error_Ca"]**2 for r in subset])))
    # PCGym's observation noise is multiplicative in physical units.
    states = np.vstack(([.8, 330.], [r["next_true_state"] for r in rows[:-1]]))
    physical_obs = physical_observations([r["obs"] for r in rows])
    noise = physical_obs / states - 1
    # The initial observation is noiseless. Each PCGym step draws one pair thereafter.
    rng = np.random.RandomState(result["seed"])
    expected_noise = np.vstack(([0., 0.], .001 * rng.normal(size=(expected-1, 2))))
    result["max_noise_replay_error"] = float(np.max(abs(noise - expected_noise)))
    if result["max_noise_replay_error"] > 16*np.finfo(noise.dtype).eps:
        raise ValueError(f"Sensor noise disagrees with the saved seed in {directory}")
    return result, noise


def persistence_errors(truth):
    origins = np.array(truth["origin_ticks"])
    last_seen = physical_observations(truth["observations_normalized"])[origins]
    trajectory = np.asarray(truth["truth_physical"])
    return {str(h): dict(zip(("Ca", "T"), np.sqrt(np.mean(
        (last_seen - trajectory[origins + h])**2, axis=0)))) for h in truth["horizons"]}


def summarize(root, campaign):
    runs = root / campaign
    results = {}
    noise = {"short": [], "long": []}
    draw_digests = {}
    truth = read(root / "forecast-truth.json")
    for arm in ARMS:
        directory = lambda phase: runs / f"{arm}-{phase}"
        q = read(directory("oracle") / "result.json")
        f = read(directory("forecast") / "result.json")
        if not q["complete"] or len(q["cases"]) != 3 or not all(c["passed"] for c in q["cases"]):
            raise ValueError(f"Incomplete affine qualification for {arm}")
        if (f["dataset_sha256"] != truth["sha256"] or f["particles"] != 512
                or [row["tick"] for row in f["origins"]] != truth["origin_ticks"]):
            raise ValueError(f"Wrong forecast protocol for {arm}")
        for row in f["origins"]:
            if row["status"] == "ok":
                digest = row["draws_sha256"]
                if draw_digests.setdefault(row["tick"], digest) != digest:
                    raise ValueError(f"Unmatched particle draws at tick {row['tick']} for {arm}")
        entry = results[arm] = dict(qualified=q["complete"], oracle=q["cases"], forecast=f,
                                   forecast_process_s=read(directory("forecast") / "process.json")["wall_s"])
        for phase, steps in (("short", 59), ("long", 599)):
            entry[phase], draws = audit_episode(directory(phase), steps)
            noise[phase].append(draws)
    noise_error = {}
    for phase, draws in noise.items():
        noise_error[phase] = float(np.max(abs(np.asarray(draws) - draws[0])))
        if (noise_error[phase] > 32*np.finfo(np.float64).eps
                or len({r[phase]["rng_digest"] for r in results.values()}) != 1):
            raise ValueError(f"Unmatched external noise stream in {phase} episodes")
    projected = {phase: 30*sum(r[phase]["process_s"] for r in results.values())/3600
                 for phase in ("short", "long")}
    forecast_seconds = sum(r["forecast_process_s"] for r in results.values())
    control_seconds = sum(r[p]["process_s"] for r in results.values() for p in ("short", "long"))
    return dict(arms=results, noise_relative_difference=noise_error, hours_per_30_seeds=projected,
                forecast_hours_per_30_batches=30 * forecast_seconds / 3600,
                measured_control_and_forecast_s=control_seconds + forecast_seconds,
                persistence_rmse=persistence_errors(truth))


def report(root, campaign, summary):
    names = dict(cffg="CFFG", rxinfer="RxInfer", numpyro="NumPyro", gtsam="GTSAM")
    arms = summary["arms"]
    p = summary["hours_per_30_seeds"]
    f_hours = summary["forecast_hours_per_30_batches"]
    n = sum(r[k]["n"] for r in arms.values() for k in ("short", "long"))
    holds = sum(r[k]["holds"] for r in arms.values() for k in ("short", "long"))
    scored = sum(r["forecast"]["n_scored"] for r in arms.values())
    body = ["<h1>Four-library reactor timing pilot</h1>",
            f"<p><strong>{n:,} control decisions, {holds} holds, {scored}/48 forecasts scored.</strong> "
            f"The eight episodes and four forecast batches took {summary['measured_control_and_forecast_s']/60:.2f} "
            "minutes of measured process time in total on one Apple M2 Max. Package installation, adapter "
            "development, Gaussian qualification and truth generation were separate.</p>",
            f"<p>Repeating both episode lengths and one forecast batch on each of 30 seeds projects to "
            f"<strong>{(sum(p.values()) + f_hours)*60:.0f} minutes</strong> with one deadline condition "
            "and a fresh process for every library and phase. This extrapolation comes from one development seed.</p>"]
    timing_rows, control_rows, forecast_rows = [], [], []
    for key, r in arms.items():
        name = names[key]
        short, long = r["short"], r["long"]
        timing_rows.append([name, f'{short["process_s"]:.2f}', f'{long["process_s"]:.2f}',
                            f'{long["previous_stage_to_episode_s"]:.2f}', f'{long["episode_s"]:.2f}',
                            *(f'{long["latency_s"][s]*1000:.2f}' for s in ("median","p95","max"))])
        for length, row in (("59",short), ("599",long)):
            link = f'{campaign}/{key}-' + ("short" if length=="59" else "long") + "/result.json"
            control_rows.append([f'<a href="{escape(link)}">{name}</a>', length,
                                 f'{row["return_sum"]:.4f}', f'{row["rmse_Ca"]:.5f}',
                                 row["missed"], row["failures"], row["holds"]])
        f = r["forecast"]
        h = f["summary"]["60"]
        metric = lambda value: "—" if value is None else f"{value:.5g}"
        forecast_rows.append([f'<a href="{escape(campaign)}/{key}-forecast/result.json">{name}</a>',
                              f'{f["n_scored"]}/{f["n_origins"]}', f'{r["forecast_process_s"]:.2f}',
                              metric(h["Ca"]["rmse"]), metric(h["T"]["rmse"]),
                              metric(h["Ca"]["coverage90"]), metric(h["T"]["coverage90"]),
                              metric(h["Ca"]["mean_width90"]), metric(h["T"]["mean_width90"])])
    body.append("<h2>Wall time</h2>" + table(
        ["Library", "59 steps, process s", "599 steps, process s", "Launch to warmed controller, s",
         "599-step control loop, s", "Decision median ms", "p95 ms", "Maximum ms"], timing_rows))
    cffg, rxinfer = arms["cffg"]["long"], arms["rxinfer"]["long"]
    body.append(f"<p>CFFG reached a warmed controller in {cffg['previous_stage_to_episode_s']:.1f} seconds "
                f"and took {1000*cffg['latency_s']['median']:.2f} ms per decision at the median. "
                f"RxInfer reached the same stage in {rxinfer['previous_stage_to_episode_s']:.1f} seconds "
                f"and took {1000*rxinfer['latency_s']['median']:.2f} ms at the median.</p>")
    body.append("<p>Startup, control-loop and decision columns come from the 599-step run. Startup includes "
                "imports, native backend setup, and two synthetic solves. Installed environments and Julia’s "
                "package precompilation were available. Decisions include input preparation, inference, "
                "status readout and clipping. Closing the workers adds to total process time.</p>")
    body.append(f"<p>At 30 seeds the four short runs cost about {p['short']*60:.1f} minutes in aggregate, "
                f"the long runs {p['long']*60:.1f}, and twelve forecast origins per seed per library "
                f"another {f_hours*60:.1f}. Repeating control under a second deadline at the same cost, "
                f"while reusing the forecast results, gives {(2*sum(p.values()) + f_hours)*60:.0f} minutes. "
                "Offline posterior sampling, additional plant conditions and retries require a separate budget.</p>")
    body.append("<h2>Control and fallback</h2>" + table(
        ["Library", "Decisions", "PCGym return", "Concentration RMSE", "Late", "Failed", "Held"], control_rows))
    body.append(f"<p>On the long run, the difference in return between CFFG and RxInfer was "
                f"{abs(cffg['return_sum'] - rxinfer['return_sum']):.1g}. "
                f"CFFG’s return minus GTSAM’s was {cffg['return_sum'] - arms['gtsam']['long']['return_sum']:+.3f}. "
                "A paired run across independent seeds is needed to measure how consistent this difference is.</p>")
    body.append("<p>Higher PCGym return and lower concentration error are better. Both episode lengths use "
                "control seed 0 and the same physical decision interval. Each controller sees up to four "
                "previous noisy measurements and applied actions, the current measurement, and fifteen future "
                "setpoints. Each arm drives its own simulator using matched sensor-noise draws. Commands are "
                "clipped to 295–302 K. A failed solve or a decision over one second holds the last command "
                "(initially 298.5 K). Each deadline starts after the controller has warmed up.</p>")
    body.append("<details><summary>Tracking error early and late in the long episode</summary>" + table(
        ["Library", "First 100, RMSE (mol/L)", "Last 100, RMSE (mol/L)"],
        [[names[k], f'{r["long"]["rmse_Ca_first100"]:.5f}', f'{r["long"]["rmse_Ca_last100"]:.5f}']
         for k, r in arms.items()]) +
        "<p>The setpoint schedule changes across these intervals. The plant parameters stay fixed, and each "
        "decision fits a fresh window. This pilot measures sustained control with a fixed model.</p></details>")
    body.append("<h2>60-step forecasts</h2>" + table(
        ["Library", "Scored", "Batch s", "Ca RMSE (mol/L)", "T RMSE (K)",
         "Ca coverage", "T coverage", "90% width (mol/L)", "90% width (K)"], forecast_rows))
    persistence = summary["persistence_rmse"]["60"]
    model = arms["cffg"]["forecast"]["summary"]["60"]
    body.append(f"<p>Holding the most recent observation fixed for 60 steps produced concentration RMSE "
                f"{persistence['Ca']:.5f} mol/L and temperature RMSE {persistence['T']:.2f} K. "
                f"CFFG with the shared reactor model achieved {metric(model['Ca']['rmse'])} mol/L and "
                f"{metric(model['T']['rmse'])} K.</p>")
    body.append("<p>Each library estimates the initial posterior with future goals masked. Every arm then "
                "uses the same controlled RK4 propagator and 512 matched draws over twelve origins from "
                "a separate open-loop simulator trajectory with known future coolant commands. This tests "
                "how the initial belief affects the forecast. Coverage is the fraction of true latent states "
                "inside the predicted 90% interval. Twelve origins on one trajectory give limited information "
                "about calibration.</p>")
    horizon_rows = []
    for h, baseline in summary["persistence_rmse"].items():
        for k, r in arms.items():
            scores = r["forecast"]["summary"][h]
            horizon_rows.append([h, names[k], metric(scores["Ca"]["rmse"]), metric(scores["T"]["rmse"])])
        horizon_rows.append([h, "Hold last observed value", metric(baseline["Ca"]), metric(baseline["T"])])
    body.append("<details><summary>Forecast accuracy by lead, with a persistence baseline</summary>" +
                table(["Steps ahead", "Method", "Ca RMSE (mol/L)", "T RMSE (K)"], horizon_rows) +
                "<p>The baseline carries the last noisy measurement forward from each origin. It provides "
                "a scale for the errors using the same truth and origins.</p></details>")
    body.append("<h2>What each controller computes</h2>" + table(["Library", "Nonlinear inference for this run"], [
        ["CFFG", "Directed Gaussian message passing with a per-decision convergence and free-energy certificate."],
        ["RxInfer", "Native streaming Gaussian message updates with matched local linearizations in a persistent Julia worker."],
        ["NumPyro", "Joint Laplace fit with autodifferentiated curvature and a line-searched Newton solver."],
        ["GTSAM", "Sparse Levenberg–Marquardt fit with a compiled RK4 factor and Gauss–Newton covariance on request."],
    ]) + "<p>Every arm uses the same reactor equations, prior, likelihoods and goals. Their nonlinear "
         "approximations and termination checks differ. The timing therefore applies to these implementations. "
         'A fixed-budget full-covariance NumPyro variational fit missed the affine tolerances during '
         'development. <a href="numpyro-development.json">NumPyro method selection</a></p>')
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
summary{color:#125d74;font-weight:600;cursor:pointer}details{margin:1.3rem 0}details table{margin-top:1rem}
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
