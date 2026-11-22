"""Render the audited four-controller PCGym analysis as a self-contained local report."""

import argparse
from html import escape
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


NAMES = {"cffg": "CFFG", "rxinfer": "RxInfer", "numpyro": "NumPyro", "gtsam": "GTSAM"}
PEERS = tuple(name for name in NAMES if name != "cffg")
VARIABLES = {"Ca": "Concentration (mol/L)", "T": "Temperature (K)"}
PHASES = {"long": 599, "short": 59}

STYLE = """
:root { color-scheme: light; --ink:#182d32; --muted:#4b6266; --rule:#c9d5d4;
        --accent:#22645b; font:16px/1.55 "Source Sans 3",system-ui,sans-serif; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:#fbfcfa; border-top:6px solid var(--accent); }
main { max-width:70rem; margin:auto; padding:3rem clamp(1rem,4vw,3rem) 5rem; }
h1,h2,h3 { line-height:1.16; text-wrap:balance; }
h1,h2 { font-family:"Spectral",Georgia,serif; font-weight:500; }
h1 { font-size:clamp(2.5rem,6vw,4.5rem); letter-spacing:-.045em; margin:.15em 0; }
h2 { margin:0 0 .5em; font-size:clamp(1.85rem,4vw,2.55rem); letter-spacing:-.03em; }
h3 { font-size:1.1rem; margin:1.6em 0 .4em; }
p { max-width:80ch; margin:.5rem 0 .9rem; }
header p { max-width:65ch; font-size:1.14rem; }
nav { display:flex; flex-wrap:wrap; gap:.5rem 1.5rem; margin:1.2rem 0 0; }
a { color:var(--accent); text-underline-offset:.18em; }
section { border-top:1px solid var(--rule); padding-top:1.8rem; margin-top:2rem; }
.takeaway { border-left:4px solid var(--accent); background:#eaf2ed; padding:1rem 1.2rem;
            max-width:none; font-size:1.1rem; }
.scroll { overflow-x:auto; margin:.8rem 0 1rem; }
table { border-collapse:collapse; min-width:100%; font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:.55rem .7rem; vertical-align:baseline; }
thead th { font-size:.85rem; font-weight:600; color:#325351; background:#eaf0ec; line-height:1.3; }
tbody tr { border-bottom:1px solid #dce5e0; }
tbody th { font-weight:600; }
thead th:first-child,tbody th { text-align:left; }
tbody td,tbody th { white-space:nowrap; }
details { padding:.6rem 0; }
summary { cursor:pointer; color:var(--accent); font-weight:600; }
.note { color:var(--muted); font-size:.93rem; }
ul { padding-left:1.3rem; } li { margin:.35rem 0; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
@media print { body { background:white; border:0; } main { padding:0; }
nav { display:none; } section { break-before:auto; } h2,h3,summary { break-after:avoid; }
tr { break-inside:avoid; } .scroll { overflow:visible; } th,td { padding:.25rem; } }
"""


def number(value, digits=5, *, signed=False):
    """Keep small paired differences visible at their measured scale."""
    if value is None:
        return "—"
    if value == 0:
        return "0"
    return format(value, f"{'+' if signed else ''}.{digits}g").replace("-", "−")


def interval(bounds):
    if not bounds or any(v is None for v in bounds):
        return "—"
    return " to ".join(number(v, signed=True) for v in bounds)


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def table(title, headings, rows):
    head = "".join(f'<th scope="col">{escape(s)}</th>' for s in headings)
    body = []
    for label, *values in rows:
        cells = "".join(f"<td>{escape(str(v))}</td>" for v in values)
        body.append(f'<tr><th scope="row">{escape(str(label))}</th>{cells}</tr>')
    return (f'<div class="scroll"><table aria-label="{escape(title)}"><thead><tr>{head}'
            f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def details(label, content):
    return f"<details><summary>{escape(label)}</summary>{content}</details>"


def paired_table(phase, data):
    rows = []
    for peer in PEERS:
        pair = data["paired"][peer]
        change = pair["relative_percent_mean"]
        rows.append([NAMES[peer], number(pair["mean"], signed=True), interval(pair["ci95"]),
                     number(pair["median"], signed=True),
                     f'{number(change, 4, signed=True)}%' if change is not None else "—",
                     pair["positive"], pair["ties"], pair["negative"]])
    return table(f'{PHASES[phase]}-decision paired return differences',
                 ["Peer", "Mean difference", "95% interval for mean", "Median difference", "Mean relative change",
                  "CFFG higher", "Tie", "Peer higher"], rows)


def primary(control, seeds, tie, draws):
    paired = control["long"]["paired"]
    findings = []
    for peer in PEERS:
        pair = paired[peer]
        n = pair["n"]
        if pair["ties"] == n:
            findings.append(f'CFFG and {NAMES[peer]} tied on all {n} seeds.')
        elif pair["positive"] == n or pair["negative"] == n:
            direction = "higher" if pair["positive"] == n else "lower"
            findings.append(f'CFFG earned a {direction} return than {NAMES[peer]} on all {n} seeds '
                            f'(mean difference {number(pair["mean"], signed=True)}).')
        else:
            findings.append(f'Against {NAMES[peer]}, CFFG earned more reward on {pair["positive"]} '
                            f'of {n} seeds, with {pair["ties"]} ties.')
    seed_table = table(
        "Individual long-episode paired returns",
        ["Seed", "CFFG return", *(f"CFFG − {NAMES[peer]}" for peer in PEERS)],
        [[row["seed"], number(row["control"]["long"]["arms"]["cffg"]["return_sum"], 8),
          *(number(row["control"]["long"]["paired"][peer]["delta"], 7, signed=True) for peer in PEERS)]
         for row in seeds])
    short = control["short"]["paired"]
    short_wins = [f'{NAMES[peer]} on {short[peer]["positive"]} of {short[peer]["n"]} seeds'
                  for peer in ("numpyro", "gtsam")]
    short_finding = "Over 59 decisions, CFFG earned a higher return than " + " and ".join(short_wins) + ". "
    if all(short[peer]["ci95"][0] < 0 < short[peer]["ci95"][1] for peer in ("numpyro", "gtsam")):
        short_finding += "Both 95% intervals for the mean return difference cross zero. "
    return "".join([
        '<section id="returns"><h2>599-decision return</h2>',
        f'<p class="takeaway">{" ".join(findings)}</p>',
        "<p>For each seed, subtract the peer’s summed PCGym reward from CFFG’s. "
        "A positive difference favors CFFG, including when both returns are negative. Each seed has equal weight.</p>",
        paired_table("long", control["long"]),
        f'<p class="note">Differences within ±{number(tie)} return units count as numerical ties. '
        'Means retain the original differences. Each 95% interval uses the percentiles of '
        f'{draws:,} resamples of whole paired seeds and applies to one comparison at a time. '
        'Relative change divides each seed’s difference by the magnitude of its peer return and averages '
        'over seeds with a nonzero peer return.</p>',
        f'<p>{short_finding}The long schedule adds setpoint changes and varies how long each target '
        'is held, so the comparison cannot isolate the effect of episode duration.</p>',
        details("Inspect every seed’s 599-decision return difference", seed_table),
        details("Compare the shorter 59-decision episodes", paired_table("short", control["short"])),
        '</section>'])


def control_section(control):
    outcomes = []
    failures = []
    timings = []
    for phase, steps in PHASES.items():
        for arm, name in NAMES.items():
            a = control[phase]["arms"][arm]
            outcomes.append([name, steps, number(a["return_sum"]["mean"], 7),
                             number(a["rmse_Ca"]["mean"]), f'{a["decisions"]:,}',
                             f'{a["decisions"] - a["holds"]:,}', a["failures"], a["missed"], a["holds"]])
            for status, count in a["status_counts"].items():
                if status != "ok":
                    failures.append([name, steps, status, count])
            latency = a["latency_s"]
            timings.append([name, steps, number(a["previous_stage_to_episode_s"]["median"]),
                            *(number(1000 * latency[metric], 4) for metric in ("median", "p95", "max")),
                            number(a["episode_s"]["median"]), number(a["process_s"]["median"])])
    long = control["long"]["arms"]["cffg"]
    segments = [[name, number(control["long"]["arms"][arm]["rmse_Ca_first100"]["mean"]),
                 number(control["long"]["arms"][arm]["rmse_Ca_last100"]["mean"])]
                for arm, name in NAMES.items()]
    return "".join([
        '<section id="control"><h2>Tracking and controller reliability</h2>',
        '<p>The controller adjusts coolant temperature to keep reactor concentration near a changing target. '
        'Tracking error is the root mean square error (RMSE) between true concentration and its target, '
        'in mol/L. Smaller is better. Return and tracking error below are means across episodes.</p>',
        table("Control quality and command counts",
              ["Controller", "Steps", "Mean return", "Tracking RMSE", "Decisions", "Applied on time",
               "Failed solves", "Late", "Held"], outcomes),
        '<p class="note">Each decision has one second after warmup. A late or failed decision repeats '
        'the previous command, initially 298.5 K. A decision can be both late and failed. '
        'Held counts each fallback once. The episode return includes its consequences.</p>',
        details("Tracking early and late in the 599-decision episodes",
                '<p>The setpoint schedule varies across these segments. Each entry is the mean '
                'across seeds of segment-level concentration RMSE (mol/L).</p>'
                + table("Long-episode concentration tracking",
                        ["Controller", "First 100 decisions", "Last 100 decisions"], segments)),
        details("Failed solver statuses", table("Solver failures", ["Controller", "Steps", "Status", "Count"],
                                                failures)) if failures else "",
        '</section><section id="timing"><h2>Startup and warmed decision time</h2>',
        f'<p>For the long episodes, CFFG took {number(long["previous_stage_to_episode_s"]["median"], 4)} s '
        f'to get ready at the median. Once warm, its median decision took '
        f'{number(1000 * long["latency_s"]["median"], 4)} ms.</p>',
        '<p>Startup measures launch through imports, adapter setup and two synthetic warmup solves. '
        'Each controller starts a fresh process for each seed and phase. Warmed decisions include '
        'window preparation, inference, reading the action and clipping it to the coolant limits.</p>',
        table("Startup and warmed latency by episode length",
              ["Controller", "Steps", "Startup (s)", "Median decision (ms)", "95th percentile (ms)",
               "Slowest (ms)", "Episode (s)", "Process (s)"], timings),
        '<p class="note">Decision times pool all ticks for that controller and episode length. '
        'Startup, episode and total process times are medians across seeds. '
        'Episode time includes the simulator step and logging. Process time includes startup and shutdown.</p></section>'])


def forecast_table(forecast, lead):
    rows = []
    for key, variable in VARIABLES.items():
        for arm, name in NAMES.items():
            scores = forecast["arms"][arm]["leads"][lead][key]
            rows.append([variable, name, scores["n"], number(scores["rmse"]), number(scores["mae"]),
                         percent(scores["coverage90"]), number(scores["mean_width90"])])
        base = forecast["baseline"][lead][key]
        rows.append([variable, "Persistence", base["n"], number(base["rmse"]), number(base["mae"]), "—", "—"])
    return table(f"{lead}-step forecast errors and interval calibration",
                 ["Variable", "Method", "Origins scored", "RMSE", "Mean absolute error",
                  "90% coverage", "Mean 90% width"], rows)


def forecast_section(forecast, distinct_trajectories):
    performance = []
    for arm, name in NAMES.items():
        record = forecast["arms"][arm]
        performance.append([name, record["n_origins"], record["n_scored"], record["n_failed"],
                            number(record["process_s"]["median"])])
    effects = []
    lead = "60"
    comparisons = forecast["paired"]
    for key, variable in VARIABLES.items():
        for peer in (*PEERS, "persistence"):
            pair = comparisons[peer][lead][key]
            effects.append([variable, NAMES.get(peer, "Persistence"), pair["n_seeds"], pair["n_origins"],
                            number(pair["rmse_delta"], signed=True), interval(pair["rmse_delta_ci95"])])
    relative = []
    for key, noun, unit in (("Ca", "concentration", "mol/L"), ("T", "temperature", "K")):
        diff = comparisons["persistence"][lead][key]["rmse_delta"]
        if diff is None:
            continue
        word = "lower" if diff < 0 else "higher"
        relative.append(f'{noun} RMSE was {number(abs(diff))} {unit} {word} than persistence'
                        if diff else f'{noun} RMSE matched persistence')
    takeaway = f'<p>At 60 steps, CFFG’s {" and its ".join(relative)} on matched origins.</p>' if relative else ""
    cffg = forecast["arms"]["cffg"]["leads"][lead]
    truth_note = ("All seeds share one physical truth trajectory. Sensor noise and particle draws "
                  "vary across seeds." if distinct_trajectories == 1 else
                  f"The archive contains {distinct_trajectories} distinct physical truth trajectories.")
    coverage = ""
    if all(cffg[key]["coverage90"] is not None for key in VARIABLES):
        coverage = (f'<p>CFFG’s nominal 90% intervals covered the true concentration '
                    f'{percent(cffg["Ca"]["coverage90"])} of the time and the true temperature '
                    f'{percent(cffg["T"]["coverage90"])} of the time.</p>')
    return "".join([
        '<section id="forecast"><h2>Forecast accuracy and calibration</h2>',
        '<p>Twelve times in each separate open-loop trajectory, each controller estimated the current '
        'state from the same observations and past commands. All four forecasts then used the same '
        'reactor propagator, known future coolant commands and 512 matched particle draws. '
        'Persistence holds the latest noisy measurement fixed for every future step.</p>',
        f'<p>{truth_note}</p>',
        '<h3>60 steps ahead</h3>', takeaway, forecast_table(forecast, lead), coverage,
        '<p class="note">Errors and interval widths use the units shown for each variable. '
        'Coverage measures how often the true state fell inside the predicted 90% interval. '
        'Read coverage alongside width, since wider intervals can cover more states. '
        'Persistence supplies a point forecast. Each method’s error uses its scored origins.</p>',
        details("Forecasts at 1, 5 and 15 steps", "".join(
            f"<h3>{h} step{'s' if h != '1' else ''} ahead</h3>{forecast_table(forecast, h)}"
            for h in ("1", "5", "15"))),
        details("Paired 60-step RMSE differences and uncertainty",
                '<p>Subtract the comparator’s RMSE from CFFG’s on origins both scored. '
                'A negative value favors CFFG. Intervals resample whole seeds together with their origins.</p>'
                + table("Paired 60-step forecast differences",
                        ["Variable", "Comparator", "Seeds", "Matched origins", "CFFG − comparator",
                         "95% interval"], effects)),
        '<h3>Forecast completion</h3>',
        table("Forecast availability", ["Controller", "Planned origins", "Scored", "Failed",
                                        "Median batch process time (s)"], performance),
        '</section>'])


def evidence(root, seeds):
    seed = int(seeds[0]["seed"])
    paths = [("summary.json", "Aggregate estimates, uncertainty and pairing audits"),
             ("paired-seeds.json", "Individual seed outcomes and paired differences"),
             ("protocol.json", "Study protocol and frozen source hashes"),
             ("provenance.json", "Verified worker environments and reference checks"),
             ("finished.json", "Campaign completion and wall time"),
             (f"seed-{seed}/forecast-truth.json", f"Forecast truth and commands for seed {seed}"),
             (f"seed-{seed}/cffg-long/ticks.jsonl", f"Every CFFG decision and applied action for seed {seed}")]
    links = [f'<li><a href="{path}">{title}</a></li>' for path, title in paths if (root / path).is_file()]
    manifest = root / "qualification/cffg-oracle/manifest.json"
    hardware = json.loads(manifest.read_text())["runtime"]["hardware"] if manifest.is_file() else None
    host = f"one {escape(hardware)} CPU" if hardware else "one CPU host"
    for arm, name in NAMES.items():
        path = f"qualification/{arm}-oracle/result.json"
        if (root / path).is_file():
            links.append(f'<li><a href="{path}">{name} linear Gaussian reference checks</a></li>')
    folders = []
    for row in seeds:
        seed_id = int(row["seed"])
        folder = f"seed-{seed_id}/"
        if (root / folder).is_dir():
            folders.append(f'<a href="{folder}">{seed_id}</a>')
    return "".join([
        '<section id="scope"><h2>What these seeds can tell us</h2>',
        '<p>The runs use one simulated continuous stirred-tank reactor, one physical parameter setting '
        'and one sensor-noise level (0.1% multiplicative noise). Each controller drives its own plant '
        'with the same noise draws within a seed. Different actions can produce different trajectories. '
        'The intervals describe variation across these noise seeds.</p>',
        '<p>All four implementations use fixed model parameters, up to four prior measurement and action '
        'pairs, the current measurement, and fifteen future setpoints for control. Each solve starts '
        'from a fresh window. No controller learns or adapts model parameters. CFFG and RxInfer use '
        'local linearizations of the dynamics. NumPyro fits a joint mode with Laplace uncertainty, '
        'and GTSAM fits a joint mode with Gauss–Newton uncertainty.</p>',
        f'<p>All timings cover the installed implementations on {host}, using float64. '
        'The one-second deadline and fixed reactor model limit comparisons to these operating conditions.</p>',
        '<p>The forecast comparison isolates the initial filtered state estimate. All methods use a '
        'common transition model and particle propagation after filtering. Nearby forecast origins '
        'share a trajectory, so each whole seed is the unit of uncertainty.</p>',
        '</section><section id="evidence"><h2>Study artifacts</h2><ul>',
        "".join(links), '</ul><p class="note">Per-seed folders contain each controller’s tick logs, '
        'native diagnostics, timing manifests, forecast scores and truth trajectory.</p>',
        details("Browse raw evidence by seed", "<p>" + " · ".join(folders) + "</p>") if folders else "",
        '</section>'])


def render(root):
    summary = json.loads((root / "summary.json").read_text())
    seed_file = json.loads((root / "paired-seeds.json").read_text())
    if summary["schema_version"] != 1 or seed_file["schema_version"] != 1:
        raise ValueError("Unsupported paired analysis schema")
    seeds = seed_file["seeds"]
    seed_ids = {row["seed"] for row in seeds}
    if (not seeds or len(seed_ids) != len(seeds) or len(seeds) != summary["n_seeds"]
            or seed_ids != set(summary["audit"]["seeds"])):
        raise ValueError("Seed records disagree with the audited analysis")
    for phase in PHASES:
        for peer in PEERS:
            if summary["control"][phase]["paired"][peer]["n"] != len(seeds):
                raise ValueError(f"{phase} {peer} pair count differs from paired-seeds.json")
    title = f"Reactor control across {len(seeds)} paired seeds"
    page = "".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{title}</title><style>{STYLE}</style></head><body><main><header><h1>{title}</h1>',
        '<p>Four fixed-model controllers drove the same PCGym reactor under matched sensor noise. '
        'The primary question was how much total reward CFFG gained or lost against each peer '
        'over 599 decisions.</p><nav aria-label="Report sections">',
        '<a href="#returns">Paired return</a><a href="#control">Tracking</a>',
        '<a href="#timing">Timing</a><a href="#forecast">Forecasts</a>',
        '<a href="#scope">Scope</a><a href="#evidence">Artifacts</a></nav></header>',
        primary(summary["control"], seeds, summary["numerical_tie_abs_return"], summary["bootstrap"]["draws"]),
        control_section(summary["control"]),
        forecast_section(summary["forecast"], summary["audit"]["distinct_truth_trajectories"]),
        evidence(root, seeds),
        '</main></body></html>'])
    destination = root / "index.html"
    with NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(page)
            tmp.close()
            tmp_path.replace(destination)
        finally:
            tmp_path.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Campaign directory containing the analysis JSON")
    args = parser.parse_args()
    print(render(args.output.expanduser().resolve()))
