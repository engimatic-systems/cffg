"""Render the paired four-planner reactor comparison from its audited artifacts."""

import argparse
from hashlib import sha256
from html import escape
import json
from os.path import relpath
from pathlib import Path
from statistics import fmean, median
from tempfile import NamedTemporaryFile
from urllib.parse import quote

if __package__:
    from .paired_report import NAMES, PEERS, STYLE, details, interval, number, table
else:
    from paired_report import NAMES, PEERS, STYLE, details, interval, number, table


REGIMES = {"frozen": "Frozen", "original": "Original (0.012)", "candidate": "Selected (0.003)",
           "informed": "Informed"}
REFERENCES = {
    "stationary_penalty": "stationary penalty",
    "informed_control_headroom": "informed control gain",
    "candidate_control_benefit": "selected control gain",
    "settled_parameter_error": "settled UA error",
    "recovery": "recovery",
    "retention": "original gain retained",
    "reliability": "reliability",
}
METHODS = [
    ("CFFG", "Directed Gaussian chain with local first-order Taylor expansions and a certified message solution."),
    ("RxInfer", "One forward joint Gaussian message per slice with RxInfer’s Linearization rule."),
    ("NumPyro", "Joint posterior mode with a Laplace approximation using the exact local Hessian."),
    ("GTSAM", "Joint posterior mode by sparse Levenberg–Marquardt with Gauss–Newton covariance."),
]


def read(path):
    return json.loads(path.read_bytes())


def pct(value, *, signed=False):
    return "—" if value is None else number(value, 4, signed=signed) + "%"


def primary(data):
    pairs = data["between_libraries"]
    rows = []
    for peer in PEERS:
        row = pairs[peer]["changed_rmse"]
        rows.append([NAMES[peer], number(row["cffg_mean"]), number(row["peer_mean"]),
                     number(row["mean"], signed=True), interval(row["ci95"]),
                     pct(row["relative_to_peer_pct"], signed=True)])
    largest = max(PEERS, key=lambda p: abs(pairs[p]["changed_rmse"]["mean"]))
    effect = pairs[largest]["changed_rmse"]
    relative = effect["relative_to_peer_pct"]
    takeaway = ("The three mean paired differences were exactly zero." if effect["mean"] == 0 else
                f'The largest absolute mean difference was against {NAMES[largest]}. CFFG’s tracking error '
                f'was {number(abs(effect["mean"]))} mol/L {"higher" if effect["mean"] > 0 else "lower"}'
                + (f' ({pct(abs(relative))} of that peer’s mean error).' if relative is not None else '.'))
    seed_rows = []
    for s in data["seeds"]:
        baseline = data["libraries"]["cffg"]["per_seed"][str(s)]["candidate"]["changed_tracking_rmse"]
        seed_rows.append([s, number(baseline, 7), *(number(pairs[p]["changed_rmse"]["per_seed"][str(s)],
                                                         7, signed=True) for p in PEERS)])
    max_gaps = []
    for left, right in (("cffg", "rxinfer"), ("numpyro", "gtsam")):
        gap = max(abs(data["libraries"][left]["per_seed"][str(s)]["candidate"]["changed_tracking_rmse"]
                      - data["libraries"][right]["per_seed"][str(s)]["candidate"]["changed_tracking_rmse"])
                  for s in data["seeds"])
        max_gaps.append(f'{number(gap, 4)} mol/L for {NAMES[left]} and {NAMES[right]}')
    secondary = []
    for key, label in (("stationary_rmse", "Stationary tracking RMSE (mol/L)"),
                       ("changed_learning_benefit", "Gain over own frozen planner (mol/L)"),
                       ("changed_gain_pct_points", "Gain over own frozen planner (percentage points)")):
        for peer in PEERS:
            row = pairs[peer][key]
            secondary.append([label, NAMES[peer], number(row["mean"], signed=True), interval(row["ci95"])])
    return "".join([
        '<section id="primary"><h2>Tracking during the hidden drop</h2>',
        f'<p class="takeaway">{takeaway}</p>',
        '<p>All four planners used the selected heat-transfer estimator. For each seed, the analysis measured '
        'the root mean square error (RMSE) between the true concentration and its setpoint during decisions '
        '178–372. It subtracted the peer’s RMSE from CFFG’s and averaged the ten paired differences. '
        'A negative difference favors CFFG.</p>',
        table("Primary paired changed-period concentration RMSE", ["Peer", "CFFG mean (mol/L)", "Peer mean (mol/L)",
              "CFFG − peer (mol/L)", "95% paired interval (mol/L)", "Difference / peer mean"], rows),
        f'<p class="note">Each descriptive 95% percentile interval uses {data["bootstrap"]["draws"]:,} '
        'resamples of whole paired seeds and applies '
        'to one comparison. An interval spanning zero is compatible with either direction. The protocol '
        'set no practical-equivalence margin.</p>',
        details("Inspect every paired seed", table("Individual primary paired differences in mol/L",
                ["Seed", "CFFG RMSE", *(f"CFFG − {NAMES[p]}" for p in PEERS)], seed_rows)),
        details("Secondary paired comparisons",
                '<p>All entries are CFFG minus peer. A smaller stationary RMSE is better. A larger gain '
                'means a greater error reduction when moving from that planner’s frozen to selected regime.</p>'
                + table("Secondary paired differences", ["Measure", "Peer", "Mean difference", "95% paired interval"],
                        secondary)),
        '<h3>Which approximations produced these results?</h3>',
        table("Benchmarked planner implementations", ["Planner", "Approximation used in this experiment"], METHODS),
        f'<p>The largest absolute within-seed differences in changed-period RMSE were '
        f'{" and ".join(max_gaps)}. These measured differences apply to the implementations and windows '
        'tested here. The same external EKF supplied the learned UA in all four implementations.</p>',
        '</section>'])


def adaptation(libraries):
    screening, outcomes, recovery, references = [], [], [], []
    passing = []
    for key, library in libraries.items():
        name = NAMES[key]
        summary, gates = library["summary"], library["gates"]
        candidate = summary["candidate"]
        if gates["stationary_penalty"]["passed"]:
            passing.append(name)
        missed = [label for gate, label in REFERENCES.items() if not gates[gate]["passed"]]
        references.append([name, ", ".join(missed) if missed else "All reference criteria met"])
        stationary_ci = library["within_library_intervals"]["arms"]["candidate"]["stationary_penalty_pct"]
        screening.append([name, pct(candidate["stationary_penalty_pct"], signed=True), interval(stationary_ci),
                          "Pass" if gates["stationary_penalty"]["passed"] else "Exceeded",
                          pct(candidate["changed_gain_pct"], signed=True), f'{candidate["changed_wins"]}/10',
                          pct(gates["retention"]["retained_gain_pct"])])
        for arm, label in REGIMES.items():
            s = summary[arm]
            outcomes.append([name, label, number(s["stationary_tracking_rmse"], 6),
                             number(s["changed_tracking_rmse"], 6), pct(s["stationary_penalty_pct"], signed=True),
                             pct(s["changed_gain_pct"], signed=True), number(s["settled_ua_mae"])])
        for stage, label in (("during", "Drop"), ("after", "Return")):
            switch = gates["recovery"]["switches"][stage]
            delay = number(switch["median_delta"], signed=True)
            if switch["upper_bound"] and switch["median_delta"] is not None:
                delay = "≤" + delay
            recovery.append([name, label, number(summary["original"]["median_recovery_delay"][stage]),
                             number(candidate["median_recovery_delay"][stage]), delay,
                             switch["original_unrecovered"], switch["candidate_unrecovered"]])
    status = (f'The selected update met the 1% stationary allowance in {len(passing)} of four planners.'
              + (f' The planners within the allowance were {", ".join(passing)}.' if 0 < len(passing) < 4 else ''))
    return "".join([
        '<section id="adaptation"><h2>What adapting UA changed for each planner</h2>',
        f'<p>{status} The allowance applies to the increase in mean full-episode tracking RMSE '
        'relative to that same planner with UA held fixed. A positive changed-period gain means learning '
        'reduced tracking error versus its own frozen baseline.</p>',
        table("Selected estimator versus each planner’s frozen baseline",
              ["Planner", "Stationary penalty", "95% paired interval (%)", "1% allowance", "Median changed gain", "Seeds with lower error",
               "Original gain retained"], screening),
        '<p class="note">The stationary allowance applies to the point estimate, using the ratio of mean per-seed errors. Changed gain takes the median '
        'of the ten seed-level percentage reductions during the drop. Retention divides the selected gain '
        'by the original learner’s positive gain.</p>',
        details("Engineering reference checks by planner",
                '<p>The reference asks for ≥1% median tracking gain on the changed plant with gains on ≥8/10 '
                'seeds for both selected and informed, ≤1% stationary penalty, ≥50% lower settled UA error '
                'than frozen, ≥90% of the original '
                'tracking gain, timely recovery at both switches, and zero '
                'holds, failed solves or missed deadlines.</p>'
                + table("Engineering checks", ["Planner", "Reference criteria missed"], references)),
        details("All four parameter regimes for every planner",
                table("Per-planner control and estimation outcomes", ["Planner", "UA source",
                      "Stationary RMSE (mol/L)", "Changed RMSE (mol/L)", "Stationary penalty", "Median changed gain",
                      "Settled mean |UA error|"], outcomes)
                + '<p>RMSE columns average episode-level values across seeds. The settled UA error excludes '
                  'the first 25 decisions after the drop.</p>'),
        '<h3>How quickly the heat-transfer estimate recovered</h3>',
        '<p>Recovery begins at the first of five consecutive pre-action estimates within 1% of true UA. '
        'A positive slowdown means the selected estimator took more ticks than the original. The comparison '
        'allows up to five ticks of median paired slowdown at each switch and requires every selected '
        'episode to recover.</p>',
        table("Recovery delay after each hidden switch", ["Planner", "Switch", "Original median (ticks)",
              "Selected median (ticks)", "Median paired slowdown", "Original unrecovered", "Selected unrecovered"],
              recovery),
        '<p class="note">Median paired slowdown uses within-seed differences. A dash marks an unavailable '
        'estimate. A ≤ sign marks an upper bound when the original failed to recover within the window.</p>',
        '</section>'])


def predictions(libraries):
    rows = []
    selected_t = []
    for key, library in libraries.items():
        for arm, label in REGIMES.items():
            forecasts = [r["switching"][arm]["windows"]["during"]["observation_prediction_rmse"]
                         for r in library["episodes"].values()]
            temperature = fmean(r["T"] for r in forecasts)
            if arm == "candidate":
                selected_t.append(temperature)
            rows.append([NAMES[key], label, number(fmean(r["Ca"] for r in forecasts)), number(temperature)])
    return ('<section id="prediction"><h2>Predicting the next observation</h2>'
            '<p>Before each plant step, the benchmark forecast the next concentration and temperature using '
            'the applied coolant command, the current UA and the common reactor model. These errors average '
            'one-step observation RMSE across seeds during the hidden drop. With the selected estimator, '
            f'the temperature errors ranged from {number(min(selected_t))} to {number(max(selected_t))} K. '
            'Each planner generated its own closed-loop trajectories and therefore its own prediction inputs.</p>'
            + details("One-step errors by planner and UA source", table("Changed-period one-step observation RMSE",
                      ["Planner", "UA source", "Concentration RMSE (mol/L)", "Temperature RMSE (K)"], rows))
            + '</section>')


def timing(libraries):
    costs, latencies, reliability = [], [], []
    for key, library in libraries.items():
        name = NAMES[key]
        worker = library["worker_cost_s"]
        clocks = library["pooled_latency_s"]["both"]["candidate"]
        dec = clocks["decision_s"]
        setups = [c[a]["warm_setup_s"] for r in library["episodes"].values() for c in r.values() for a in REGIMES]
        costs.append([name, number(worker["adapter_setup_s"]["median"], 4),
                      number(worker["first_warmup_s"]["median"], 4),
                      number(median(setups), 4), number(worker["process_wall_s"]["median"], 4)])
        counts = library["totals"]
        latencies.append([name, dec["n"], *(number(1000*dec[k], 4) for k in ("median", "p95", "max")),
                         number(1000*clocks["update_s"]["median"], 4)])
        total = sum(library["pooled_latency_s"]["both"][arm]["decision_s"]["n"] for arm in REGIMES)
        reliability.append([name, total, total - counts["holds"], counts["failures"], counts["misses"], counts["holds"]])
    return "".join([
        '<section id="timing"><h2>Decision time and reliability</h2>',
        '<p>The warmed clock includes the online estimator update where applicable, installation of UA, '
        'preparing the observation window, inference, and clipping the proposed command. The one-step '
        'forecast and simulator step follow that clock.</p>',
        table("Selected estimator warmed latency, both plant conditions", ["Planner", "Selected decisions",
              "Median (ms)", "95th percentile (ms)", "Maximum (ms)", "Median update (ms)"], latencies),
        '<p>Across all regimes, a failed solve or a decision over one second held the previous coolant '
        'command. A late failure contributes to both diagnostic counts and one hold.</p>',
        table("All-regime reliability across both conditions", ["Planner", "All decisions", "Applied on time",
              "Failed solves", "Deadline misses", "Commands held"], reliability),
        details("Setup and process time", '<p>Each library used a fresh single-threaded process per seed. '
                'Its eight episodes shared compiled code and began with an independent plant and estimator. '
                'Adapter setup, first warmup and process wall time are medians over ten processes. '
                'Repeated setup is the median over the four regimes and both conditions.</p>'
                + table("Measured setup stages", ["Planner", "Adapter setup (s)", "First warmup (s)",
                        "Repeated episode setup (s)", "Eight-episode process (s)"], costs)),
        '</section>'])


def evidence(root):
    frozen = read(root / "protocol.json")["frozen"]
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    sources = {Path(path): digest for path, digest in frozen["sha256"].items()}
    sources.update({repo / path: digest for path, digest in frozen["experiment_sha256"].items()})
    source_links = []
    for source in ("continual_comparison.py", "continual_estimator.py", "cffg_adapter.py", "rxinfer_worker.jl",
                   "numpyro_adapter.py", "gtsam_adapter.py", "gtsam_cstr.cpp"):
        path = here / source
        if path in sources and path.is_file() and sha256(path.read_bytes()).hexdigest() == sources[path]:
            href = quote(relpath(path, root))
            source_links.append(f'<li><a href="{href}">{source}</a></li>')
    paths = [("analysis.json", "Paired estimates, seed-level scores, audits and raw trace digests"),
             ("protocol.json", "Frozen design, numerical source paths, dependency hashes and prior selection hashes"),
             ("qualified.json", "Qualification record and check-file digests"),
             ("completed.json", "Campaign completion record")]
    links = [f'<li><a href="{path}">{escape(label)}</a></li>' for path, label in paths if (root / path).is_file()]
    for key, name in NAMES.items():
        files = ((f'qualification/{key}.json', 'reference checks'), (f'workers/{key}.json', 'worker timings'),
                 (f'{key}/', 'all raw episodes'))
        available = [f'<a href="{path}">{label}</a>' for path, label in files if (root / path).exists()]
        if available:
            links.append(f'<li>{name} · {" · ".join(available)}</li>')
    return "".join([
        '<section id="scope"><h2>What the comparison covers</h2>',
        '<p>The simulated reactor ran for 599 decisions under 0.1% multiplicative sensor noise. '
        'UA remained 50,000 in stationary episodes. In switching episodes it fell to 48,150 for '
        'decisions 178–372, then returned to 50,000. Frozen kept the initial value throughout. Informed '
        'received the true value before each decision and provides a reference for tracking with known UA.</p>',
        '<p>In the original and selected regimes, the benchmark supplied the same scalar extended Kalman filter '
        'to every planner. It updated log UA from the observed temperature after each realized transition. '
        'The original used a per-transition process standard deviation of 0.012, and the selected setting '
        'from the earlier CFFG study used 0.003. Each library planned with the resulting estimate, so '
        'differences measure these planner implementations in a shared learning loop.</p>',
        '<p>Seeds 6601–6610 supplied matched sensor-noise draws. Each planner chose its own commands. '
        'As trajectories diverged, the shared update rule could produce different estimates. '
        'The intervals describe variation across these noise seeds for one plant and one change schedule. '
        'The earlier one-percent study supplied the tuning choice and engineering reference.</p>',
        '<p>Qualification used linear Gaussian problems with known answers and checked the action after a '
        'live UA change against a freshly constructed model at the same value. For timing, library order '
        'rotated by seed and regime order rotated within workers.</p>',
        '</section><section id="evidence"><h2>Evidence</h2><ul>', "".join(links), '</ul>',
        details("Local source files matching the frozen hashes", "<ul>" + "".join(source_links) + "</ul>")
        if source_links else "",
        '</section>'])


def render(root):
    data = read(root / "analysis.json")
    digest = sha256((root / "protocol.json").read_bytes()).hexdigest()
    if data["raw_sha256"]["protocol.json"] != digest:
        raise ValueError("Analysis refers to a different protocol")
    for marker in ("qualified.json", "completed.json"):
        if read(root / marker)["protocol_sha256"] != digest:
            raise ValueError(f"{marker} refers to a different protocol")
    if sorted(data["seeds"]) != list(range(6601, 6611)) or set(data["libraries"]) != set(NAMES):
        raise ValueError("Expected all four planners and the ten fresh paired seeds")
    libraries = {key: data["libraries"][key] for key in NAMES}
    page = "".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>Four planners tracking a changing reactor</title><style>{STYLE}</style></head><body><main>',
        '<header><h1>Four planners tracking a changing reactor</h1>',
        '<p>CFFG, RxInfer, NumPyro and GTSAM controlled the same simulated reactor. '
        'The primary question was how closely each planner could track target concentration after '
        'a hidden drop in heat transfer, using the same online estimator of the heat-transfer coefficient (UA).</p>',
        '<nav aria-label="Report sections"><a href="#primary">Paired comparison</a>',
        '<a href="#adaptation">Adaptation and recovery</a><a href="#prediction">Predictions</a>',
        '<a href="#timing">Timing and reliability</a><a href="#evidence">Evidence</a></nav></header>',
        primary(data), adaptation(libraries), predictions(libraries), timing(libraries), evidence(root),
        '</main></body></html>'])
    dest = root / "index.html"
    with NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as stream:
        path = Path(stream.name)
        try:
            stream.write(page)
            stream.close()
            path.replace(dest)
        finally:
            path.unlink(missing_ok=True)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Completed comparison directory")
    print(render(parser.parse_args().output.expanduser().resolve()))
