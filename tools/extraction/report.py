"""Render the local extraction-column study. Run with the study directory as ROOT."""
import argparse
from hashlib import sha256
from html import escape
import json
from math import isfinite
from pathlib import Path
import re
from statistics import fmean, median
from tempfile import NamedTemporaryFile


STYLE = """
:root{color-scheme:light;font:16px/1.55 "Source Sans 3",system-ui,sans-serif;color:#172c30;background:#f3f5f3}
*{box-sizing:border-box}body{margin:0}main{max-width:66rem;margin:auto;background:#fff;padding:clamp(1rem,4vw,3.4rem)}
header{border-top:5px solid #205d60;padding-top:1.3rem}h1,h2{font-family:"Source Serif 4",Georgia,serif;line-height:1.15}
h1{font-size:clamp(2.1rem,5vw,3.4rem);max-width:22ch;margin:.4rem 0 1rem}h2{font-size:1.65rem;margin:0 0 .75rem}
h3{font-size:1.08rem;margin:1.3rem 0 .35rem}p{max-width:79ch;margin:.5rem 0 1rem}
section{padding:1.65rem 0;border-top:1px solid #cbd6d4}header{padding-bottom:1rem}
.status{font-weight:700;color:#205d60;margin:0}.finding{font-size:1.18rem;font-weight:600}
a{color:#165e73;text-underline-offset:.16em}a:focus-visible,summary:focus-visible{outline:3px solid #bd7826}
figure{margin:1rem 0}figure img{display:block;width:100%;max-width:52rem;height:auto}figcaption{max-width:79ch;margin-top:.65rem}
.scroll{overflow-x:auto;margin:.9rem 0}table{border-collapse:collapse;width:100%;font-size:.92rem;font-variant-numeric:tabular-nums}
th,td{padding:.5rem .65rem;border-bottom:1px solid #dde3df;text-align:left;vertical-align:top}thead th{background:#e8efec}
details{margin:.8rem 0}summary{cursor:pointer;font-weight:650}li{margin:.25rem 0}ul{padding-left:1.35rem}
@media print{main{max-width:none;padding:0}.scroll{overflow:visible}tr{break-inside:avoid}}
"""
LIBS = ("cffg", "rxinfer")
NAMES = {"cffg": "CFFG", "rxinfer": "RxInfer"}
ARMS = {"adaptive": "Estimated transfer rate", "frozen": "Fixed nominal rate",
        "informed": "Given true rate", "open_loop": "No sensor updates"}


def read(path):
    return json.loads(path.read_bytes())


def txt(value):
    return escape(str(value), quote=True)


def fmt(value):
    return f"{value:.5g}".replace("-", "−") if numeric(value) else "—"


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def table(heading, columns, rows):
    labels = "".join(f'<th scope="col">{txt(name)}</th>' for name in columns)
    body = []
    for label, *values in rows:
        cells = "".join(f"<td>{txt(value)}</td>" for value in values)
        body.append(f'<tr><th scope="row">{txt(label)}</th>{cells}</tr>')
    return (f'<div class="scroll"><table aria-label="{txt(heading)}"><thead><tr>{labels}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def section(title, content):
    return f"<section><h2>{txt(title)}</h2>{content}</section>"


def disclosure(title, content):
    return f"<details><summary>{txt(title)}</summary>{content}</details>"


def pretty(key):
    return str(key).replace("_", " ").strip()


def gate_rows(pilot):
    rows = []
    labels = {"affine": "Exact test on a linear system", "plant": "Simulated plant and sensor checks",
              "information": "Do the outlets reveal enough about the interior?",
              "nonlinear": "Short nonlinear reference check", "reliability": "Failures and fallback commands",
              "feasible": "Long-run cost and feasibility"}
    for key, gate in pilot.get("gates", {}).items():
        label = labels.get(key, pretty(key).capitalize())
        for lib in LIBS:
            if key in {f"information_{lib}", f"reliability_{lib}"}:
                label = f'{NAMES[lib]} · {labels[key.split("_")[0]]}'
        if isinstance(gate, dict):
            passed = gate.get("passed")
            detail = gate.get("detail", "")
        else:
            passed, detail = gate, ""
        result = {True: "Passed", False: "Failed", None: "Unverified"}.get(passed, "Unverified")
        rows.append((label, result, detail))
    return rows


def primary(data, margin, opening=""):
    score = data.get("primary") or {}
    if score.get("comparison") not in (None, "cffg_minus_rxinfer"):
        raise ValueError("Unexpected primary comparison direction")
    pairs = score.get("per_seed") or []
    effect = score.get("mean_difference")
    ci = score.get("ci95")
    if not pairs or not numeric(effect) or not ci or len(ci) != 2 or not all(map(numeric, ci)):
        return section("Hidden concentration error", "<p>The analysis does not provide a complete paired estimate and interval.</p>")
    lo, hi = ci
    if lo > hi:
        raise ValueError("Reversed primary interval")
    if not all(numeric(row[k]) for row in pairs for k in ("cffg", "rxinfer", "delta")):
        raise ValueError("Invalid primary seed score")
    if numeric(margin) and margin > 0:
        if hi < -margin:
            verdict = "CFFG had meaningfully lower error."
        elif lo > margin:
            verdict = "RxInfer had meaningfully lower error."
        elif -margin <= lo <= hi <= margin:
            verdict = "The methods were practically tied under the frozen margin."
        else:
            verdict = "The comparison was inconclusive at the frozen margin."
        threshold = f"The study defined a difference of {fmt(margin)} concentration units as practically meaningful."
    else:
        verdict = ("CFFG had lower error." if hi < 0 else "RxInfer had lower error." if lo > 0
                   else "The interval does not distinguish a winner.")
        threshold = "The analysis supplies no positive practical margin, so it cannot establish practical equivalence."
    avg_cffg = fmean(r["cffg"] for r in pairs)
    avg_rxinfer = fmean(r["rxinfer"] for r in pairs)
    relative = f" ({fmt(100*abs(effect)/avg_rxinfer)}% of RxInfer’s error)" if avg_rxinfer > 0 else ""
    rows = [(r["seed"], fmt(r["cffg"]), fmt(r["rxinfer"]), fmt(r["delta"])) for r in pairs]
    return section("Hidden concentration error",
        f'<p class="finding">{verdict}</p>{opening}<p>Across {len(pairs)} paired seeds, average error was '
        f'{fmt(avg_cffg)} for CFFG and {fmt(avg_rxinfer)} '
        f'for RxInfer. Smaller is better. CFFG minus RxInfer was {fmt(effect)} concentration units{relative} '
        f'(95% interval {fmt(lo)} to {fmt(hi)}). {threshold}</p>'
        '<p>Each pair saw the same flows and sensor readings. Each seed contributes one root mean squared error '
        'across the eight hidden concentrations during plant changes. Larger misses count more heavily. The interval resamples whole seed pairs.</p>'
        + disclosure("Results for each paired seed", table("Primary paired results",
            ["Seed", "CFFG error", "RxInfer error", "CFFG − RxInfer"], rows)))


# Direction is part of the result. Calibration is judged by distance from the advertised coverage.
METRICS = {
    "hidden_rmse": ("Hidden concentration error", "lower", "Concentration units. Zero is exact."),
    "outlet_log_score": ("Next outlet reading, log score", "higher", "Higher means more probability on the reading that followed."),
    "kla_rmse": ("Transfer-rate error", "lower", "Physical transfer-rate units. Zero is exact."),
    "hidden_coverage90": ("Hidden concentration coverage", "coverage", "Target 0.90 for the individual 90% ranges."),
    "joint_coverage90": ("Joint hidden-state coverage", "coverage", "Target 0.90 for the whole hidden profile."),
    "hidden_width90": ("Width of 90% ranges", "width", "Concentration units. A narrow range is useful when coverage is close to 0.90."),
    "joint_log_score": ("Probability of the true hidden profile, log score", "higher", "Higher rewards accurate, well-calibrated joint predictions."),
    "full_minus_diag_log_score": ("Gain from retaining correlations", "higher", "Above zero favors using correlations for the same predicted means."),
    "tracking_rmse": ("X5 target tracking error", "lower", "Concentration units. Zero means the outlet followed the target exactly."),
    "effort": ("Flow effort", "lower", "Squared departures from nominal flows, scaled by 15 for liquid and 8 for gas. Lower means closer to nominal."),
    "flow_movement": ("Flow movement", "lower", "Average squared change in the two flows, scaled by their allowed ranges. Smaller means smoother control."),
    "latency_p50_s": ("Typical processing time (seconds)", "lower", "Median elapsed time per measured step."),
    "latency_p95_s": ("95th-percentile processing time (seconds)", "lower", "About 95% of measured steps finished within this time."),
    "failed": ("Failed steps", "lower", "Zero means every scheduled output succeeded."),
    "late": ("Late steps", "lower", "Zero means every measured step met the deadline."),
    "holds": ("Held commands", "lower", "Zero means the controller never had to reuse its previous command."),
    "joint_log_score_undefined": ("Unavailable joint scores", "lower", "Zero means every hidden profile received a valid joint probability score."),
    "outlet_log_score_undefined": ("Unavailable outlet scores", "lower", "Zero means every outlet reading received a valid forecast score."),
    "flow_violation_ticks": ("Steps outside flow limits", "lower", "Zero means both commands stayed within their allowed limits."),
}


def observed(a, b, direction):
    if not all(map(numeric, (a, b))):
        return "Insufficient paired measurements"
    if direction == "width":
        if a == b:
            return "Same width. Judge with coverage."
        return f'{"CFFG" if a < b else "RxInfer"} narrower. Judge with coverage.'
    if direction == "coverage":
        a, b = abs(a-.9), abs(b-.9)
        description = "closer to 90%"
    else:
        description = "lower" if direction == "lower" else "higher"
    if a == b:
        return "Tied on the reported values"
    if fmt(a) == fmt(b):
        return "Same to the displayed precision"
    first = a < b if direction != "higher" else a > b
    return f'{"CFFG" if first else "RxInfer"} {description} in these measurements'


def same_seeds(runs, mode, condition, arm):
    sets = [{r["seed"] for r in runs if (r["mode"], r["condition"], r["arm"], r["library"]) ==
             (mode, condition, arm, lib)} for lib in LIBS]
    return bool(sets[0]) and sets[0] == sets[1]


def highlights(groups, runs, protocol, *, complete):
    arms = {(g["mode"], g["condition"], g["library"]): g for g in groups if g["arm"] == "adaptive"}
    rows = []
    measures = [("tape", "changed", "hidden_rmse", "How far off were the eight hidden estimates during changes?"),
                ("tape", "overall", "hidden_coverage90", "How often did a 90% range contain the true concentration?"),
                ("control", "overall", "tracking_rmse", "How closely did the controllers track the X5 target?"),
                ("control", "overall", "latency_p95_s", "How long did a slower decision take (95th percentile, seconds)?"),
                ("control", "overall", "holds", "How often did a controller reuse the previous flow commands?")]
    for mode, period, metric, question in measures:
        pair = [arms.get((mode, "changing", lib), {}) for lib in LIBS]
        values = [(g.get(period) or {}).get(metric) for g in pair]
        if not any(map(numeric, values)):
            continue
        _, direction, scale = METRICS[metric]
        result = observed(*values, direction) if same_seeds(runs, mode, "changing", "adaptive") else "Incomplete pairing. No winner."
        if metric == "latency_p95_s" and numeric(protocol.get("deadline_s")):
            scale += f' Deadline {fmt(protocol["deadline_s"])} s.'
        rows.append((question, fmt(values[0]), fmt(values[1]), result, scale))
    if not rows:
        return ""
    title = "Control and estimation on fresh seeds" if complete else "What the development runs showed"
    return section(title, "<p>These results use each library’s adaptive filter on the changing plant. "
        "The hidden-state comparison uses identical inputs. The target-tracking and timing comparisons "
        "use each controller’s own actions.</p>" +
        table(title, ["Question", "CFFG", "RxInfer", "Observed result", "Reference"], rows))


def learning_errors(data, seeds):
    """Keep every frozen seed in every library/arm mean and observed range."""
    arms = ("frozen", "adaptive", "informed")
    errors = {(lib, arm): {} for lib in LIBS for arm in arms}
    for row in data["runs"]:
        if (row["mode"], row["condition"]) != ("tape", "changing"):
            continue
        cell = errors[row["library"], row["arm"]]
        error = (row.get("changed") or {}).get("hidden_rmse")
        if not numeric(error) or error < 0 or row["seed"] in cell:
            raise ValueError("Invalid or duplicate changed-period hidden error")
        cell[row["seed"]] = error
    if not seeds or len(set(seeds)) != len(seeds) or any(set(cell) != set(seeds) for cell in errors.values()):
        raise ValueError("Hidden-error figure requires all frozen seeds for both libraries and all three filters")
    return {key: list(cell.values()) for key, cell in errors.items()}


def learning_summary(errors):
    changes = []
    for lib in LIBS:
        fixed = fmean(errors[lib, "frozen"])
        learned = fmean(errors[lib, "adaptive"])
        delta = learned - fixed
        if fmt(fixed) == fmt(learned):
            changes.append(f"gave the same mean error to the displayed precision for {NAMES[lib]}")
            continue
        size = f"{abs(delta)/fixed:.1%}" if fixed > 0 else f"{fmt(abs(delta))} concentration units"
        changes.append(f'{"reduced" if delta < 0 else "increased"} mean error by {size} for {NAMES[lib]}')
    return ("Compared with keeping the nominal transfer rate fixed, learning it from the two outlets "
            + " and ".join(changes) + ". These changes are relative to each library’s own fixed-rate error.")


def speed_summary(data, cold):
    warm = {g["library"]: g["overall"]["latency_p50_s"] for g in data.get("groups", [])
            if (g["arm"], g["mode"], g["condition"]) == ("adaptive", "control", "changing")}
    medians_ms = [1000*warm[lib] if numeric(warm.get(lib)) else None for lib in LIBS]
    lines = []
    for values, boundary, unit in [(cold, "time to the first real action from a fresh process", "s"),
                                   (medians_ms, "median time per warmed adaptive decision on the changing plant", "ms")]:
        if not all(map(numeric, values)):
            lines.append(f"The {boundary} comparison is unavailable.")
            continue
        times = f"CFFG {fmt(values[0])} {unit}, RxInfer {fmt(values[1])} {unit}"
        if fmt(values[0]) == fmt(values[1]):
            lines.append(f"The {boundary} was the same to the displayed precision ({times}).")
        else:
            winner = NAMES[LIBS[0 if values[0] < values[1] else 1]]
            lines.append(f"{winner} had the shorter {boundary} ({times}).")
    return " ".join(lines)


def learning_figure(root, errors, n_seeds):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    arms = ("frozen", "adaptive", "informed")
    labels = ("Assume the nominal rate", "Learn the rate from outlets", "Supply the true rate")
    fig = Figure(figsize=(8.0, 4.2), dpi=210, facecolor="white")
    FigureCanvasAgg(fig)
    fig.subplots_adjust(left=.31, right=.965, top=.75, bottom=.24)
    ax = fig.subplots()
    fig.text(.035, .92, "Hidden concentration error as the transfer rate changes",
             fontsize=15, fontweight="bold", color="#183438")
    fig.text(.035, .844, f"Same outlet recordings for both filters  ·  {n_seeds} paired seeds",
             fontsize=10, color="#425658")
    max_error = max(max(values) for values in errors.values())
    for lib, offset, color, marker in (("cffg", -.13, "#006c73", "o"), ("rxinfer", .13, "#8d4e11", "D")):
        for j, arm in enumerate(arms):
            values = errors[lib, arm]
            y = j+offset
            ax.plot([min(values), max(values)], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(fmean(values), y, marker=marker, markersize=7, color=color, linestyle="none",
                    label=NAMES[lib] if j == 0 else None)
    ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(2.5,-.5), xlim=(0, max_error*1.09 or 1.),
           xlabel="Hidden-state RMSE in concentration units   ·   zero is best")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(axis="x", length=3, color="#a2b2b2")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#a2b2b2")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dce3e2", linewidth=.7)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower left", bbox_to_anchor=(.022,.025), ncol=2, frameon=False)
    fig.text(.36,.072,"Marker = seed mean     Line = smallest to largest seed error", fontsize=9, color="#425658")
    name = "hidden-concentration-error.png"
    stream = NamedTemporaryFile(dir=root, prefix=".figure-", suffix=".png", delete=False)
    tmp = Path(stream.name)
    try:
        with stream:
            fig.savefig(stream, format="png", facecolor="white")
        tmp.replace(root/name)
    finally:
        tmp.unlink(missing_ok=True)
    alt = "Hidden error during changed periods. " + " ".join(
        f'{label}: CFFG {fmt(fmean(errors["cffg", arm]))}, RxInfer {fmt(fmean(errors["rxinfer", arm]))}.'
        for arm, label in zip(arms, labels))
    best = {}
    for lib in LIBS:
        means = [fmean(errors[lib, arm]) for arm in arms]
        best[lib] = " and ".join(arm for arm, val in zip(arms, means) if fmt(val) == fmt(min(means)))
    if best["cffg"] == best["rxinfer"]:
        finding = f'The {best["cffg"]} filter gave the lowest mean error for both libraries.'
        if " and " in best["cffg"]:
            finding = f'The {best["cffg"]} filters tied for lowest mean error to the displayed precision for both libraries.'
    else:
        finding = f'The lowest observed mean error came from {best["cffg"]} for CFFG and {best["rxinfer"]} for RxInfer.'
    oracle = ", ".join(f'{NAMES[lib]} {fmt(fmean(errors[lib, "informed"]))}' for lib in LIBS)
    return section("How much did learning the transfer rate help?",
        f'<figure><a href="{name}"><img src="{name}" width="1680" height="882" alt="{txt(alt)}"></a>'
        f'<figcaption>{txt(finding)} The informed filter received the true rate and still had to infer the eight '
        f'hidden concentrations ({txt(oracle)}). Each marker averages one error '
        f'per seed across changed periods. The lines show variation among seeds.</figcaption></figure>')


def visit_results(data):
    pairs = {g["library"]: g for g in data.get("groups", [])
             if (g["arm"], g["condition"], g["mode"]) == ("adaptive", "changing", "tape")}
    if not all(lib in pairs for lib in LIBS) or not same_seeds(data.get("runs", []), "tape", "changing", "adaptive"):
        return ""
    rows = []
    labels = {"step1": "First abrupt visit", "step2": "Second abrupt visit",
              "ramp_down": "Gradual departure", "ramp_hold": "Time at the new level", "ramp_up": "Gradual return"}
    for key, label in labels.items():
        values = [(pairs[lib].get("by_event", {}).get(key) or {}).get("hidden_rmse") for lib in LIBS]
        if all(map(numeric, values)):
            rows.append((label, *(fmt(n) for n in values), observed(*values, "lower")))
    if not rows:
        return ""
    return disclosure("Hidden error across repeated changes",
        "<p>Smaller error means a better estimate on the shared recording. A second visit measures how "
        "well the filter adapts again. The reference and disturbances also affect how difficult each segment is.</p>"
        + table("Hidden concentration error by changed segment", ["Segment", "CFFG", "RxInfer", "Observed result"], rows))


def recovery(runs):
    matches = [r for r in runs if r["arm"] == "adaptive" and r["condition"] == "changing" and r["library"] in LIBS]
    rows = []
    for mode in sorted({r["mode"] for r in matches}):
        by_library = {}
        for lib in LIBS:
            visits = [v for r in matches if r["mode"] == mode and r["library"] == lib
                      for v in r.get("recovery", []) if v["changed"]]
            delays = [v["delay_ticks"] for v in visits if numeric(v["delay_ticks"])]
            by_library[lib] = (len(visits), len(delays), median(delays) if delays else None)
        a, b = (by_library[lib] for lib in LIBS)
        if a[0] or b[0]:
            comparison = "Unequal visit counts"
            if a[0] == b[0]:
                comparison = observed(a[1], b[1], "higher")
            rows.append((pretty(mode), f"{a[1]}/{a[0]}", f"{b[1]}/{b[0]}", comparison))
            comparable = a[1] == a[0] == b[0] == b[1] and a[0] > 0
            rows.append(("Median ticks among recovered visits", fmt(a[2]), fmt(b[2]),
                         observed(a[2], b[2], "lower") if comparable else "Unrecovered visits limit the speed comparison"))
    if not rows:
        return ""
    return section("Recovery after a change",
        "<p>Recovery starts when the transfer rate reaches the changed level. The estimate must stay "
        "within 2% of it for five consecutive successful updates. More recovered visits and fewer ticks "
        "to recovery are better. Visits that ended first are counted as unrecovered.</p>"
        + table("Recovery on changed visits", ["Measure", "CFFG", "RxInfer", "Observed result"], rows))


def group_results(groups, runs):
    pairs = {}
    for row in groups:
        library = row["library"]
        if library in LIBS:
            key = (row["mode"], row["condition"], row["arm"])
            pair = pairs.setdefault(key, {})
            if library in pair:
                raise ValueError(f"Duplicate scored group: {key}/{library}")
            pair[library] = row
    bodies = []
    for (mode, condition, arm), pair in sorted(pairs.items()):
        first, second = (pair.get(lib, {}) for lib in LIBS)
        rows = []
        for key, (label, direction, meaning) in METRICS.items():
            if mode == "tape" and key in {"tracking_rmse", "effort", "flow_movement", "holds", "late", "flow_violation_ticks"}:
                continue
            values = [r.get("overall", {}).get(key) for r in (first, second)]
            if any(map(numeric, values)):
                verdict = observed(*values, direction)
                if not same_seeds(runs, mode, condition, arm):
                    verdict = "Incomplete pairing. Comparison inconclusive."
                rows.append((label, fmt(values[0]), fmt(values[1]), verdict, meaning))
        if rows:
            title = f"{pretty(mode)} · {pretty(condition)} · {ARMS.get(arm, pretty(arm))}"
            counts = f'CFFG {first.get("n_seeds", 0)} seeds, RxInfer {second.get("n_seeds", 0)} seeds.'
            bodies.append(disclosure(title, f"<p>{txt(counts)}</p>"+table(title,
                ["Whole-run measure", "CFFG", "RxInfer", "Observed result", "How to read it"], rows)))
    if not bodies:
        return ""
    return section("Results by plant condition and filter",
        "<p>The filter can estimate the transfer rate, keep it fixed at its nominal value, or receive its "
        "true value. A tape replays the same recorded flows. In control runs each method sets its own flows.</p>"
        + "".join(bodies))


def readiness(pilot, protocol, development):
    rows = gate_rows(pilot)
    criteria = protocol.get("gates", {})
    if rows:
        body = table("Recorded pilot gates", ["Development check", "Outcome", "Evidence"], rows)
    else:
        body = "<p>The pilot has not recorded its gate decisions.</p>"
    if isinstance(criteria, dict) and criteria:
        body += disclosure("Criteria set before development", "<ul>" + "".join(
            f"<li><strong>{txt(pretty(k).capitalize())}.</strong> {txt(v)}</li>" for k, v in criteria.items()) + "</ul>")
    headroom = []
    dev = development.get("groups") or []
    for lib in LIBS:
        errors = {g["arm"]: (g.get("changed") or {}).get("hidden_rmse") for g in dev
                  if (g["library"], g["mode"], g["condition"]) == (lib, "tape", "changing")}
        fixed, informed = (errors.get(k) for k in ("frozen", "informed"))
        if not all(map(numeric, (fixed, informed))):
            continue
        gains = (pilot.get("development", {}).get(lib, {}).get("headroom_percent") or [])
        gain = f"{fmt(fmean(gains))}%" if gains and all(map(numeric, gains)) else "—"
        if fmt(informed) == fmt(fixed):
            result = "Same to displayed precision"
        else:
            result = "Knowing the rate reduced error" if informed < fixed else "The fixed rate gave lower error"
        headroom.append((NAMES[lib], fmt(fixed), fmt(informed), gain, result))
    if headroom:
        body += ("<h3>How much could knowing the transfer rate help?</h3>"
                 "<p>On development recordings, the informed filter received the true transfer rate. "
                 "Lower changed-period hidden error is better. Each error is in concentration units. "
                 "The pilot averages the percentage improvement over individual seed pairs.</p>" +
                 table("Development headroom", ["Library", "Fixed-rate error", "Informed error",
                       "Mean paired reduction", "Observed result"], headroom))
    forecast = pilot.get("projected_serial_episode_hours")
    if numeric(forecast):
        body += (f"<p>The proposed fresh-seed episodes were projected to take {fmt(forecast)} hours "
                 "running one at a time. This estimate scales development episode times and excludes "
                 "worker startup, scoring and file writes.</p>")
    return section("Pilot readiness checks", body)


def coverage(data):
    inv = data.get("inventory") or {}
    rows = []
    for unit, numerator, denominator in [
            ("Library runs", "scored_runs", "expected_runs"),
            ("Steps across runs", "scored_ticks", "expected_ticks")]:
        done, expected = inv.get(numerator), inv.get(denominator)
        if numeric(done) and numeric(expected):
            share = f"{100*done/expected:.1f}%" if expected else "—"
            rows.append((unit, done, expected, share))
    end = data.get("endurance_inventory") or {}
    if numeric(end.get("scored_runs")) and numeric(end.get("expected_runs")):
        rows.append(("Separate long development runs", end["scored_runs"], end["expected_runs"],
                     f'{100*end["scored_runs"]/end["expected_runs"]:.1f}%' if end["expected_runs"] else "—"))
    if not rows:
        return ""
    return section("How much evidence was scored", table("Scored inventory",
        ["Unit", "Scored", "Scheduled", "Share"], rows) +
        "<p>A scored step can include a failed update. The fallback keeps that step in the error calculation. "
        "A predictive density that could not be calculated needs a separate count.</p>")


def cold_times(root, sealed_hashes=None):
    cold = []
    for lib in LIBS:
        path = root/"qual"/f"{lib}.json"
        if sealed_hashes is not None:
            expected = sealed_hashes.get(path.name)
            if not expected:
                cold.append(None)
                continue
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Qualification clock differs from the freeze: {path.name}")
        record = read(path) if path.is_file() else {}
        cold.append(record.get("cold_real_action_s") if record.get("passed") else None)
    return cold


def compute(cold, data):
    rows = []
    if any(map(numeric, cold)):
        rows.append(("Qualification · first real action (s)", fmt(cold[0]), fmt(cold[1]), observed(*cold, "lower"),
                     "Fresh process through first action including physics, graph setup and compilation"))

    regular = [r for r in data.get("runs", []) if
               (r["mode"], r["condition"], r["arm"]) == ("control", "changing", "adaptive")]
    long_runs = data.get("endurance", [])
    for title, runs in (("Study runs", regular), ("Development endurance test", long_runs)):
        if not runs:
            continue
        for key, label, how, reduce in (
                ("cold_s", "Process launch to first synthetic plan (s)", "median over workers", median),
                ("peak_rss_mib", "Peak sampled working memory (MiB)", "maximum over workers and their child processes", max),
                ("episode_s", "Episode wall time (s)", "median across complete runs including simulation", median)):
            pair, counts = [], []
            for lib in LIBS:
                values = [r["meta"][key] for r in runs if r["library"] == lib and numeric(r.get("meta", {}).get(key))]
                counts.append(len(values))
                pair.append(reduce(values) if values else None)
            if any(counts):
                verdict = observed(*pair, "lower") if counts[0] == counts[1] else "Unequal measurement counts"
                rows.append((f"{title} · {label}", fmt(pair[0]), fmt(pair[1]), verdict, how.capitalize()))
    if not rows:
        return ""
    return section("Startup, memory and run time",
        "<p>The cold qualification times each library once. Episode costs use changing-plant runs with the adaptive controller. Lower is better for "
        "each cost. Sampled memory gives a lower bound on a process tree’s instantaneous peak. "
        "A synthetic plan uses test inputs before the episode starts.</p>" + table("Measured resources",
        ["Quantity", "CFFG", "RxInfer", "Observed result", "Measurement boundary"], rows))


def design(protocol):
    dt, preview = protocol.get("dt_model_hours"), protocol.get("horizon")
    clock = []
    if numeric(dt):
        clock.append(f'One step represents {fmt(dt)} model hours ({fmt(60*dt)} simulated minutes).')
    sd = protocol.get("sensor_sd")
    if isinstance(sd, list) and len(sd) == 2 and all(map(numeric, sd)):
        clock.append(f"Additive sensor noise has standard deviation {fmt(sd[0])} on X5 and {fmt(sd[1])} on Y1 "
                     "(about half a percent at nominal concentrations). The earlier feasibility sketch used 5%.")
    short, long = protocol.get("short_steps"), protocol.get("long_steps")
    seeds = protocol.get("reserved_heldout_seeds")
    if numeric(short) and numeric(long) and isinstance(seeds, list):
        clock.append(f"The design uses {short:,} steps for short development episodes, {long:,} for "
                     f"long episodes and {len(seeds)} reserved fresh seeds.")
    if numeric(preview):
        clock.append(f'Controllers look ahead {preview} steps when choosing the two flow commands.')
    if numeric(protocol.get("deadline_s")):
        clock.append(f'The decision deadline is {fmt(protocol["deadline_s"])} wall-clock seconds.')
    return section("What the methods can see and do",
        '<p>Liquid and gas pass through five stages. Each stage has two concentrations. Sensors measure '
        'the liquid leaving stage five (X5) and the gas leaving stage one (Y1). The scorekeeper sees the '
        'other eight concentrations and the rate of transfer between liquid and gas (Kla).</p>'
        '<p>The adaptive filter updates its estimate of concentrations and transfer rate using the newest '
        'reading and the flows actually applied. The fixed-rate filter assumes the nominal rate throughout. '
        'The informed filter gets the current true rate and still has to estimate the concentrations. '
        'Shared Python code makes a local straight-line approximation to the plant dynamics and passes these coefficients to each library. CFFG and RxInfer then compute their own Gaussian updates and plans.</p>'
        '<p>CFFG uses its production directed Gaussian message operations in a prearranged graph, checked '
        'against the public Engine. Full Engine.run dispatch falls outside this clock. RxInfer uses cached RxInfer.jl factor graphs '
        'in a Julia worker that stays running. Warm RxInfer times include the Python–Julia round trip. '
        'The probabilities in the report come from these local Gaussian approximations. A separate short-window '
        'nonlinear reference checks their accuracy at selected development inputs.</p>'
        '<p>On shared recordings, the two filters get identical evidence. In a separate control test, each '
        'planner chooses liquid and gas flows to track a target for X5. Those choices change what each '
        'controller sees next. Future targets enter the planner; only past and current sensor readings '
        'inform the filter.</p>' + (f'<p>{txt(" ".join(clock))}</p>' if clock else '') +
        '<p>The warm control clock covers updating the belief, preparing the local model, planning, '
        'applying flow bounds and reading the commands. Simulation, file writes and scoring happen '
        'outside that clock. On failure or deadline miss, the plant reuses its previous feasible flows.</p>')


def evidence(root, finished, stopped, revision):
    paths = [("protocol.json", "Experiment design and decision criteria"),
             ("pilot.json", "Pilot decision and gate evidence"),
             ("qual/cffg.json", "CFFG numerical and runtime checks"),
             ("qual/rxinfer.json", "RxInfer numerical and runtime checks"),
             ("qual/nonlinear.json", "Short nonlinear reference and numerical checks"),
             ("qual/plant-qualification.json", "Outlet information, hidden-state headroom and plant verification"),
             ("development-completed.json", "Development run inventory and hashes"),
             ("endurance-completed.json", "Long development run inventory and hashes"),
             ("endurance/", "Long development recordings"),
             ("workers/development/", "Development worker records"),
             ("workers/endurance/", "Endurance worker records"),
             ("development/analysis.json", "Development scores, timings and failures by run"),
             ("development/", "Development recordings and results"),
             ("sources.json", "Source file digests"), ("sources/", "Measurement source and dependency manifests"),
             ("environment.json", "Software and hardware environment"),
             ("freeze.json", "Frozen fresh-seed design")]
    if finished:
        paths.extend([("completed.json", "Fresh-seed completion record"), ("analysis.json", "Final analysis and paired seeds"),
                      ("heldout/", "Fresh-seed recordings"), ("workers/heldout/", "Fresh-seed worker records")])
    elif stopped:
        paths.append(("analysis.json", "Final pilot-only analysis"))
    links = [f'<li><a href="{path}">{txt(label)}</a></li>' for path, label in paths if (root/path).exists()]
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
        url = f"https://github.com/engimatic-systems/cffg/tree/{revision}/tools/extraction"
        links.append(f'<li><a href="{url}">Benchmark source at recorded revision {revision[:7]}</a></li>')
    return section("Evidence and sources", "<ul>" + "".join(links) + "</ul>") if links else ""


def render(root):
    root = Path(root)
    protocol = read(root/"protocol.json")
    pilot = read(root/"pilot.json") if (root/"pilot.json").is_file() else {}
    status = pilot.get("status")
    complete = (root/"completed.json").is_file()
    dev_path = root/"development/analysis.json"
    development = read(dev_path) if dev_path.is_file() else {}
    if development and development.get("stage") != "development":
        raise ValueError("Expected development scores under development/")
    if complete:
        if status != "go" or not (root/"freeze.json").is_file():
            raise ValueError("Fresh-seed completion requires a successful pilot and a frozen design")
        freeze = read(root/"freeze.json")
        marker = read(root/"completed.json")
        for name, owner in (("freeze", marker), ("protocol", freeze), ("pilot", freeze)):
            if owner.get(f"{name}_sha256") != sha256((root/f"{name}.json").read_bytes()).hexdigest():
                raise ValueError(f"Frozen {name} hash differs from completion evidence")
        data = read(root/"analysis.json")
        if data.get("stage") != "heldout" or data.get("status") != "complete":
            raise ValueError("Completion marker and fresh-seed analysis disagree")
        if data["inventory"]["sha256"] != marker["runs"]:
            raise ValueError("Scored trace hashes differ from the completion record")
        label = "Fresh-seed benchmark"
        cold = cold_times(root, freeze.get("qualification_sha256", {}))
        seeds = freeze["seeds"]
        errors = learning_errors(data, seeds)
        opening = f"<p>{txt(learning_summary(errors))}</p><p>{txt(speed_summary(data, cold))}</p>"
        margin = freeze.get("practical_margin", (data.get("primary") or {}).get("margin"))
        main_result = primary(data, margin, opening) + learning_figure(root, errors, len(seeds))
        intro = ""
    else:
        cold = cold_times(root)
        data = development
        main_result = ""
        if status == "no_go":
            label = "Development pilot · stopped"
            failed = [name for name, outcome, _ in gate_rows(pilot) if outcome == "Failed"]
            reason = pilot.get("reason")
            if not reason and failed:
                reason = "The unmet checks were " + ", ".join(name.rstrip("?") for name in failed) + "."
            why = " " + txt(reason) if reason else ""
            intro = f'<p class="finding">The pilot stopped before fresh-seed evaluation.{why}</p>'
            if (root/"analysis.json").is_file():
                stopped = read(root/"analysis.json")
                if stopped.get("stage") != "stopped":
                    raise ValueError("Pilot stop and root analysis disagree")
                data = stopped
        elif status == "go":
            label = "Development pilot · ready for evaluation"
            intro = '<p class="finding">The pilot passed. A completed fresh-seed comparison is pending.</p>'
        else:
            label = "Development pilot · in progress"
            intro = '<p class="finding">The study has not yet recorded a final pilot decision.</p>'
        intro += '<p>Development runs test whether this experiment can distinguish the methods. A winner on fresh seeds remains undetermined.</p>'
    body = (f'<header><p class="status">{txt(label)}</p><h1>Inferring the inside of an extraction column</h1>'
            '<p>Can CFFG or RxInfer estimate eight hidden concentrations more accurately from two noisy '
            'outlet readings as the transfer rate changes?</p>' + intro + '</header>' + main_result +
            highlights(data.get("groups") or [], data.get("runs") or [], protocol, complete=complete) + readiness(pilot, protocol, development if complete else data) +
            coverage(data) + visit_results(data) + recovery(data.get("runs") or []) + compute(cold, data) + group_results(data.get("groups") or [], data.get("runs") or []) +
            design(protocol) + evidence(root, complete, status == "no_go", pilot.get("source_revision")))
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Extraction column · CFFG and RxInfer</title><style>{STYLE}</style></head><body><main>{body}'
            '</main></body></html>')
    dest = root/"index.html"
    stream = NamedTemporaryFile(dir=root, prefix=".report-", delete=False, mode="w", encoding="utf-8")
    tmp = Path(stream.name)
    try:
        with stream:
            stream.write(page)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, metavar="ROOT")
    print(render(parser.parse_args().root.expanduser().resolve()))
