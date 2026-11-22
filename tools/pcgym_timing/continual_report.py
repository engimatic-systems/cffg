"""Render the changing-reactor development results from audited numerical artifacts."""
import argparse
from html import escape
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    def cells(tag, values):
        return "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values)
    return ('<div class="scroll"><table><thead><tr>' + cells("th", headers) + "</tr></thead><tbody>"
            + "".join("<tr>" + cells("td", row) + "</tr>" for row in rows) + "</tbody></table></div>")


def chart(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    rows = [json.loads(l) for l in (out/"seed-6401/switching-learned.jsonl").read_text().splitlines()]
    figure, (ua, gains) = plt.subplots(2, 1, figsize=(8.4, 5.4), constrained_layout=True,
                                      gridspec_kw={"height_ratios": [1.5, 1]})
    ua.plot([r["ua"] for r in rows], color="#14647a", linewidth=1, label="Estimate before each action")
    ua.step(range(len(rows)), [r["true_ua"] for r in rows], where="post", color="#af4c2b",
            linewidth=2, label="Hidden plant coefficient")
    ua.set(xlabel="Decision tick", ylabel="Heat transfer (UA)", title="Parameter tracking on seed 6401")
    ua.legend(frameon=False, fontsize=9, ncol=2)
    for arm, marker, offset, color in (("learned", "o", -.08, "#14647a"), ("informed", "s", .08, "#af4c2b")):
        pairs = summary["paired"]["switching"][arm]["during"]["seeds"]
        seeds = sorted(pairs)
        gains.scatter(np.arange(len(seeds)) + offset, [pairs[s]["tracking_reduction_pct"] for s in seeds],
                      color=color, marker=marker, label=arm.title(), zorder=3)
    gains.set(xticks=range(len(seeds)), xticklabels=seeds, ylabel="Reduction in RMSE (%)",
              xlabel="Paired development seed", title="Concentration tracking relative to the frozen controller")
    gains.axhline(0, linewidth=.8, color="#48545a")
    gains.axhline(1, linewidth=.8, linestyle=":", color="#69767b", label="1% screening threshold")
    gains.legend(frameon=False, fontsize=9, ncol=3)
    for axis in (ua, gains):
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    figure.savefig(out/"parameter-and-control.png", dpi=165)
    plt.close(figure)


def render(out):
    summary = read(out/"analysis.json")
    study = summary["study"]
    seeds = list(map(str, study["seeds"]))
    sections = []
    def heading(title, content):
        sections.append(f"<section><h2>{escape(title)}</h2>{content}</section>")

    checks = summary["gates"]
    learned = checks["learned_control_benefit"]
    informed = checks["informed_control_headroom"]
    ratio = checks["stationary_penalty"]["learned_over_frozen"]
    ident = checks["settled_parameter_error"]
    t = checks["changed_tape_temperature"]["rmse_K"]
    reliability = checks["reliability"]
    decision = "Proceed to a larger control comparison" if summary["proceed"] else "Hold the larger control comparison"
    changed_direction = "lower" if learned["median_reduction_pct"] >= 0 else "higher"
    stationary_direction = "higher" if ratio >= 1 else "lower"
    sections.append(f"""<p class="lead"><b>{decision}.</b> With a hidden 3.7% drop in heat transfer, the learner’s
    median concentration tracking error was {abs(learned['median_reduction_pct']):.2f}% {changed_direction}
    than the frozen controller’s during the change. On the stationary plant, its mean tracking error was
    {abs(100*(ratio-1)):.2f}% {stationary_direction} over the full run.</p>""")
    sections.append(f"""<p>Thirty uninterrupted episodes covered {len(seeds)*6*study['steps']:,} decisions across
    five paired development seeds. The warmed one-second deadline included the parameter update, installation
    in CFFG, planning and action clipping. The controllers recorded {reliability['failures']} failed solves,
    {reliability['misses']} missed deadlines and {reliability['holds']} held commands.</p>""")
    gate_values = [
        ("informed_control_headroom", "Value of knowing the true coefficient",
         f"{informed['median_reduction_pct']:+.2f}% tracking improvement, {informed['lower_seeds']}/5 wins", "≥1% median and ≥4 wins"),
        ("learned_control_benefit", "Value of learning the coefficient",
         f"{learned['median_reduction_pct']:+.2f}% tracking improvement, {learned['lower_seeds']}/5 wins", "≥1% median and ≥4 wins"),
        ("stationary_penalty", "Stationary learned / frozen tracking RMSE", f"{ratio:.4f}", "≤1.02"),
        ("settled_parameter_error", "Settled parameter error reduction", f"{ident['reduction_pct']:.1f}%", "≥50%"),
        ("changed_tape_temperature", "Temperature prediction RMSE on shared tape",
         f"{t['learned']:.3f} K learned, {t['frozen']:.3f} K frozen", "Learned < frozen"),
        ("reliability", "Control reliability", f"{reliability['holds']} holds", "Zero holds or failures")]
    missed = [label for k, label, _, _ in gate_values if not checks[k]["passed"]]
    if missed:
        sections.append("<p><b>Criteria outside the prespecified bounds</b> · "
                        + escape(", ".join(missed)) + ".</p>")
    heading("Decision criteria", table(["Criterion", "Measured", "Required", "Outcome"],
        [[label, value, threshold, "Pass" if checks[k]["passed"] else "Miss"]
         for k, label, value, threshold in gate_values]) +
        "<p>The thresholds were fixed before collecting outcomes. Positive tracking improvement means a reduction "
        "in true concentration RMSE relative to frozen. Each seed supplies one matched comparison. On the "
        "stationary plant the frozen controller starts with the correct coefficient, so the stationary test "
        "measures the cost of keeping learning active on an already calibrated system.</p>")
    sections.append('<figure><img src="parameter-and-control.png" alt="Heat-transfer estimates over time on seed 6401 '
                    'and paired concentration tracking improvements on all five seeds."></figure>')

    def mean(condition, arm, period, metric):
        return np.mean([summary["episodes"][s][condition][arm]["windows"][period][metric] for s in seeds])
    control = []
    recovery_rows = []
    for arm in ("frozen", "learned", "informed"):
        lat = summary["pooled_latency_s"]["switching"][arm]["decision_s"]
        control.append([arm.title(), f'{mean("stationary", arm, "full", "tracking_rmse"):.6f}',
                        f'{mean("switching", arm, "during", "tracking_rmse"):.6f}',
                        f'{mean("switching", arm, "during", "return_sum"):.4f}',
                        f'{mean("switching", arm, "after", "tracking_rmse"):.6f}',
                        f'{lat["median"]*1000:.2f}', f'{lat["p95"]*1000:.2f}'])
    for s in seeds:
        a = summary["episodes"][s]["switching"]["learned"]
        pair = summary["paired"]["switching"]["learned"]["during"]["seeds"][s]
        recovery_rows.append([s, a["recovery"]["during"]["delay"], a["recovery"]["after"]["delay"],
                              f'{a["windows"]["changed_plateau"]["ua_mae"]:.1f}',
                              f'{pair["tracking_reduction_pct"]:+.2f}'])
    heading("Closed-loop performance", table(
        ["Controller", "Stationary RMSE", "Changed RMSE", "Changed return ↑", "After return RMSE",
         "Median decision (ms)", "95th percentile (ms)"], control) +
        "<p>Concentration RMSE is in mol/L. The changed segment spans 195 decisions. The return segment "
        "starts when heat transfer returns to nominal. Each controller saw its own state trajectory with "
        "the same exogenous sensor-noise draws per seed.</p>" +
        table(["Seed", "Recovery after drop (ticks)", "Recovery after return (ticks)",
               "Settled mean absolute UA error", "Changed tracking improvement (%)"],
              [["Unreached" if v is None else v for v in row] for row in recovery_rows]) +
        "<p>Recovery begins at the first of five consecutive pre-action estimates within 1% of the current "
        "coefficient. The settled window begins 25 ticks after the drop.</p>")

    prediction_rows = []
    for c in ("stationary","switching"):
        for a in ("frozen","learned","grid","informed"):
            w = summary["tapes"][c]["methods"][a]["windows"]["during"]
            f = summary["forecasts"][c]
            prediction_rows.append([c.title(), a.title(), f'{w["observation_prediction_rmse"]["T"]:.3f}',
                f'{w["ua_mae"]:.1f}', f'{f["15"]["methods"][a]["T"]:.3f}', f'{f["60"]["methods"][a]["T"]:.3f}'])
    heading("Predictions on a shared stream", table(["Tape", "Estimate", "One-step sensor RMSE (K)",
            "Mean absolute UA error", "15-step state RMSE (K)", "60-step state RMSE (K)"], prediction_rows) +
            "<p>One-step error and coefficient error use ticks 178–372 on both tapes. Longer forecasts use all "
            "prespecified origins, known future coolant commands and the estimate available at the origin. "
            "Forecasts spanning a switch have no advance warning of it, including the informed comparator.</p>")

    heading("What was learned", "<p>The only unknown quantity was UA, the reactor’s heat-transfer coefficient "
            "times area. A scalar extended Kalman filter estimated it continuously from temperature prediction "
            "errors. Its prior centered on 50,000. The hidden changed value, 48,150, was also between nodes of "
            "the independent grid reference. Each update used one newly observed transition and the actual "
            "coolant command. The predictor propagates the previous sensors’ uncertainty through the reactor "
            "dynamics. Adjacent transitions share a noisy endpoint, so the sequence likelihood is approximate.</p>"
            "<p>The learner belongs to the benchmark harness. It installs its estimate into the 19 tied transitions "
            "through CFFG’s public parameter API. All three controllers retain the same certified directed planner, "
            "concentration bias, four-step history, fifteen-step goal preview, clipping and fallback. These five "
            "development seeds vary sensor noise on one plant and one change schedule.</p>")
    if missed == ["Stationary learned / frozen tracking RMSE"]:
        heading("Next experiment", "<p>Keep this estimator as the development baseline. Test whether the learner "
                "can hold a steady estimate when measurements look like ordinary sensor noise, while still "
                "reacting quickly when consecutive temperature errors point in the same direction. Freeze that "
                "rule on a development tape, then repeat the stationary and switching comparisons on fresh seeds. "
                "The existing stationary-cost and recovery criteria measure the tradeoff.</p>")
    links = [("analysis.json","Complete numerical results"), ("report.txt","Numerical report"),
             ("study.json","Frozen protocol and source hashes"), ("qualified.json","Plant and planner checks"),
             ("tape-switching.jsonl","Chronological changing-plant tape"), ("completed.json","Campaign completion")]
    links.extend((f"seed-{s}/",f"Seed {s} traces") for s in seeds)
    heading("Evidence", "<ul>"+"".join(f'<li><a href="{p}">{label}</a></li>' for p,label in links)+"</ul>")
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Learning a changing reactor</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#1b2e36;max-width:1050px;
margin:3rem auto;padding:0 1rem}h1,h2{line-height:1.2}h2{margin-top:2.2rem}p{max-width:88ch}.lead{font-size:1.2rem}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid #c6d7d6;padding:.46em;text-align:right}th{background:#edf3f1}td:first-child,th:first-child{text-align:left}
img{max-width:100%;height:auto}figure{margin:2rem 0}a{color:#075a71}</style><main><h1>Learning a changing reactor</h1>"""
    chart(out, summary)
    (out/"index.html").write_text(html+"".join(sections)+"</main></html>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args().output.expanduser().resolve())
