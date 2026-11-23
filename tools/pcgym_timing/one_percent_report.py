"""Render the frozen one-percent reactor comparison from its audited JSON results."""

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

if __package__:
    from .paired_report import STYLE, details, table
else:
    from paired_report import STYLE, details, table


def read(path):
    return json.loads(path.read_bytes())


def fmt(value, decimals=2, *, signed=False):
    if value is None:
        return "—"
    return format(value, f"{'+' if signed else ''}.{decimals}f").replace("-", "−")


def pct(value, *, signed=False, decimals=2):
    return "—" if value is None else fmt(value, decimals, signed=signed) + "%"


def result(passed):
    return "Pass" if passed else "Hold"


def development_section(dev, selected):
    selected_row = next(row for row in dev["candidates"] if row["label"] == selected["label"])
    if selected_row["q"] != selected["motion_sd"]:
        raise ValueError("Frozen process SD disagrees with development result")
    original = selected_row["summary"]["original"]
    rows = [["0.0120 (original)", pct(original["stationary_penalty_pct"], signed=True),
             pct(original["changed_gain_pct"], signed=True), "Reference", "Reference", "Reference", "Reference"]]
    for row in sorted(dev["candidates"], key=lambda r: r["q"], reverse=True):
        label = fmt(row["q"], 4)
        if row["label"] == selected["label"]:
            label += " (selected)"
        rows.append([label, pct(row["stationary_penalty_pct"], signed=True),
                     pct(row["changed_gain_pct"], signed=True), pct(row["retained_gain_pct"]),
                     slowdown(row, "during"), slowdown(row, "after"),
                     result(row["eligible"])])
    selection = (f'The {fmt(selected["motion_sd"], 4)} setting passed the development screen.'
                 if selected_row["eligible"] else
                 f'All four settings missed at least one development requirement. The fallback rule '
                 f'advanced {fmt(selected["motion_sd"], 4)} for an independent validation.')
    return "".join([
        '<section id="development"><h2>How the candidate was chosen</h2>',
        '<p>All learners estimated the reactor’s heat-transfer coefficient (UA) from each new temperature '
        'measurement. The original learner allowed a process standard deviation of 0.012 in log UA per '
        'transition. A smaller value expresses a belief that UA usually moves less between observations '
        'and gives a single noisy temperature reading less influence.</p>',
        '<p>Four gentler settings were screened on the five development seeds. ', selection,
        ' The ranking favored fewer failed requirements, then less excess stationary penalty above 1%, '
        'then the largest process SD. The choice was frozen before the fresh data were opened.</p>',
        table("Five-seed development screen", ["Process SD", "Stationary penalty", "Changed tracking gain",
              "Original gain retained", "Drop slowdown (ticks)", "Return slowdown (ticks)", "Joint screen"], rows),
        '<p class="note">Positive stationary penalty means worse tracking. Positive changed gain means better '
        'tracking than the frozen controller. Positive slowdown means more ticks to recover than the original '
        'learner. An unrecovered candidate on either switch fails the recovery requirement.</p></section>'])


def tradeoff_figure(path, dev, validation):
    """Compare the same settings on development and validation at the two measured outcomes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.35), layout="constrained")
    cases = sorted(dev["candidates"], key=lambda r: r["q"])
    original, candidate = (validation["summary"][name] for name in ("original", "candidate"))
    for ax, key, title, ylabel in zip(
        axes, ("stationary_penalty_pct", "changed_gain_pct"),
        ("Stationary plant", "Hidden UA drop"),
        ("Tracking penalty vs frozen (%)", "Median tracking gain vs frozen (%)"), strict=True,
    ):
        ax.scatter([r["q"] for r in cases] + [.012], [r[key] for r in cases] + [cases[0]["summary"]["original"][key]],
                   label="Development settings",
                   marker="o", s=38, edgecolor="#496969", facecolor="none", linewidth=1.4, zorder=3)
        ax.scatter([validation["q"]], [candidate[key]], color="#196453", marker="D", s=48,
                   label="Selected · fresh seeds", zorder=4)
        ax.scatter([.012], [original[key]], color="#9a4729", marker="s", s=44,
                   label="Original · fresh seeds", zorder=4)
        ax.set(title=title, xlabel="Assumed log-UA process SD per tick", ylabel=ylabel)
        ax.set_xticks([.0015, .003, .006, .010, .012], [".0015", ".003", ".006", ".010", ".012"])
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].axhline(1, color="#535e5b", linestyle="--", linewidth=1, label="1% allowance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncols=2, fontsize=9)
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def recovery_text(value):
    return "Unrecovered" if value is None else str(value)


def slowdown(row, switch):
    delta = row["recovery_delay_delta"][switch]
    if delta is None:
        return "—"
    qualifier = "≤" if row["gates"]["recovery"]["switches"][switch]["upper_bound"] else ""
    return qualifier + fmt(delta, 1, signed=True)


def seed_evidence(per_seed):
    rows = []
    for seed, records in sorted(per_seed.items(), key=lambda item: int(item[0])):
        old, new = records["original"], records["candidate"]
        rows.append([seed, pct(old["stationary_penalty_pct"], signed=True),
                     pct(new["stationary_penalty_pct"], signed=True), pct(old["changed_gain_pct"], signed=True),
                     pct(new["changed_gain_pct"], signed=True),
                     recovery_text(old["recovery_delay"]["during"]), recovery_text(new["recovery_delay"]["during"]),
                     recovery_text(old["recovery_delay"]["after"]), recovery_text(new["recovery_delay"]["after"])])
    return details("Inspect all paired validation seeds",
                   table("Per-seed tracking and recovery", ["Seed", "Original penalty", "Candidate penalty",
                         "Original gain", "Candidate gain",
                         "Original drop (ticks)", "Candidate drop (ticks)", "Original return (ticks)",
                         "Candidate return (ticks)"], rows))


def gate_table(val):
    gates = val["gates"]
    arms = val["summary"]
    new, base = arms["candidate"], arms["frozen"]
    count = len(val["per_seed"])

    def tracking(arm):
        return f'{pct(arm["changed_gain_pct"])} gain, {arm["changed_wins"]}/{count} wins'

    missing = new["unrecovered"]
    recovery = (f'{slowdown(val, "during")} / {slowdown(val, "after")} ticks, '
                f'{missing["during"]} / {missing["after"]} unrecovered')
    reliability = gates["reliability"]
    rows = [
        ("stationary_penalty", "Stationary penalty vs frozen", pct(val["stationary_penalty_pct"], signed=True, decimals=3),
         "At most 1%"),
        ("retention", "Original changed benefit retained", pct(val["retained_gain_pct"]), "At least 90%"),
        ("recovery", "Slowdown after drop / return", recovery, "Each ≤5 ticks, all recovered"),
        ("candidate_control_benefit", "Candidate gain vs frozen", tracking(new), "≥1% gain, ≥80% wins"),
        ("informed_control_headroom", "Informed gain vs frozen", tracking(arms["informed"]), "≥1% gain, ≥80% wins"),
        ("settled_parameter_error", "Settled UA error vs frozen",
         f'{fmt(new["settled_ua_mae"], 1)} vs {fmt(base["settled_ua_mae"], 1)}', "At least 50% lower"),
        ("changed_tape_temperature", "Changed-tape temperature RMSE",
         f'{fmt(new["changed_tape_T_rmse_K"], 4)} vs {fmt(base["changed_tape_T_rmse_K"], 4)} K',
         "Lower than frozen"),
        ("reliability", "Closed-loop reliability",
         f'{reliability["holds"]} holds, {reliability["failures"]} failures, {reliability["misses"]} misses',
         "All zero"),
    ]
    missed = [name for key, name, _, _ in rows if not gates[key]["passed"]]
    outcome = ("The selected learner passed every primary requirement." if not missed else
               "The primary requirements missed were " + ", ".join(missed) + ".")
    return (f"<p><strong>{escape(outcome)}</strong></p>"
            + table("Prespecified validation requirements", ["Requirement", "Measured", "Target", "Decision"],
                    [[name, value, target, result(gates[key]["passed"])] for key, name, value, target in rows]))


def validation_sections(val, image_name):
    arms = val["summary"]
    original, candidate = arms["original"], arms["candidate"]
    budget = 1
    threshold = val["thresholds"][str(budget)]
    if threshold != val["eligible"]:
        raise ValueError("Validation eligibility disagrees with the primary stationary budget")
    old_penalty = original["stationary_penalty_pct"]
    penalty = candidate["stationary_penalty_pct"]
    boot = val["paired_intervals"]
    stationary_intervals = []
    for arm, label in (("candidate", "Selected learner"), ("original", "Original learner")):
        low, high = boot["arms"][arm]["stationary_penalty_pct"]
        bounds = f'{pct(low, signed=True, decimals=3)} to {pct(high, signed=True, decimals=3)}'
        stationary_intervals.append([label, pct(arms[arm]["stationary_penalty_pct"], signed=True, decimals=3), bounds])
    if old_penalty <= budget and penalty <= budget:
        meaning = "Both settings met the stationary allowance on these seeds."
    elif old_penalty > budget and penalty <= budget:
        meaning = "Only the gentler setting met the stationary allowance on these seeds."
    elif old_penalty <= budget:
        meaning = "Only the original setting met the stationary allowance on these seeds."
    else:
        meaning = "Both settings exceeded the stationary allowance on these seeds."
    title = "Proceed to the next comparison" if threshold else "Hold the larger comparison"
    names = {"frozen": "Frozen", "original": "Original learner", "candidate": "Selected learner",
             "informed": "Informed"}
    arm_rows = []
    absolute = []
    for arm, label in names.items():
        scores = arms[arm]
        arm_rows.append([label, pct(scores["stationary_penalty_pct"], signed=True),
                         pct(scores["changed_gain_pct"], signed=True), scores["changed_wins"],
                         fmt(scores["settled_ua_mae"], 1), fmt(scores["changed_tape_T_rmse_K"], 4)])
        absolute.append([label, fmt(scores["stationary_tracking_rmse"], 6),
                         fmt(scores["changed_tracking_rmse"], 6),
                         fmt(scores["median_recovery_delay"]["during"], 1),
                         fmt(scores["median_recovery_delay"]["after"], 1)])
    checks = gate_table(val)
    retained = (f'The candidate retained {pct(val["retained_gain_pct"])} of the original’s measured gain.'
                if val["retained_gain_pct"] is not None else
                'The original had no positive gain to express as a retention percentage.')
    return "".join([
        f'<section id="decision"><h2>{title}</h2>',
        f'<p class="takeaway">On the ten fresh seeds, the selected learner’s stationary tracking penalty '
        f'was <strong>{pct(penalty, signed=True, decimals=3)}</strong>. The original learner’s penalty on those same '
        f'seeds was <strong>{pct(old_penalty, signed=True, decimals=3)}</strong>. {meaning}</p>',
        table("Stationary tracking penalty and descriptive paired uncertainty",
              ["Controller", "Stationary penalty vs frozen", "95% paired bootstrap interval"], stationary_intervals),
        f'<p class="note">The 1% decision uses the point estimate. The descriptive intervals use '
        f'{boot["draws"]:,} percentile bootstrap resamples of whole validation seeds (fixed seed {boot["seed"]}). '
        'Each resample keeps all controllers paired and recomputes the ratio of mean tracking RMSEs.</p>',
        f'<p>Gentler tuning changed the stationary penalty by {fmt(penalty - old_penalty, signed=True)} '
        f'percentage points on the fresh seeds. During the hidden change, median tracking gain versus '
        f'frozen was {pct(candidate["changed_gain_pct"], signed=True)} for the candidate and '
        f'{pct(original["changed_gain_pct"], signed=True)} for the original. {retained}</p>',
        '<p>The original learner’s 2.80% penalty in the earlier five-seed pilot motivated this study. '
        'The fresh-seed comparison above gives both learners the same noise draws. Their separate '
        'closed-loop actions can still send the two reactors along different trajectories.</p>',
        '<h3>What the 1% allows</h3><p>The frozen controller keeps UA at its correct initial value. '
        'On the stationary plant, the learner may increase the mean full-episode concentration '
        'tracking RMSE by at most 1% relative to frozen. The penalty uses the ratio of the two '
        'means across seeds. RMSE is the root mean square distance '
        'between true concentration and its setpoint. This allowance measures control quality. '
        'The estimator’s motion setting and the planner’s one-second decision deadline stay separate.</p>',
        checks,
        '<p class="note">Recovery starts at the first of five consecutive pre-action UA estimates within '
        '1% of the new true coefficient. Slowdown is the median across paired seeds of candidate delay '
        'minus original delay, scored separately after the drop and after the return. A ≤ sign marks '
        'an upper bound when an original recovery was unobserved. This second use '
        'of 1% refers to coefficient estimation accuracy.</p>',
        '<h3>Would a different stationary allowance change the decision?</h3>',
        table("Validation decision with the stationary allowance varied",
              ["Maximum stationary penalty", "Joint validation verdict"],
              [[str(limit) + "%", result(val["thresholds"][str(limit)])] for limit in (1, 2, 3)]),
        '<p class="note">The 2% and 3% sensitivity checks change only the allowable stationary tracking '
        'penalty. Every other requirement and the frozen estimator stay fixed.</p></section>',
        '<section id="validation"><h2>All four controllers on fresh seeds</h2>',
        '<p>Each controller used the same planner and received noisy concentration and temperature. '
        'Frozen kept UA at 50,000. Informed received the actual UA and shows the benefit available '
        'from knowing the change. Original and selected inferred UA from realized transitions.</p>',
        table("Held-out tracking, parameter estimation and temperature prediction",
              ["Controller", "Stationary penalty", "Changed tracking gain", "Changed wins vs frozen",
               "Settled mean |UA error|", "Tape temperature RMSE (K)"], arm_rows),
        '<p class="note">Changed tracking gain is the median of each seed’s percentage reduction in '
        'concentration RMSE during the drop. Settled UA error excludes the first 25 changed ticks. '
        'On the fresh switching tape, each method forecasts the next noisy temperature before seeing '
        'it, using the same observations and applied coolant commands. Lower prediction RMSE is better.</p>',
        details("Absolute concentration errors and recovery delays", table("Absolute errors and delays",
                ["Controller", "Stationary mean RMSE (mol/L)", "Changed mean RMSE (mol/L)",
                 "Median drop delay (ticks)", "Median return delay (ticks)"], absolute)),
        f'<figure><img src="{image_name}" alt="Stationary penalty and changed tracking gain by process SD. '
        'The development and validation tables give the plotted values."><figcaption>The original and four '
        'gentler settings on development seeds, with the original and selected settings on fresh seeds.</figcaption></figure>',
        seed_evidence(val["per_seed"]), '</section>'])


def evidence(root):
    items = [
        ("protocol.json", "Prespecified protocol and scoring windows"),
        ("development.json", "Four development candidates, gate results and ranking"),
        ("freeze.json", "Frozen learner and input hashes"),
        ("validation.json", "Fresh-seed analysis, individual outcomes and audits"),
        ("validation/", "Fresh control traces and prediction tape"),
        ("dev/", "Development controller traces and prediction tapes"),
    ]
    links = [f'<li><a href="{path}">{escape(text)}</a></li>' for path, text in items if (root / path).exists()]
    return ('<section id="evidence"><h2>Scope and evidence</h2><p>This experiment used one reactor, one '
            'sensor-noise level and one hidden change schedule. UA fell from 50,000 to 48,150 for '
            'decisions 178–372 and returned to 50,000 at decision 373. Each episode ran for 599 decisions. '
            'Validation used ten paired control seeds (6501–6510) and one independent tape (6500). '
            'The screening verdict applies to this plant and change size.</p><ul>'
            + "".join(links) + '</ul></section>')


def render(root):
    dev_path = root / "development.json"
    dev = read(dev_path)
    freeze = read(root / "freeze.json")
    if sha256(dev_path.read_bytes()).hexdigest() != freeze["development_sha256"]:
        raise ValueError("Development JSON differs from the frozen selection")
    val = read(root / "validation.json")
    selected = freeze["selected"]
    if val["freeze"]["freeze_sha256"] != sha256((root / "freeze.json").read_bytes()).hexdigest():
        raise ValueError("Validation was scored against a different freeze")
    if (val["label"], val["q"]) != (selected["label"], selected["motion_sd"]) or dev["ranking"][0] != selected["label"]:
        raise ValueError("Development ranking or validation setting disagrees with the frozen selection")
    if set(map(int, val["per_seed"])) != set(range(6501, 6511)):
        raise ValueError("Expected the ten fresh validation seeds")
    selection = development_section(dev, selected)
    name = "tracking-tradeoff.png"
    body = "".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Learning reactor heat transfer under a 1% tracking budget</title>',
        f'<style>{STYLE}figure{{margin:1.5rem 0}}figure img{{width:100%;max-width:56rem;height:auto}}'
        'figcaption{color:var(--muted);font-size:.93rem}</style></head><body><main>',
        '<header><h1>Learning reactor heat transfer under a 1% tracking budget</h1>',
        '<p>Can a gentler update spare the controller unnecessary adjustments when the reactor stays '
        'stable and still recover when heat transfer changes?</p><nav aria-label="Report sections">',
        '<a href="#decision">Decision</a><a href="#validation">Fresh-seed results</a>',
        '<a href="#development">Development screen</a><a href="#evidence">Evidence</a></nav></header>',
        validation_sections(val, name), selection, evidence(root), '</main></body></html>'])
    tradeoff_figure(root / name, dev, val)
    target = root / "index.html"
    with NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(body)
            stream.close()
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory containing the three audited JSON files")
    print(render(parser.parse_args().output.expanduser().resolve()))
