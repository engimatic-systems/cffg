"""Render explode.html and its reward chart from the sealed Explode development pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path
import statistics
import tempfile

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import evaluation_report as frozen


STUDY_SHA256 = "b42e006cfe7c88c2f1310b7ca2eb8fe953aeacb7000dd5f5f6819dcdee23041a"
CHART = "explode-block-rewards.png"
BLOCK = 1000
GAME_SOURCE = ("https://github.com/VersesTech/gameworld/blob/85493c477004ecfd3299b0b053884f23d59a5ec5/"
               "gameworld/envs/base/explode.py#L69-L90")


def pilot_evidence(root):
    receipt = root / "pilot_passed.json"
    frozen.require(receipt.is_file() and not receipt.is_symlink(), "waiting for the sealed pilot_passed.json")
    study_path = root / "study.json"
    frozen.require(not study_path.is_symlink(), "linked study manifest")
    study, study_hash = frozen.read_json(study_path)
    frozen.require(study_hash == STUDY_SHA256, "study.json differs from the registered pilot")
    protocol_path = frozen.HERE / "reference_protocol.json"
    frozen.require(frozen.reference.sha(protocol_path) == frozen.REFERENCE_FILES[protocol_path.name],
                   "frozen reference protocol differs")
    protocol = json.loads(protocol_path.read_bytes())
    case, = frozen.inventory(study, protocol)["development"]
    name = f"{case['game']}-{case['model_seed']}"
    frozen.require(name == "Explode-88101" and case["steps"] == 10000 and case["game_seed"] == 180400,
                   "unexpected pilot assignment")
    row, rewards = frozen.read_run(root, "development", case, study["source_sha256"])
    frozen.require(row["status"] == "complete", f"incomplete or invalid pilot: {row.get('reason', row['status'])}")
    receipt_hash = frozen.verify_marker(receipt, "development", [row], study["source_sha256"])
    game_index = protocol["games"].index("Explode")
    historical = protocol["table1"]["untuned"][game_index]
    return study, row, rewards, historical, receipt_hash


def score_blocks(rewards):
    frozen.require(len(rewards) == 10 * BLOCK, "the pilot requires ten complete blocks")
    frozen.require(all(r in (-1, 0, 1) for r in rewards), "unexpected native Explode reward")
    blocks = []
    for first in range(0, len(rewards), BLOCK):
        window = Counter(rewards[first:first + BLOCK])
        blocks.append(dict(first=first + 1, last=first + BLOCK, catches=window[1], misses=window[-1],
                           zero=window[0], net=window[1] - window[-1]))
    return blocks


def clocks(path, expected_hash):
    decisions, observations = [], []
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            checksum.update(line)
            action = json.loads(line)
            decisions.append(action["decision_s"])
            observations.append(action["observation_s"])
    frozen.require(checksum.hexdigest() == expected_hash, "step file changed while reading timing data")
    return decisions, observations


def chart(blocks, path):
    scores = [b["net"] for b in blocks]
    fig = Figure(figsize=(9, 3.9), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dbe3e4", linewidth=.6)
    ax.bar(range(1, 11), scores, color=["#146670" if n >= 0 else "#a84e2f" for n in scores], width=.65)
    ax.axhline(0, color="#4f6163", linewidth=.9)
    lo, hi = min(0, *scores), max(0, *scores)
    pad = max(1, .22 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    for x, score in enumerate(scores, 1):
        ax.annotate(f"{score:+d}" if score else "0", (x, score), xytext=(0, 5 if score >= 0 else -5),
                    textcoords="offset points", ha="center", va="bottom" if score >= 0 else "top",
                    fontsize=10, color="#22393f")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Action at the end of each 1,000-action block (thousands)")
    ax.set_ylabel("Net reward in block (points)")
    fig.savefig(path, dpi=160)


def signed(n):
    return f"{n:+,d}" if n else "0"


def point_comparison(score, reference):
    delta = score - reference
    if delta == 0:
        return "The pilot matched that printed mean."
    noun = "point" if abs(delta) == 1 else "points"
    return f"The pilot finished {abs(delta):,} {noun} {'above' if delta > 0 else 'below'} that mean."


def clock_table(decisions, observations):
    rows = []
    for label, times in (("Choose an action", decisions), ("Learn from its outcome", observations),
                         ("Both intervals together", [a+b for a, b in zip(decisions, observations, strict=True)])):
        rows.append((label, f"{statistics.mean(times):,.3f}", f"{times[0]:,.3f}",
                     f"{statistics.median(times[1:]):,.3f}"))
    return frozen.table(("Measured interval", "Mean · all 10,000 (s)", "First action (s)",
                         "Median · actions 2–10,000 (s)"), rows)


def render(study, row, blocks, historical, decisions, observations, receipt_hash, host_note):
    catches = sum(b["catches"] for b in blocks)
    misses = sum(b["misses"] for b in blocks)
    resolved = catches + misses
    score = catches - misses
    frozen.require(score == row["observed_return"], "block scores disagree with the certified raw total")
    zero = row["steps"] - resolved
    mean, sd = historical
    first, last = blocks[0]["net"], blocks[-1]["net"]
    change = last - first
    if change == 0:
        trend = f"The opening and closing blocks each earned {signed(first)} points."
    else:
        trend = (f"The opening block earned {signed(first)} points. The closing block earned {signed(last)}, "
                 f"{abs(change)} {'more' if change > 0 else 'fewer'} than the opening block.")
    early = sum(b["net"] for b in blocks[:5])
    late = sum(b["net"] for b in blocks[5:])
    catch_rate = (f"Of the {resolved:,} bombs that reached an outcome, {100*catches/resolved:.1f}% were caught."
                  if resolved else "No bomb reached a catch or miss during this run.")
    block_table = frozen.table(("Real actions", "Catches (+1)", "Misses (−1)", "No score (0)", "Net points"), [
        (f"{b['first']:,}–{b['last']:,}", str(b["catches"]), str(b["misses"]), str(b["zero"]), signed(b["net"]))
        for b in blocks])
    sec = row["process_s"]
    if row["resumed_from"]:
        wall = (f"The final worker resumed from action {row['resumed_from']:,} and ran for {sec:,.1f} seconds. "
                "The receipt records wall time for that attempt only. The action trace spans every attempt.")
    else:
        wall = (f"The worker completed all 10,000 actions in <strong>{sec:,.1f} seconds "
                f"({sec/3600:.2f} hours)</strong>, or {sec/10000:.3f} seconds of wall time per action.")
    path = row["path"]
    chart_alt = "Net points in successive 1,000-action blocks: " + ", ".join(signed(b["net"]) for b in blocks)
    hashes = frozen.table(("Verified file", "SHA-256"), [
        (f'<a href="{name}">{escape(name)}</a>', f'<code>{fingerprint}</code>') for name, fingerprint in
        (("study.json", STUDY_SHA256), ("pilot_passed.json", receipt_hash), (f"{path}/result.json", row["result_sha256"]),
         (f"{path}/steps.jsonl", row["steps_sha256"]))])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>10,000 actions in Explode</title>
<style>{frozen.CSS}
main{{max-width:59rem}}.lead{{font-size:1.25rem}}figure{{margin:1.25rem 0}}img{{width:100%;height:auto}}
figcaption,footer{{font-size:.9rem}}td code{{font-size:.8rem;overflow-wrap:anywhere}}summary{{padding:.25rem 0}}
</style></head><body><main>
<header><h1>10,000 actions in Explode</h1>
<p class="lead">CFFG’s implementation of AXIOM caught <strong>{catches:,}</strong> falling bombs and missed <strong>{misses:,}</strong>,
earning <strong>{signed(score)} points in 10,000 real actions</strong>.</p>
<p>In Explode, the agent moves a bucket left or right, or stays still. Catching a bomb earns one point.
Letting it reach the bottom costs one point. Higher totals are better. {catch_rate}
In {zero:,} of the 10,000 actions, no bomb was caught or missed. <a href="{GAME_SOURCE}">Game rules</a></p>
<p>The published untuned AXIOM agent averaged <strong>{mean} points</strong> per 10,000 actions across ten seeds,
with a standard deviation of {sd} points. {point_comparison(score, mean)}
This single development run gives one observed score. Comparing average performance across seeds requires
more independent runs. <a href="https://arxiv.org/html/2505.24784v1#S3.T1">Published Table 1</a></p></header>
<section><h2>How the score changed during play</h2>
<p>{trend} Across actions 1–5,000 the total was {signed(early)} points, compared with {signed(late)} for actions 5,001–10,000.
Each bar counts catches minus misses in one successive block of 1,000 real actions.</p>
<figure><a href="{CHART}"><img src="{CHART}" width="1440" height="624" alt="{chart_alt}"></a>
<figcaption>Actions 1–1,000 form the first block. Actions 9,001–10,000 form the last.</figcaption></figure>
<details><summary>Catches, misses and net points for each block</summary>{block_table}</details></section>
<section><h2>Time to play and learn</h2><p>{wall} Starting the simulator and constructing the agent took
{row["initialization_s"]:,.3f} seconds of that worker time. {host_note}</p>
{clock_table(decisions, observations)}
<p>The decision clock covers choosing a real action. The learning clock covers processing the returned observation,
including scheduled rule merges and any terminal reset. Simulator stepping, checkpoint writes and other overhead
also contribute to wall time. Later actions can still trigger compilation.</p></section>
<section><h2>What this run covered</h2>
<p>The run used the untuned model with 32-bit floating-point arithmetic and compensated updates, which carry small rounding corrections across actions. The agent learned from the game’s pixels, its applied actions and the resulting rewards. Before each real action
it evaluated 512 candidate action sequences, each 32 steps long, with three simulated draws per sequence.
Imagined rewards do not enter the reported score.</p>
<p>The agent reviewed its learned interaction rules after every 500 real actions.
Across {row["reductions"]:,} scheduled reviews in this run, it accepted {row["accepted_merges"]:,} merges from
{row["proposed_pairs"]:,} recorded candidate pairs. A pair can be considered more than once.</p>
<p>The study used model seed <strong>{row["model_seed"]}</strong> and simulator seed <strong>{row["game_seed"]}</strong>
for one development run on Explode. The planned 20 engineering runs and 100 primary evaluation runs were
deferred at this reporting milestone.</p>
<details><summary>Verified receipts and raw action log</summary>{hashes}
<p>Candidate revision <code>{escape(study["revision"])}</code>. Source inventory SHA-256
<code>{study["source_sha256"]}</code>. The receipt links the complete 10,000-action result to the raw log.</p>
</details></section></main></body></html>"""


def publish(root):
    study, row, rewards, published, receipt_hash = pilot_evidence(root)
    blocks = score_blocks(rewards)
    decisions, observations = clocks(root / row["path"] / "steps.jsonl", row["steps_sha256"])
    host_note = ("The same host ran a 2,808-test integration suite from 15:49:47 to 16:12:00 UTC on "
                 "24 November 2026 (1,333.136 seconds of dashboard wall time). The per-action measurements "
                 "include that period of shared CPU use.")
    if (root / "host-activity.log").is_file():
        host_note += ' <a href="host-activity.log">Host activity log</a>'
    html = render(study, row, blocks, published, decisions, observations, receipt_hash, host_note)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        tmp = Path(temp)
        chart(blocks, tmp / CHART)
        (tmp / "explode.html").write_text(html, encoding="utf-8")
        for name, checksum in (("study.json", STUDY_SHA256), ("pilot_passed.json", receipt_hash),
                               (f"{row['path']}/result.json", row["result_sha256"]),
                               (f"{row['path']}/steps.jsonl", row["steps_sha256"])):
            frozen.require(frozen.reference.sha(root / name) == checksum, f"{name} changed during rendering")
        (tmp / CHART).replace(root / CHART)
        (tmp / "explode.html").replace(root / "explode.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="sealed campaign archive")
    publish(parser.parse_args().root.expanduser())
