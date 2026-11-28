"""Render the archived online-precision comparisons. No learner or JAX runtime is required."""

import argparse
import hashlib
import json
import tempfile
from html import escape
from pathlib import Path


ROOT = Path.home() / "agent-share/axiom-online-precision"
OLD = "fd8bfd164cbb274b201d5b41f79419058e99eb11"
FIRST = "a72cabc821e7c486b4f13064f8e34144c4e5d4eb"
COMPUTE = "089f3777bc825032301f80ef40e797131a5f9cf7"
COMBINED = "b7f1fd89049ce3d3c235603878d73565713e178b"
IO = "checkpoint-io/combined-confirmation-2"
DECLARED = "../cffg-6wpk"


def read(path, expected_hash=None):
    data = path.read_bytes()
    if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"Source checksum mismatch for {path}")
    return json.loads(data)


def link(path, label):
    return f'<a href="{escape(path, quote=True)}">{escape(label)}</a>'


def table(caption, headers, rows):
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = []
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"Wrong column count in {caption}")
        body.append('<tr><th scope="row">' + escape(str(row[0])) + "</th>"
                    + "".join(f"<td>{escape(str(v))}</td>" for v in row[1:]) + "</tr>")
    return (f'<div class="scroll" tabindex="0" role="region" aria-label="{escape(caption)}">'
            f"<table><caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def details(title, body):
    return f"<details><summary>{escape(title)}</summary>{body}</details>"


def percent(values):
    mid, lo, hi = values
    return f"{mid:+.1f}% [{lo:+.1f}, {hi:+.1f}]"


def ratio(effect):
    return percent([100 * (x - 1) for x in [effect["ratio"], *effect["ci95"]]])


def shape(row):
    return f"{row['n']} × {row['k']:,}"


def accuracy(root):
    seal = read(root / "qualification-manifest.json")
    def records(rev, prec):
        fname = f"accuracy-{prec}.json"
        folder = Path(rev) / "final" if rev == OLD else Path(rev)
        return read(root / folder / fname, seal[rev]["files"][fname])
    old = records(OLD, "float32")["checks"]
    new = records(FIRST, "float32")["checks"]
    for dtype in ("float32", "float64"):
        if not records(FIRST, dtype)["passed"]:
            raise ValueError(f"Compensated {dtype} accuracy gate failed")
    def mean_scale(row):
        return [f"{row['mean'][0][0]:,.6f}", f"{row['psi'][0][0][0]:,.6f}"]
    def row(test, route):
        return mean_scale(test["rows"][route])
    stress = new["translated"]
    comparison = table("Translated float32 stream after 1,024 observations", ["Method", "Mean", "Scale Ψ"], [
        ["Independent reference", *mean_scale({"mean": stress["expected"]["mu"], "psi": stress["expected"]["psi"]})],
        ["Rounded, one batch", *row(old["translated"], "batch")],
        ["Rounded, sequential", *row(old["translated"], "stream")],
        ["Compensated, uninterrupted", *row(stress, "stream")],
        ["Compensated, exact reload every observation", *row(stress, "exact_every_1")],
        ["Compensated, compact reload every observation", *row(stress, "compact_every_1")],
        ["Compensated, compact reload every 128", *row(stress, "compact_rare")],
    ])
    return f"""<section id="accuracy"><h2>Where rounding loses evidence</h2>
<p>In a six-dimensional mixture bank, each observation can add fractional mass to several components.
The scalar control isolates the numerical effect. It starts the mean at 10,000 and adds 1,024
observations of 10,001. Each individual update is smaller than a float32 spacing at that mean.
Rounding after every update leaves the mean at 10,000 and inflates the estimated scale.
Compensation keeps the small remainder for the next update.</p>{comparison}
<p><strong>Saving is harmless to the live learner. Reloading determines the precision.</strong>
A compact file saves rounded values. An exact file also saves the remainder for each parameter.
Reloading every 128 observations recovered the visible mean in this case, but the scale missed the
reference by {stress["rows"]["compact_rare"]["error"]["psi"]:.3f}. Exact reload after every
observation matched uninterrupted learning bit for bit. The other compact controls with short streams
reloaded at their final tick, after the small increments had accumulated. The error depends on which fractional updates
have accumulated when the file is reloaded.</p>
<p>At an initial count of 2²², four fractional updates should add masses of 1 and 3 to the two
active components. The rounded stream added 0 and 4. Both the batch and compensated stream added 1
and 3. Sixteen increments of 0.25 to a scale of 2²⁴ should add 4. The rounded stream lost all 4.
Compensation also passed a two-observation test one floating-point spacing above the prior mean, the
missing-evidence checks, and the invalid-update checks. The CPU scale controls covered
480 successive valid score/update pairs across six shapes and both precisions.</p>
<p class="sources">{link(OLD + '/final/accuracy-float32.json', 'Rounded controls')} ·
{link(FIRST + '/accuracy-float32.json', 'Compensated controls')} ·
{link('qualification-manifest.json', 'Qualification manifest')} ·
{link('compute-development/089f377/results.json', 'Optimized compute checks')}</p></section>"""


def compute(initial, confirmed, raw):
    warm, cold, original = [], [], []
    for r in confirmed["results"]:
        m = r["ms_per_tick"]
        route = "Device scan" if r["route"] == "scan" else "Host API"
        warm.append([shape(r), route, *(f"{m[a]:.3f}" for a in ("rounded", "compensated", "optimized")),
                     percent(r["vs_rounded_pct"]), percent(r["vs_compensated_pct"])])
    for cell in initial["cells"]:
        for k, route in [("scan", "Device scan"), ("public", "Host API")]:
            m = cell["speed_ms_per_tick"][k]
            original.append([shape(cell), route, f"{m['old_median']:.3f}", f"{m['new_median']:.3f}",
                             percent([m["change_pct"], *m["ci95_pct"]])])
    for c in raw["results"]:
        for route, label in [("scan", "Device scan"), ("public", "Host API")]:
            cold.append([shape(c), label, *(f"{c['workers'][a]['cold_block_ms'][route]:,.1f}"
                                             for a in ("rounded", "compensated", "optimized"))])
    changes = [r["vs_compensated_pct"][0] for r in confirmed["results"]]
    return f"""<section id="compute"><h2>The recurrent update is still more expensive</h2>
<p>Optimization cut the time per tick by {abs(max(changes)):.1f}–{abs(min(changes)):.1f}% relative to the first fix.
One tick scores a new observation batch and updates the same learned bank. <i>N</i> is the
batch size and <i>K</i> is the number of components. Each has six continuous features and two
categorical heads with three and five outcomes. Each fresh CPU worker trained for 32 ticks before
measurement. The device scan runs eight dependent ticks and synchronizes at the end. The host API
accepts NumPy inputs and materializes scores, weights, validity flags and the whole posterior each tick.</p>
{table("Confirmation · warmed milliseconds per tick and paired time changes", ["N × K", "Route", "Rounded", "First fix", "Optimized", "vs rounded [95% CI]", "vs first fix [95% CI]"], warm)}
<p>Times are absolute medians. Percentages are medians of paired time ratios, so dividing the
displayed medians can give a slightly different result. The confirmation balanced all six orders
of the three workers over 12 blocks after three warmups. Intervals resample whole paired blocks
10,000 times. Positive percentages mean more time.</p>
{details("Initial two-arm campaign", table("Original warmed milliseconds per tick", ["N × K", "Route", "Rounded", "First fix", "Change [95% CI]"], original))}
{details("Cold first blocks in the confirmation", '<p>Milliseconds for the first eight-tick block on each route, after import and pretraining. One observation per cell and arm.</p>' + table("First block including compilation, ms", ["N × K", "Route", "Rounded", "First fix", "Optimized"], cold))}
<p class="sources">{link('timing/summary.json', 'Initial summary')} · {link('timing/timing.json', 'Initial raw blocks')} ·
{link('compute-confirmation/summary.json', 'Compute confirmation')} · {link('compute-confirmation/timing.json', 'Confirmation raw blocks')} ·
{link('predeclared-timing.json', 'Frozen timing protocol')}</p></section>"""


def checkpoint(initial, io):
    compact, optional, sizes, cold, original = [], [], [], [], []
    for case in io["results"]:
        workers = case["workers"]
        receipts = workers["candidate"]["checks"]
        b = receipts["compact"]["file_bytes"]
        e = receipts["exact"]["file_bytes"]
        for mode in ("compact", "exact"):
            if receipts[mode] != workers["base"]["checks"][mode]:
                raise ValueError(f"Checkpoint parity failed at {shape(case)}")
        if b != workers["legacy"]["checks"]["compact"]["file_bytes"]:
            raise ValueError("Legacy compact length differs")
        sizes.append([shape(case), f"{b:,}", f"{e:,}", f"{e-b:,}", f"{100*(e/b-1):+.1f}%"])
        cold.append([shape(case), *(f"{workers[k]['cold_first_compact_restore_ms']:,.1f}"
                                    for k in ("legacy", "base", "candidate"))])
        for op in ("write_ms", "read_ms"):
            label = "Save" if op == "write_ms" else "Restore"
            c = case["effects"]["compact"]
            base = c["candidate_over_base"][op]
            legacy = c["candidate_over_legacy"][op]
            compact.append([shape(case), label, f"{legacy['old_median']:.2f}", f"{base['old_median']:.2f}",
                            f"{base['new_median']:.2f}", ratio(legacy), ratio(base)])
            ex = case["effects"]["exact"]["candidate_over_base"][op]
            optional.append([shape(case), label, f"{ex['old_median']:.2f}", f"{ex['new_median']:.2f}", ratio(ex)])
    for case in initial["cells"]:
        for op, label in [("write_ms", "Save"), ("read_ms", "Restore")]:
            v = case["archive_ms"]["same_visible_compact"][op]
            original.append([shape(case), label, f"{v['old_median']:.2f}", f"{v['new_median']:.2f}",
                             percent([v["change_pct"], *v["ci95_pct"]])])
    return f"""<section id="files"><h2>Compact files kept their size. Restores got faster.</h2>
<p>All three workers used the same learned arrays for each comparison. A save includes public
validation and compression to a buffered local NPZ file. A restore reads the same frozen file,
calls the public constructor and synchronizes the returned bank. The 12 paired blocks compare
the rounded baseline, initial compensation and the combined compute and checkpoint revision.</p>
{table("Compact files · warmed milliseconds per operation", ["N × K", "Operation", "Rounded", "First fix", "Combined", "vs rounded [95% CI]", "vs first fix [95% CI]"], compact)}
<p>The combined revision reduced compact restore time by 27–50% relative to the first fix.
Against the rounded baseline, the 32 × 5,000 restore cost 10.6% more time. Its interval spans
the 10% threshold. The other eleven compact save or restore comparisons had upper bounds below 10%.</p>
{table("Compressed bytes for identical learned values", ["N × K", "Compact, all arms", "Exact", "Optional extra bytes", "Extra share"], sizes)}
<p>Equal visible values produced the same compact file length in all six cells. The first and
combined fixes also produced identical whole files. The older writer used a different ZIP member
order. Every named member matched byte for byte. Independently trained rounded and compensated
banks gave slightly different values and compressed to between 0.11% fewer and 0.13% more bytes.
Exact files retain the corrections and are optional. The live compensated bank also stores those
corrections in memory. At 5,000 components the rounded bank used 1,065,000 bytes and the
compensated bank used 2,125,000 bytes before serialization.</p>
{details("Initial compact file measurements", table("First fix versus rounded at identical visible values, ms", ["N × K", "Operation", "Rounded", "First fix", "Change [95% CI]"], original))}
{details("Optional exact file timings", table("Exact save and restore, warmed milliseconds", ["N × K", "Operation", "First fix", "Combined", "Change [95% CI]"], optional))}
{details("Cold compact restores", '<p>First read after import and device initialization, one fresh worker per shape and revision.</p>' + table("First read including validation compilation, ms", ["N × K", "Rounded", "First fix", "Combined"], cold))}
<p>Public save and restore outcomes matched the initial compensated revision on 1,800 near-rank
matrices, 20 malformed records and 16 learned files. The restored files also reproduced the next
exact update bit for bit. An initial file preflight stopped before timing when it encountered the
older ZIP order. All cells then ran from the beginning with file lengths and named members checked.</p>
<p class="sources">{link(IO + '/timing.json', 'File timing and byte receipts')} · {link(IO + '/warm.csv', 'Paired file timings')} ·
{link(IO + '/cold.csv', 'Cold reads')} · {link(IO + '/bytes.csv', 'Compact and exact bytes')} ·
{link('checkpoint-io/qualification-' + COMBINED + '/manifest.json', 'Checkpoint qualification')}</p></section>"""


def recordings(root):
    data = read(root / "frames.json")
    tapes = data["runs"]
    rows = []
    for mode, label in [("compact_episode", "Compact at episode boundaries"),
                        ("compact_every_16", "Compact every 16 frames"),
                        ("exact_every_16", "Exact every 16 frames")]:
        alternatives = [t["comparisons"][mode] for t in tapes]
        rows.append([label, f"{max(a['max_assignment_difference'] for a in alternatives):.3g}",
                     sum(a["changed_patch_labels"] for a in alternatives),
                     f"{sum(a['all_posterior_arrays_identical'] for a in alternatives)} / {len(tapes)}"])
    slots = read(root / "slot-replay.json")
    slot_rows = []
    for game in slots:
        frames = game["frames"]["float32"]
        disagreements = game["pixel_disagreements"]
        if len(frames) != len(disagreements):
            raise ValueError("Missing pixel comparison")
        slot_rows.append([game["game"], len(frames), sum(len(x["growth"]) for x in frames),
                          f"{sum(sum(f['occupancy']) for f in frames):,}", sum(disagreements)])
    boundaries = sum(t["episode_reloads"] > 0 for t in tapes)
    return f"""<section id="frames"><h2>What changed on recorded game frames?</h2>
<p><strong>Fixed-action patch replay.</strong> A six-component learner classified 16 RGB patches
from each of 48 pre-action frames per recording across ten games and three recording conditions.
It made {sum(t["assignments"] for t in tapes):,} patch assignments per schedule. Each schedule
used the same frames and was compared with uninterrupted compensated learning. Compact reloads
changed some assignment probabilities while preserving every winning label. Only {boundaries} of
{len(tapes)} recordings crossed an episode boundary, so the episode schedule exercised fewer reloads.</p>
{table("Patch replay relative to uninterrupted learning", ["Reload schedule", "Largest probability change", "Label changes", "Identical final banks"], rows)}
<p><strong>Full-frame spatial slots.</strong> A separate replay assigned every pixel of six
consecutive frames to one of 32 possible spatial slots. Float32 and float64 selected the same slot
for every pixel in these recordings. Positions from the preceding frame supplied the next
predictions. This measures frame-local fitting and short-term slot reuse on recorded actions.
Policy choice and return are outside this recorded-action comparison.</p>
{table("Per-pixel float32 versus float64 replay", ["Game", "Frames", "Slots added (float32)", "Pixels compared", "Different labels"], slot_rows)}
<p class="sources">{link('frames.json', 'Fixed-action patch runs')} ·
{link('compute-development/089f377/frames.json', 'Optimized patch replay')} ·
{link('frame-regression.json', 'Single-frame appearance check')} · {link('spatial-frame.json', 'Spatial fit check')} ·
{link('slot-replay.json', 'Per-frame slot assignments and growth')}</p></section>"""


SCENE_FILES = {
    "final-scene-f32.json": "8de52cc16d0e7d3c7c15c380763678ba066e9c376de84411f8f6774c884bc0f5",
    "final-scene-f64.json": "47652dde6d23217f1bae4c291b74b6f53abf286cd670dca59076881dea97d402",
    "bounce-common-init/replay.json": "2014f2535970ab7d1bfcb6964777b286a3b0eb5a01c3959141076ce4c3c966f2",
}
ACTED = "acted-smoke.json"


def scene_replays(root):
    pairs = {}
    rows = []
    for bits in (32, 64):
        name = f"final-scene-f{bits}.json"
        records = read(root / name, SCENE_FILES[name])
        pairs[bits] = {record["game"]: record for record in records}
    for game in ("Explode", "Bounce", "Aviate"):
        for bits in (32, 64):
            record = pairs[bits][game]
            updates = [entry for entry in record["trace"] if entry["update"]]
            last = updates[-1]
            if ([x["tick"] for x in updates] != list(range(48)) or
                    record["exact_reloaded_at_tick"] != 24 or record["final_all_arrays_equal"] is not True):
                raise ValueError(f"Incomplete {game} float{bits} replay or restart")
            rows.append([game, f"float{bits}", len(updates), record["reset_observations"],
                         sum(len(x["tracked"]) for x in updates), last["active_identities"],
                         last["active_motion"], last["active_interactions"]])
    common_path = "bounce-common-init/replay.json"
    common = read(root / common_path, SCENE_FILES[common_path])
    final = common["final_components"]
    mismatch_count = common["mismatching_track_ticks"]
    comparisons = common["n"] * pairs[32]["Bounce"]["config"]["slots"]
    cli = ("https://github.com/VersesTech/axiom/blob/"
           "b99f62599e73294639f1944827bf427909f6c698/defaults.py#L255")
    return f"""<section id="scenes"><h2>Tracking and learning from whole scenes</h2>
<p>The observation model followed the same 48 recorded actions per game and precision. It saw
each new image, applied action and reward, tracked visible slots, and updated its identity, motion
and interaction components. Each run reloaded the scene after action 24 and finished with all 101
state arrays byte-identical to its uninterrupted counterpart. Aviate also processed one reset
observation per precision.</p>
{table("Fixed-action development replay · cumulative tracking and final component counts",
       ["Game", "Precision", "Actions", "Resets", "Tracked slot-steps", "Identities", "Motion", "Interactions"], rows)}
<p>A tracked slot-step counts one tracked slot on one actual transition. The final three columns
count occupied components. The same seed (17) initialized each native run, and random draws can
differ by precision. On Bounce the native tracking masks first differed at tick
{common["original_native_first_track_difference"]}. Slot 4 appeared only in the float32 mask there. Their final interaction counts were {pairs[32]["Bounce"]["trace"][-1]["active_interactions"]}
and {pairs[64]["Bounce"]["trace"][-1]["active_interactions"]}, respectively.</p>
<p><strong>The tracking difference disappeared with a common start.</strong> The control initialized
Bounce once in float32 and widened each floating-point array exactly for the float64 run. Tracking
masks differed on {mismatch_count} of {common["n"]} transitions across {comparisons:,} individual
slot decisions. The final interaction counts were {final["32"]} and {final["64"]}. These recordings
have no ground-truth object labels for assessing tracking quality. The authors’ command-line runner
defaults to float32 and allows float64. The training seeds and precision of the published results
remain unspecified by this evidence.</p>
<p class="sources">{link("final-scene-f32.json", "Native float32 scenes")} ·
{link("final-scene-f64.json", "Native float64 scenes")} · {link(common_path, "Common-start Bounce trace")} ·
{link(cli, "Authors’ command-line setting")}</p>
{acted_smoke(root)}</section>"""


def acted_smoke(root):
    path = root / ACTED
    pending = "<h3>Acted development smoke</h3><p>Pending a complete, validated 24-action Explode run.</p>"
    if not path.exists():
        return pending
    data = path.read_bytes()
    report = json.loads(data)
    if report.get("phase") != "complete" or report.get("passed") is not True:
        return pending
    # The producer verified all trace entries, the disk restart and Git source hashes.
    audited_sha = "c37fed8473489526a858365c21df5f6b9cad95b04586fdfbe03ad4dd2afaa1ea"
    if hashlib.sha256(data).hexdigest() != audited_sha or report["source_reverified"] is not True:
        raise ValueError("Acted smoke differs from the independently validated result")
    summary = report["summary"]
    histogram = summary["applied_histogram"]
    active = summary["final_active"]
    warm = summary["warm_decision_s"]
    cold = summary["first_by_shape_decision_s"]
    return f'''<h3 data-acted-status="complete">One acted development run</h3><p>In a separate float32 Explode run, the
planner considered 16 candidate action sequences over eight steps with three sampled rollouts per
sequence. The simulator applied all 24 selected commands. Actions 0, 1 and 2 occurred
{histogram[0]}, {histogram[1]} and {histogram[2]} times. Each new observation used the applied
command and returned reward. All observed rewards were zero. The novelty term distinguished
candidate scores on {summary["decisions_with_score_spread"]} of 24 decisions. The observed model’s
bytes remained identical across each planning call. Reloading the 104 scene, proposal and key arrays
at decision 12 reproduced action 2, the full decision output and the next random key.</p>
<p>The run finished with {active["identity"]} identity, {active["motion"]} motion and
{active["interaction"]} interaction components. The 24 actions preceded the
first scheduled model reduction. Across {warm["n"]} decisions that reused compiled shapes, the
median decision took {1000*warm["median"]:.2f} ms including model preparation. First use of
{cold["n"]} shapes took {cold["minimum"]:.2f}–{cold["maximum"]:.2f} s. Updating the scene after the first action took {summary["first_observe_s"]:.2f} s and the median for the remaining 23 was
{1000*summary["warm_observe_s"]["median"]:.2f} ms. These timings describe one development
trajectory. A policy-quality comparison would require separate evaluation.</p>
<p class="sources">{link(ACTED, "Acted smoke trace and checkpoint receipt")}</p>'''


def declared(root):
    inventory = read(root / DECLARED / "inventory.json")
    def artifact(filename):
        return read(root / DECLARED / filename, inventory["artifacts"][filename])
    measures = artifact("compact-timing.json")["cases"]
    before = artifact("translated_base.json")
    after = artifact("translated_candidate.json")
    rows = []
    for key, label in [("mix3d2", "3 components, 2 features"), ("mix32d6", "32 components, 6 features")]:
        result = measures[key]
        rows.append([label, *(f"{v:.2f}" for v in result["warm_median_ms"]),
                     f"{100*(result['paired_ratio_median']-1):+.1f}%"])
    return f"""<section id="declared"><h2>A separate result for graph-declared learners</h2>
<p>The graph API replaces a declared learner’s posterior using a converged inference result.
A centered calculation recovered an expected translated scale of 3.25 in a case where the
rounded path returned {before['0.25']['a_state'][0]:.2f}. Both the ordinary and dynamics
calculations returned {after['0.25']['a_state'][0]:.2f}. A separate timing with original float32
host priors measured just the public replacement after inference, using eight alternating paired
blocks of three replacements. These improvements apply to the declared learner operation.</p>
{table("Warmed public Learnables.replace, ms", ["Graph", "Original", "Candidate", "Paired time change"], rows)}
<p class="sources">{link(DECLARED + '/inventory.json', 'Declared-learning evidence inventory')} ·
{link(DECLARED + '/compact-timing.json', 'Replacement times')} ·
{link(DECLARED + '/translated_candidate.json', 'Translated scale controls')}</p></section>"""


CSS = """
:root{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#192d35;background:#f6f7f4;
 --rule:#ccd6d3;--teal:#18574f;font-synthesis:none}
*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:auto;padding:56px 42px 76px}
h1,h2{font-family:"Iowan Old Style",Georgia,serif;letter-spacing:-.025em;font-weight:600}
h1{font-size:clamp(2.5rem,6vw,4.4rem);line-height:1.04;margin:0 0 .35em;max-width:900px;text-wrap:balance}
h2{font-size:clamp(1.7rem,3vw,2.3rem);line-height:1.12;margin:.2em 0 .5em}
p{max-width:81ch;font-size:1rem;line-height:1.62;margin:0 0 1em}
.lede{font-size:1.19rem;max-width:72ch}.decision{border-left:4px solid var(--teal);padding:.8rem 1rem;
background:#e6efea;max-width:none}.decision strong{color:#174c44}
nav{display:flex;gap:9px 22px;flex-wrap:wrap;padding:10px 0 0}a{color:var(--teal);text-underline-offset:3px}
a:hover{color:#071f20}section{padding:34px 0 12px;border-top:1px solid var(--rule);margin-top:28px}
.scroll{overflow-x:auto;max-width:100%;margin:15px 0 19px;background:#fff;border:1px solid var(--rule);
border-radius:7px}table{width:100%;border-collapse:collapse;font-size:.86rem;font-variant-numeric:tabular-nums}
caption{text-align:left;font-weight:650;color:var(--teal);padding:12px}th,td{padding:7px 10px;
text-align:right;border-top:1px solid #dfe6e4;white-space:nowrap}thead{background:#eaf0ee;color:#294943}
tbody th{text-align:left}th:first-child{text-align:left}tbody tr:nth-child(even){background:#f7faf8}
details{margin:8px 0;padding:.55rem .8rem;border:1px solid var(--rule);border-radius:5px;background:#f0f4f1}
summary{cursor:pointer;font-weight:600}details p{margin-top:1rem}.sources{font-size:.85rem;line-height:1.8;
margin:17px 0 0;max-width:none}footer{border-top:1px solid var(--rule);margin-top:25px;padding-top:14px;
font-size:.85rem;color:#485756}code{font-size:.92em}a:focus-visible,summary:focus-visible,.scroll:focus-visible{
outline:3px solid #a96312;outline-offset:3px}@media(max-width:640px){main{padding:30px 15px 45px}th,td{padding:6px 8px}}
@media print{body{background:white}main{max-width:none;padding:0}.scroll{overflow:visible}section{break-inside:auto}
table{font-size:8pt}tr{break-inside:avoid}nav{display:none}}
"""


def render(root):
    initial = read(root / "timing/summary.json")
    read(root / "timing/timing.json", initial["raw_sha256"])
    confirmed = read(root / "compute-confirmation/summary.json")
    raw = read(root / "compute-confirmation/timing.json", confirmed["timing_sha256"])
    io = read(root / IO / "timing.json")
    if (confirmed["source"] != dict(rounded=OLD, compensated=FIRST, optimized=COMPUTE)
            or io["sources"] != dict(legacy=OLD, base=FIRST, candidate=COMBINED)
            or initial["candidate"] != FIRST or initial["baseline"] != OLD):
        raise ValueError("Source revisions differ from qualified comparisons")
    cells = {(n, k) for k in (32, 500, 5000) for n in (1, 32)}
    for rows in (initial["cells"], raw["results"], io["results"]):
        if len(rows) != len(cells) or {(r["n"], r["k"]) for r in rows} != cells:
            raise ValueError("Missing or duplicated measured shape")
    routes = {(r["n"], r["k"], r["route"]) for r in confirmed["results"]}
    if len(confirmed["results"]) != 12 or routes != {(n, k, kind) for n, k in cells for kind in ("scan", "public")}:
        raise ValueError("Incomplete compute confirmation")
    costs = [r["vs_rounded_pct"][0] for r in confirmed["results"]]
    if any(r["vs_rounded_pct"][1] <= 10 for r in confirmed["results"]):
        raise ValueError("The decision summary needs review against the timing threshold")
    nav = " ".join(link("#" + dest, label) for dest, label in (
        ("accuracy", "Accuracy and restarts"), ("compute", "Update time"), ("files", "File costs"),
        ("frames", "Recorded frames"), ("scenes", "Scene tracking and actions"), ("declared", "Declared learners")))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Online learning precision and its cost</title><style>{CSS}</style></head><body><main>
<header><h1>Online learning precision and its cost</h1>
<p class="lede">Keeping small rounding remainders fixed an error that built up across repeated
float32 updates. When 1,024 observations accumulated near 10,000, the compensated stream recovered
the expected scale of 950.2. The original stream reached 1,154.8.</p>
<p class="decision"><strong>Public default decision pending.</strong> Even after optimization,
each recurrent score and update took {min(costs):.1f}–{max(costs):.1f}% more time than the rounded
implementation in the CPU benchmark. All twelve 95% intervals lie above the predeclared 10%
cost threshold. Required compact storage grew by zero bytes for identical values. Optional exact
files were 67–88% larger than compact files and enabled bitwise continuation.</p>
<nav aria-label="Report sections">{nav}</nav></header>
{accuracy(root)}{compute(initial, confirmed, raw)}{checkpoint(initial, io)}{recordings(root)}{scene_replays(root)}{declared(root)}
<footer>Apple M2 Max CPU · JAX 0.10.2 · Timed learner inputs in float32 · Rounded baseline
<code>fd8bfd1</code> · Initial compensation <code>a72cabc</code> · Optimized compute <code>089f377</code> ·
Combined checkpoint validation <code>b7f1fd8</code></footer></main></body></html>
"""


def publish(root):
    # Detect a replaced producer result, renderer or destination during a render.
    paths = [Path(__file__), root / ACTED, *(root / name for name in SCENE_FILES), root / "index.html"]

    def fingerprint():
        return [hashlib.sha256(p.read_bytes()).digest() if p.exists() else None for p in paths]

    destination = root / "index.html"
    before = fingerprint()
    html = render(root)
    complete = 'data-acted-status="complete"'
    if complete not in html and destination.exists() and complete in destination.read_text(encoding="utf-8"):
        raise ValueError("Preserving the completed report while acted evidence is incomplete")
    with tempfile.TemporaryDirectory(dir=root) as directory:
        draft = Path(directory) / "index.html"
        draft.write_text(html, encoding="utf-8")
        if destination.exists():
            draft.chmod(destination.stat().st_mode & 0o777)
        if fingerprint() != before:
            raise RuntimeError("Scene evidence, renderer or report changed during rendering")
        draft.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    root = parser.parse_args().root.expanduser()
    publish(root)


if __name__ == "__main__":
    main()
