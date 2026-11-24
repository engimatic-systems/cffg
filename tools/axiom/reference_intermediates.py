"""Independent scalar AXIOM checkpoints and scorer-only visible-object reference tapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import gameworld_replay as gw
import numpy as np

HERE = Path(__file__).resolve().parent
DEV_MANIFEST = "abca225803b4b79801c7fcf2a5823cc639237f12b707f779dea3f3815f3ed376"


def harmonic(n):
    return math.fsum(1 / k for k in range(1, n + 1))


def integer_dirichlet_kl_uniform(alpha):
    """KL to Dirichlet(1,...,1), using the integer digamma recurrence."""
    total = sum(alpha)
    normalizer = math.lgamma(total) - sum(map(math.lgamma, alpha)) - math.lgamma(len(alpha))
    return normalizer + sum((a - 1) * (harmonic(a - 1) - harmonic(total - 1)) for a in alpha)


def scalar_checks(cases):
    """Cross-check the stored targets using explicit one- or two-component calculations."""
    checks = []

    def check(name, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, err_msg=name)
        checks.append(name)

    prior = cases["component_prior"]
    alpha = np.array(prior["slot_alpha"])
    check("finite_slot_prior", alpha / alpha.sum(), prior["expected_finite"])
    first, second = prior["illustrative_mean_breaks"]
    check(
        "ordered_slot_prior",
        [first, (1 - first) * second, (1 - first) * (1 - second)],
        prior["expected_ordered_breaks"],
    )

    spatial = cases["spatial_sweep"]
    pixels = np.array(spatial["tokens"])[:, None]
    means = np.array(spatial["previous_means"])
    logits = -0.5 * ((pixels - means) ** 2 + spatial["previous_variances"])
    logits += [harmonic(a - 1) for a in spatial["mix_alpha"]]
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    qz = weights / weights.sum(axis=1, keepdims=True)
    mass = qz.sum(axis=0)
    variance = 1 / (1 + mass)
    check("pixel_responsibility", qz, spatial["expected_qz"])
    check(
        "slot_mean",
        (spatial["prior_means"] + (pixels * qz).sum(axis=0)) * variance,
        spatial["expected_qx_means"],
    )
    check("slot_variance", variance, spatial["expected_qx_variances"])
    precision = np.eye(2) * (1 + mass[0]) + spatial["release_precision_scalar_added_to_every_entry"]
    eta = np.column_stack((spatial["prior_means"] + (pixels * qz).sum(axis=0), np.zeros(len(mass))))
    check(
        "release_slot_covariance",
        np.linalg.solve(precision, np.eye(2)),
        spatial["expected_release_covariance"],
    )
    check("release_slot_means", np.linalg.solve(precision, eta.T).T, spatial["expected_release_means"])
    check(
        "occupied_by_winner",
        np.bincount(qz.argmax(axis=1), minlength=len(mass)) > 0,
        spatial["expected_hard_used"],
    )
    check("soft_write_mask", mass > 0, spatial["expected_positive_mass"])

    b = cases["birth"]
    best = max(b["occupied_scores"])
    bounds = [*b["thresholds"].values(), b["log_predictive_prior"] + math.log(b["concentration"])]
    check(
        "identity_birth",
        [best <= bounds[0], best <= bounds[1], best < bounds[2]],
        [
            b["expected"][key]
            for key in ("release_identity_birth", "paper_table_birth", "paper_predictive_birth")
        ],
    )

    a = cases["assignment"]
    allowed = [i for i, used in enumerate(a["active"]) if used]
    for key in ("release", "paper"):
        unscaled = np.zeros(len(a["alpha"]))
        for i in allowed:
            correction = harmonic(int(a["alpha"][i]) - 1) if key == "paper" else 0.0
            unscaled[i] = math.exp(a["log_likelihood"][i] + correction)
        check("assignment_" + key, unscaled / unscaled.sum(), a["expected"][key])

    s = cases["soft_update"]

    def stats(pair):
        x, category = pair
        return np.array([1, x, x * x, int(category == 0), int(category == 1)])

    w = s["weight"]
    past = sum(map(stats, s["old_observations"]))
    for key, retained in (("release", w), ("additive", 1.0)):
        record = retained * past + w * stats(s["new_observation"])
        check("soft_" + key, [s["prior_count"] + record[0], *record], s["expected"][key])
    g = s["write_gate"]
    check(
        "write_gate",
        [q > 0.25 and n < 32 for q, n in zip(g["weight"], g["proposed_count"], strict=True)],
        g["expected"],
    )

    s = cases["slot_growth"]
    for i, row in enumerate(s["cases"]):
        worst = int(np.argmin(row["best_token_scores"]))
        free = sorted(set(range(row["capacity"])) - set(row["winners"]))
        chosen = [worst, free[0]] if free and row["best_token_scores"][worst] < s["threshold"] else None
        if chosen != row["expected_forced_token_slot"]:
            raise ValueError(f"slot growth case {i}")
    checks.append("slot_growth")
    f = cases["full_bank"]
    vacant = [i for i, active in enumerate(f["active"]) if not active]
    if f["score"] >= b["thresholds"]["release_identity"] or vacant:
        raise ValueError("full-bank fixture must trigger a miss with no free row")
    if {
        "identity_or_recurrent_selected": int(np.argmax(np.logical_not(f["active"]))),
        "transition_allocated": None,
        "slot_allocated": None,
    } != f["expected"]:
        raise ValueError("full bank")
    checks.append("full_bank")

    t = cases["track_gates"]
    identity, motion = [], []
    for background, visible, foreground, moving, used, unused, update in t["rows"]:
        identity.append(update and visible and foreground)
        motion.append(
            update and not background and moving > 0.003 and used > 0.02 and unused < 10 * 0.05
        )
    check("identity_gate", identity, t["expected"]["identity_update"])
    check("motion_gate", motion, t["expected"]["motion_update"])

    m = cases["motion"]
    # The same Gaussian normalization appears in both the score and the release threshold.
    variance = m["variance"]
    check(
        "motion_birth",
        [-sse / (2 * variance) < m["raw_log_threshold"] * variance / 2 for sse in m["squared_residual"]],
        m["expected_birth"],
    )
    x, y, unused, vx, vy, _ = m["state"]
    du = m["unused_increment"]
    check(
        "seed_motion",
        [
            [x + m["dt"] * vx, y + m["dt"] * vy, 0, vx, vy, 0],
            [x, y, unused + du, vx, vy, 0],
            [x, y, du, 0, 0, 0],
            [x, y, 0, 0, 0, 0],
        ],
        m["expected_initial_predictions"],
    )
    kinds = [
        "absolute_target" if abs(dx) > m["threshold"] else "velocity_offset"
        for dx in m["novel_displacements"]
    ]
    if kinds != m["expected_kind"]:
        raise ValueError("novel motion")
    check(
        "first_velocity_offset",
        m["novel_displacements"][0] - m["previous_vx"],
        m["expected_first_velocity_offset"],
    )
    r = cases["relation"]
    own = [*r["own"][:3], *(v * r["velocity_scale"] for v in r["own"][3:])]
    check(
        "default_features", own + r["neighbor_displacement_feature"], r["expected"]["default_features"]
    )
    check("fixed_features", own, r["expected"]["fixed_features"])

    circles = r["stationary_circles"]
    overlaps = []
    for scale in circles["scale"]:
        radius = scale * circles["unscaled_radius"]
        grid = np.linspace(-radius, radius, circles["grid_width"])
        x, y = np.meshgrid(grid, grid)
        other_x = x + circles["center_separation"]
        overlaps.append(bool(np.any((x * x + y * y <= radius**2) & (other_x**2 + y * y <= radius**2))))
    check("tuned_radius_overlap", overlaps, circles["overlap"])
    for switch, name in ((0, "fixed_default_switch_neighbor"), (1, "fixed_nondefault_switch_neighbor")):
        neighbor = circles["neighbor_index"] if switch and overlaps[-1] else None
        if neighbor != r["expected"][name]:
            raise ValueError(name)
        checks.append(name)

    r = cases["reduction"]
    check("context_only_choice", np.argmax(r["context_scores"], axis=1), [0, 1])
    alpha, prior = np.array(r["mix_before"]), np.array(r["mix_prior"])
    after = [int(alpha.sum() - prior[1]), int(prior[1])]
    check("merge_mass", after, r["expected_mix_after"])
    head = np.asarray(r["head_before"])
    merged = 2 * head - r["head_prior"]
    check("merge_head", merged, r["expected_head_after"])
    check("donor_reset", r["head_prior"], r["expected_donor_head"])

    def score(mass, counts, outputs):
        # Both selected rows have identical heads. Each output contributes its expected log probability.
        return sum(
            harmonic(counts[o] - 1) - harmonic(sum(counts) - 1) for o in outputs
        ) - integer_dirichlet_kl_uniform(mass)

    for i, row in enumerate(r["cases"]):
        outputs = row["heldout_reward_and_switch"]
        delta = score(after, merged, outputs) - score(alpha, head, outputs)
        check(f"merge_delta_{i}", delta, row["expected_delta"])
        check(f"merge_accept_{i}", delta >= 0, row["accepted"])
    e = r["eligibility"]
    check(
        "live_modal_nondecreasing",
        [
            a and b and d >= 0
            for a, b, d in zip(e["same_modes"], e["both_live"], e["score_delta"], strict=True)
        ],
        e["accepted"],
    )

    n = cases["novelty"]
    for setting in (n, n["novel_override"]):
        alpha, assignment = setting["alpha"], setting["assignment"]
        bonus = sum(q / a for q, a in zip(assignment, alpha, strict=True)) - 1 / sum(alpha)
        targets = setting.get("expected", setting)
        check("matched_bonus_" + str(alpha), bonus, targets["matched"])
        check("threshold_bonus_" + str(alpha), 1 - 1 / sum(alpha), targets["novel"])
    v = n["expected"]
    outside = max(map(abs, n["predicted_xy"])) > 1.1
    check("outside_bonus", -1 if outside else v["matched"], v["out_of_frame"])
    check("all_slot_average", v["matched"] / len(n["slots"]), v["mean_over_slots"])
    check("novelty_sign", n["bonus_weight"] * v["mean_over_slots"], v["score_at_zero_reward"])
    check("one_observation_KL", integer_dirichlet_kl_uniform([2, 1]), v["exact_one_count_KL"])
    p = cases["planner"]
    discount = [sum(z * p["gamma"] ** i for i, z in enumerate(policy)) for policy in p["step_scores"]]
    total = list(map(sum, p["step_scores"]))
    for name, scores, choice in (
        ("discounted", discount, "elite_first_action"),
        ("undiscounted", total, "executed_first_action"),
    ):
        check(name, scores, p["expected"][name])
        check(choice, p["first_actions"][np.argmax(scores)], p["expected"][choice])
    return checks


def learner_events(tape):
    """Chronological model boundary over a verified tape; scorer fields stay outside each event."""
    for i, act in enumerate(tape["action"]):
        if i == 0 or tape["pre_frame"][i] != tape["post_frame"][i - 1]:
            yield "reset", tape["pixels"][tape["pre_frame"][i]].copy(), None, None
        yield (
            "transition",
            tape["pixels"][tape["post_frame"][i]].copy(),
            int(act),
            float(tape["reward"][i]),
        )


def pixel_statistics(image, color):
    ys, xs = np.nonzero(np.all(image == color, axis=-1))
    if not len(xs):
        return {"pixels": 0, "box_xyxy_exclusive": None, "mean_xy": None}
    return {
        "pixels": len(xs),
        "box_xyxy_exclusive": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "mean_xy": [float(xs.mean()), float(ys.mean())],
    }


def explode_reference(archive):
    if gw.digest((archive / "manifest.json").read_bytes()) != DEV_MANIFEST:
        raise ValueError("expected the recorded car-8rgl development archive")
    runs = {r["file"]: r for r in gw.load_manifest(archive)["runs"]}
    run = runs["Explode/base.npz"]
    tape = gw.load_tape(archive, run)
    palette = {
        "background": [50, 50, 100],
        "bucket": [255, 255, 0],
        "bomb": [255, 0, 0],
        "bomber": [0, 255, 0],
    }
    rows = []
    for frame in tape["pixels"]:
        objects = {name: pixel_statistics(frame, rgb) for name, rgb in palette.items()}
        if sum(slot["pixels"] for slot in objects.values()) != frame.shape[0] * frame.shape[1]:
            raise ValueError("unrecognized Explode palette")
        rows.append({"frame_sha256": hashlib.sha256(frame).hexdigest(), "visible": objects})
    check = rows[0]["visible"]
    if check["bucket"]["box_xyxy_exclusive"] != [65, 170, 95, 182] or check["bomb"]["pixels"] != 0:
        raise ValueError("native reset geometry differs")
    # Reuse the terminal-to-reset case from the original archive.
    aviate = gw.load_tape(archive, runs["Aviate/base.npz"])
    events = list(learner_events(aviate))
    if [i for i, (kind, *_) in enumerate(events) if kind == "reset"] != [0, 48]:
        raise ValueError("unexpected reset positions")
    for index, event in enumerate(e for e in events if e[0] == "transition"):
        _, frame, action, reward = event
        if frame.tobytes() != aviate["pixels"][aviate["post_frame"][index]].tobytes() or (
            action,
            reward,
        ) != (aviate["action"][index], aviate["reward"][index]):
            raise ValueError("causal transition mismatch")
    return {
        "origin": "MIT visible pixels; no inferred or hidden simulator states",
        "source": "https://github.com/VersesTech/gameworld/blob/"
        + gw.COMMIT
        + "/gameworld/envs/base/explode.py#L99-L130",
        "tape_sha256": run["npz_sha256"],
        "rows": rows,
    }


def object_fixture():
    """Eight hand-constructed frames for a motion reversal, disappearance and second contact."""
    path = [1, 2, 4, 5, 4, 5, None, 1]
    actions = [1, 1, 1, 0, 1, 0, 1]
    colors = {"background": [3, 5, 8], "wall": [200, 100, 50], "mover": [20, 180, 200]}
    frames, labels = [], []
    for t, x in enumerate(path):
        frame = np.empty((5, 9, 3), np.uint8)
        frame[:] = colors["background"]
        frame[1:4, 6] = colors["wall"]
        if x is not None:
            frame[2, x] = colors["mover"]
        visible = {name: pixel_statistics(frame, color) for name, color in colors.items()}
        labels.append(
            {
                "visible": visible,
                "contact": x == 5,
                "dx": None if not t or x is None or path[t - 1] is None else x - path[t - 1],
            }
        )
        frames.append(frame)
    rewards = [int(row["contact"]) for row in labels[1:]]
    if rewards != [0, 0, 1, 0, 1, 0, 0] or [row["dx"] for row in labels] != [
        None,
        1,
        2,
        1,
        -1,
        1,
        None,
        None,
    ]:
        raise ValueError("object fixture landmarks differ")
    return (
        np.stack(frames),
        {"actions": actions, "rewards": rewards},
        {"palette": colors, "frames": labels},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="recorded development tapes")
    parser.add_argument("--output", type=Path, required=True, help="fresh reference directory")
    args = parser.parse_args()
    cases_path = HERE / "reference_cases.json"
    checks = scalar_checks(json.loads(cases_path.read_text()))
    native = explode_reference(args.archive)
    frames, stream, labels = object_fixture()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames.rgb").write_bytes(frames.tobytes())
    gw.json_write(
        args.output / "object_stream.json",
        {
            "layout": "uint8 RGB HWC",
            "shape": list(frames.shape),
            "frames_sha256": hashlib.sha256(frames).hexdigest(),
            **stream,
        },
    )
    gw.json_write(args.output / "scorer_labels.json", labels)
    gw.json_write(args.output / "explode_visible.json", native)
    gw.json_write(
        args.output / "checks.json",
        {
            "checks": checks,
            "source_sha256": gw.digest(Path(__file__).read_bytes()),
            "cases_sha256": gw.digest(cases_path.read_bytes()),
            "development_manifest_sha256": DEV_MANIFEST,
            "native_frames": len(native["rows"]),
            "object_frames": len(frames),
            "files_sha256": {p.name: gw.digest(p.read_bytes()) for p in sorted(args.output.iterdir())},
        },
    )


if __name__ == "__main__":
    main()
