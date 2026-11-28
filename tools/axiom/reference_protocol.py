"""Freeze the Gameworld reference design and score complete, separately verified raw reward streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = {
    "tapes/manifest.json": "abca225803b4b79801c7fcf2a5823cc639237f12b707f779dea3f3815f3ed376",
    "replay.json": "5b7461d497c76abef33146a13a41920b610a36ab827a812f3cada95a193a9059",
    "score.json": "e03979e75fe857384aff14bde18c5356569412bebf92c37d88a489f4a26713a3",
    "check.json": "b35879e652c177b790119ec0b04e324e442c46a88c47470575606aab521ef0a6",
    "overlay/uv.lock": "035d1f9b20a9d490a55f142c644cd27bb2789d03b0806523c1f85d64de720937",
}
INTERMEDIATE_CHECKS_SHA256 = "27755723842ab2910fc5b75584326975a055ce88902cef27b478a595e5033cd1"
TOOLS = (
    "reference_protocol.json",
    "reference_cases.json",
    "reference_intermediates.py",
    "reference_protocol.py",
    "gameworld_replay.py",
    "gameworld_replay_check.py",
)
# Two-sided 95% union bound for 20 means and 9 degrees of freedom.
SIMULTANEOUS_T9 = 4.1457888284431945
PAIRED_T9 = 2.262157162798205
MAIN_ARMS = ("untuned", "fixed_r", "no_bmr", "no_ig", "color", "shape")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_pairs(protocol, stage):
    settings = protocol["splits"][stage]
    for game_index, game in enumerate(protocol["games"]):
        if stage == "development" and game != settings["pilot_game"]:
            continue
        for index, seed in enumerate(settings["model_seeds"]):
            yield {
                "game": game,
                "model_seed": seed,
                "game_seed": settings["game_seed_base"] + 100 * game_index + index,
                "steps": settings["steps"],
            }


def frozen_inventory(protocol):
    pairs = {stage: list(seed_pairs(protocol, stage)) for stage in protocol["splits"]}
    evaluation = pairs["evaluation"]
    return {
        "stage_pairs": pairs,
        "main_arms": {name: evaluation for name in MAIN_ARMS},
        "privileged_color_remap": [r for r in evaluation if r["game"] == "Explode"],
        "illustrations": [r for r in evaluation if r["model_seed"] == evaluation[0]["model_seed"]],
        "optional_random": evaluation,
        "optional_rollout_sweep": {
            "policies": [64, 128, 256, 512],
            "samples": [1, 3, 5],
            "pairs": [
                r
                for r in evaluation
                if r["model_seed"] in protocol["splits"]["evaluation"]["model_seeds"][:3]
            ],
            "default_cell": "Reuse matched 512x3 untuned runs.",
        },
    }


def freeze(root, intermediate):
    protocol = json.loads((HERE / "reference_protocol.json").read_text())
    for name, expected in EVIDENCE.items():
        if sha(root / name) != expected:
            raise ValueError(f"changed development evidence: {name}")
    tools = {name: sha(HERE / name) for name in TOOLS}
    if sha(intermediate / "checks.json") != INTERMEDIATE_CHECKS_SHA256:
        raise ValueError("changed v1 intermediate receipt")
    checks = json.loads((intermediate / "checks.json").read_text())
    intermediate_files = {
        "frames.rgb",
        "object_stream.json",
        "scorer_labels.json",
        "explode_visible.json",
    }
    if (
        checks["source_sha256"] != tools["reference_intermediates.py"]
        or checks["cases_sha256"] != tools["reference_cases.json"]
        or checks["development_manifest_sha256"] != EVIDENCE["tapes/manifest.json"]
        or set(checks["files_sha256"]) != intermediate_files
    ):
        raise ValueError("intermediate provenance mismatch")
    for name, expected in checks["files_sha256"].items():
        if sha(intermediate / name) != expected:
            raise ValueError(f"changed intermediate: {name}")
    return {
        "version": protocol["version"],
        "stage": "reference_only",
        "protocol": protocol,
        "tools_sha256": tools,
        "development_sha256": EVIDENCE,
        "intermediate_checks_sha256": sha(intermediate / "checks.json"),
        "intermediate_sha256": checks["files_sha256"],
        "inventory": frozen_inventory(protocol),
    }


def verdict(low, high, margin):
    if low >= -margin and high <= margin:
        return "within_margin"
    if low > margin:
        return "above_margin"
    if high < -margin:
        return "below_margin"
    return "inconclusive"


def printed_target(deltas, margin):
    mean = statistics.mean(deltas)
    half = PAIRED_T9 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return {
        "delta": mean,
        "ci": [mean - half, mean + half],
        "verdict": verdict(mean - half, mean + half, margin),
    }


def estimate(values, published, margin):
    n = len(values)
    if n != 10 or not all(math.isfinite(x) for x in values):
        raise ValueError("the published comparison requires 10 finite returns")
    average, sd = statistics.mean(values), statistics.stdev(values)
    old_mean, old_sd = published
    delta = average - old_mean
    historical_sd_upper = (old_sd + 0.5) * math.sqrt(10 / 9)
    half = 0.5 + SIMULTANEOUS_T9 * (historical_sd_upper + sd) / math.sqrt(10)
    low, high = delta - half, delta + half
    return {
        "mean": average,
        "sample_sd": sd,
        "published_mean_sd": published,
        "printed_target": printed_target([v - old_mean for v in values], margin),
        "delta": delta,
        "ci": [low, high],
        "verdict": verdict(low, high, margin),
    }


def score_primary(protocol, records):
    """Scorer boundary after the campaign owner verifies the closed run manifest and raw tape hashes.

    Each supplied record has the four fields in seed_pairs plus status and a 1D sequence
    of raw rewards. Include a partial reward sequence for failed runs. Duplicate attempts
    must be resolved against the campaign's recovery log before crossing this boundary.
    """
    expected = {(r["game"], r["model_seed"]): r for r in seed_pairs(protocol, "evaluation")}
    seen, failed, totals = set(), [], {}
    for record in records:
        key = record["game"], record["model_seed"]
        if key in seen or key not in expected or any(record[k] != v for k, v in expected[key].items()):
            raise ValueError(f"duplicate or mismatched run: {key}")
        if record["arm"] != "untuned" or record["condition"] != "base":
            raise ValueError(f"unexpected primary arm or condition: {key}")
        seen.add(key)
        rewards = record["rewards"]
        if (
            record["status"] != "ok"
            or len(rewards) != expected[key]["steps"]
            or not all(math.isfinite(x) for x in rewards)
        ):
            failed.append(
                {
                    "game": key[0],
                    "model_seed": key[1],
                    "native_status": record["status"],
                    "observed_steps": len(rewards),
                }
            )
            continue
        totals[key] = math.fsum(rewards)
    missing = [r for key, r in expected.items() if key not in seen]
    by_seed = [{**ident, "return": totals[key]} for key, ident in expected.items() if key in totals]
    result = {
        "planned": len(expected),
        "scored": len(totals),
        "failed": failed,
        "missing": missing,
        "per_seed": by_seed,
        "games": {},
        "macro": None,
        "whole_suite_agreement": None,
    }
    if failed or missing:
        return result
    margin = protocol["scoring"]["margin"]
    for game, published in zip(protocol["games"], protocol["table1"]["untuned"], strict=True):
        values = [totals[key] for key in expected if key[0] == game]
        result["games"][game] = estimate(values, published, margin)
    games = list(result["games"].values())
    low, high = (statistics.mean(g["ci"][index] for g in games) for index in (0, 1))
    result["macro"] = {
        "delta": statistics.mean(g["delta"] for g in games),
        "ci": [low, high],
        "verdict": verdict(low, high, margin),
    }
    macro_deltas = [
        statistics.mean(
            totals[(game, seed)] - result["games"][game]["published_mean_sd"][0]
            for game in protocol["games"]
        )
        for seed in protocol["splits"]["evaluation"]["model_seeds"]
    ]
    result["macro"]["printed_target"] = printed_target(macro_deltas, margin)
    result["whole_suite_agreement"] = all(g["verdict"] == "within_margin" for g in games)
    return result


def paired_contrast(differences):
    """One difference per matched seed; also applies to ten within-seed equal-game averages."""
    if len(differences) != 10 or not all(map(math.isfinite, differences)):
        raise ValueError("paired contrast requires ten complete seed differences")
    mean = statistics.mean(differences)
    half = PAIRED_T9 * statistics.stdev(differences) / math.sqrt(10)
    return {"differences": list(differences), "mean": mean, "ci95": [mean - half, mean + half]}


def perturbation_scores(base, altered):
    if (
        len(base) != 10000
        or len(altered) != 10000
        or not all(map(math.isfinite, base))
        or not all(map(math.isfinite, altered))
    ):
        raise ValueError("two complete 10000-step reward streams required")
    differences = [y - x for x, y in zip(base, altered, strict=True)]
    pre, post = {}, {}
    for name, rewards in (("base", base), ("altered", altered), ("difference", differences)):
        for row, chunk in ((pre, rewards[:5000]), (post, rewards[5000:])):
            total = math.fsum(chunk)
            row[name] = {"sum": total, "rate": total / 5000}
    # Compare sums directly. A 250-action window can lose at most 2.5 raw units.
    recovered = [math.fsum(differences[i : i + 250]) >= -2.5 for i in range(5000, 10000, 250)]
    delay = next((i * 250 for i in range(len(recovered) - 1) if recovered[i] and recovered[i + 1]), None)
    return {"pre": pre, "post": post, "recovery_delay": delay, "censored": delay is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="existing development evidence root")
    parser.add_argument(
        "--intermediates",
        required=True,
        type=Path,
        help="independent reference_intermediates.py outputs",
    )
    parser.add_argument("--output", required=True, type=Path, help="fresh JSON reference manifest")
    args = parser.parse_args()
    result = freeze(args.archive, args.intermediates)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
