"""Seal the development checks and run the preregistered corridor cells."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import digamma

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKER = HERE / "changing_corridor.py"
PROTOCOL = HERE / "changing_corridor_protocol.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, item):
    with path.open("x") as f:
        json.dump(item, f, indent=2, allow_nan=False)
        f.write("\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def cell_order(p, seeds):
    for i, seed in enumerate(seeds):
        for j in range(len(p["worlds"])):
            world = p["worlds"][(i + j) % len(p["worlds"])]
            for k in range(len(p["arms"])):
                yield seed, p["arms"][(i + j + k) % len(p["arms"])], world


def stamps(root):
    return {
        "protocol_sha256": sha(root / "protocol.json"),
        "models_sha256": sha(root / "models.json"),
    }


def verify_models(root):
    p = load(root / "protocol.json")
    manifest = load(root / "models.json")
    if sha(PROTOCOL) != sha(root / "protocol.json") or manifest[
        "protocol_sha256"
    ] != sha(PROTOCOL):
        raise ValueError("protocol source mismatch")
    if set(manifest["arms"]) != set(p["arms"]):
        raise ValueError("model arm inventory mismatch")
    for arm in p["arms"]:
        entries = manifest["arms"][arm]
        if set(entries) != {"plan", "filter"}:
            raise ValueError("missing model window")
        for item in entries.values():
            if sha(root / item["file"]) != item["sha256"]:
                raise ValueError("model digest mismatch")
    return p, manifest


def inputs(revision):
    names = (
        "src/cffg",
        "uv.lock",
        WORKER.relative_to(REPO),
        PROTOCOL.relative_to(REPO),
    )
    return {str(name): git("rev-parse", f"{revision}:{name}") for name in names}


def warm_facts(trace, p):
    mode, action, source = p["observed_column"]
    warm = trace[: p["warmup"]]
    visits = sum((r["m"], r["u"], r["x"]) == (mode, action, source) for r in warm)
    bits = []
    for row in warm[-p["saturation_window"] :]:
        a = np.asarray(row["forecast"])[3 * mode + action, :, source]
        mu = a / a.sum()
        bits.append(
            float(-mu @ np.log(mu) + mu @ digamma(a + 1) - digamma(a.sum() + 1))
        )
    return visits, bits


def qualify(root):
    p, manifest = verify_models(root)
    if (root / "cells").exists() or (root / "freeze.json").exists():
        raise ValueError("development qualification must precede the formal stage")
    common = stamps(root)
    rows = {}
    prefixes = {}
    dev_heads = set()
    for seed, arm, world in cell_order(p, p["development_seeds"]):
        key = f"{seed}/{arm}-{world}"
        stem = root / "development" / "cells" / key
        result = load(stem.with_suffix(".json"))
        trace = [
            json.loads(line)
            for line in stem.with_suffix(".jsonl").read_text().splitlines()
        ]
        if (
            (seed, arm, world) != (result["seed"], result["arm"], result["world"])
            or any(result[k] != v for k, v in common.items())
            or result["completed_steps"] != len(trace)
        ):
            raise ValueError(f"development identity mismatch: {key}")
        dev_heads.add(result["source_commit"])
        for kind in ("plan", "filter"):
            if result[kind + "_hash"] != manifest["arms"][arm][kind]["structure_hash"]:
                raise ValueError(f"development model mismatch: {key}")
        prefix = []
        for t, r in enumerate(trace):
            current = (r["x"], r["m"])
            if r["t"] != t or current != (
                tuple(p["initial_state"]) if not prefix else prefix[-1][-2:]
            ):
                raise ValueError(f"development chronology mismatch: {key} at {t}")
            if t < p["warmup"]:
                expected = 2 if current[1] == 1 or current[0] == 0 else 0
                if r["plan"] is not None or r["u"] != expected:
                    raise ValueError(
                        "development acquisition differed from registered policy"
                    )
            elif (
                r["plan"]["reason"] != "root_converged"
                or r["plan"]["outer_residual"] is None
            ):
                raise ValueError("development row lacks a planning root")
            for fact in [r["filter"]] + ([r["plan"]] if r["plan"] else []):
                if (
                    fact["inner_status"] != "converged"
                    or fact["status"] != "converged"
                    or not np.isfinite(fact["free_energy"])
                ):
                    raise ValueError("invalid development solver classification")
            prefix.append((*current, r["u"], r["x_next"], r["m_next"]))
        warm = prefix[: p["warmup"]]
        comparison = prefixes.setdefault(seed, warm)
        n = min(len(comparison), len(warm))
        if warm[:n] != comparison[:n]:
            raise ValueError(
                "paired development physical tapes disagree during shared acquisition"
            )
        if len(prefix) > p["warmup"]:
            first = prefix[p["warmup"]][:3]
            if prefixes.setdefault((seed, arm), first) != first:
                raise ValueError(
                    "first development policy action differs before changed-world observation"
                )
        visits, d1 = warm_facts(trace, p)
        if result["status"] != "failed":
            if visits != result["warm_visits"]:
                raise ValueError("exposure count differs from the applied warm actions")
            np.testing.assert_allclose(
                result["warm_last_d1"], d1, atol=1e-13, rtol=1e-13
            )
        if result["status"] == "unsaturated":
            if len(trace) != p["warmup"] or (
                visits >= p["saturation_min_visits"]
                and max(d1) <= p["saturation_max_d1"]
            ):
                raise ValueError("unjustified unsaturated development receipt")
        ready = result["status"] == "complete"
        if ready and (
            len(trace) != p["warmup"] + p["development_followup"]
            or result["warm_visits"] < p["saturation_min_visits"]
            or max(result["warm_last_d1"]) > p["saturation_max_d1"]
        ):
            raise ValueError("invalid completed development cell")
        rows[key] = {
            k: result[k]
            for k in (
                "status",
                "reason",
                "warm_visits",
                "warm_last_d1",
                "completed_steps",
            )
        }
        rows[key].update(
            json=sha(stem.with_suffix(".json")), jsonl=sha(stem.with_suffix(".jsonl"))
        )
    qualified_inputs = inputs("HEAD")
    for head in dev_heads:
        if inputs(head) != qualified_inputs:
            raise ValueError(
                "development runtime or protocol differs from the current candidate"
            )
    passed = all(r["status"] == "complete" for r in rows.values())
    record = dict(
        status="passed" if passed else "no_go",
        development_heads=sorted(dev_heads),
        qualified_inputs=qualified_inputs,
        cells=rows,
        **common,
    )
    save(root / "qualification.json", record)
    print(record["status"])
    return passed


def readiness(root):
    """Independently replay the development gate on either archived protocol."""
    p = load(root / "protocol.json")
    result = load(root / "qualification.json")
    if (
        result["status"] != "no_go"
        or (root / "freeze.json").exists()
        or (root / "cells").exists()
    ):
        raise ValueError("this review covers development stops before held-out access")
    manifests = load(root / "models.json")
    protocol_hash = sha(root / "protocol.json")
    if (
        manifests["protocol_sha256"] != protocol_hash
        or result["protocol_sha256"] != protocol_hash
    ):
        raise ValueError("development protocol digest mismatch")
    for windows in manifests["arms"].values():
        for window in windows.values():
            if sha(root / window["file"]) != window["sha256"]:
                raise ValueError("development model digest mismatch")
    dev_seeds = (
        p["development_seeds"] if "development_seeds" in p else [p["development_seed"]]
    )
    cells = {}
    warm_by_seed = {}
    for seed, arm, world in cell_order(p, dev_seeds):
        key = f"{seed}/{arm}-{world}"
        stem = root / "development" / "cells" / key
        raw = load(stem.with_suffix(".json"))
        tape = [
            json.loads(line)
            for line in stem.with_suffix(".jsonl").read_text().splitlines()
        ]
        version = subprocess.check_output(
            [
                "git",
                "show",
                raw["source_commit"] + ":tools/changing_corridor_protocol.json",
            ],
            cwd=REPO,
        )
        if (
            hashlib.sha256(version).hexdigest() != protocol_hash
            or (seed, arm, world) != (raw["seed"], raw["arm"], raw["world"])
            or raw["protocol_sha256"] != protocol_hash
            or raw["models_sha256"] != sha(root / "models.json")
            or len(tape) != raw["completed_steps"]
        ):
            raise ValueError(f"incorrect development identity: {key}")
        for kind in ("plan", "filter"):
            if raw[kind + "_hash"] != manifests["arms"][arm][kind]["structure_hash"]:
                raise ValueError(f"model mismatch: {key}")
        visits, d1 = warm_facts(tape, p)
        np.testing.assert_allclose(d1, raw["warm_last_d1"], rtol=1e-13, atol=1e-13)
        if visits != raw["warm_visits"]:
            raise ValueError(f"wrong exposure count: {key}")
        warm = tape[: p["warmup"]]
        signature = [
            (r["t"], r["x"], r["m"], r["u"], r["x_next"], r["m_next"]) for r in warm
        ]
        if warm_by_seed.setdefault(seed, signature) != signature:
            raise ValueError("development arms did not see the same warm trajectory")
        for t, r in enumerate(warm):
            expected = (
                2 * (t % 2)
                if p["version"] == 1
                else (2 if r["m"] == 1 or r["x"] == 0 else 0)
            )
            if r["t"] != t or r["u"] != expected:
                raise ValueError(f"incorrect acquisition action: {key} at {t}")
        eligible_visits = visits >= p["saturation_min_visits"]
        violations = sum(x > p["saturation_max_d1"] for x in d1)
        ready = (
            eligible_visits and len(d1) == p["saturation_window"] and violations == 0
        )
        if raw["status"] == "unsaturated" and (ready or len(tape) != p["warmup"]):
            raise ValueError(f"wrong saturation stop: {key}")
        if raw["status"] == "complete" and (
            not ready or len(tape) != p["warmup"] + p["development_followup"]
        ):
            raise ValueError(f"incorrect completed cell: {key}")
        cells[key] = dict(
            seed=seed,
            arm=arm,
            world=world,
            status=raw["status"],
            reason=raw["reason"],
            visits=visits,
            preaction_novelty_nats=d1,
            above_threshold_ticks=violations,
            completed_steps=len(tape),
            native_post_actions=max(0, len(tape) - p["warmup"]),
            trace_sha256=sha(stem.with_suffix(".jsonl")),
            receipt_sha256=sha(stem.with_suffix(".json")),
        )
    if all(row["status"] == "complete" for row in cells.values()):
        raise ValueError("all development cells passed")
    report = dict(
        version=p["version"],
        status="no_go",
        scored_cells=0,
        qualified=sum(row["status"] == "complete" for row in cells.values()),
        total=len(cells),
        protocol_sha256=protocol_hash,
        models_sha256=sha(root / "models.json"),
        qualification_sha256=sha(root / "qualification.json"),
        cells=cells,
    )
    existing = root / "readiness.json"
    if existing.exists():
        if load(existing) != report:
            raise ValueError("archived readiness report does not match the raw tapes")
    else:
        save(existing, report)
    print(
        f"version {report['version']}: {report['qualified']}/{report['total']} development cells passed, zero held-out cells opened"
    )
    return report


def freeze(root):
    p, _ = verify_models(root)
    check = load(root / "qualification.json")
    if check["status"] != "passed" or (root / "cells").exists():
        raise ValueError("formal opening requires a passed development inventory")
    if check["qualified_inputs"] != inputs("HEAD"):
        raise ValueError("candidate inputs changed after development qualification")
    expected = {f"{s}/{a}-{w}" for s, a, w in cell_order(p, p["development_seeds"])}
    if set(check["cells"]) != expected:
        raise ValueError("incomplete development inventory")
    for key, record in check["cells"].items():
        stem = root / "development" / "cells" / key
        if record["status"] != "complete" or any(
            sha(stem.with_suffix("." + ext)) != record[ext] for ext in ("json", "jsonl")
        ):
            raise ValueError("development evidence differs from qualification")
    common = stamps(root)
    if any(check[k] != v for k, v in common.items()):
        raise ValueError("unqualified manifest or protocol")
    if (
        git("status", "--porcelain", "--untracked-files=no")
        or sha(PROTOCOL) != common["protocol_sha256"]
    ):
        raise ValueError("freeze requires committed sources")
    save(
        root / "freeze.json",
        dict(
            source_commit=git("rev-parse", "HEAD"),
            source_tree=git("rev-parse", "HEAD^{tree}"),
            qualification_sha256=sha(root / "qualification.json"),
            **common,
        ),
    )


def campaign(root):
    p, _ = verify_models(root)
    freeze_record = load(root / "freeze.json")
    common = stamps(root)
    commit = git("rev-parse", "HEAD")
    if (
        commit != freeze_record["source_commit"]
        or any(freeze_record[k] != v for k, v in common.items())
        or sha(root / "qualification.json") != freeze_record["qualification_sha256"]
    ):
        raise ValueError("candidate identity differs from the qualification and freeze")
    cells = {}
    for seed, arm, world in cell_order(p, p["seeds"]):
        key = f"{seed}/{arm}-{world}"
        stem = root / "cells" / key
        if not stem.with_suffix(".json").exists():
            if stem.with_suffix(".jsonl").exists():
                raise ValueError(f"interrupted cell needs review: {key}")
            command = [
                sys.executable,
                str(WORKER),
                "cell",
                str(root),
                "--arm",
                arm,
                "--world",
                world,
                "--seed",
                str(seed),
            ]
            subprocess.run(command, cwd=REPO, check=True)
        r = load(stem.with_suffix(".json"))
        if (
            r["source_commit"] != commit
            or any(r[k] != v for k, v in common.items())
            or (r["seed"], r["arm"], r["world"]) != (seed, arm, world)
        ):
            raise ValueError(f"foreign cell {key}")
        cells[key] = {
            ext: sha(stem.with_suffix("." + ext)) for ext in ("json", "jsonl")
        }
    save(
        root / "complete.json",
        dict(
            source_commit=commit,
            freeze_sha256=sha(root / "freeze.json"),
            cells=cells,
            **common,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("qualify", "freeze", "campaign", "readiness")
    )
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    with (args.root / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "qualify":
            qualify(args.root)
        elif args.command == "readiness":
            readiness(args.root)
        elif args.command == "freeze":
            freeze(args.root)
        else:
            campaign(args.root)
