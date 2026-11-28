"""Exercise the changing-corridor scorer on generated records and analytic probabilities."""

import hashlib
import importlib.util
import json
import math
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "changing_corridor_score.py"
spec = importlib.util.spec_from_file_location("changing_corridor_score", SCRIPT)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def save(path, document):
    payload = json.dumps(document).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def harmonic(n):
    return sum(1 / k for k in range(1, n + 1))


def integer_dirichlet_d1(a, b):
    p = a / (a + b)
    return (-p * math.log(p) - (1 - p) * math.log(1 - p)
            + p * harmonic(a) + (1 - p) * harmonic(b) - harmonic(a + b))


def rows(arm, world, seed):
    base = np.full((6, 2, 2), 19.0)
    moved = base.copy()
    moved[2, :, 0] = [2.0, 2.0]
    predicted = base.copy()
    predicted[2, :, 0] = [4.0, 1.0]
    real_change = arm != "stationary" and world == "change"
    base, moved, predicted = [a.tolist() for a in (base, moved, predicted)]
    filt = dict(status="converged", reason="converged", inner_status="converged",
                outer_residual=None, free_energy=0.0, iterations=1)
    plan = dict(filt, reason="root_converged", outer_residual=0.0)
    x = 0
    for t in range(176):
        i = t - 128
        # Seed 1 has only one opportunity, the other seeds all have 48.
        nx = int(i >= 0 and seed == scorer.SEEDS[0])
        u = 2 if i < 0 or arm != "no_epistemics" or i == 0 else 0
        yield dict(t=t, x=x, m=0, u=u, x_next=nx, m_next=0,
                   alpha=moved if real_change and i > 0 else base,
                   forecast=predicted if real_change and i > 0 and i != 3 else base,
                   after=moved if real_change and i >= 0 else base,
                   plan=plan if i >= 0 else None, filter=filt,
                   plan_s=2.0 if i >= 0 else None, observe_s=4.0, update_s=1.0,
                   plan_peak_rss_bytes=1000 + seed if i >= 0 else None,
                   update_peak_rss_bytes=100 + t)
        x = nx


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    old = np.full((2, 3, 2, 2), 0.5)
    # Distinguish the target's two axes and its neighbor by source and mode.
    old[0, 2, :, 1] = [0.25, 0.75]
    old[1, 2, :, 0] = [0.9, 0.1]
    new = old.copy()
    new[0, 2, :, 0] = [0.8, 0.2]
    protocol = dict(version=2, seeds=list(scorer.SEEDS), arms=list(scorer.ARMS), worlds=list(scorer.WORLDS),
                    initial_state=[0, 0], warmup=128, followup=48, saturation_window=8, saturation_max_d1=0.08,
                    saturation_min_visits=16, observed_column=[0, 2, 0], old_position=old.tolist(),
                    new_position=new.tolist(), mode_transition=old[1].tolist(), gate_thresholds=scorer.GATES)
    protocol_sha = save(root / "protocol.json", protocol)
    models = {}
    for arm in scorer.ARMS:
        models[arm] = {}
        for role in ("plan", "filter"):
            name = f"{arm}-{role}.synthetic"
            asset_hash = save(root / name, [arm, role])
            models[arm][role] = dict(file=name, sha256=asset_hash, structure_hash=asset_hash[::-1])
    models_sha = save(root / "models.json", dict(protocol_sha256=protocol_sha, arms=models))
    binding = dict(protocol_sha256=protocol_sha, models_sha256=models_sha, source_commit="1" * 40)
    qualification_sha = save(root / "qualification.json", dict(binding, status="passed"))
    freeze_sha = save(root / "freeze.json", dict(binding, qualification_sha256=qualification_sha))
    seal = dict(binding, cells={}, freeze_sha256=freeze_sha)
    for seed, arm, world in product(scorer.SEEDS, scorer.ARMS, scorer.WORLDS):
        key = scorer.cell_key(seed, arm, world)
        path = root / "cells" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(json.dumps(r).encode() + b"\n" for r in rows(arm, world, seed))
        path.with_suffix(".jsonl").write_bytes(data)
        receipt = dict(binding, seed=seed, arm=arm, world=world, status="complete", reason=None,
                       completed_steps=176, plan_hash=models[arm]["plan"]["structure_hash"],
                       filter_hash=models[arm]["filter"]["structure_hash"])
        seal["cells"][key] = {"jsonl": hashlib.sha256(data).hexdigest(),
                              "json": save(path.with_suffix(".json"), receipt)}
    save(root / "complete.json", seal)
    return root


def kl(p):
    return p * math.log(2 * p) + (1 - p) * math.log(2 * (1 - p))


def test_complete_grid_estimands_and_reports(archive):
    report = scorer.write_report(archive)
    assert report["complete_grid"]
    assert len(report["cells"]) == 36
    group = report["arm_world"]["changing"]["change"]
    mean = group["means"]
    assert mean["target_trace_gain"] == pytest.approx(15 / 16 * (0.5 / 5 - 0.5 / 39))
    assert mean["target_d1_gain"] == pytest.approx(14 / 16 * (integer_dirichlet_d1(4, 1)
                                                             - integer_dirichlet_d1(19, 19)))
    assert mean["unchanged_trace_gain"] == pytest.approx(0)
    assert mean["unchanged_d1_gain"] == pytest.approx(0)
    assert mean["target_kl_first24"] == pytest.approx(2 / 24 * kl(0.8))
    assert mean["target_kl_all48"] == pytest.approx(2 / 48 * kl(0.8))
    assert mean["unchanged_kl_first24"] == pytest.approx((kl(0.25) + kl(0.9)) / 11)
    assert mean["mode_kl"] == 0
    assert group["mean_recovery_delay"] == 4
    assert group["recovery_censored_count"] == 0
    stationary = report["arm_world"]["stationary"]["change"]
    assert stationary["mean_recovery_delay"] is None
    assert stationary["recovery_censored_count"] == 6
    assert report["arm_world"]["changing"]["sham"]["mean_recovery_delay"] == 0
    paired = report["equal_seed_mean_comparisons"]["change"]
    assert paired["probe_visit_fraction_advantage"] == pytest.approx(5 / 6 * 47 / 48)
    assert paired["true_target_kl_advantage_nats"] == pytest.approx(22 / 24 * kl(0.8))
    assert mean["visit_fraction"] == pytest.approx((5 + 1 / 48) / 6)
    assert report["arm_world"]["no_epistemics"]["change"]["means"]["visits_per_opportunity"] == pytest.approx(
        (1 + 5 / 48) / 6
    )
    gates = report["change_world_gates"]
    assert {k: v["status"] for k, v in gates.items()} == {
        k: "fail" if k == "goal_occupancy" else "pass" for k in scorer.GATES
    }
    assert group["mean_applied_seconds"] == dict(plan_s=96, observe_s=704, update_s=176)
    assert group["pooled_peak_rss_bytes"] == dict(plan_peak_rss_bytes=20706, update_peak_rss_bytes=275)
    assert json.loads((archive / "score.json").read_text())["cells"].keys() == report["cells"].keys()
    summary = (archive / "score.txt").read_text()
    assert all(key in summary for key in report["cells"])
    assert "largest sampled whole-process resident size" in summary


def test_zero_opportunities_and_a_truncated_recovery_streak():
    sample = list(rows("stationary", "change", 19701))
    truth = np.full((12, 2), 0.5)
    truth[scorer.TARGET_INDEX] = [0.8, 0.2]
    for row in sample[128:]:
        row["x"] = 1
    trace, d1, forecast = scorer.uncertainty(sample)
    forecast[-2:, scorer.TARGET_INDEX] = truth[scorer.TARGET_INDEX]
    result = scorer.score_cell(sample, 128, 8, truth, trace, d1, forecast)
    assert result["metrics"]["visit_fraction"] == 0
    assert result["metrics"]["visits_per_opportunity"] is None
    assert result["recovery"] == dict(delay=None, right_censored=True, followup_steps=48)


def replace_cell(root, key, data, status, reason, failure=None):
    base = root / "cells" / key
    tape = b"".join(json.dumps(r).encode() + b"\n" for r in data)
    base.with_suffix(".jsonl").write_bytes(tape)
    receipt = json.loads(base.with_suffix(".json").read_text())
    receipt.update(status=status, reason=reason, completed_steps=len(data))
    if failure is not None:
        receipt["failure"] = failure
    seal = json.loads((root / "complete.json").read_text())
    seal["cells"][key] = {"jsonl": hashlib.sha256(tape).hexdigest(), "json": save(base.with_suffix(".json"), receipt)}
    save(root / "complete.json", seal)


def test_failed_and_unsaturated_cells_do_not_contribute_survivor_means(archive, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(archive, root)
    key = "19701/changing-change"
    failure = {"phase": "planning", "plan": {"status": "budget_exhausted"}}
    replace_cell(root, key, list(rows("changing", "change", 19701))[:128], "failed", "root solve", failure)
    key2 = "19702/no_epistemics-sham"
    data = list(rows("no_epistemics", "sham", 19702))[:128]
    data[-8]["forecast"] = np.ones((6, 2, 2)).tolist()
    replace_cell(root, key2, data, "unsaturated", "warmup diagnostic")
    report = scorer.score(root)
    assert not report["complete_grid"]
    assert report["arm_world"] is report["equal_seed_mean_comparisons"] is None
    assert all(gate["status"] == "unscored" and gate["estimate"] is None
               for gate in report["change_world_gates"].values())
    assert report["cells"][key]["receipt"]["failure"] == failure
    assert report["cells"][key2]["metrics"] is None
    assert not report["cells"][key2]["saturation"]["passed"]
    assert report["per_seed_comparisons"]["19701"]["change"]["goal_occupancy"] is None
    assert report["cells"]["19701/stationary-sham"]["metrics"] is not None


@pytest.mark.parametrize("defect, message", [
    ("row digest", "SHA-256 mismatch"),
    ("model digest", "SHA-256 mismatch"),
    ("freeze digest", "SHA-256 mismatch"),
    ("inventory", "inventory differs"),
    ("missing cell", "fixed cell grid"),
    ("wrong solve", "Uncertified planning root"),
    ("unpaired", "Mismatched first post-switch action"),
])
def test_rejects_damaged_or_inconsistent_archives(archive, tmp_path, defect, message):
    root = tmp_path / "copy"
    shutil.copytree(archive, root)
    key = "19701/changing-change"
    base = root / "cells" / key
    if defect == "row digest":
        base.with_suffix(".jsonl").write_text("")
    elif defect == "model digest":
        (root / "changing-plan.synthetic").write_text("replaced")
    elif defect == "freeze digest":
        (root / "freeze.json").write_text("{}")
    elif defect == "inventory":
        (base.parent / "extra.json").write_text("{}")
    elif defect == "missing cell":
        seal = json.loads((root / "complete.json").read_text())
        del seal["cells"][key]
        save(root / "complete.json", seal)
    else:
        data = list(rows("changing", "change", 19701))
        if defect == "wrong solve":
            data[128]["plan"] = dict(data[128]["plan"], reason="budget_exhausted")
        else:
            data[128]["u"] = 1
        replace_cell(root, key, data, "complete", None)
    with pytest.raises(ValueError, match=message):
        scorer.score(root)
