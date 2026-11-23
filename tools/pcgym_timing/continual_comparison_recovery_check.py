"""Exercise campaign interruption boundaries without running a reactor or a solver."""
import fcntl
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import continual_comparison as runner
import continual_comparison_analysis as scorer


SEED = 6601


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    seeds = (SEED, SEED + 1)
    for module in (runner, scorer):
        monkeypatch.setattr(module, "LIBRARIES", ("cffg",))
        monkeypatch.setattr(module, "SEEDS", seeds)
    monkeypatch.setattr(scorer, "N_EPISODES", 16)
    plan = dict(libraries=["cffg"], arms=list(runner.SETTINGS), motion_sd=runner.SETTINGS,
                seeds=list(seeds), conditions=list(runner.CONDITIONS), steps=runner.STEPS,
                decision_deadline_s=1, stationary_budget_pct=1, block_timing="full blocks only",
                starting_method={"initial_ua": 50000, "learner": {"motion_log_sd": .012}},
                schedule=dict(nominal=50000, changed=48150, changed_interval=[178, 373]),
                scoring=dict(before=[0, 178], changed=[178, 373], returned=[373, runner.STEPS],
                             settled_changed=[203, 373], transition_window=25),
                frozen={"library_tree": "test"})
    runner.save(tmp_path/"protocol.json", plan)
    monkeypatch.setattr(runner, "freeze", lambda *args: None)
    monkeypatch.setattr(runner, "sources", lambda peers: plan["frozen"])
    data = None, {}, {}, Mock(), None, plan["frozen"]
    monkeypatch.setattr(runner, "context", lambda *args: data)
    checks = dict(affine=[{"passed": True}]*3, updated=1, fresh=1, nominal=0, restored=0,
                  plant=dict(passed=True, switch_ticks=[178, 373], continuous_state_time_and_integrator=True,
                             learner_observation=["Ca_norm", "T_norm"]))
    monkeypatch.setattr(runner, "qualification", lambda *args: checks)
    planner = Mock(metadata={})
    planner.solve.return_value = {"status": "ok"}
    monkeypatch.setattr(runner, "factory", lambda lib: lambda *args: planner)
    monkeypatch.setitem(sys.modules, "continual_estimator", SimpleNamespace(warmup=lambda: None))
    monkeypatch.setitem(sys.modules, "cffg_adapter", SimpleNamespace(__file__="synthetic"))
    state = SimpleNamespace(fail=None, calls=[], launches=[])

    def episode(env, params, arrays, fixture, window, seed, condition, arm, out, *, motion_sd, planner, provenance):
        folder = out/f"seed-{seed}"
        folder.mkdir(exist_ok=True)
        stem = folder/f"{condition}-{arm}"
        trace = stem.with_suffix(".jsonl")
        assert "attempts" in trace.relative_to(tmp_path).parts
        assert not trace.exists()
        state.calls.append((seed, condition, arm))
        trace.write_text('{"tick":0}\n')
        if state.fail == "trace" and condition == "stationary" and arm == "candidate":
            raise InterruptedError("trace")
        runner.save(stem.with_suffix(".json"), dict(seed=seed, condition=condition, arm=arm, steps=runner.STEPS,
                    motion_sd=motion_sd, trace_sha256=runner.digest(trace), episode_s=.5, warm_setup_s=.1, **provenance))

    def launch(out, lib, peers, lock_fd, seed=None):
        state.launches.append(seed)
        if seed == SEED and state.fail == "before setup":
            raise InterruptedError("before setup")
        runner.worker(out, lib, seed, peers)
        if seed == SEED and state.fail == "after worker":
            raise InterruptedError("after worker")
        return 11.

    monkeypatch.setattr(runner, "episode", episode)
    monkeypatch.setattr(runner, "run_child", launch)
    state.run = lambda: runner.campaign(tmp_path, None, None, None)
    return state


@pytest.mark.parametrize(("point", "remaining"), [("before setup", 8), ("trace", 6), ("after worker", 8)])
def test_interrupted_campaign(tmp_path, campaign, point, remaining):
    campaign.fail = point
    with pytest.raises(InterruptedError, match=point):
        campaign.run()
    if point == "trace":
        assert len(list(tmp_path.glob("attempts/*/*/*/stationary-candidate.jsonl"))) == 1
        assert not list(tmp_path.glob("cffg/*/stationary-candidate.jsonl"))
    campaign.fail = None
    campaign.run()
    _, workers = scorer.metadata(tmp_path, {})
    retry, fresh = workers["cffg"]
    assert retry["episodes_run"] == remaining and retry["process_wall_s"] is None
    assert fresh["episodes_run"] == 8 and fresh["process_wall_s"] == 11.
    assert runner.read(tmp_path/"completed.json")["worker_s"] is None
    assert campaign.launches.count(SEED) == (1 if point == "after worker" else 2)
    expected = {f"{c}-{a}.{ext}" for c in runner.CONDITIONS for a in runner.SETTINGS for ext in ("json", "jsonl")}
    episodes = {}
    for seed in runner.SEEDS:
        folder = tmp_path/"cffg"/f"seed-{seed}"
        assert {p.name for p in folder.iterdir()} == expected
        episodes[str(seed)] = {c: {a: runner.read(folder/f"{c}-{a}.json") for a in runner.SETTINGS}
                               for c in runner.CONDITIONS}
    assert scorer.worker_costs(episodes, [retry, fresh])["process_wall_s"]["n"] == 1
    assert scorer.worker_costs(episodes, [retry])["process_wall_s"]["median"] is None
    with pytest.raises(ValueError, match="below measured stages"):
        scorer.worker_costs(episodes, [dict(fresh, process_wall_s=1.)])
    sealed = (tmp_path/"completed.json").read_bytes()
    with pytest.raises(ValueError, match="sealed"):
        campaign.run()
    with pytest.raises(ValueError, match="sealed"):
        runner.worker(tmp_path, "cffg", None, None)
    assert (tmp_path/"completed.json").read_bytes() == sealed


def test_recover_interrupted_publication_and_legacy_names(tmp_path, campaign, monkeypatch):
    original = Path.rename

    def stop_before_sidecar(path, target):
        if target.name == "stationary-candidate.json":
            raise InterruptedError("publish")
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", stop_before_sidecar)
        with pytest.raises(InterruptedError, match="publish"):
            campaign.run()
    folder = tmp_path/"cffg"/f"seed-{SEED}"
    # Old runners could leave several partials alongside a valid sidecar.
    old = folder/"stationary-frozen.interrupted-12345.0.jsonl"
    old.write_bytes(b"historical partial")
    bad = folder/"stationary-frozen.interrupted-wrong.jsonl"
    bad.touch()
    with pytest.raises(ValueError, match="Unexpected episode file"):
        campaign.run()
    assert old.is_file()
    bad.unlink()
    campaign.run()
    partials = list((tmp_path/"attempts").glob("*/interrupted-*/*"))
    assert {p.name: p.read_bytes() for p in partials} == {
        old.name: b"historical partial", "stationary-candidate.jsonl": b'{"tick":0}\n'}
    assert not old.exists()
    _, workers = scorer.metadata(tmp_path, {})
    assert workers["cffg"][0]["process_wall_s"] is None


@pytest.mark.parametrize("bad", ["missing trace", "wrong sidecar", "missing pair"])
def test_final_worker_rejects_damaged_episode(tmp_path, campaign, bad):
    campaign.fail = "after worker"
    with pytest.raises(InterruptedError):
        campaign.run()
    stem = tmp_path/"cffg"/f"seed-{SEED}"/"stationary-frozen"
    if bad == "wrong sidecar":
        row = runner.read(stem.with_suffix(".json"))
        runner.save(stem.with_suffix(".json"), dict(row, seed=0))
    else:
        stem.with_suffix(".jsonl").unlink()
        if bad == "missing pair":
            stem.with_suffix(".json").unlink()
    campaign.fail = None
    with pytest.raises((ValueError, FileNotFoundError)):
        campaign.run()
    assert campaign.launches.count(SEED) == 1
    assert not (tmp_path/"completed.json").exists()


def test_interruption_after_all_episodes_before_worker_record(tmp_path, campaign, monkeypatch):
    save = runner.save

    def interrupt(path, row):
        if path.name == f"cffg-{SEED}.json":
            raise InterruptedError("worker record")
        return save(path, row)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "save", interrupt)
        with pytest.raises(InterruptedError):
            campaign.run()
    campaign.run()
    _, workers = scorer.metadata(tmp_path, {})
    assert workers["cffg"][0]["episodes_run"] == 0
    assert workers["cffg"][0]["process_wall_s"] is None
    assert len(campaign.calls) == 16


def test_reject_missing_seal_and_changed_qualification(tmp_path, campaign):
    campaign.fail = "before setup"
    with pytest.raises(InterruptedError):
        campaign.run()
    opened = tmp_path/"opened.json"
    seal = opened.read_bytes()
    opened.unlink()
    with pytest.raises(ValueError, match="seal"):
        campaign.run()
    assert not opened.exists()
    opened.write_bytes(seal)
    (tmp_path/"qualification/cffg.json").write_bytes(b'{}')
    with pytest.raises((KeyError, ValueError)):
        campaign.run()
    assert opened.read_bytes() == seal


def test_child_retains_lock_after_parent_handle_closes(tmp_path, monkeypatch):
    script = tmp_path/"worker.py"
    script.write_text("import os, sys, fcntl\nfrom pathlib import Path\n"
                      "root = Path(sys.argv[sys.argv.index('--output')+1])\n"
                      "fd = int(sys.argv[sys.argv.index('--lock-fd')+1])\n"
                      "assert os.path.samestat(os.fstat(fd), (root/'study.lock').stat())\n"
                      "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\nsys.stdin.readline()\n")
    monkeypatch.setattr(runner, "__file__", str(script))
    (tmp_path/"logs").mkdir()
    lock = tmp_path/"study.lock"
    with lock.open("a") as parent:
        fcntl.flock(parent, fcntl.LOCK_EX)

        def lose_parent(cmd, *, check, **options):
            with subprocess.Popen(cmd, stdin=subprocess.PIPE, **options) as child:
                parent.close()
                try:
                    with lock.open("a") as retry, pytest.raises(BlockingIOError):
                        fcntl.flock(retry, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    child.communicate(input=b"exit\n", timeout=5)
                assert child.returncode == 0

        monkeypatch.setattr(runner.subprocess, "run", lose_parent)
        runner.run_child(tmp_path, "cffg", tmp_path, parent.fileno(), SEED)
