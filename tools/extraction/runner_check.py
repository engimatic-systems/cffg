"""Synthetic fault and restart checks for the extraction campaign runner."""

import fcntl
from contextlib import nullcontext
from itertools import count
from pathlib import Path
import select
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import analysis
import campaign
import decision
import episode
import qual_bundle
from study import ENDURANCE, LIBRARIES, read, save, sha


class EpisodeFailureTest(unittest.TestCase):
    def test_reset_and_post_action_failures_are_scored_at_the_causal_decision(self):
        n = 5
        nominal = np.linspace(.3, .5, 10)
        reset_mean = np.r_[nominal, np.log(5.)]
        offset = np.full(11, .001)
        proposal = np.array([80., 35.])
        tape = dict(initial=nominal, kla=np.full(n, 5.), phase=np.zeros(n, int),
                    event=np.full(n, "nominal"), target=np.full(n, .31),
                    process=np.zeros((n, 10)), noise=np.zeros((n, 2)), y0=nominal[[8, 1]])
        simulator = Mock()
        simulator.step.side_effect = lambda x, u, kla: x
        backend = Mock()
        calls = count()

        def update(m, P, A, b, Q, H, R, y):
            if next(calls) in (0, 2):  # Reset, followed by the reading after command 1.
                raise BrokenPipeError("synthetic transport failure")
            return A @ m + b + .002, A @ P @ A.T + Q

        backend.update.side_effect = update
        backend.plan.side_effect = [proposal, proposal, proposal + 1., RuntimeError("synthetic plan failure"), proposal + 2.]
        with (patch.object(episode.plant, "nominal_profile", return_value=nominal),
              patch.object(episode, "transition", return_value=(np.eye(11), np.zeros((11, 2)), offset)),
              patch.object(episode, "ps_memory", return_value=1.),
              patch.object(episode, "perf_counter", side_effect=count(0., .001).__next__)):
            result = episode.simulate(backend, simulator, tape, "adaptive", "control")

        self.assertTrue(result["reset_failed"].item())
        np.testing.assert_array_equal(result["filter_failed"], [False, True, False, False, False])
        np.testing.assert_array_equal(result["planner_failed"], [False, False, False, True, False])
        np.testing.assert_array_equal(result["failed"], [True, True, False, True, False])
        np.testing.assert_array_equal(result["held"], [True, False, True, True, False])
        np.testing.assert_array_equal(result["u"], [episode.NOMINAL_U, proposal, proposal, proposal, proposal + 2.])
        np.testing.assert_array_equal(backend.plan.call_args_list[0].args[0], reset_mean)
        np.testing.assert_array_equal(backend.plan.call_args_list[0].args[1], episode.PRIOR)
        np.testing.assert_allclose(result["m"][1], result["m"][0] + offset, atol=1e-14, rtol=0)
        q = np.diag([.001**2] * 10 + [episode.MOTION_SD**2])
        np.testing.assert_allclose(result["P"][1], result["P"][0] + q, atol=1e-14, rtol=0)
        with TemporaryDirectory() as td:
            path = Path(td) / "faults.npz"
            np.savez(path, seed=np.asarray(1), **result)
            trace = analysis.load_trace(path, 1, n, "stationary", "control")
            metrics = analysis.metrics(trace, analysis.score_arrays(trace), slice(None))
            self.assertEqual((metrics["n"], metrics["holds"]), (n, 3))
            self.assertTrue(np.isfinite(metrics["hidden_rmse"]))


class CampaignRecoveryTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        save(self.root / "protocol.json", {"test": True})
        self.manifest = {"source": {"synthetic": "unchanged"}}
        self.signature = campaign.identity(self.manifest)
        self.enterContext(patch.object(campaign, "runtime", return_value=self.manifest))

    def test_recover_receipts_only_from_intact_completed_episodes(self):
        paths = []
        for lib in LIBRARIES:
            path = self.root / "endurance" / f"seed-{ENDURANCE}" / "changing" / f"control-adaptive-{lib}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, seed=ENDURANCE, sample=np.asarray([.3]))
            save(path.with_suffix(".json"), dict(stage="endurance", seed=ENDURANCE, library=lib,
                 arm="adaptive", mode="control", condition="changing", runtime_sha256=self.signature, sha256=sha(path)))
            paths.append(path)
        expected = {str(p.relative_to(self.root)): sha(p) for p in paths}
        with (self.root / "campaign.lock").open("a") as lock, patch.object(campaign.subprocess, "run") as launch:
            campaign.run(self.root, "endurance", lock.fileno())
            for lib in LIBRARIES:
                receipt = read(self.root / "workers" / "endurance" / f"{ENDURANCE}-{lib}.json")
                self.assertTrue(receipt["recovered_from_sealed_episodes"])
                self.assertEqual(receipt["runtime_sha256"], self.signature)
            self.assertEqual(read(self.root / "endurance-completed.json")["runs"], expected)
            campaign.run(self.root, "endurance", lock.fileno())
            with paths[0].open("ab") as changed:
                changed.write(b"modified")
            with self.assertRaisesRegex(ValueError, "Broken episode seal"):
                campaign.run(self.root, "endurance", lock.fileno())
            launch.assert_not_called()

    def test_worker_keeps_exclusive_lock_after_parent_closes_descriptor(self):
        lock_path = self.root / "campaign.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(campaign.subprocess, "run", side_effect=InterruptedError) as launch:
                with self.assertRaises(InterruptedError):
                    campaign.run(self.root, "endurance", lock.fileno())
            args, kwargs = launch.call_args
            # Exercise the runner's real worker entry point with the standard lock handoff.
            shim = """import campaign, sys
campaign.configure = lambda: None
def worker(*args):
    print('locked', flush=True)
    sys.stdin.readline()
campaign.worker = worker
campaign.main()
"""
            proc = subprocess.Popen([sys.executable, "-B", "-c", shim, *args[0][2:]],
                                    cwd=Path(campaign.__file__).parent, pass_fds=kwargs["pass_fds"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([proc.stdout], [], [], 5)[0], "Worker did not reach its entry point")
            self.assertEqual(proc.stdout.readline().strip(), "locked")
            with lock_path.open("a") as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            try:
                proc.communicate("\n", timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
        self.assertEqual(proc.returncode, 0)
        with lock_path.open("a") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


class PilotGateTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.folder = self.root / "qual"
        manifest = {"stage": "synthetic"}
        self.signature = campaign.identity(manifest)
        for module in (qual_bundle, decision):
            self.enterContext(patch.object(module, "runtime", return_value=manifest))
        for stage in ("development", "endurance"):
            save(self.root / (stage + "-completed.json"), dict(runtime_sha256=self.signature, runs={}))

        def synthetic_qualification(command, **kwargs):
            script = Path(command[1]).stem
            if script == "qualify":
                path = self.folder / (command[2] + ".json")
            else:
                path = Path(command[-1])
            save(path, dict(passed=True, runtime_sha256=self.signature))

        with patch.object(qual_bundle.subprocess, "run", side_effect=synthetic_qualification):
            qual_bundle.qualify(self.root)

    def test_stale_native_result_is_rejected_even_with_updated_attestation(self):
        name = "cffg.json"
        save(self.folder / name, dict(passed=True, runtime_sha256="earlier runtime"))
        att = read(self.folder / "attestations.json")
        att["results"][name] = sha(self.folder / name)
        save(self.folder / "attestations.json", att)
        with self.assertRaisesRegex(ValueError, "different measurement runtime"):
            decision.decide(self.root)

    def test_incomplete_and_changed_qualification_evidence_is_rejected(self):
        path = self.folder / "attestations.json"
        for section, key in (("checks", "tools/extraction/dynamics_check.py"), ("results", "nonlinear.json")):
            for omission in (True, False):
                with self.subTest(section=section, omission=omission):
                    att = read(path)
                    original = att[section][key]
                    if omission:
                        del att[section][key]
                    else:
                        att[section][key] = "wrong digest"
                    save(path, att)
                    with self.assertRaisesRegex(ValueError, "(?i)qualification"):
                        decision.decide(self.root)
                    att[section][key] = original
                    save(path, att)

    def test_isolated_update_failures_count_once_per_affected_command(self):
        def arrays(path):
            endurance = "endurance" in path.parts
            n = 6000 if endurance else 600
            m = np.full((n, 11), .01 if "informed" in path.name else .03)
            fails = np.zeros(n, bool)
            if not endurance and "control-adaptive" in path.name:
                # Four or seven isolated failures, followed by their corresponding held commands.
                count_failed = 4 if path.stem.endswith("cffg") else 7
                fails[1:2*count_failed:2] = True
            held = np.r_[False, fails[:-1]]
            return nullcontext(dict(held=held, filter_failed=fails, failed=fails, reset_failed=False,
                                    m=m, x=np.zeros((n+1, 10)), phase=np.arange(n) > 0,
                                    obs=np.zeros((n, 2)), u=np.zeros((n, 2)), P=np.broadcast_to(np.eye(11), (n, 11, 11))))

        def metadata(path):
            if path.suffix == ".json" and "development" in path.parts:
                return {"episode_s": 1.}
            return read(path)

        for lib in LIBRARIES:
            save(self.root / "workers" / "endurance" / f"{ENDURANCE}-{lib}.json", {"completed": True})
        with patch.object(np, "load", side_effect=arrays), patch.object(decision, "read", side_effect=metadata):
            pilot = decision.decide(self.root)
        self.assertTrue(pilot["gates"]["reliability_cffg"]["passed"])
        self.assertFalse(pilot["gates"]["reliability_rxinfer"]["passed"])
        self.assertEqual(pilot["development"]["cffg"]["worst_hold_fraction"], 4/600)
        self.assertEqual(pilot["development"]["cffg"]["worst_update_failure_fraction"], 4/601)


if __name__ == "__main__":
    unittest.main()
