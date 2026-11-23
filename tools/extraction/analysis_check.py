"""Exercise the extraction scorer using only synthetic paths and analytical Gaussians."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import analysis as a
from study import save


def trace(seed, n, condition, mode, arm, lib):
    x = .1 + np.arange(n + 1)[:, None] * .001 + np.arange(10)[None, :] * .01
    x += .00001 * (seed % 10)
    phase = np.zeros(n, dtype=int)
    kla = np.full(n, 5.)
    event = np.full(n, "nominal", dtype="<U12")
    if condition == "changing":
        phase[4:15] = 1
        kla[4:15] = 4.5
        event[4:15] = "step1"
        event[15:] = "return1"
    m = np.column_stack((x[1:], np.log(kla if arm == "informed" else np.full(n, 5.))))
    m[:, a.HIDDEN] += {"adaptive": .01, "frozen": .03, "informed": .005}[arm] * (1 if lib == "cffg" else 2)
    P = np.tile(np.eye(11) * .01, (n, 1, 1))
    i, j = a.HIDDEN[:2]
    P[:, i, j] = P[:, j, i] = .005
    obs = x[1:, a.OUTLETS] + [.001, -.002]
    flags = np.zeros(n, dtype=bool)
    return dict(x=x, obs=obs, phase=phase, event=event, kla=kla, m=m, P=P, pred=obs.copy(),
                pred_cov=np.tile(np.eye(2) * .001, (n, 1, 1)), u=np.tile(a.NOMINAL_FLOW, (n, 1)),
                target=x[1:, 8].copy(), failed=flags.copy(), filter_failed=flags.copy(), planner_failed=flags.copy(),
                late=flags.copy(), held=flags.copy(), latency=np.full(n, .005), seed=np.asarray(seed))


def archive(root, seeds, split="development", n=24):
    save(root/"protocol.json", dict(short_steps=n, long_steps=n))
    files = {}
    for spec in a.run_paths(seeds, a.CONDITIONS):
        seed, cond, mode, arm, lib = spec
        relative = f"{split}/seed-{seed}/{cond}/{mode}-{arm}-{lib}.npz"
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(stage=split, seed=seed, steps=n, condition=cond, mode=mode, arm=arm, library=lib,
                    protocol_sha256=a.digest(root/"protocol.json"), runtime_sha256="synthetic")
        np.savez_compressed(path, **trace(seed, n, cond, mode, arm, lib), meta=np.asarray(json.dumps(meta)))
        files[relative] = a.digest(path)
        save(path.with_suffix(".json"), dict(**meta, sha256=files[relative]))
    if split == "heldout":
        save(root/"pilot.json", dict(status="go"))
        save(root/"freeze.json", dict(seeds=seeds, steps=n, protocol_sha256=a.digest(root/"protocol.json"),
                                      pilot_sha256=a.digest(root/"pilot.json"), runtime_sha256="synthetic",
                                      source_sha256={"tools/extraction/analysis.py": a.digest(Path(a.__file__))}))
        save(root/"completed.json", dict(seeds=seeds, steps=n, runs=files, freeze_sha256=a.digest(root/"freeze.json"),
                                         protocol_sha256=a.digest(root/"protocol.json"), runtime_sha256="synthetic"))
    return files


class ScorerTest(unittest.TestCase):
    def test_joint_and_outlet_densities(self):
        r = trace(17101, 24, "changing", "tape", "adaptive", "cffg")
        s = a.score_arrays(r)
        i = a.HIDDEN
        covariance = r["P"][0][np.ix_(i, i)]
        err = r["m"][0, i] - r["x"][1, i]
        expected = -.5 * (8*np.log(2*np.pi) + np.linalg.slogdet(covariance)[1] + err@np.linalg.solve(covariance, err))
        self.assertAlmostEqual(s["joint"][0], expected)
        diagonal = -.5 * np.sum(np.log(2*np.pi*np.diag(covariance)) + err**2/np.diag(covariance))
        self.assertAlmostEqual(s["diag"][0], diagonal)
        self.assertAlmostEqual(s["outlet"][0], -np.log(2*np.pi*.001))
        r["P"][1, i[0], i[0]] = -1
        r["pred"][2] = np.nan
        r["filter_failed"][2] = r["failed"][2] = True
        scored = a.metrics(r, a.score_arrays(r), slice(None))
        self.assertIsNone(scored["joint_log_score"])
        self.assertIsNone(scored["outlet_log_score"])
        self.assertEqual(scored["joint_log_score_undefined"], 1)
        self.assertEqual(scored["outlet_log_score_undefined"], 1)
        self.assertAlmostEqual(scored["hidden_rmse"], .01)
        self.assertAlmostEqual(scored["tracking_rmse"], 0)
        self.assertAlmostEqual(scored["joint_coverage90"], 23/24)
        self.assertAlmostEqual(scored["hidden_coverage90"], 191/192)
        self.assertEqual(scored["n"], 24)

    def test_ramp_recovery_and_censoring(self):
        r = trace(17101, 24, "changing", "tape", "informed", "cffg")
        r["kla"][4:15] = [4.8, 4.6, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.6, 4.7, 4.9]
        r["m"][:, 10] = np.log(r["kla"])
        rec = a.recovery(r, a.score_arrays(r))
        self.assertEqual((rec[0]["arrival"], rec[0]["delay_ticks"], rec[0]["confirmation_tick"]), (6, 0, 10))
        r["filter_failed"][8] = True
        self.assertTrue(a.recovery(r, a.score_arrays(r))[0]["censored"])

    def test_interval_resamples_whole_seed_differences(self):
        # Two equally weighted seeds give endpoint quantiles at the two extremes.
        result = a.interval([{"seed": 1, "delta": -3.}, {"seed": 2, "delta": 1.}])
        self.assertEqual(result["ci95"], [-3., 1.])
        self.assertEqual(result["mean_difference"], -1.)
        self.assertIsNone(a.interval([{"seed": 1, "delta": -3.}])["ci95"])

    def test_sealed_pairing_and_seed_interval(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            files = archive(root, [17301, 17302], "heldout")
            result = a.analyze(root, "heldout")
            self.assertEqual(result["inventory"]["scored_runs"], 48)
            self.assertEqual(result["primary"]["n_seeds"], 2)
            np.testing.assert_allclose(result["primary"]["ci95"], [-.01, -.01], atol=1e-12)
            self.assertEqual(result["status"], "complete")
            adaptive = next(r for r in result["within_library"] if
                            (r["library"], r["mode"], r["condition"], r["contrast"]) ==
                            ("cffg", "tape", "changing", "adaptive_minus_frozen"))
            self.assertAlmostEqual(adaptive["mean_difference"], -.02)
            path = root/next(iter(files))
            with path.open("ab") as f:
                f.write(b"altered")
            with self.assertRaisesRegex(ValueError, "digest"):
                a.analyze(root, "heldout")

    def test_partial_stop_never_opens_heldout(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            files = archive(root, [17101])
            missing = root/"development/seed-17101/changing/tape-adaptive-cffg.npz"
            missing.rename(missing.with_suffix(".partial"))
            def guard(func):
                def wrapped(path, *args, **kwargs):
                    if "heldout" in path.parts:
                        raise AssertionError("Inspected held-out path")
                    return func(path, *args, **kwargs)
                return wrapped
            with patch.object(Path, "is_file", guard(Path.is_file)), patch.object(Path, "open", guard(Path.open)):
                dev = a.analyze(root, "development", seeds=[17101])
                self.assertEqual(dev["status"], "partial")
                self.assertIsNone(dev["primary"]["mean_difference"])
                self.assertEqual(dev["inventory"]["scored_runs"], len(files)-1)
                save(root/"pilot.json", {"status": "no_go", "reason": "Synthetic feasibility stop"})
                stopped = a.analyze(root, "stopped", seeds=[17101])
                self.assertEqual(stopped["status"], "stopped")
                self.assertEqual(stopped["decision"]["status"], "no_go")

    def test_noise_and_tape_mismatch(self):
        r = trace(17101, 24, "changing", "tape", "adaptive", "cffg")
        refs = {}
        a.check_pairing(r, refs, 17101, "changing", "tape")
        r["u"][1, 0] += 1
        with self.assertRaisesRegex(ValueError, "probing"):
            a.check_pairing(r, refs, 17101, "changing", "tape")
        r["obs"][1, 0] += .01
        with self.assertRaisesRegex(ValueError, "Sensor noise"):
            a.check_pairing(r, refs, 17101, "changing", "control")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Write a synthetic development report fixture to a new directory")
    args = parser.parse_args()
    if args.fixture is not None:
        args.fixture.mkdir()
        archive(args.fixture, [17101, 17102])
        save(args.fixture/"development/analysis.json", a.analyze(args.fixture, "development", seeds=[17101, 17102]))
    else:
        unittest.main(argv=[__file__])
