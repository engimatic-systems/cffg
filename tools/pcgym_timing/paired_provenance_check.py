"""Isolated relocation and tamper checks for the historical paired-study verifier."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import paired_provenance as verifier
import run_paired
from pilot import ARMS, save


def make_files(root, names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{root.name}/{name}")


class PairedProvenanceTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.repo, self.peers, self.experiment = (base / r for r in ("repo", "peers", "experiment"))
        self.archive = base / "archive"
        self.archive.mkdir()
        self.harness = self.repo / "tools/pcgym_timing"
        self.worker_files = [f"{arm}_adapter.py" for arm in ARMS] + [
            "pilot.py", "forecast.py", "qualification.py", "run_paired.py", "rxinfer_worker.jl", "gtsam_cstr.cpp"]
        make_files(self.harness, self.worker_files)
        runtime_repo = ["uv.lock", "tools/cstr_reactor_pilot.py", "tools/cstr_runtime_fixture.py",
                        "tools/benchmark_cstr_chain.py"]
        make_files(self.repo, runtime_repo)
        make_files(self.experiment, verifier.EXPERIMENT_INPUTS)
        binary = "gtsam/site/_gtsam_cstr.test.so"
        make_files(self.peers, [n for names in verifier.PEER_DEPS.values() for n in names] + [binary])

        # Use the unchanged recorder to exercise the original absolute-path format.
        with patch.multiple(run_paired, HERE=self.harness, ROOT=self.repo), \
                patch.object(run_paired.subprocess, "check_output", return_value="library-tree\n"):
            frozen = run_paired.signature(self.peers)
        self.plan = dict(frozen=frozen, seeds=list(run_paired.SEEDS), arms=list(ARMS), phases=list(run_paired.PHASES))
        save(self.archive / "protocol.json", self.plan)
        save(self.archive / "finished.json", {"completed": 360})
        hashes = frozen["sha256"]
        worker_hashes = {name: hashes[str(self.harness / name)] for name in self.worker_files}
        runtime = dict(modules={}, pcgym_source={}, environment={}, hardware="", devices=[], python_version="3.12",
                       reference_revision="reference", nested_checkout="simulator", head="measured-head",
                       library_tree="library-tree", reference_path=str(self.repo / runtime_repo[-1]),
                       experiment=str(self.experiment))
        inputs = [self.repo / name for name in runtime_repo] + [self.experiment / n for n in verifier.EXPERIMENT_INPUTS]
        runtime["file_sha256"] = {str(p): sha256(p.read_bytes()).hexdigest() for p in inputs}
        for seed in (None, *run_paired.SEEDS):
            for arm in ARMS:
                phases = ["oracle"] if seed is None else list(run_paired.PHASES)
                if seed is not None and arm == "cffg":
                    phases.append("truth")
                deps = {name: hashes[str(self.peers / name)] for name in verifier.PEER_DEPS[arm]}
                for phase in phases:
                    directory = self.archive / ("qualification" if seed is None else f"seed-{seed}") / f"{arm}-{phase}"
                    directory.mkdir(parents=True)
                    manifest = dict(runtime=runtime, source_sha256=worker_hashes, arm=arm, phase=phase,
                                    seed=seed, dependency_sha256=deps)
                    if arm == "gtsam":
                        manifest["native_binary_sha256"] = {Path(binary).name: hashes[str(self.peers / binary)]}
                    save(directory / "manifest.json", manifest)
                    if phase == "oracle":
                        save(directory / "result.json", dict(complete=True, cases=[{"passed": True}]*3))

    def relocate(self):
        roots = {}
        for arg, path in (("source_root", self.repo), ("peer_root", self.peers), ("experiment_root", self.experiment)):
            roots[arg] = path.rename(path.with_name("relocated-" + path.name))
        return roots

    def test_all_workers_checked_before_and_after_relocation(self):
        path = self.archive / "seed-1001/cffg-short/manifest.json"
        original = path.read_text()
        for relocated in (False, True):
            roots = self.relocate() if relocated else {}
            report = verifier.audit(self.archive, **roots)
            self.assertEqual(report["worker_manifests"], 394)
            self.assertEqual(report["worker_source_checks"], 3940)
            self.assertEqual(report["worker_source_files"], 10)
            self.assertEqual(report["frozen_files"], 20)
            self.assertEqual(report["runtime_files"], 12)
            self.assertEqual(set(report["worker_sources"]), {f"tools/pcgym_timing/{f}" for f in self.worker_files})
            for missing in (False, True):
                with self.subTest(relocated=relocated, missing=missing):
                    manifest = json.loads(original)
                    if missing:
                        del manifest["source_sha256"]["cffg_adapter.py"]
                    else:
                        manifest["source_sha256"]["cffg_adapter.py"] = "0"*64
                    save(path, manifest)
                    with self.assertRaisesRegex(ValueError, "cffg_adapter.py.*cffg-short"):
                        verifier.audit(self.archive, **roots)
                    path.write_text(original)
        with patch.object(verifier, "ROOT", roots.pop("source_root")):
            self.assertEqual(verifier.audit(self.archive, **roots)["worker_source_checks"], 3940)

    def test_seed_schedule_is_complete_unique_and_ordered(self):
        seeds = self.plan["seeds"]
        for altered in ([seeds[0], *seeds[:-1]], seeds[:-1], [seeds[1], seeds[0], *seeds[2:]]):
            with self.subTest(first=altered[:2], last=altered[-1]):
                save(self.archive / "protocol.json", {**self.plan, "seeds": altered})
                with self.assertRaisesRegex(ValueError, "Seed schedule"):
                    verifier.audit(self.archive)

    def test_manifest_paths_are_unique(self):
        duplicate = self.archive / "seed-1030/gtsam-long/manifest.json"
        duplicate.unlink()
        duplicate.symlink_to(self.archive / "seed-1001/gtsam-long/manifest.json")
        with self.assertRaisesRegex(ValueError, "Repeated path"):
            verifier.audit(self.archive)

    def test_incomplete_or_ambiguous_freeze_fails(self):
        path = str(self.harness / "cffg_adapter.py")
        for issue in ("missing", "another repository", "same normalized name", "missing peer lock"):
            with self.subTest(issue=issue):
                hashes = self.plan["frozen"]["sha256"].copy()
                if issue == "missing":
                    del hashes[path]
                elif issue == "another repository":
                    hashes[str(self.repo.parent / "other/tools/pcgym_timing/cffg_adapter.py")] = hashes[path]
                elif issue == "same normalized name":
                    hashes[path.replace("/cffg_adapter.py", "/./cffg_adapter.py")] = hashes[path]
                else:
                    del hashes[str(self.peers / "numpyro-arm.lock")]
                plan = {**self.plan, "frozen": {**self.plan["frozen"], "sha256": hashes}}
                save(self.archive / "protocol.json", plan)
                with self.assertRaisesRegex(ValueError, "Frozen inventory mismatch|Ambiguous frozen file"):
                    verifier.audit(self.archive)

    def test_duplicate_worker_key_fails(self):
        path = self.archive / "seed-1001/cffg-short/manifest.json"
        key = '"cffg_adapter.py":'
        raw = path.read_text()
        self.assertEqual(raw.count(key), 1)
        path.write_text(raw.replace(key, key + '"altered", ' + key))
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key.*cffg-short"):
            verifier.audit(self.archive)

    def test_missing_and_changed_historical_bytes_fail(self):
        roots = self.relocate()
        file = roots["source_root"] / "tools/pcgym_timing/forecast.py"
        file.write_text("newer version")
        with self.assertRaisesRegex(ValueError, "Digest mismatch.*forecast.py.*measured-head"):
            verifier.audit(self.archive, **roots)
        file.unlink()
        with self.assertRaisesRegex(ValueError, "Missing historical checkout file.*forecast.py.*measured-head"):
            verifier.audit(self.archive, **roots)


if __name__ == "__main__":
    unittest.main()
