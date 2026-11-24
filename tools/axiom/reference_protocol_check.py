"""Focused synthetic checks for the prospective design and raw-reward estimands."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import reference_protocol as ref


class ReferenceDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ref.HERE / "reference_protocol.json").read_text())

    def test_inventory(self):
        plan = ref.frozen_inventory(self.protocol)
        splits = plan["stage_pairs"]
        self.assertEqual(
            [len(splits[k]) for k in ("development", "pre_evaluation", "evaluation")], [1, 20, 100]
        )
        pairs = [row for rows in splits.values() for row in rows]
        self.assertEqual(len({row["game_seed"] for row in pairs}), len(pairs))
        self.assertEqual(len({row["model_seed"] for row in pairs}), 13)
        self.assertEqual(len(plan["privileged_color_remap"]), 10)
        self.assertEqual(len(plan["optional_rollout_sweep"]["pairs"]), 30)
        self.assertEqual({len(rows) for rows in plan["main_arms"].values()}, {100})
        self.assertEqual(len(plan["main_arms"]), 6)
        self.assertEqual(sum(row[0] for row in self.protocol["table1"]["untuned"]) / 10, 71.2)
        self.assertTrue(all(f"Fig{i}" in self.protocol["inventory"] for i in range(1, 13)))
        self.assertTrue(all(f"Table{i}" in self.protocol["inventory"] for i in range(1, 4)))

    def test_t9_critical_value(self):
        # For t9, the substitution t=3*tan(theta) gives an integral proportional to cos(theta)^8.
        def integral(theta):
            value = theta
            for degree in (2, 4, 6, 8):
                value = (
                    math.sin(theta) * math.cos(theta) ** (degree - 1) + (degree - 1) * value
                ) / degree
            return value

        for critical, p in ((ref.SIMULTANEOUS_T9, 1 - 0.05 / 40), (ref.PAIRED_T9, 0.975)):
            cdf = 0.5 + 0.5 * integral(math.atan(critical / 3)) / integral(math.pi / 2)
            self.assertAlmostEqual(p, cdf, places=10)

    def test_raw_primary(self):
        # Arbitrary synthetic totals; share one zero buffer per game to keep this check small.
        historical = dict(zip(self.protocol["games"], self.protocol["table1"]["untuned"], strict=True))
        rewards = {game: [0] * 9999 + [old[0] + 3] for game, old in historical.items()}
        rows = [
            {**r, "arm": "untuned", "condition": "base", "status": "ok", "rewards": rewards[r["game"]]}
            for r in ref.seed_pairs(self.protocol, "evaluation")
        ]
        result = ref.score_primary(self.protocol, reversed(rows))
        self.assertEqual(result["scored"], 100)
        self.assertEqual(result["macro"]["delta"], 3)
        self.assertEqual(result["games"]["Drive"]["verdict"], "within_margin")
        self.assertEqual(result["games"]["Impact"]["verdict"], "inconclusive")
        self.assertFalse(result["whole_suite_agreement"])
        self.assertEqual(
            result["macro"]["printed_target"], {"delta": 3, "ci": [3, 3], "verdict": "within_margin"}
        )
        self.assertEqual(result["games"]["Impact"]["printed_target"]["ci"], [3, 3])
        self.assertEqual(ref.verdict(-10, 10, 10), "within_margin")
        self.assertEqual(ref.verdict(10, 20, 10), "inconclusive")
        self.assertEqual(ref.verdict(10.1, 20, 10), "above_margin")
        self.assertEqual(ref.verdict(-20, -10.1, 10), "below_margin")
        for edits in ({"arm": "no_ig"}, {"condition": "shape"}, {"game_seed": -1}):
            with self.assertRaises(ValueError):
                ref.score_primary(self.protocol, [dict(rows[0], **edits)])
        with self.assertRaises(ValueError):
            ref.score_primary(self.protocol, [rows[0], rows[0]])
        failed = ref.score_primary(
            self.protocol, [dict(rows[0], status="nonfinite", rewards=[1]), dict(rows[1], rewards=[])]
        )
        self.assertEqual((len(failed["failed"]), len(failed["missing"]), failed["scored"]), (2, 98, 0))
        self.assertIsNone(failed["macro"])
        self.assertIsNone(failed["whole_suite_agreement"])
        self.assertEqual(
            ref.score_primary(self.protocol, [dict(rows[0], rewards=[float("nan")] * 10000)])["scored"],
            0,
        )

    def test_matched_effects(self):
        self.assertEqual(ref.paired_contrast([-2] * 10)["ci95"], [-2, -2])
        with self.assertRaises(ValueError):
            ref.paired_contrast([4] * 9)
        base = [0] * 10000
        altered = [2] * 5000 + [-1] * 250 + [0] * 4750
        scores = ref.perturbation_scores(base, altered)
        self.assertEqual(scores["pre"]["difference"]["sum"], 10000)
        self.assertEqual(scores["post"]["difference"]["sum"], -250)
        self.assertEqual(scores["recovery_delay"], 250)
        self.assertFalse(scores["censored"])
        alternating = [(-1 if (i // 250) % 2 else 0) for i in range(5000)]
        self.assertTrue(ref.perturbation_scores(base, base[:5000] + alternating)["censored"])

    def test_freeze_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "intermediates"
            folder.mkdir()
            (root / "simulated.json").write_text("{}")
            evidence = {
                "simulated.json": ref.sha(root / "simulated.json"),
                "tapes/manifest.json": ref.sha(root / "simulated.json"),
            }
            (root / "tapes").mkdir()
            (root / "tapes/manifest.json").write_text("{}")
            names = ("frames.rgb", "object_stream.json", "scorer_labels.json", "explode_visible.json")
            for name in names:
                (folder / name).write_bytes(name.encode())
            checks = {
                "cases_sha256": ref.sha(ref.HERE / "reference_cases.json"),
                "source_sha256": ref.sha(ref.HERE / "reference_intermediates.py"),
                "development_manifest_sha256": evidence["tapes/manifest.json"],
                "files_sha256": {name: ref.sha(folder / name) for name in names},
            }
            check_file = folder / "checks.json"
            check_file.write_text(json.dumps(checks))
            with (
                mock.patch.object(ref, "EVIDENCE", evidence),
                mock.patch.object(ref, "INTERMEDIATE_CHECKS_SHA256", ref.sha(check_file)),
            ):
                actual = ref.freeze(root, folder)
                self.assertEqual(actual["stage"], "reference_only")
                (folder / names[0]).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    ref.freeze(root, folder)
                checks["files_sha256"][names[0]] = ref.sha(folder / names[0])
                check_file.write_text(json.dumps(checks))
                with self.assertRaises(ValueError):
                    ref.freeze(root, folder)
                (root / "simulated.json").write_text("[]")
                with self.assertRaises(ValueError):
                    ref.freeze(root, folder)


if __name__ == "__main__":
    unittest.main()
