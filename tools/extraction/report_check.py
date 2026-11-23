"""Exercise report states and practical-effect boundaries on synthetic evidence."""
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from report import learning_errors, observed, primary, render, speed_summary


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value))
    return sha256(path.read_bytes()).hexdigest()


def score(ci=(-.004, -.003)):
    return dict(comparison="cffg_minus_rxinfer", mean_difference=sum(ci)/2, ci95=list(ci),
                margin=.001, per_seed=[dict(seed=i, cffg=.01+x, rxinfer=.01, delta=x)
                                       for i, x in enumerate(ci)])


class Links(HTMLParser):
    def __init__(self, root):
        super().__init__()
        self.root = root

    def handle_starttag(self, tag, attrs):
        if tag in {"a", "img"}:
            target = dict(attrs)["href" if tag == "a" else "src"]
            if tag == "img":
                assert dict(attrs).get("alt"), "Figure is missing its text alternative"
            if not target.startswith("https://"):
                assert (self.root/target).exists(), target


class ReportChecks(unittest.TestCase):
    def test_practical_decision_boundaries(self):
        for ci, text in [((-.002, -.0011), "CFFG had meaningfully lower"),
                         ((.0011, .002), "RxInfer had meaningfully lower"),
                         ((-.001, .001), "practically tied"), ((.0005, .002), "inconclusive"),
                         ((-.003, .002), "inconclusive")]:
            with self.subTest(ci=ci):
                self.assertIn(text, primary({"primary": score(ci)}, .001))
        self.assertIn("RxInfer", observed(.84, .93, "coverage"))
        self.assertIn("CFFG", observed(.7, .2, "higher"))
        self.assertIn("precision", observed(1., 1.+1e-12, "lower"))

    def test_unclosed_report_skips_even_present_fresh_outcomes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write(root/"protocol.json", {"horizon": 6})
            write(root/"pilot.json", {"status":"go", "gates": {}})
            (root/"analysis.json").write_text("unopened held-out data")
            page = render(root).read_text()
            self.assertIn("pending", page)
            self.assertNotIn('href="analysis.json"', page)
            self.assertNotIn('href="heldout/', page)
            self.assertFalse(list(root.glob("*.png")))
            Links(root).feed(page)

    def test_stopped_report_preserves_reason_and_missing_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write(root/"protocol.json", {})
            write(root/"pilot.json", {"status":"no_go", "reason":"Fewer than 5% improved <test>",
                                      "gates":{"information":{"passed":False, "detail":"Insufficient headroom"}}})
            write(root/"analysis.json", {"stage":"stopped", "inventory":{"scored_ticks":0, "expected_ticks":2400}})
            page = render(root).read_text()
            self.assertIn("stopped before fresh-seed", page)
            self.assertIn("&lt;test&gt;", page)
            self.assertIn("Insufficient headroom", page)
            self.assertIn("2400", page)
            self.assertNotIn("CFFG had meaningfully", page)
            self.assertFalse(list(root.glob("*.png")))
            Links(root).feed(page)

    def test_complete_requires_matching_outcome_inventory(self):
        from analysis import analyze
        from analysis_check import archive
        from PIL import Image
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = [17301, 17302]
            archive(root, seeds, split="heldout")
            freeze = json.loads((root/"freeze.json").read_text())
            marker = json.loads((root/"completed.json").read_text())
            freeze["qualification_sha256"] = {
                f"{lib}.json": write(root/"qual"/f"{lib}.json", dict(passed=True, cold_real_action_s=seconds))
                for lib, seconds in (("cffg", 2.), ("rxinfer", 5.))}
            marker["freeze_sha256"] = write(root/"freeze.json", freeze)
            write(root/"completed.json", marker)
            data = analyze(root, "heldout")
            write(root/"analysis.json", data)
            page = render(root).read_text()
            self.assertIn("CFFG had meaningfully lower", page)
            self.assertIn("first real action from a fresh process (CFFG 2 s, RxInfer 5 s)", page)
            self.assertIn("reduced mean error by 66.7% for CFFG", page)
            self.assertIn("CFFG 5 ms, RxInfer 5 ms", page)
            Links(root).feed(page)
            figure = root/"hidden-concentration-error.png"
            with Image.open(figure) as im:
                self.assertEqual(im.size, (1680, 882))
                im.verify()
            samples = learning_errors(data, seeds)
            self.assertEqual(len(samples), 6)
            for arm, expected in (("adaptive", .01), ("informed", .005), ("frozen", .03)):
                for lib, scale in (("cffg", 1), ("rxinfer", 2)):
                    self.assertEqual(len(samples[lib,arm]), 2)
                    for error in samples[lib,arm]:
                        self.assertAlmostEqual(error, expected*scale)
            with self.assertRaisesRegex(ValueError, "all frozen seeds"):
                learning_errors(data, seeds+[17303])
            old_figure = figure.read_bytes()
            data["inventory"]["sha256"] = {}
            write(root/"analysis.json", data)
            with self.assertRaisesRegex(ValueError, "hashes"):
                render(root)
            self.assertEqual((root/"index.html").read_text(), page)
            self.assertEqual(figure.read_bytes(), old_figure)

    def test_timing_can_favor_different_libraries(self):
        groups = [dict(mode="control", condition="changing", arm="adaptive", library=lib,
                       overall={"latency_p50_s": ms/1000}) for lib, ms in (("cffg", 5.), ("rxinfer", 1.))]
        summary = speed_summary({"groups": groups}, [3., 10.])
        self.assertIn("CFFG had the shorter time to the first", summary)
        self.assertIn("RxInfer had the shorter median time", summary)



if __name__ == "__main__":
    unittest.main()
