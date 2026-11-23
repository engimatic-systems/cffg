"""Exercise report states and practical-effect boundaries on synthetic evidence."""
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from report import observed, primary, render


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
        if tag == "a":
            target = dict(attrs)["href"]
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
            Links(root).feed(page)

    def test_complete_requires_matching_outcome_inventory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = write(root/"protocol.json", {})
            pilot = write(root/"pilot.json", {"status":"go", "gates":{}})
            freeze = write(root/"freeze.json", {"protocol_sha256":protocol, "pilot_sha256":pilot})
            write(root/"completed.json", {"freeze_sha256":freeze,"runs":{"record":"testhash"}})
            analysis = {"stage":"heldout","status":"complete","primary":score(),"inventory":{"sha256":{"record":"testhash"}}}
            write(root/"analysis.json", analysis)
            page = render(root).read_text()
            self.assertIn("CFFG had meaningfully lower", page)
            self.assertIn("95% interval", page)
            Links(root).feed(page)
            analysis["inventory"]["sha256"] = {}
            write(root/"analysis.json", analysis)
            with self.assertRaisesRegex(ValueError, "hashes"):
                render(root)
            self.assertEqual((root/"index.html").read_text(), page)


if __name__ == "__main__":
    unittest.main()
