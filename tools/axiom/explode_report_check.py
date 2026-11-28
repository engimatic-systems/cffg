"""Synthetic archive tests for the standalone Explode pilot report."""

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import explode_report as report
from evaluation_report import REFERENCE_FILES, STAGES, reference


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    data = encode(value) + b"\n"
    path.write_bytes(data)
    return fingerprint(data)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(v for k, v in attrs if k in ("src", "href") and "://" not in v)


class PilotReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.root = Path(cls.scratch.name)
        cls.protocol = json.loads((reference.HERE / "reference_protocol.json").read_bytes())
        files = {f"tools/axiom/{name}": reference.sha(reference.HERE / name)
                 for name in ("evaluation_report.py", *REFERENCE_FILES)}
        source = fingerprint(encode(files))
        study = dict(revision="synthetic", source_files=files, source_sha256=source, arm="untuned",
                     dtype="float32", precision="compensated", horizon=32, candidates=512, draws=3,
                     reduction_cadence=500, protocol_sha256=REFERENCE_FILES["reference_protocol.json"],
                     reference_sha256="ef5141f70e4ef6792a40e7ade7086109fc9e04b68afed19764294f9b53282caf",
                     stages={stage: list(reference.seed_pairs(cls.protocol, stage)) for stage in STAGES})
        cls.study_hash = write_json(cls.root / "study.json", study)
        ident, = study["stages"]["development"]
        cls.run_dir = cls.root / "development/Explode-88101"
        cls.run_dir.mkdir(parents=True)
        trace = cls.run_dir / "steps.jsonl"
        with trace.open("wb") as stream:
            for tick in range(10000):
                block, pos = divmod(tick, 1000)
                catches, misses = 3*(block+1), 2*(10-block)
                reward = 1 if pos < catches else -1 if pos < catches + misses else 0
                native = dict(tick=tick, event="step", applied=2, reward=reward, episode=0, terminated=False,
                              truncated=False, rgb_sha256="a"*64)
                scheduled = (tick+1) % 500 == 0
                row = dict(tick=tick, action=2, reward=reward, game=native, reset=None, terminated=False,
                           truncated=False, frame_sha256="a"*64, decision_s=1 if tick == 0 else .25,
                           observation_s=2 if tick == 0 else .5, decision={"total_score_range": 0},
                           valid=True, reduction_performed=scheduled,
                           reduction=dict(step=tick+1, status="ok", pairs=7, accepted=2) if scheduled else None)
                stream.write(encode(row) + b"\n")
        result = dict(**ident, status="complete", stage="development", source_sha256=source, return_sum=55,
                      steps_sha256=reference.sha(trace), process_s=9000., initialization_s=30., resumed_from=0)
        cls.result = cls.run_dir / "result.json"
        cls.receipt = cls.root / "pilot_passed.json"
        write_json(cls.receipt, dict(stage="development", source_sha256=source,
                                    runs={cls.run_dir.name: write_json(cls.result, result)}))
        (cls.root / "index.html").write_bytes(b"original campaign page\n")
        (cls.root / "host-activity.log").write_text("synthetic host context")

    def setUp(self):
        override = patch.object(report, "STUDY_SHA256", self.study_hash)
        override.start()
        self.addCleanup(override.stop)

    def test_complete_report(self):
        opened = []
        original = Path.open
        def record(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", record):
            report.publish(self.root)
        self.assertFalse(any(p.is_relative_to(self.root / stage) for p in opened for stage in
                             ("pre_evaluation", "evaluation", "pre_evaluation_passed.json", "completed.json")))
        self.assertEqual((self.root / "index.html").read_bytes(), b"original campaign page\n")
        page = (self.root / "explode.html").read_text()
        for text in ("<strong>165</strong>", "<strong>110</strong>", "<strong>+55 points in 10,000 real actions",
                     "125 points below", "60.0%", "9,725", "45 more than the opening block", "−1",
                     "180 points", "deviation of 30 points", "9,000.0 seconds", "0.900 seconds",
                     "20 scheduled reviews", "40 merges from", "140 recorded candidate pairs", "88101", "180400"):
            self.assertIn(text, page)
        self.assertIn("-17, -12, -7, -2, +3, +8, +13, +18, +23, +28", page)
        self.assertIn("0.250</td><td>1.000</td><td>0.250", page)
        self.assertIn("0.750</td><td>3.000</td><td>0.750", page)
        links = Links()
        links.feed(page)
        self.assertTrue(all((self.root / target).is_file() for target in links.targets))
        with Image.open(self.root / report.CHART) as img:
            self.assertEqual(img.size, (1440, 624))
            img.verify()

    def test_reject_before_opening_partial_trace(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(report.frozen, "read_run") as read:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "waiting for the sealed"):
                report.publish(root)
            read.assert_not_called()
            self.assertFalse((root / "explode.html").exists())
            self.assertFalse((root / report.CHART).exists())

    def test_tampered_inputs(self):
        for path in (self.root / "study.json", self.receipt, self.result, self.run_dir / "steps.jsonl"):
            data = path.read_bytes()
            try:
                with self.subTest(path=path.name):
                    if path == self.receipt:
                        marker = json.loads(data)
                        marker["runs"] = {}
                        write_json(path, marker)
                    else:
                        path.write_bytes(data[:-1] if path.name == "steps.jsonl" else data+b" ")
                    with self.assertRaises(ValueError):
                        report.publish(self.root)
            finally:
                path.write_bytes(data)

    def test_scope_and_restarts(self):
        with self.assertRaisesRegex(ValueError, "unexpected native Explode"):
            report.score_blocks([0]*9999+[2])
        self.assertIn("matched", report.point_comparison(180, 180))
        self.assertIn("1 point above", report.point_comparison(181, 180))
        study, run, _, historical, receipt = report.pilot_evidence(self.root)
        samples = [1.]*10000
        zero_run = dict(run, resumed_from=500, observed_return=0)
        page = report.render(study, zero_run, report.score_blocks([0] * 10000), historical,
                             samples, samples, receipt, "")
        self.assertIn("No bomb reached a catch or miss", page)
        self.assertIn("resumed from action 500", page)
        self.assertNotIn("seconds of wall time per action", page)


if __name__ == "__main__":
    unittest.main()
