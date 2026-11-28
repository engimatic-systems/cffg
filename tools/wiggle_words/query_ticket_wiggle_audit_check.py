"""Focused CLI regression for compact ticket review and the structured formats."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import yaml


class ReviewOutput(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        repo = Path(directory.name)
        tools = repo / "tools" / "wiggle_words"
        tools.mkdir(parents=True)
        self.script = tools / "query_ticket_wiggle_audit.py"
        shutil.copyfile(Path(__file__).with_name(self.script.name), self.script)
        tickets = repo / ".tickets"
        tickets.mkdir()
        (tickets / "cff-demo.md").write_text("Fast size and fast queries.\nSize is measured in bytes.\n")
        lexicon = dict(schema_version=2, candidates=[
            dict(id=word, category="example", pattern=word, meanings=["a listed meaning"])
            for word in ("fast", "size")])
        lexicon_path = tools / "ticket-wiggle-words-v2.yaml"
        lexicon_path.write_text(yaml.safe_dump(lexicon))
        hit = dict(ticket=".tickets/cff-demo.md", line=7, column=3, matched_text="fast",
                   section="Historical heading", meaning="unlisted_meaning",
                   observed_meaning="latency", rationale="The claim omitted a duration.")
        listed = dict(hit, line=10, meaning="a listed meaning")
        del listed["observed_meaning"]
        audit = dict(schema_version=2,
                     provenance=dict(lexicon={"path": str(lexicon_path.relative_to(repo))},
                                     ticket_corpus={"git_commit": "pinned-revision"}),
                     candidates=[dict(id="fast", category="example", pattern="fast",
                                      occurrences=[hit, listed])])
        (tools / "ticket-wiggle-word-audit-v2.yaml").write_text(yaml.safe_dump(audit))

    def query(self, *args):
        return subprocess.check_output([sys.executable, self.script, *args], text=True)

    def test_scan_groups_each_complete_line_and_discloses_full_details(self):
        compact = self.query("scan", "cff-demo")
        self.assertIn(".tickets/cff-demo.md · matches 4 · candidates 2", compact)
        self.assertEqual(compact.count("Fast size and fast queries."), 1)
        self.assertIn("L1 [fast ×2, size]", compact)
        self.assertIn("L2 [size] Size is measured in bytes.", compact)
        self.assertIn("--detail", compact)
        self.assertNotIn("a listed meaning", compact)
        self.assertIn("matches 0", self.query("scan", "cff-demo", "--candidate", "absent"))
        detail = self.query("scan", "cff-demo", "--detail")
        self.assertEqual(detail, self.query("--format", "yaml", "scan", "cff-demo"))
        full = json.loads(self.query("--format", "json", "scan", "cff-demo"))
        self.assertEqual(yaml.safe_load(detail), full)
        self.assertEqual([hit["column"] for hit in full["candidates"][0]["occurrences"]], [1, 15])

    def test_slice_uses_archived_context_and_respects_filters(self):
        args = ("slice", "cff-demo", "--candidate", "fast", "--problems-only")
        compact = self.query(*args)
        self.assertIn("pinned-revision", compact)
        self.assertIn("problem matches 1", compact)
        self.assertIn("L7 [fast “fast” → unlisted_meaning (latency)", compact)
        self.assertIn("The claim omitted a duration.", compact)
        self.assertIn("Historical heading", compact)
        self.assertNotIn("L10", compact)
        self.assertNotIn("Fast size and fast queries.", compact)
        self.assertEqual(yaml.safe_load(self.query(*args, "--detail")),
                         json.loads(self.query("--format", "json", *args)))


if __name__ == "__main__":
    unittest.main()
