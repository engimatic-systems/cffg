"""Fault and restart checks for the evaluation journal, seed receipts and stage gates."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

import evaluation_campaign as campaign
import evaluation_episode as runner


class CounterActor:
    def __init__(self, image, game, actions, seed):
        self.game, self.seed = game, seed
        self.tick = 0
        self.total = seed
        self.last_decision = None

    def choose(self):
        action = self.total % 3
        self.last_decision = {"action": action, "plan_sha256": hashlib.sha256(str(self.total).encode()).hexdigest()}
        return action

    def observe(self, image, applied, reward, reset=False):
        self.total += int(image) + applied
        self.tick += int(not reset)
        due = not reset and self.tick % 500 == 0
        return dict(valid=True, reduction_performed=due, update=not reset,
                    reduction=dict(step=self.tick, status="ok") if due else None,
                    tracked=0, active_interaction=0, active_motion=4, identity_writes=0, interaction_writes=0)

    def state_dict(self):
        return {"scene[0].tick": np.int32(self.tick), "balance": np.asarray(self.total, np.int64)}

    def load_state_dict(self, fields):
        self.tick = int(fields["scene[0].tick"])
        self.total = int(fields["balance"])


@contextmanager
def simulator(reference, game, seed):
    tick = value = 0
    def request(cmd):
        nonlocal tick, value
        if cmd == "reset":
            value = seed % 11
            return value, dict(tick=tick, event="reset", rgb_sha256=str(value))
        assert cmd["tick"] == tick
        value = (value * 7 + cmd["action"]) % 19
        tick += 1
        return value, dict(tick=tick-1, event="step", applied=cmd["action"], reward=float(value % 3-1),
                           rgb_sha256=str(value), truncated=tick%127==0, terminated=False)
    yield request, {"game":game, "actions":["stay", "left", "right"]}


def save_fixture(path, actor, metadata):
    np.savez(path, **actor.state_dict(), envelope=np.array(json.dumps(metadata)))


def restore_fixture(path, actor, metadata):
    with np.load(path, allow_pickle=False) as data:
        if json.loads(data["envelope"].item()) != metadata:
            raise ValueError("fixture checkpoint identity mismatch")
        actor.load_state_dict({name:data[name] for name in data.files if name != "envelope"})


class CampaignTest(unittest.TestCase):
    def test_checkpoint_and_entire_qualification_continuation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = dict(game="Explode", model_seed=33, game_seed=17, steps=1000)
            source = "1" * 64
            study = dict(source_sha256=source, reference=tmp, stages={"pre_evaluation":[case]})
            stage = "pre_evaluation"
            folder = root / stage / campaign.run_name(case)
            folder.mkdir(parents=True)
            with (patch.object(runner,"verify_source"), patch.object(runner,"Actor",CounterActor),
                  patch.object(runner,"save",save_fixture), patch.object(runner,"restore",restore_fixture),
                  patch.object(runner,"simulator",simulator)):
                runner.episode(root, study, stage, case)
                first = campaign.read(folder / "result.json")
                self.assertEqual(first["resume_matched_steps"],500)
                marker = campaign.stage_receipt(root, study, stage)
                self.assertEqual(marker["runs"][campaign.run_name(case)], campaign.digest(folder / "result.json"))
                # Simulate a killed worker after tick 503 with an incomplete JSON line.
                tape = folder / "steps.jsonl"
                lines = tape.read_bytes().splitlines(keepends=True)
                tape.write_bytes(b"".join(lines[:503]) + b'{"tick":')
                (folder/"result.json").unlink()
                (folder/"checkpoints/1000.npz").unlink()
                runner.episode(root,study,stage,case)
                second=campaign.read(folder/"result.json")
                self.assertEqual(first["final_state_sha256"],second["final_state_sha256"])
                self.assertEqual(first["return_sum"],second["return_sum"])
                self.assertEqual(second["resume_matched_steps"],500)
                self.assertEqual(len(list(folder.glob("partial-line-*.bin"))),1)
                self.assertEqual(len(campaign.stage_receipt(root,study,stage)["runs"]),1)
                tape.write_bytes(tape.read_bytes().replace(b'"tick":9,',b'"tick":8,',1))
                with self.assertRaisesRegex(ValueError,"digest"):
                    campaign.audit_run(root,study,stage,case)

    def test_stage_closure_requires_exact_qualifying_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = dict(game="Explode", model_seed=33, game_seed=17, steps=10000)
            study = dict(source_sha256="1"*64,stages={"development":[case],"pre_evaluation":[]})
            campaign.publish(root/"pilot_passed.json",dict(stage="development",
                source_sha256=study["source_sha256"],runs={"wrong-seed":"0"*64}))
            with self.assertRaises(FileNotFoundError):
                campaign.verify_before(root,study,"pre_evaluation")

    def test_applied_action_retained_after_failed_observation(self):
        class FailedActor(CounterActor):
            def observe(self,*args,**kwargs):
                raise ArithmeticError("bad observation")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = dict(game="Explode",model_seed=33,game_seed=17,steps=10000)
            dest=root/"development"/campaign.run_name(case)
            dest.mkdir(parents=True)
            with (patch.object(runner,"Actor",FailedActor),patch.object(runner,"simulator",simulator),
                  self.assertRaisesRegex(ArithmeticError,"bad observation")):
                runner.episode(root,dict(source_sha256="1"*64,reference=tmp),"development",case)
            failed = json.loads((dest/"result.json").read_bytes())
            self.assertEqual(failed["status"],"failed")
            actual=failed["applied_before_failure"]["game"]
            self.assertEqual(actual["tick"],0)
            self.assertEqual(actual["applied"],0)
            self.assertEqual(actual["reward"],0.)
            self.assertEqual(failed["steps_sha256"],hashlib.sha256().hexdigest())


if __name__ == "__main__":
    unittest.main()
