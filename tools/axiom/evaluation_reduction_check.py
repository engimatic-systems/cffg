"""Exercise the acted reduction boundary against the direct host API on synthetic data."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from cffg.experiments import axiom_reduction_schedule as boundary
from cffg.component_bank_device import from_host, to_host
from cffg.experiments import _axiom_reduction_scores as scoring
from cffg.experiments.axiom_banks import BankSettings, initial_banks, seeded_dormant
from cffg.experiments.axiom_mixture import learn_frame
from cffg.experiments.axiom_reduction import ReductionState, reduce_interactions
from cffg.mixed_learning import MixedBatch


def exhaustive_bounds(tables, tape, ids):
    """Force the reference competition to evaluate every fixed component."""
    return np.zeros(len(ids), bool), -np.inf


def equal_fields(left, right):
    if set(left) != set(right):
        raise AssertionError("checkpoint inventory differs")
    for name, field in left.items():
        np.testing.assert_array_equal(field, right[name], err_msg=name)


def fixture():
    settings = BankSettings(identities=3, motions=4, interactions_per_motion=2)
    bank = initial_banks(3, settings=settings)[1]
    virgin = seeded_dormant(bank.prior, jax.random.key(987))
    return bank, virgin


def learn(bank, positions):
    count = len(positions)
    x = np.zeros((count, 7), np.float32)
    x[:, 0] = positions
    labels = jnp.zeros(count, jnp.int32)
    data = MixedBatch(jnp.asarray(x), jnp.zeros((count,7,7)), jnp.ones(count,bool), (labels,)*6)
    learned = learn_frame(bank, data, jnp.ones(count, bool), threshold=100.)
    if not bool(learned.valid):
        raise AssertionError("synthetic observed interaction failed")
    return learned.bank


class ScheduledTests(unittest.TestCase):
    def test_receipts_resume_and_terminal_reduction_match_host(self):
        bank, virgin = fixture()
        bank = learn(bank, [0.,0.,.4])
        initial_key = jax.random.key(722)
        history = boundary.start(bank, initial_key)
        reference_bank = to_host(bank)
        reference = ReductionState.start(reference_bank, initial_key)

        with mock.patch.object(boundary, "to_host", side_effect=AssertionError("early transfer")):
            for step in (0, 1, 499):
                unchanged, same, receipt = boundary.after_action(bank,history,step=step,dormant_initial=virgin)
                self.assertIs(unchanged, bank)
                self.assertIs(same, history)
                self.assertIsNone(receipt)
            with self.assertRaisesRegex(ValueError, "history ends"):
                boundary.after_action(bank,history,step=501,dormant_initial=virgin)

        for step in (500, 1000, 10000):
            if step == 1000:
                # Both clocks see one more learned observation between reductions.
                bank = learn(bank, [.8])
                reference_bank = to_host(learn(from_host(reference_bank), [.8]))
            elif step == 10000:
                # Supply a complete retained tape at 9500 to isolate the terminal boundary.
                records = type(history.records)(*(np.resize(a, (38_000, a.shape[1])) for a in history.records))
                history = ReductionState(history.key, 9500, records)
                reference = ReductionState(reference.key, 9500, records)
            with mock.patch.object(scoring, "_fixed_bounds", exhaustive_bounds):
                direct = reduce_interactions(reference_bank, reference, step=step, dormant_initial=virgin)
            bank, history, receipt = boundary.after_action(bank,history,step=step,dormant_initial=virgin)
            self.assertEqual(direct.status, "ok")
            self.assertEqual(receipt["step"], step)
            self.assertEqual(receipt["records_total"], step // 500 * 2000)
            self.assertEqual(receipt["initial_score"], direct.initial_score)
            self.assertEqual(receipt["final_score"], direct.final_score)
            self.assertEqual(receipt["decisions"], [e._asdict() for e in direct.decisions])
            self.assertEqual(receipt["pairs"], len(direct.decisions))
            json.dumps(receipt, allow_nan=False)
            reference_bank, reference = direct.bank, direct.state
            equal_fields(to_host(bank).state_dict(exact=True), reference_bank.state_dict(exact=True))
            equal_fields(boundary.archive(history), reference.state_dict())
            self.assertIsNone(boundary.after_action(bank,history,step=step,dormant_initial=virgin)[2])
            if step == 500:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "reduction.npz"
                    np.savez(path, **boundary.archive(history))
                    with np.load(path, allow_pickle=False) as arrays:
                        history = boundary.restore(arrays, bank, step=step)

    def test_failed_or_truncated_history_preserves_state(self):
        bank, virgin = fixture()
        history = boundary.start(bank, jax.random.key(12))
        before = boundary.archive(history)
        with self.assertRaisesRegex(ArithmeticError, "empty_support"):
            boundary.after_action(bank,history,step=500,dormant_initial=virgin)
        equal_fields(before, boundary.archive(history))
        with self.assertRaisesRegex(ValueError, "seeded dormant"):
            boundary.after_action(bank,history,step=500,dormant_initial=None)
        with self.assertRaisesRegex(ValueError, "history ends"):
            boundary.restore(before, bank, step=500)
        bogus = {**before, "last_step":np.int64(500), "features":np.zeros((1,7),np.float32),
                 "categories":np.zeros((1,6),np.int32)}
        with self.assertRaisesRegex(ValueError, "2000 generated records"):
            boundary.restore(bogus,bank,step=700)
        with self.assertRaisesRegex(ValueError, "integer count"):
            boundary.after_action(bank,history,step=True,dormant_initial=virgin)


def full_size_work():
    """Count actual context evaluations on a sparse full-width development fixture."""
    model = initial_banks(3)[1]
    virgin = seeded_dormant(model.prior, jax.random.key(170))
    model = learn(model, [0., .05, 1.25, 1.5, 2., 4.])
    pending = boundary.start(model, jax.random.key(123))
    original = scoring.Tables.context

    def run(bank, history, step, *, exhaustive=False):
        count = 0
        def counted(self, tape, slots):
            nonlocal count
            count += len(tape.features)*len(slots)
            return original(self, tape, slots)
        bounds = exhaustive_bounds if exhaustive else scoring._fixed_bounds
        with mock.patch.object(scoring, "_fixed_bounds", bounds), mock.patch.object(scoring.Tables, "context", counted):
            result = boundary.after_action(bank, history, step=step, dormant_initial=virgin)
        return result, count

    (first, history, early), work = run(model, pending, 500)
    (check, saved, receipt), exhaustive = run(model, pending, 500, exhaustive=True)
    equal_fields(to_host(first).state_dict(exact=True), to_host(check).state_dict(exact=True))
    equal_fields(boundary.archive(history), boundary.archive(saved))
    if early != receipt:
        raise AssertionError("large sparse reduction decisions changed")
    changed = learn(first, [3., 3.25, 4.5])
    # Keep the first real batch as the replay population to control the large
    # input size. The final call draws 2,000 new points from the changed bank.
    retained = type(history.records)(*(np.tile(a, (19,1)) for a in history.records))
    previous = ReductionState(history.key, 9500, retained)
    (_, final, late), final_work = run(changed, previous, 10000)
    print(json.dumps({
        "slots": len(model.posterior.active), "early": early["records_total"],
        "early_active": early["active_before"], "early_pairs": early["pairs"],
        "early_accepted": early["accepted"], "early_exhaustive_contexts": exhaustive,
        "early_certified_contexts": work, "late": late["records_total"],
        "late_active": late["active_before"], "late_pairs": late["pairs"],
        "late_accepted": late["accepted"], "late_certified_contexts": final_work,
        "late_initial_dense_contexts": late["records_total"]*len(model.posterior.active),
        "late_tape_bytes": sum(a.nbytes for a in final.records),
    }, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--scale"]:
        full_size_work()
    else:
        unittest.main()
