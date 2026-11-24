"""Exercise the acted reduction boundary against the direct host API on synthetic data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import evaluation_reduction as boundary
from cffg.component_bank_device import from_host, to_host
from cffg.experiments.axiom_banks import BankSettings, initial_banks, seeded_dormant
from cffg.experiments.axiom_mixture import learn_frame
from cffg.experiments.axiom_reduction import ReductionState, reduce_interactions
from cffg.mixed_learning import MixedBatch


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


if __name__ == "__main__":
    unittest.main()
