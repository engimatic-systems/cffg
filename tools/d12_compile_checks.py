"""Focused instrumentation checks; no D12 lowering, compilation or solve execution.

Run directly, outside pytest discovery: uv run python tools/d12_compile_checks.py
One scalar arithmetic program tests the compilation API and output evidence.
"""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import d12_compile_scaling as measurement

jax = measurement.native_jax()
import numpy as np

from cffg import hybrid_joint as joint, hybrid_joint_solver as solver
from d12_compile_fixture import context, fixture_arrays, inference_outputs


class InstrumentationChecks(unittest.TestCase):
    def test_frozen_inventory_and_budgets(self):
        spec = measurement.load_spec()
        self.assertEqual(spec['budgets'], asdict(solver.Budgets()))
        self.assertEqual([c['id'] for c in spec['cases']], ['t32-full', 't32-inference', 't100-full'])
        self.assertEqual(spec['limits']['worker_seconds'] * len(spec['cases']), 3600)

    def test_exact_fixture_formulas_and_shapes(self):
        for horizon in (32, 100):
            parameters, sites = fixture_arrays(horizon)
            self.assertEqual(parameters['weights'].shape, (horizon - 1, 4, 3, 2))
            for i in range(4):
                for stick in range(3):
                    expected = np.float32([.12 * (i - 1.5) + .04 * (stick - 1),
                                           -.08 * (i - 1.5) + .03 * (stick - 1)])
                    np.testing.assert_array_equal(parameters['weights'][:, i, stick],
                                                  np.broadcast_to(expected, (horizon - 1, 2)))
                    np.testing.assert_array_equal(parameters['offsets'][:, i, stick],
                                                  np.full(horizon - 1, .10 * (i - 1.5) - .15 * (stick - 1), np.float32))
                for name, expected in (
                    ('matrix', [[.65 + .04 * i, .03], [-.02, .80 - .03 * i]]),
                    ('bias', [.10 - .05 * i, -.15 + .04 * i]),
                    ('noise', [[.8 + .1 * i, 0], [0, 1.1 + .05 * i]]),
                ):
                    expected = np.asarray(expected, np.float32)
                    np.testing.assert_array_equal(parameters[name][:, i],
                                                  np.broadcast_to(expected, (horizon - 1, *expected.shape)))
                for t in range(horizon):
                    self.assertEqual(sites['constant'][t, i], np.float32(-.2 - .1 * i))
                    np.testing.assert_array_equal(sites['linear'][t, i], np.float32([
                        .1 * (i - 1.5) + .05 * (-1) ** t, -.08 * (i - 1.5) + .03 * ((t % 3) - 1)]))
                    np.testing.assert_array_equal(sites['precision'][t, i],
                                                  np.float32([[.9 + .1 * i, 0], [0, 1.1 + .05 * i]]))
            for value in (*parameters.values(), *sites.values()):
                self.assertIn(value.dtype, (np.dtype('float32'), np.dtype('int32')))
            self.assertTrue(np.all(parameters['xi'] == 0))
            self.assertTrue(np.all(sites['center'] == 0))
            self.assertTrue(np.all(sites['status'] == 0))

    def test_native_context_without_evaluation(self):
        self.assertFalse(jax.config.x64_enabled)
        self.assertEqual(jax.config.jax_explicit_x64_dtypes.name, 'ERROR')
        with self.assertRaises(ValueError):
            jax.numpy.asarray(1, dtype='float64')
        with self.assertRaises(ValueError):
            measurement.checked_arrays(np.asarray(1, np.float64))
        for horizon in (32, 100):
            coordinate, parameters, sites = context(horizon)
            joint.validate_context(coordinate, parameters, sites)
            self.assertEqual(coordinate.size, 58 * (horizon - 1))
            self.assertEqual(coordinate.residual_size, 106 * (horizon - 1))
            records = measurement.checked_arrays((parameters, sites, coordinate.metadata()))
            self.assertTrue(records)
            self.assertTrue(all(r['dtype'] in ('float32', 'int32', 'bool') for r in records))

    def test_output_selection_preserves_status_and_coordinates(self):
        state = SimpleNamespace(**{name: object() for name in solver.State._fields})
        evaluation = SimpleNamespace(status=object(), certificates=object(),
                                     local=SimpleNamespace(free_energy=object(), dual_energy=object(), status=object()),
                                     auxiliary=SimpleNamespace(status=object(), linear=object(), precision=object()))
        state.evaluation = evaluation
        selected = inference_outputs(state)
        self.assertEqual(set(selected), set(measurement.load_spec()['policy']['inference_outputs']))
        self.assertIs(selected['coordinate'], state.coordinate)
        self.assertIs(selected['certificates'], evaluation.certificates)
        self.assertIs(selected['evaluation_status'], evaluation.status)
        self.assertIs(selected['local_status'], evaluation.local.status)
        self.assertIs(selected['auxiliary_status'], evaluation.auxiliary.status)
        for name in solver.State._fields:
            if name != 'evaluation':
                self.assertIs(selected[name], getattr(state, name))

    def test_limits_and_memory_units(self):
        limits = measurement.load_spec()['limits']
        self.assertEqual(measurement.rss_bytes(1024, 'Darwin'), 1024)
        self.assertEqual(measurement.rss_bytes(1024, 'Linux'), 1048576)
        self.assertIsNone(measurement.limit_reason(1, 100, {}, limits, 0))
        self.assertEqual(measurement.limit_reason(1201, 100, {}, limits, 0), 'worker_seconds')
        self.assertEqual(measurement.limit_reason(1, limits['rss_bytes'] + 1, {}, limits, 0), 'rss_bytes')
        for stage, seconds in (('lower', 181), ('compile', 901)):
            progress = dict(stage=stage, stage_started_ns=0)
            self.assertEqual(measurement.limit_reason(1, 100, progress, limits, seconds * 10**9), f'{stage}_seconds')

    def test_supervisor_preserves_real_resource_stop(self):
        limits = dict(measurement.load_spec()['limits'], worker_seconds=.1, sample_seconds=.02,
                      terminate_grace_seconds=.05)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = measurement.supervise([sys.executable, '-c', 'import time; time.sleep(5)'],
                                            output, limits, measurement.child_environment())
            self.assertEqual(report['stop'], 'worker_seconds')
            self.assertLess(report['exit_code'], 0)
            self.assertGreater(report['process_high_water']['bytes'], 0)
            self.assertTrue((output / 'resource-stop.json').exists())
            self.assertTrue((output / 'rss-samples.jsonl').read_text())
            self.assertEqual(json.loads((output / 'process.json').read_text()), report)

    def test_scalar_compilation_evidence_and_rejected_reports(self):
        # This compiles scalar addition only. The executable is never called.
        report = {}
        stages = []
        retained = measurement.compile_program(lambda x: x + jax.numpy.float32(2),
                    (np.asarray(1, np.float32),), report, stages.append)
        self.assertEqual(len(retained), 2)
        self.assertEqual(stages, ['lower', 'lower_inspection', 'compile', 'compiled_inspection'])
        self.assertEqual(report['calls'], dict(lower=1, compile=1, execute=0))
        self.assertEqual(set(report['stablehlo_dtypes']['value_type_occurrences']), {'f32'})
        process = dict(exit_code=0, stop=None, process_high_water=dict(bytes=100))
        completed = dict(schema=measurement.REPORT_SCHEMA, status='complete', execution='not run',
                         measurements=report, runtime=dict(backend='cpu', config=dict(jax_enable_x64=False,
                             jax_explicit_x64_dtypes='ERROR', jax_enable_compilation_cache=False,
                             jax_persistent_cache_enable_xla_caches='')))
        measurement.validate_completed(completed, process)
        for field, bad in (('status', 'started'), ('execution', 'run')):
            with self.assertRaises(ValueError):
                measurement.validate_completed(dict(completed, **{field: bad}), process)
        for bad in (dict(process, stop='rss_bytes'), dict(process, exit_code=1)):
            with self.assertRaises(ValueError):
                measurement.validate_completed(completed, bad)
        for field, bad in (('lower_seconds', -1), ('compile_seconds', float('nan')), ('compiled_outputs', []),
                           ('calls', dict(lower=1, compile=1, execute=1))):
            changed = copy.deepcopy(completed)
            changed['measurements'][field] = bad
            with self.assertRaises(ValueError):
                measurement.validate_completed(changed, process)
        changed = copy.deepcopy(completed)
        changed['measurements']['compiled_outputs'][0]['dtype'] = 'float64'
        with self.assertRaises(ValueError):
            measurement.validate_completed(changed, process)


if __name__ == '__main__':
    unittest.main(verbosity=2)
