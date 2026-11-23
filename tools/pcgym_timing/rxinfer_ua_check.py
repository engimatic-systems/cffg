"""Check live RxInfer UA delivery and the fixed affine oracle on synthetic windows.

From the repo root run PYTHONPATH=tools uv run --frozen --no-sync python
tools/pcgym_timing/rxinfer_ua_check.py.
"""
from contextlib import closing
import json

import numpy as np

import cstr_runtime_fixture as fixture
from qualification import qualify
from rxinfer_adapter import Adapter


def readouts(adapter, windows):
    results = []
    for data in windows:
        result = adapter.solve(data, details=True)
        if result["status"] != "ok":
            raise AssertionError(result)
        results.append(np.r_[result["action"], result["mean"], np.ravel(result["cov"]),
                             result["next_action_cov"]])
    return np.stack(results)


def check():
    _, arrays = fixture.model()
    forecast = fixture.observations(history=True)
    forecast.update({f"e{i}": None for i in range(5, 20)})
    windows = [fixture.observations(), fixture.observations(2, history=True, goal=.3), forecast]
    nominal, changed = 50_000., 51_500.
    theta = arrays["theta"].copy()
    theta[2] = np.log(changed)
    with closing(Adapter(arrays, fixture.CONFIGURATION)) as live:
        initial = readouts(live, windows)
        worker_pid = live.proc.pid
        live.install(changed)
        updated = readouts(live, windows)
        with closing(Adapter({**arrays, "theta": theta}, fixture.CONFIGURATION)) as fresh:
            rebuilt = readouts(fresh, windows)
        np.testing.assert_allclose(updated, rebuilt, rtol=0, atol=1e-10)
        if abs(updated[0, 0] - initial[0, 0]) < 1e-4:
            raise AssertionError("changed UA must change the planned action")
        live.install(nominal)
        for invalid in (0., -1., np.nan, np.inf, [changed]):
            with np.testing.assert_raises(ValueError):
                live.install(invalid)
        restored = readouts(live, windows)
        np.testing.assert_allclose(restored, initial, rtol=0, atol=1e-10)
        if live.proc.pid != worker_pid:
            raise AssertionError("worker restarted during UA changes")
        metadata = live.metadata
    def affine_adapter(arrays, config, *, affine):
        adapter = Adapter(arrays, config, affine=affine)
        adapter.install(changed)
        return adapter

    oracle = qualify(affine_adapter, arrays)
    if not all(case["passed"] for case in oracle):
        raise AssertionError(oracle)
    return dict(metadata=metadata, nominal_ua=nominal, changed_ua=changed,
                fresh_max_abs=float(np.max(abs(updated - rebuilt))),
                restored_max_abs=float(np.max(abs(restored - initial))),
                action_change=float(updated[0, 0] - initial[0, 0]), affine=oracle)


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, allow_nan=False))
