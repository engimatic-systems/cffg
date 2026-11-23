"""Qualify the persistent extraction worker on synthetic correlated affine slices."""

import json
from time import perf_counter

import numpy as np

from cffg_native_check import dense_plan, dense_update, fixture
from rxinfer_native import Backend


def run():
    update, plan = fixture()
    start = perf_counter()
    backend = Backend()
    times = {"launch_s": perf_counter() - start}
    try:
        def checked_update(args):
            expected = dense_update(*args)
            mean, cov = backend.update(*args)
            np.testing.assert_allclose(mean, expected[0], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(cov, expected[1], rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(backend.last_innovation_nll, expected[2], rtol=1e-9, atol=1e-9)
            return mean, cov

        start = perf_counter()
        first = checked_update(update)
        times["first_observed_update_s"] = perf_counter() - start
        reference = dense_plan(*plan)
        start = perf_counter()
        original = backend.plan(*plan)
        times["first_plan_s"] = perf_counter() - start
        np.testing.assert_allclose(original, reference, rtol=1e-9, atol=1e-8)
        changed = list(plan)
        changed[7] = changed[7] + .015
        other = backend.plan(*changed)
        np.testing.assert_allclose(other, dense_plan(*changed), rtol=1e-9, atol=1e-8)
        assert np.linalg.norm(other - original) > 1e-3

        repeat = checked_update(update)
        np.testing.assert_array_equal(repeat[0], first[0])
        np.testing.assert_array_equal(repeat[1], first[1])
        checked_update((*update[:-1], None))
        m, P, A, b, Q, H, R, y = update
        checked_update((first[0], first[1], A*.97, b+.01, Q*1.1, H[::-1].copy(), R*1.2, y[::-1]))
        checked_update((m, P, A, b, np.zeros_like(Q), H, R, None))
        as_float32 = [a.astype(np.float32) for a in update]
        promoted = dense_update(*(a.astype(np.float64) for a in as_float32))
        for actual, expected in zip(backend.update(*as_float32), promoted[:2], strict=True):
            assert actual.dtype == np.float64
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)
        with np.testing.assert_raises(RuntimeError):
            backend.update(m, P, A, b, -Q, H, R, y)
        with np.testing.assert_raises(ValueError):
            backend.update(m, P, A, b, Q, H, R, np.full_like(y, np.inf))
        checked_update(update)
        short = list(plan)
        for index in (2, 3, 4, 5, 7, 9):
            short[index] = short[index][:1]
        np.testing.assert_allclose(backend.plan(*short), dense_plan(*short), rtol=1e-9, atol=1e-8)
        empty = list(plan)
        empty[6] = np.zeros_like(plan[6])
        np.testing.assert_allclose(backend.plan(*empty), plan[9][0], atol=1e-8)
        for name, method, args in (("filter", backend.update, update), ("plan", backend.plan, plan)):
            durations = []
            for _ in range(30):
                start = perf_counter()
                method(*args)
                durations.append((perf_counter() - start) * 1000)
            times[name + "_warm_median_ms"] = float(np.median(durations))
            times[name + "_warm_p95_ms"] = float(np.percentile(durations, 95))
        return dict(action=original.tolist(), max_dense_action_error=float(max(abs(original-reference))),
                    state_parameter_cross_cov=float(np.max(abs(first[1][:-1, -1]))), timings=times)
    finally:
        backend.close()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
