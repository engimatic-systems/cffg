"""Check live UA delivery and JIT reuse on a synthetic NumPyro planning window."""
import json

import jax
import numpy as np

import cstr_runtime_fixture as fixture
from numpyro_adapter import Adapter
from qualification import qualify


def checked_solve(adapter, data):
    result = adapter.solve(data, details=True)
    if result["status"] != "ok":
        raise AssertionError(result)
    return result


def compare(actual, expected):
    errors = {}
    for name in ("action", "mean", "cov", "next_action_cov", "objective"):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-8, rtol=1e-10)
        errors[name] = float(np.max(np.abs(np.subtract(actual[name], expected[name]))))
    return errors


def check():
    _, arrays = fixture.model()
    original_theta = arrays["theta"].copy()
    changed_ua = 48150.
    data = fixture.observations()
    live = Adapter(arrays, fixture.CONFIGURATION)
    first = checked_solve(live, data)

    # A warmed solve must accept each coefficient without tracing any new JIT.
    with jax.no_tracing():
        live.install(changed_ua)
        changed = checked_solve(live, data)
        live.install(50000.)
        restored = checked_solve(live, data)
        live.install(changed_ua)
        repeated = checked_solve(live, data)

    fresh_theta = original_theta.copy()
    fresh_theta[2] = np.log(changed_ua)
    rebuilt = Adapter({**arrays, "theta": fresh_theta}, fixture.CONFIGURATION)
    equivalent = checked_solve(rebuilt, data)
    np.testing.assert_array_equal(arrays["theta"], original_theta)
    if abs(changed["action"] - first["action"]) < 1e-4:
        raise AssertionError("UA change did not affect the proposed action")

    for value in (0., -1., np.nan, np.inf, [50000.]):
        try:
            live.install(value)
        except ValueError:
            continue
        raise AssertionError(f"Invalid UA accepted: {value}")
    invalid_input_recovery = checked_solve(live, data)

    affine = qualify(Adapter, arrays)
    if not all(case["passed"] for case in affine):
        raise AssertionError(affine)
    return dict(nominal_action=first["action"], changed_action=changed["action"],
                fresh_errors=compare(changed, equivalent), restored_errors=compare(restored, first),
                repeated_errors=compare(repeated, changed), after_invalid_errors=compare(invalid_input_recovery, changed),
                affine=affine)


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
