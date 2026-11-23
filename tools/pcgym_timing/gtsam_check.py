"""Compare the native GTSAM factor and solver to the fixture and affine oracle."""

import json
import time

import gtsam
import jax
import jax.numpy as jnp
import numpy as np

import cstr_runtime_fixture as fixture
from _gtsam_cstr import Reactor
from gtsam_adapter import Adapter
from qualification import qualify


def check_transition():
    # Test the physical interior, clipped inputs, clip ties and nonzero disturbances.
    ordinary = np.array([
        [0., 0., 0., 0.], [-.7, .3, -.02, -.95], [.2, -.8, .13, .9],
        [-5.1, -.7, .15, -8.], [5.2, -.7, -.08, 8.],
        [.25, -5.2, .1, .5], [-5., .1, -.01, 0.],
        [.1, -5., .01, (302. + 20. - 295.) * 2. / 7. - 1.],
    ])
    # At the upper temperature guard the fixture's unguarded RK4 stages run away.
    # Retain this comparison and report its magnitude separately.
    states = np.vstack((ordinary, [-1.1, 5.1, .08, .75]))

    def step(s, theta):
        return fixture.cstr_rk4(s, theta, fixture.CONFIGURATION)
    reference = jax.jit(jax.vmap(lambda s, t: (step(s, t), jax.jacfwd(step)(s, t)),
                                  in_axes=(0, None)))
    groups = {name: dict(value_abs=0., jacobian_abs=0., value_scaled=0., jacobian_scaled=0.)
              for name in ('ordinary_and_clipped', 'runaway')}
    for theta in (fixture.PARAMETERS, fixture.PARAMETERS + np.array([.06, .015, -.03, .02])):
        predicted, jacobians = map(np.asarray, reference(jnp.asarray(states), jnp.asarray(theta)))
        native = Reactor(theta, fixture.CONFIGURATION).factor(0, gtsam.noiseModel.Unit.Create(4))
        for i, (s, target, jac) in enumerate(zip(states, predicted, jacobians, strict=True)):
            values = gtsam.Values()
            values.insert(0, s)
            values.insert(1, np.zeros(4))
            forward = -native.unwhitenedError(values)
            linear = native.linearize(values)
            np.testing.assert_allclose(forward, target, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(linear.getb(), target, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(linear.getA()[:, :4], -jac, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(linear.getA()[:, 4:], np.eye(4))
            group = groups['ordinary_and_clipped' if i < len(ordinary) else 'runaway']
            for key, delta, expected in [('value', forward - target, target),
                                         ('jacobian', linear.getA()[:, :4] + jac, jac)]:
                absolute = np.abs(delta)
                scaled = absolute / np.maximum(1., np.abs(expected))
                group[key + '_abs'] = max(group[key + '_abs'], float(absolute.max()))
                group[key + '_scaled'] = max(group[key + '_scaled'], float(scaled.max()))
    return dict(states=len(states), theta_vectors=2, **groups)


def check_install(arrays):
    # A nondefault reaction rate detects resets of parameters other than UA.
    theta = arrays["theta"].copy()
    theta[0] += .01
    model = {**arrays, "theta": theta.copy()}
    data = fixture.observations()
    adapter = Adapter(model, fixture.CONFIGURATION)
    ua = np.exp(theta[2])
    changed_ua = 1.03 * ua

    def posterior(planner):
        result = planner.solve(data, details=True)
        assert result["status"] == "ok", result
        return np.r_[result["action"], result["mean"], result["cov"].ravel(), result["next_action_cov"]]

    baseline = posterior(adapter)
    adapter.install(changed_ua)
    changed = posterior(adapter)
    rebuilt_theta = theta.copy()
    rebuilt_theta[2] = np.log(changed_ua)
    rebuilt = posterior(Adapter({**model, "theta": rebuilt_theta}, fixture.CONFIGURATION))
    np.testing.assert_allclose(changed, rebuilt, rtol=1e-12, atol=1e-12)
    assert abs(changed[0] - baseline[0]) > 1e-4
    for invalid in (0., -1., np.inf, np.nan, [ua]):
        with np.testing.assert_raises(ValueError):
            adapter.install(invalid)
    np.testing.assert_array_equal(posterior(adapter), changed)
    adapter.install(ua)
    restored = posterior(adapter)
    np.testing.assert_allclose(restored, baseline, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(model["theta"], theta)

    def installed_affine(arrays, config, *, affine):
        planner = Adapter(arrays, config, affine=affine)
        planner.install(changed_ua)
        return planner

    affine = qualify(installed_affine, arrays)
    assert all(case["passed"] for case in affine), affine
    return dict(ua=float(ua), changed_ua=float(changed_ua), baseline_action=baseline[0],
                changed_action=changed[0], rebuilt_error=float(np.max(abs(changed - rebuilt))),
                restored_error=float(np.max(abs(restored - baseline))), affine=affine)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    arrays = fixture.model()[1]
    affine = qualify(Adapter, arrays)
    assert all(case["passed"] for case in affine), affine
    adapter = Adapter(arrays, fixture.CONFIGURATION)
    planned = fixture.observations(2, history=True, goal=.3)
    filtered = {key: value if int(key[1:]) <= fixture.PAST else None
                for key, value in planned.items()}
    real = {}
    for name, data in [("fresh", fixture.observations()), ("changed_goal", planned),
                       ("filter", filtered), ("fresh_again", fixture.observations())]:
        start = time.perf_counter()
        answer = adapter.solve(data, details=True)
        elapsed = time.perf_counter() - start
        assert answer["status"] == "ok", answer
        np.linalg.cholesky(answer["cov"])
        real[name] = {k: v for k, v in answer.items() if k not in ("mean", "cov")}
        real[name]["details_s"] = elapsed
    assert real["fresh"]["action"] == real["fresh_again"]["action"]
    print(json.dumps(dict(affine=affine, real=real, physics=check_transition(),
                          install=check_install(arrays)), indent=2))
