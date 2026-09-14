"""Exact-input regressions for strong-cavity assembly and Gaussian inversion."""

from decimal import Decimal, localcontext
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_covariance as hc, hybrid_natural as hn, hybrid_separator as hs
from hybrid_kernel_cases import tolerance
from hybrid_joint_reference import Model, Natural, evaluate as reference_evaluate


FIXTURE = json.loads((Path(__file__).parent / "fixtures/hybrid-strong-cavity.json").read_text())
AUXILIARY = jax.jit(hs.auxiliary_statistics)


def exact_scalar(xi, source=0):
    """Decimal80 scalar integration, projection and cancellation-free quotient.

    J_out = (a² + J(a-b))/(J+a+b), where a=omega*w² and
    b=w²*p*(1-p). The input authority is the stored site precision, not
    a covariance rounded by a prior normalization or an inverse readout.
    """
    D = Decimal.from_float
    with localcontext() as ctx:
        ctx.prec = 80
        J = D(FIXTURE["sites"]["precision"][0][source][0][0])
        w = D(FIXTURE["parameters"]["weights"][0][source][0][0])
        offset = D(FIXTURE["parameters"]["offsets"][0][source][0])
        offset += w * D(FIXTURE["sites"]["center"][0][source][0])
        x = D(float(xi))
        c = x.sqrt()
        omega = Decimal(1)/4 if x == 0 else (c.exp()-1)/(2*c*(c.exp()+1))
        a = omega*w*w
        L = J+a
        p = 1/(1+(-offset*J/L).exp())
        b = w*w*p*(1-p)
        value = (a*a+J*(a-b))/(J+a+b)
        Lp = -w*w/48
        pp = p*(1-p)*(-offset*J*Lp/L**2)
        bp = w*w*(1-2*p)*pp
        derivative = ((L*L+2*L*b)*Lp-L*L*bp)/(L+b)**2
        return float(value), float(derivative), 1/L


def inputs(dtype):
    parameters = hs.TransitionParameters(*(jnp.asarray(FIXTURE["parameters"][k], dtype)
                                          for k in hs.TransitionParameters._fields))
    values = FIXTURE["sites"]
    sites = hn.Natural(*(jnp.asarray(values[k], dtype) for k in ("constant", "linear", "precision")),
                        jnp.asarray(values["status"], jnp.int32), jnp.asarray(values["center"], dtype))
    return parameters, sites


def unpack(vector, sites):
    constants = jnp.zeros((2, 2), vector.dtype).at[:, 1].set(vector[:2])
    linear = vector[2:6].reshape(2, 2, 1)
    precision = vector[6:10].reshape(2, 2, 1, 1)
    return tuple(hn.Natural(constants[i:i+1], linear[i:i+1], precision[i:i+1],
                            jnp.zeros(1, jnp.int32), sites.center[i:i+1]) for i in (0, 1))


def held(vector, parameters, sites):
    parameters = parameters._replace(xi=vector[10:].reshape(1, 2, 1))
    return hs.evaluate(parameters, sites, unpack(vector, sites))


def source_precision(vector, parameters, sites):
    """The T=2 region has no neighboring message in either exterior cavity."""
    parameter = jax.tree.map(lambda x: x[0], parameters)._replace(xi=vector[10:].reshape(2, 1))
    left, right = (jax.tree.map(lambda x: x[i], sites) for i in (0, 1))
    factor = hs.transition(*parameter, left_center=left.origin, right_center=right.origin)
    region = hn.normalize(factor+hn.lift(left, right))
    return hn.quotient(hn.marginal(region, 0), left).precision[:, 0, 0]


def reference(vector):
    p, s = FIXTURE["parameters"], FIXTURE["sites"]
    arrays = {k: np.array(v, dtype=float) for k, v in p.items()}
    model = Model([Natural(np.array(a), np.array(h), np.array(j)) for a, h, j in
                   zip(s["constant"], s["linear"], s["precision"])], np.array(s["center"]),
                  *(arrays[k] for k in ("weights", "offsets", "matrix", "bias", "noise")))
    constants = np.zeros((2, 2))
    constants[:, 1] = vector[:2]
    messages = [[Natural(constants[i], vector[2:6].reshape(2, 2, 1)[i],
                          vector[6:10].reshape(2, 2, 1, 1)[i]) for i in (0, 1)]]
    return reference_evaluate(model, messages, vector[10:].reshape(1, 2, 1))


def check_held(result, vector, dtype):
    hn.raise_for_status(result.status)
    expected = [exact_scalar(vector[10+i], i)[0] for i in range(2)]
    np.testing.assert_allclose(result.targets[0].precision[0, :, 0, 0], expected,
                               atol=tolerance(dtype), rtol=tolerance(dtype))
    ref = reference(np.asarray(vector, float))
    for energy in ("free_energy", "dual_energy"):
        np.testing.assert_allclose(getattr(result, energy), ref[energy], atol=tolerance(dtype, 10), rtol=0)
    for name in ("regions", "separators"):
        actual = getattr(result, name)
        for field in ("probability", "displacement", "full_covariance"):
            field_ref = "covariance" if field == "full_covariance" else field
            np.testing.assert_allclose(getattr(actual, field), [getattr(b, field_ref) for b in ref[name]],
                                       atol=tolerance(dtype), rtol=tolerance(dtype))
    params, _ = inputs(dtype)
    region = jax.tree.map(lambda x: x[0], result.regions)
    aux = AUXILIARY(region, params.weights[0], params.offsets[0],
                    jnp.asarray(vector[10:], dtype).reshape(2, 1))
    hn.raise_for_status(aux.status)
    np.testing.assert_allclose(aux.target, ref["auxiliary_target"][0],
                               atol=tolerance(dtype), rtol=tolerance(dtype))
    for field, name in (("exposure", "exposure"), ("linear", "centered_linear"),
                        ("precision", "centered_precision")):
        np.testing.assert_allclose(getattr(aux, field), ref["learning"][0][name],
                                   atol=tolerance(dtype), rtol=tolerance(dtype))
    assert all(np.isfinite(np.asarray(x)).all() for x in aux[:5])
    assert result.status.dtype == jnp.int32


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_original_precision_failure_and_frozen_held_grid(dtype):
    """Host/JIT and single/vmap values use exact exported coordinates."""
    parameters, sites = inputs(dtype)
    vectors = np.array([FIXTURE["x"]] + [p["coordinate"] for p in FIXTURE["grid"][np.dtype(dtype).name]], dtype)
    value = lambda vector: held(vector, parameters, sites)
    # Preserve the failed evidence independently of the now-corrected implementation.
    truth = exact_scalar(0)[0]
    old = FIXTURE["failed_base_precision"][np.dtype(dtype).name]
    assert abs(old-truth) > tolerance(dtype)*(1+abs(truth))
    results = jax.jit(jax.vmap(value))(jnp.asarray(vectors))
    for i, vector in enumerate(vectors):
        check_held(jax.tree.map(lambda x: x[i], results), vector, dtype)
    precision = lambda vector: source_precision(vector, parameters, sites)
    expected = np.array([[exact_scalar(vector[10+i], i)[0] for i in (0, 1)] for vector in vectors])
    for call in (precision, jax.jit(precision)):
        np.testing.assert_allclose(call(jnp.asarray(vectors[0])), expected[0],
                                   atol=tolerance(dtype), rtol=tolerance(dtype))
    for call in (jax.vmap(precision), jax.jit(jax.vmap(precision))):
        np.testing.assert_allclose(call(jnp.asarray(vectors)), expected,
                                   atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_held_precision_derivative_uses_analytic_exact_input_reference(dtype):
    parameters, sites = inputs(dtype)
    point = jnp.asarray(FIXTURE["x"], dtype)
    direction = jnp.asarray(FIXTURE["v"], dtype)*jnp.asarray(FIXTURE["coordinate_scales"], dtype)
    def residual(vector):
        return source_precision(vector, parameters, sites)-vector[6:8]
    expected = np.array([exact_scalar(0, i)[1]*float(direction[10+i])-float(direction[6+i]) for i in (0, 1)])
    def derivative(x, v):
        return jax.jvp(residual, (x,), (v,))[1]
    for call in (derivative, jax.jit(derivative)):
        np.testing.assert_allclose(call(point, direction), expected,
                                   atol=tolerance(dtype), rtol=tolerance(dtype))
    actual = jax.jit(jax.vmap(derivative))(jnp.stack([point, point]), jnp.stack([direction, direction]))
    np.testing.assert_allclose(actual, np.broadcast_to(expected, (2, 2)),
                               atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_larger_strong_cavity_preserves_signed_assembly_inverse_and_readers(dtype):
    dim = 8
    eye = jnp.eye(dim, dtype=dtype)
    cavity = hn.Natural(jnp.zeros((), dtype), jnp.zeros(dim, dtype), eye*10000)
    increment = hn.Natural(jnp.zeros((), dtype), jnp.zeros(dim, dtype),
                           eye*jnp.asarray(-0.0125764053902527907, dtype))
    aggregate = cavity+increment
    assert np.any(np.asarray(aggregate.correction) != 0)
    for run in (hn.normalize, jax.jit(hn.normalize)):
        belief = run(aggregate)
        hn.raise_for_status(belief.status)
        target = hn.quotient(belief, cavity)
        np.testing.assert_allclose(target.precision, increment.precision,
                                   atol=tolerance(dtype), rtol=tolerance(dtype))
        # A second canonical round trip transports the derived precision word.
        canonical = hn.canonical(belief)
        repeated = hn.quotient(hn.normalize(canonical), cavity)
        np.testing.assert_allclose(repeated.precision, increment.precision,
                                   atol=tolerance(dtype), rtol=tolerance(dtype))
        assert float(belief.expectation(aggregate)) == pytest.approx(-dim/2, abs=tolerance(dtype))
        assert np.isfinite(float(belief.entropy()))
        shifted = hn.rebase(aggregate, jnp.full(dim, 0.01, dtype))
        np.testing.assert_array_equal(shifted.correction, aggregate.correction)
        np.testing.assert_allclose(hn.normalize(shifted).mean, belief.mean,
                                   atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_precision_remainder_reaches_lift_adapters_and_derivatives(dtype):
    eye = jnp.eye(1, dtype=dtype)
    cavity = hn.Natural(jnp.zeros((), dtype), jnp.zeros(1, dtype), eye*10000)
    small = hn.Natural(cavity.constant, cavity.linear, eye*jnp.asarray(-0.0125764053902527907, dtype))
    ordinary = cavity+small
    hybrid = hn.Natural(jnp.log(jnp.asarray([.4, .6], dtype)), jnp.zeros((2, 1), dtype),
                        jnp.zeros((2, 1, 1), dtype))
    for call in (hn.gaussian_adapter, jax.jit(hn.gaussian_adapter)):
        belief, _, target = call(hybrid, ordinary)
        hn.raise_for_status(belief.status | target.status)
        np.testing.assert_allclose(target.precision, 0, atol=tolerance(dtype), rtol=0)
    labeled = ordinary._replace(constant=jnp.broadcast_to(ordinary.constant, (2,)),
                                linear=jnp.broadcast_to(ordinary.linear, (2, 1)),
                                precision=jnp.broadcast_to(ordinary.precision, (2, 1, 1)),
                                precision_correction=jnp.broadcast_to(ordinary.correction, (2, 1, 1)))
    lifted = hn.lift(labeled, labeled)
    np.testing.assert_array_equal(lifted.correction[..., 0, 0],
                                  jnp.broadcast_to(ordinary.correction[0, 0], (2, 2)))
    def inverse_trace(shift):
        high, low = hc.inverse(ordinary.precision+eye*shift, ordinary.correction)
        return (high+low).sum()
    precision = float(ordinary.precision[0, 0])+float(ordinary.correction[0, 0])
    for derivative, expected in ((jax.grad(inverse_trace), -1/precision**2),
                                  (jax.grad(jax.grad(inverse_trace)), 2/precision**3)):
        np.testing.assert_allclose(jax.jit(derivative)(jnp.zeros((), dtype)), expected,
                                   atol=0, rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_malformed_precision_remainder_fails_before_normalization(dtype):
    potential = hn.Natural(jnp.zeros((), dtype), jnp.zeros(1, dtype), jnp.ones((1, 1), dtype))
    for correction in (potential.precision/2, potential.precision*jnp.nan):
        bad = potential._replace(precision_correction=correction)
        for call in (hn.normalize, jax.jit(hn.normalize)):
            result = call(bad)
            assert int(result.status) & hn.Status.INVALID
            assert np.isnan(result.covariance).all()
    with pytest.raises(ValueError, match="precision correction shapes or dtypes"):
        hn.normalize(potential._replace(precision_correction=jnp.zeros(2, dtype)))
    left = potential._replace(precision_correction=potential.precision/2)
    right = potential._replace(precision_correction=-potential.precision/2)
    for call in (lambda a, b: hn.normalize(a+b), jax.jit(lambda a, b: hn.normalize(a+b))):
        assert int(call(left, right).status) & hn.Status.INVALID


def test_original_precision_values_with_x64_disabled():
    program = textwrap.dedent('''
        import sys
        sys.path.insert(0, 'tests')
        import jax
        jax.config.update('jax_enable_x64', False)
        import numpy as np
        from test_hybrid_precision import test_original_precision_failure_and_frozen_held_grid
        assert not jax.config.x64_enabled
        test_original_precision_failure_and_frozen_held_grid(np.float32)
    ''')
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_inverse_derivatives_and_sparse_status_with_x64_disabled():
    program = textwrap.dedent('''
        import sys
        sys.path.insert(0, 'tests')
        import jax
        jax.config.update('jax_enable_x64', False)
        import jax.numpy as jnp
        import numpy as np
        from cffg import hybrid_natural as hn
        from test_hybrid_precision import (
            test_held_precision_derivative_uses_analytic_exact_input_reference,
            test_precision_remainder_reaches_lift_adapters_and_derivatives)
        test_held_precision_derivative_uses_analytic_exact_input_reference(np.float32)
        test_precision_remainder_reaches_lift_adapters_and_derivatives(np.float32)
        cavity = hn.Natural(jnp.array([0., -jnp.inf]), jnp.zeros((2, 1)),
                            jnp.full((2, 1, 1), 10000.))
        small = hn.Natural(jnp.zeros(2), jnp.zeros((2, 1)), jnp.full((2, 1, 1), .0125))
        for call in (hn.normalize, jax.jit(hn.normalize)):
            sparse = call(cavity+small)
            hn.raise_for_status(sparse.status)
            assert sparse.status.dtype == jnp.int32
            assert sparse.probability[1] == 0 and sparse.correction[1, 0, 0] == 0
            empty = call(cavity._replace(constant=jnp.full(2, -jnp.inf))+small)
            assert int(empty.status) & hn.Status.EMPTY_SUPPORT
            tiny = call(cavity._replace(constant=jnp.array([0., -1000.]))+small)
            assert int(tiny.status) & hn.Status.UNDERFLOW
            assert np.isfinite(tiny.log_probability[1])
    ''')
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def resolution_cases(dtype, addition):
    """Exact 2x2 eigenvalues distinguish strict failure from interval overlap."""
    eps = np.finfo(dtype).eps
    if addition:
        high = np.array([[1, 1-128*eps], [1-128*eps, 1]], dtype)
        low = -np.eye(2, dtype=dtype)*(16384*eps**2)
        minimum = Decimal(float(high[0, 0]))+Decimal(float(low[0, 0]))-Decimal(float(high[0, 1]))
        maximum = Decimal(float(high[0, 0]))+Decimal(float(low[0, 0]))+Decimal(float(high[0, 1]))
    else:
        above = np.nextafter(dtype(64*eps), dtype(np.inf))
        high = np.diag(np.array([above, 1], dtype))
        low = np.diag(np.array([-eps*above, eps], dtype))
        minimum = Decimal(float(high[0, 0]))+Decimal(float(low[0, 0]))
        maximum = Decimal(float(high[1, 1]))+Decimal(float(low[1, 1]))
    assert minimum > 0
    assert minimum-64*Decimal(float(eps))*maximum < 0
    valid = np.diag(np.array([128*eps, 1], dtype))
    repeated = np.eye(2, dtype=dtype)
    return jnp.asarray(np.stack([high, valid, repeated])), jnp.asarray(np.stack([
        low, np.zeros_like(low), repeated*(eps/4)]))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_represented_spectral_boundary_is_not_certified_from_its_high_word(dtype):
    with localcontext() as context:
        context.prec = 100
        for addition in (False, True):
            high, low = resolution_cases(dtype, addition)
            def normalize(matrix, correction):
                potential = hn.Natural(jnp.zeros((), dtype), jnp.zeros(2, dtype), matrix)
                if addition:
                    potential += potential._replace(precision=correction)
                else:
                    potential = potential._replace(precision_correction=correction)
                hn_low_status = hn.correction_status(potential.precision, potential.correction)
                belief = hn.normalize(potential)
                return belief.status, belief.log_normalizer, hn_low_status
            for call in (normalize, jax.jit(normalize)):
                results = [call(h, l) for h, l in zip(high, low)]
                assert int(results[0][0]) & hn.Status.NONFINITE
                assert not int(results[0][0]) & hn.Status.IMPROPER
                assert np.isnan(results[0][1])
                assert all(int(r[2]) == 0 for r in results)
                assert all(int(r[0]) == 0 and np.isfinite(r[1]) for r in results[1:])
            for call in (jax.vmap(normalize), jax.jit(jax.vmap(normalize))):
                status, logz, low_status = call(high, low)
                np.testing.assert_array_equal(status, [hn.Status.NONFINITE, 0, 0])
                assert status.dtype == jnp.int32 and np.isnan(logz[0])
                np.testing.assert_array_equal(low_status, 0)
    # A dense matrix with twice the required spectral margin stays resolvable.
    eps = np.finfo(dtype).eps
    dense = jnp.asarray([[1, 1-256*eps], [1-256*eps, 1]], dtype)
    assert bool(jax.jit(hc.resolved)(dense, -jnp.eye(2, dtype=dtype)*(16384*eps**2)))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_resolution_certificate_preserves_declared_dense_condition_controls(dtype):
    rng = np.random.default_rng(8117)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    u = rng.normal(size=(4, 4))
    increment = ((u+u.T)*.006).astype(dtype)
    matrices = np.array([q @ np.diag(np.geomspace(10000/c, 10000, 4)) @ q.T
                         for c in (3, 30, 300, 3000)], dtype)
    matrices = (matrices+matrices.swapaxes(-1, -2))/2
    def round_trip(matrix):
        cavity = hn.Natural(jnp.zeros((), dtype), jnp.zeros(4, dtype), matrix)
        belief = hn.normalize(cavity+cavity._replace(precision=jnp.asarray(increment)))
        return hn.quotient(belief, cavity)
    result = jax.jit(jax.vmap(round_trip))(jnp.asarray(matrices))
    hn.raise_for_status(result.status)
    np.testing.assert_allclose(result.precision, np.broadcast_to(increment, matrices.shape),
                               atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_repeated_eigenvalue_status_does_not_enter_inverse_derivatives(dtype):
    eye = jnp.eye(2, dtype=dtype)
    low = eye*(np.finfo(dtype).eps/4)
    def covariance(shift):
        potential = hn.Natural(jnp.zeros((), dtype), jnp.zeros(2, dtype), eye*(1+shift),
                               precision_correction=low)
        return hn.normalize(potential).full_covariance.sum()
    precision = 1+float(low[0, 0])
    for differentiate, expected in ((jax.grad, -2/precision**2),
                                    (lambda f: jax.grad(jax.grad(f)), 4/precision**3)):
        np.testing.assert_allclose(jax.jit(differentiate(covariance))(jnp.zeros((), dtype)),
                                   expected, atol=0, rtol=tolerance(dtype))


def test_represented_resolution_boundary_with_x64_disabled():
    program = textwrap.dedent('''
        import sys
        sys.path.insert(0, 'tests')
        import jax
        jax.config.update('jax_enable_x64', False)
        import numpy as np
        from test_hybrid_precision import (
            test_represented_spectral_boundary_is_not_certified_from_its_high_word,
            test_resolution_certificate_preserves_declared_dense_condition_controls,
            test_repeated_eigenvalue_status_does_not_enter_inverse_derivatives)
        test_represented_spectral_boundary_is_not_certified_from_its_high_word(np.float32)
        test_resolution_certificate_preserves_declared_dense_condition_controls(np.float32)
        test_repeated_eigenvalue_status_does_not_enter_inverse_derivatives(np.float32)
    ''')
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
