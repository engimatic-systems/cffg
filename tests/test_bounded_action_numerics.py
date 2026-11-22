"""Scalar and independent-block mathematical anchors for bounded continuous actions.

The oracle integrates a normalized Gaussian, or unit Lebesgue density for the
explicitly vacuous local message. Only the Gaussian has a prior box probability.
The inputs are finite, and a positive-width result requires distinguishable
bounds, a representable interior mean, and sufficient resolution in both the
variance and the log probability for a 24-epsilon relative comparison.
"""

import math
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import pytest
from scipy.special import erfcx


class Interval(NamedTuple):
    log_probability: float | np.floating | None
    mean: float | np.floating | None
    variance: float | np.floating | None
    entropy: float | np.floating | None


# Independent closed-form references evaluated with 120 decimal digits. For
# standardized bounds a,b, Z = (erfc(a/sqrt(2)) - erfc(b/sqrt(2)))/2.
# m = (phi(a)-phi(b))/Z, s = 1 + (a*phi(a)-b*phi(b))/Z.
# The columns are log Z, m, s-m², log Z + (log(2*pi)+s)/2.
TAIL = Interval(-75.410676444989301, 12.082210753093125,
                0.0066673285815292757, -1.4984959064142887)
FAR_TAIL = Interval(-804.60844201375379, 40.024968847207264,
                    0.00062266837859138626, -2.6901265364038411)
NARROW_TAIL = Interval(-81.238169195973006, 12.000122010707258,
                       4.9670515897430656e-9, -8.3177665243544102)
WIDE_TAIL = Interval(-8197.7710298229701, 128.00781154661659,
                     0.000061012815869709684, -3.8521523063036509)
CENTRAL = Interval(-0.22153785462948558, 0.19430451238634803,
                   0.47183879325666128, 0.95219719697036604)

# Standard Gauss-Legendre rule on [0,1]. Each panel is evaluated in the test dtype.
_nodes, _weights = np.polynomial.legendre.leggauss(24)
NODES, WEIGHTS = (_nodes + 1) / 2, _weights / 2
DTYPES = (np.float32, np.float64)
RELATIVE_ERROR_IN_EPS = 24


def _side(distance, mode_z, dtype):
    """Integrate outwards from the density maximum in standardized coordinates."""
    width = abs(distance)
    if not np.isfinite(width) or not np.isfinite(mode_z) or width == 0:
        raise FloatingPointError("standardized interval is unresolved")
    # In a tail the density changes over a distance of order 1/|mode_z|.
    edges = [dtype(0), min(width, 1 / max(dtype(1), abs(mode_z)))]
    while edges[-1] < width:
        edges.append(min(width, 2 * edges[-1]))
    widths = np.diff(edges)[:, None]
    offsets = np.asarray(edges[:-1])[:, None] + widths * NODES.astype(dtype)
    offsets = np.copysign(offsets, distance)
    # Subtract the mode energy algebraically before rounding either energy.
    penalties = offsets * (mode_z + offsets / 2)
    weights = widths * WEIGHTS.astype(dtype) * np.exp(-penalties)
    return offsets.ravel(), penalties.ravel(), weights.ravel()


def _bounded(precision, information, lower, upper, dtype, *, local_message=False):
    """Test-only quadrature in the scalar cavity's natural parameters."""
    if not all(math.isfinite(value) for value in (precision, information, lower, upper)):
        raise ValueError("nonfinite cavity or bound")
    if precision < 0 or (precision == 0 and (information != 0 or not local_message)):
        raise ValueError("improper Gaussian cavity")
    if upper <= lower:
        return Interval(None if precision == 0 else dtype(-np.inf), None, None, None)
    lo, hi = dtype(lower), dtype(upper)
    if not np.isfinite([lo, hi]).all():
        raise FloatingPointError("finite bounds required in the working dtype")
    if lo >= hi or np.nextafter(lo, hi) == hi:
        raise FloatingPointError("no representable interior")

    # A subnormal's ULP is constant. Below this magnitude one ULP exceeds
    # the relative error budget, even if the rounded result is nonzero.
    resolution_limit = np.finfo(dtype).smallest_normal / RELATIVE_ERROR_IN_EPS
    if precision == 0:
        width = hi - lo
        mean = lo + width / 2
        variance = (width / np.sqrt(dtype(12))) ** 2
        entropy = np.log(width)
        log_probability = None
    else:
        j, h = dtype(precision), dtype(information)
        if not np.isfinite([j, h]).all():
            raise FloatingPointError("natural parameter is unresolved")
        if j == 0 or (information != 0 and h == 0):
            raise FloatingPointError("natural parameter rounded to zero")
        location, scale = h / j, 1 / np.sqrt(j)
        pivot = np.clip(location, lo, hi)
        mode_z = (pivot - location) / scale
        sides = [_side((end - pivot) / scale, mode_z, dtype) for end in (lo, hi) if end != pivot]
        offsets, penalties, weights = (np.concatenate(parts) for parts in zip(*sides))
        integral = weights.sum()
        probabilities = weights / integral
        shift = np.dot(probabilities, offsets)
        mean = pivot + scale * shift
        variance = np.dot(probabilities, ((offsets - shift) * scale) ** 2)
        entropy = np.log(scale * integral) + np.dot(probabilities, penalties)
        log_probability = np.log(integral) - mode_z ** 2 / 2 - dtype(math.log(2 * math.pi) / 2)
        if lo < location < hi:
            # Preserve a representable log probability when the mass rounds to one.
            distances = np.array([location - lo, hi - location]) / scale
            tails = erfcx(distances / np.sqrt(dtype(2))) * np.exp(-distances ** 2 / 2) / 2
            missing = tails.sum()
            if missing < 0.5:
                log_probability = np.log1p(-missing)
        if not np.isfinite(log_probability) or -log_probability < resolution_limit:
            raise FloatingPointError("log probability is unresolved")

    if not lo < mean < hi or not np.isfinite(entropy):
        raise FloatingPointError("conditional mean or entropy is unresolved")
    if not np.isfinite(variance) or variance < resolution_limit:
        raise FloatingPointError("conditional variance is unresolved")
    return Interval(log_probability, mean, variance, entropy)


def _assert_gaussian(actual, reference, dtype):
    assert all(value.dtype == np.dtype(dtype) for value in actual)
    eps = np.finfo(dtype).eps
    # Coordinate rounding is unavoidable at a large offset. A tolerance relative
    # to that offset would also accept the unconditioned mode at the endpoint.
    mean_atol = abs(float(np.spacing(dtype(reference.mean)))) + 4 * eps * math.sqrt(reference.variance)
    assert abs(float(actual.mean) - reference.mean) <= mean_atol
    for measured, expected in ((actual.log_probability, reference.log_probability),
                               (actual.variance, reference.variance),
                               (actual.entropy, reference.entropy)):
        np.testing.assert_allclose(measured, expected, rtol=RELATIVE_ERROR_IN_EPS * eps, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("direction", (1, -1), ids=("right", "left"))
@pytest.mark.parametrize(("a", "b", "reference"), (
    (12, 13, TAIL), (40, 41, FAR_TAIL), (12, 12 + 2 ** -12, NARROW_TAIL),
), ids=("12-sigma", "40-sigma", "narrow-12-sigma"))
def test_tail_mass_centered_moments_and_entropy(dtype, direction, a, b, reference):
    lower, upper = sorted((direction * a, direction * b))
    result = _bounded(1, 0, lower, upper, dtype)
    _assert_gaussian(result, reference._replace(mean=direction * reference.mean), dtype)
    # At width 2^-12 the f32 variance is just five ULPs below the uniform limit.
    assert result.variance < dtype((upper - lower) ** 2 / 12)
    # The entropy and the expected normalized prior energy must also give -log Z.
    prior_energy = (dtype(math.log(2 * math.pi)) + result.mean ** 2 + result.variance) / 2
    np.testing.assert_allclose(prior_energy - result.entropy, -reference.log_probability,
                               rtol=4 * np.finfo(dtype).eps, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_narrow_interval_below_cdf_resolution(dtype):
    width = 2.0 ** -55
    result = _bounded(1, 0, 0, width, dtype)
    # Throughout this interval the density relative to its peak is in
    # [exp(-width²/2),1]. Corrections to the uniform limit are below either epsilon.
    reference = Interval(math.log(width) - math.log(2 * math.pi) / 2, width / 2,
                         width ** 2 / 12, math.log(width))
    _assert_gaussian(result, reference, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_large_location_and_wide_tail_resolve_boundary_layer(dtype):
    location, scale = 2 ** 22, 2 ** 9
    lo, hi = location + 128 * scale, location + 192 * scale
    result = _bounded(1 / scale ** 2, location / scale ** 2, lo, hi, dtype)
    reference = Interval(WIDE_TAIL.log_probability, location + scale * WIDE_TAIL.mean,
                         scale ** 2 * WIDE_TAIL.variance, math.log(scale) + WIDE_TAIL.entropy)
    _assert_gaussian(result, reference, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_near_full_mass_retains_the_excluded_tails(dtype):
    result = _bounded(1, 0, -10, 10, dtype)
    # log1p(-erfc(10/sqrt(2))) at 120 decimal digits.
    reference = Interval(-1.5239706048321052e-23, 0., 1., math.log(2 * math.pi * math.e) / 2)
    _assert_gaussian(result, reference, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vacuous_local_message_is_uniform_on_finite_interval(dtype):
    lower, upper = 5, 13
    result = _bounded(0, 0, lower, upper, dtype, local_message=True)
    assert result.log_probability is None
    # Integral of the unit incoming message is 8. Its normalized density is 1/8.
    assert result.mean == 9
    np.testing.assert_allclose(result.variance, 16 / 3, rtol=4 * np.finfo(dtype).eps)
    np.testing.assert_allclose(result.entropy, math.log(8), rtol=2 * np.finfo(dtype).eps)

    # A weak proper prior can look uniform after truncation but still supplies Z.
    eps = np.finfo(dtype).eps
    weak = _bounded(eps ** 2, 0, 0, 2, dtype)
    _assert_gaussian(weak, Interval(math.log(2 * float(eps)) - math.log(2 * math.pi) / 2,
                                    1., 1 / 3, math.log(2)), dtype)


@pytest.mark.parametrize(("precision", "information"), [(-1, 0), (0, 0), (0, -2), (0, 2)])
def test_improper_physical_cavity_has_no_gaussian_box_probability(precision, information):
    with pytest.raises(ValueError, match="improper Gaussian"):
        _bounded(precision, information, -1, 1, np.float64)


@pytest.mark.parametrize(("precision", "information", "lo", "hi"), (
    (np.nan, 0, 1, 1), (np.inf, 0, 0, 1), (1, np.inf, -1, 1), (1, 0, 0, np.inf),
))
def test_nonfinite_input_cannot_certify_support(precision, information, lo, hi):
    with pytest.raises(ValueError, match="nonfinite"):
        _bounded(precision, information, lo, hi, np.float64)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("lower", "upper"), ((2, 2), (3, 2)), ids=("singleton", "empty"))
def test_zero_measure_has_no_conditioned_action(dtype, lower, upper):
    result = _bounded(1, 0, lower, upper, dtype)
    assert result == Interval(-np.inf, None, None, None)
    local = _bounded(0, 0, lower, upper, dtype, local_message=True)
    assert local == Interval(None, None, None, None)


@pytest.mark.parametrize("dtype", DTYPES)
def test_precision_loss_does_not_classify_support_or_vacuity(dtype):
    info = np.finfo(dtype)
    # Exact integer endpoints, differing before the requested conversion.
    big = 2 ** (info.nmant + 1)
    with pytest.raises(FloatingPointError, match="no representable interior"):
        _bounded(1, 0, big, big + 1, dtype)
    with pytest.raises(FloatingPointError, match="no representable interior"):
        _bounded(1, 0, 1, np.nextafter(dtype(1), dtype(2)), dtype)
    with pytest.raises(FloatingPointError, match="variance"):
        _bounded(1, 0, 0, info.tiny, dtype)
    with pytest.raises(FloatingPointError, match="log probability"):
        _bounded(1, 0, -64, 64, dtype)
    # These positive subnormals carry too few bits for the comparison, even
    # though reporting either zero mass or a collapsed posterior would be false.
    narrow = 16 * np.sqrt(info.smallest_subnormal)
    with pytest.raises(FloatingPointError, match="variance"):
        _bounded(1, 0, 0, narrow, dtype)
    far = 14 if dtype is np.float32 else 38
    assert 0 < dtype(math.erfc(far / math.sqrt(2))) < info.smallest_normal
    with pytest.raises(FloatingPointError, match="log probability"):
        _bounded(1, 0, -far, far, dtype)
    tiny_precision = Fraction.from_float(float(info.smallest_subnormal)) / 2
    with pytest.raises(FloatingPointError, match="parameter rounded to zero"):
        _bounded(tiny_precision, 0, -1, 1, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("horizon", (20, 100))
def test_independent_action_horizon_keeps_additive_free_energy(dtype, horizon):
    # Half the factors have negligible mass, the others have asymmetric central
    # support. The joint probability is below the smallest subnormal even in f64.
    bounds = [(12, 13), (-1, 1.75)] * (horizon // 2)
    blocks = [_bounded(1, 0, lo, hi, dtype) for lo, hi in bounds]
    prior_energy = sum((dtype(math.log(2 * math.pi)) + b.variance + b.mean ** 2) / 2 for b in blocks)
    entropy = np.sum([b.entropy for b in blocks], dtype=dtype)
    joint_log_mass = np.sum([b.log_probability for b in blocks], dtype=dtype)
    expected_log_mass = horizon / 2 * (TAIL.log_probability + CENTRAL.log_probability)
    expected_entropy = horizon / 2 * (TAIL.entropy + CENTRAL.entropy)
    assert expected_log_mass < math.log(float(np.finfo(dtype).smallest_subnormal))
    for actual, expected in ((joint_log_mass, expected_log_mass), (entropy, expected_entropy),
                             (prior_energy - entropy, -expected_log_mass)):
        np.testing.assert_allclose(actual, expected, rtol=RELATIVE_ERROR_IN_EPS * np.finfo(dtype).eps, atol=0)
