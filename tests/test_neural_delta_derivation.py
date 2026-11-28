"""Independent D13a oracle for a separable, exact two-slope neural activation.

All integrals use lengths in the input coordinate. Within an affine segment
a uniform density pushes to another uniform density, divided by the slope.
An output interval likelihood pulls back by moving its two endpoints.
"""

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass(frozen=True)
class Unit:
    left: float
    right: float
    knot: float = 0.0
    offset: float = 0.0

    def __post_init__(self):
        if not np.isfinite([self.left, self.right, self.knot, self.offset]).all():
            raise ValueError("parameters must be finite")
        if min(self.left, self.right) <= 0:
            raise ValueError("both slopes must be strictly positive")

    def forward(self, x):
        step = x - self.knot
        return self.offset + step * np.where(step < 0, self.left, self.right)

    def inverse(self, y):
        step = y - self.offset
        return self.knot + step / np.where(step < 0, self.left, self.right)


def interval_tree(unit, prior, evidence, dtype):
    """Uniform input and unit-height output interval, each optionally vacuous.

    Return signed node/edge terms and the piecewise uniform output density.
    Z is the integral of these declared potentials. With a vacuous prior it
    has the normalization convention of the unit input potential.
    """
    if prior is None and evidence is None:
        return {"status": "improper"}
    for interval in (prior, evidence):
        if interval is not None:
            bounds = np.asarray(interval, dtype=dtype)
            if not np.isfinite(bounds).all() or bounds.shape != (2,) or bounds[0] >= bounds[1]:
                raise ValueError("interval endpoints must be finite and ordered at working precision")
    low, high = prior if prior is not None else (-np.inf, np.inf)
    if evidence is not None:
        a, b = unit.inverse(np.asarray(evidence, dtype=dtype))
        low, high = max(low, a), min(high, b)
    if low >= high:
        return {"status": "infeasible", "free_energy": np.inf}
    cuts = [low, high]
    if low < unit.knot < high:
        cuts.insert(1, unit.knot)
    x = np.asarray(cuts, dtype=dtype)
    y = unit.forward(x)
    widths = np.diff(x)
    output_widths = np.diff(y)
    if not np.isfinite(y).all() or np.any(output_widths <= 0):
        raise ValueError("transformed interval is unresolved at working precision")
    length = widths.sum()
    masses = widths / length
    density = masses / output_widths
    hx = np.log(length)
    hy = -masses @ np.log(density)
    prior_width = dtype(1) if prior is None else dtype(prior[1]) - dtype(prior[0])
    terms = np.array([np.log(prior_width) - hx, -hx, -hy, hx, hy], dtype=dtype)
    # Conditional first and second moments of a uniform interval.
    a, b = y[:-1], y[1:]
    mean = masses @ ((a + b) / 2)
    variance = masses @ (output_widths ** 2 / 12 + ((a + b) / 2 - mean) ** 2)
    return {"status": "proper", "free_energy": terms.sum(), "terms": terms,
            "breaks": y, "density": density, "mean": mean, "variance": variance}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bent_neural_map_transports_density_and_charges_the_tree_once(dtype):
    unit = Unit(dtype(0.5), dtype(2))
    base = interval_tree(unit, (-2, 2), None, dtype)
    assert base["status"] == "proper"
    tol = 32 * np.finfo(dtype).eps
    np.testing.assert_allclose(base["breaks"], [-1, 0, 4], atol=tol, rtol=0)
    np.testing.assert_allclose(base["density"], [0.5, 0.125], atol=tol, rtol=0)
    np.testing.assert_allclose([base["mean"], base["variance"], base["free_energy"]],
                               [0.75, 109/48, 0], atol=tol, rtol=tol)
    # Independently integrate the previously calculated output densities over
    # the two observed pieces: half a unit on the left, one on the right.
    z = 0.5 * 0.5 + 1.0 * 0.125
    observed = interval_tree(unit, (-2, 2), (-0.5, 1), dtype)
    hx, hy = np.log(1.5), np.log(1.5) - np.log(2) / 3
    np.testing.assert_allclose(observed["terms"], [np.log(4)-hx, -hx, -hy, hx, hy],
                               atol=tol, rtol=tol)
    np.testing.assert_allclose(observed["free_energy"], -np.log(z), atol=tol, rtol=tol)
    assert observed["terms"].dtype == dtype


def test_support_vacuity_and_affine_special_case():
    unit = Unit(2., 2., 1., -3.)
    result = interval_tree(unit, (0, 4), None, np.float64)
    np.testing.assert_allclose(result["density"], 1/8)
    np.testing.assert_allclose([result["mean"], result["variance"]], [-1, 16/3])
    flat_input = interval_tree(unit, None, (-3, 1), np.float64)
    np.testing.assert_allclose(flat_input["free_energy"], -np.log(2))
    assert interval_tree(unit, None, None, np.float64)["status"] == "improper"
    # Touching endpoints occupy zero Lebesgue mass.
    for window in ((3, 4), (4, 5)):
        impossible = interval_tree(unit, (0, 4), window, np.float64)
        assert impossible == {"status": "infeasible", "free_energy": np.inf}
    for slope in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            Unit(1., slope)
    with pytest.raises(ValueError, match="unresolved"):
        interval_tree(Unit(np.float32(1), np.float32(1), offset=np.float32(2**25)),
                      (0, 1), None, np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_product_support_at_32_dimensions(dtype):
    # Each coordinate has a different map and translation. An equivalent
    # Cartesian-grid enumeration would have at least 2**32 affine regions.
    values = []
    for i in range(32):
        unit = Unit(dtype(0.5 + i/16), dtype(2 + i/8), dtype(i), dtype(-i))
        prior = (i-2, i+2)
        window = (-i - float(unit.left), -i + float(unit.right)/2)
        result = interval_tree(unit, prior, window, dtype)
        assert result["status"] == "proper"
        values.append(result["free_energy"])
    atol = 64 * np.finfo(dtype).eps * len(values)
    np.testing.assert_allclose(np.sum(values, dtype=dtype), -32 * np.log(3/8), atol=atol, rtol=0)
