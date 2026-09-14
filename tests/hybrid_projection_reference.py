"""Independent stored-region projection, using 70-digit decimal arithmetic.

No production mathematics is imported. Tiny matrix elimination avoids rounding
an inverse to float64 before subtracting a large cavity precision.
"""

from fractions import Fraction

import numpy as np
from decimal import Decimal, getcontext


getcontext().prec = 70
number = Decimal


def array(value):
    return np.vectorize(lambda x: x if isinstance(x, Decimal) else Decimal(float(x)), otypes=[object])(value)


def log(value):
    return np.vectorize(lambda x: x.ln(), otypes=[object])(array(value))


def inverse_logdet(matrix):
    matrix = array(matrix).copy()
    dim = matrix.shape[-1]
    inverse = np.broadcast_to(array(np.eye(dim)), matrix.shape).copy()
    logdet = array(np.zeros(matrix.shape[:-2]))
    for k in range(dim):
        pivot = matrix[..., k, k].copy()
        assert np.all(pivot > 0)
        logdet += log(pivot)
        matrix[..., k, :] /= pivot[..., None]
        inverse[..., k, :] /= pivot[..., None]
        for row in range(dim):
            if row != k:
                multiplier = matrix[..., row, k].copy()
                matrix[..., row, :] -= multiplier[..., None] * matrix[..., k, :]
                inverse[..., row, :] -= multiplier[..., None] * inverse[..., k, :]
    return inverse, logdet


def project(probability, mean, covariance, axis):
    p, m, c = (array(x) for x in (probability, mean, covariance))
    mass = p.sum(axis=axis)
    weights = p / np.expand_dims(mass, axis)
    projected_mean = (weights[..., None] * m).sum(axis=axis)
    delta = m - np.expand_dims(projected_mean, axis)
    projected_covariance = (weights[..., None, None] *
                            (c + delta[..., :, None] * delta[..., None, :])).sum(axis=axis)
    return mass, projected_mean, projected_covariance


def canonical(probability, mean, covariance):
    precision, logdet = inverse_logdet(covariance)
    linear = (precision @ mean[..., None])[..., 0]
    constant = log(probability) - (logdet + mean.shape[-1] * log(2 * Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164"))
                                     + (mean * linear).sum(axis=-1)) / 2
    return constant, linear, precision


def quotient(moments, cavity):
    result = [a - array(b) for a, b in zip(canonical(*moments), cavity)]
    result[0] -= np.asarray(result[0]).flat[0]
    return result


def scalar_precision(probability, mean, covariance, cavity_precision):
    """Exact rational source projection from the stored scalar pair region."""
    results = []
    for probabilities, means, covariances, precision in zip(probability, mean, covariance, cavity_precision):
        p = [Fraction(float(x)) for x in probabilities]
        m = [Fraction(float(x)) for x in np.asarray(means).ravel()]
        c = [Fraction(float(x)) for x in np.asarray(covariances).ravel()]
        mass = sum(p)
        mu = sum(w * x for w, x in zip(p, m)) / mass
        variance = sum(w * (v + (x - mu) ** 2) for w, x, v in zip(p, m, c)) / mass
        results.append(float(1 / variance - Fraction(float(np.asarray(precision).item()))))
    return np.array(results)


def entropy(probability, mean, covariance):
    p = array(probability)
    _, determinant = inverse_logdet(covariance)
    gaussian = (mean.shape[-1] * (1 + log(2 * number('3.1415926535897932384626433832795028841971693993751')))
                + determinant) / 2
    return np.sum(p * (gaussian - log(p)))


def expectation(probability, mean, covariance, natural):
    p, m, c = map(array, (probability, mean, covariance))
    a, h, precision = map(array, natural)
    second = c + m[..., :, None] * m[..., None, :]
    return np.sum(p * (a + (h * m).sum(axis=-1) - (precision * second).sum(axis=(-2, -1)) / 2))
