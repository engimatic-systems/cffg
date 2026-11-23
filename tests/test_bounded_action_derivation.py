"""Exact Gaussian box-conditioning identities and independent float64 planning oracles."""

from typing import NamedTuple

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from scipy.linalg import block_diag
from scipy.optimize import lsq_linear
from scipy.sparse.csgraph import connected_components
from scipy.special import ndtr, roots_legendre
from scipy.stats import multivariate_normal, norm, truncnorm


MEAN = np.array([0.0, 2.0])
PRECISION = np.array([[2.0, -1.0], [-1.0, 2.0]])
COV = np.linalg.inv(PRECISION)
BOX = np.array([[-1.0, 1.0], [-1.0, 1.0]])


class BoxMoments(NamedTuple):
    mass: float
    mean: np.ndarray
    covariance: np.ndarray
    entropy: float


def box_quadrature(mean, covariance, bounds):
    """Integrate the normalized cavity density and its statistics on a finite box."""
    nodes, weights = roots_legendre(32)
    lower, upper = bounds.T
    half_width = (upper - lower) / 2
    axes = ((upper + lower) / 2)[:, None] + half_width[:, None] * nodes
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(mean))
    measures = np.meshgrid(*(half_width[:, None] * weights), indexing="ij")
    log_density = multivariate_normal.logpdf(points, mean, covariance)
    masses = np.prod(measures, axis=0).ravel() * np.exp(log_density)
    mass = masses.sum()
    normalized = masses / mass
    center = normalized @ points
    centered = points - center
    return BoxMoments(mass, center, (centered.T * normalized) @ centered,
                      -normalized @ (log_density - np.log(mass)))


def conditional_integrals(mean, covariance, bounds):
    """Integrate out the second coordinate analytically, then adaptively integrate the first."""
    slope = covariance[1, 0] / covariance[0, 0]
    sigma = np.sqrt(covariance[1, 1] - slope * covariance[0, 1])
    sigma_first = np.sqrt(covariance[0, 0])

    def integrand(first):
        conditional_mean = mean[1] + slope * (first - mean[0])
        lo, hi = (bounds[1] - conditional_mean) / sigma
        mass = ndtr(hi) - ndtr(lo)
        weighted_second_mean = conditional_mean * mass + sigma * (norm.pdf(lo) - norm.pdf(hi))
        return norm.pdf(first, mean[0], sigma_first) * np.array([mass, first * mass, weighted_second_mean])

    return quad_vec(integrand, *bounds[0], epsabs=1e-13, epsrel=1e-13)[0]


def test_scalar_interval_moments_entropy_and_support():
    mu, sigma = 0.45, 0.7
    bounds = np.array([[-1.3, 0.8]])
    lo, hi = (bounds[0] - mu) / sigma
    mass = ndtr(hi) - ndtr(lo)
    shift = (norm.pdf(lo) - norm.pdf(hi)) / mass
    boundary = (lo * norm.pdf(lo) - hi * norm.pdf(hi)) / mass
    exact = box_quadrature([mu], [[sigma**2]], bounds)
    assert_allclose(exact.mass, mass, atol=1e-13, rtol=0)
    assert_allclose(exact.mean, [mu + sigma * shift], atol=1e-13, rtol=0)
    assert_allclose(exact.covariance, [[sigma**2 * (1 + boundary - shift**2)]], atol=1e-13, rtol=0)
    entropy = np.log(sigma * mass * np.sqrt(2 * np.pi * np.e)) + boundary / 2
    assert_allclose(exact.entropy, entropy, atol=1e-13, rtol=0)
    points = np.array([-1.31, -1.3, 0.1, 0.8, 0.81])
    density = truncnorm.pdf(points, lo, hi, loc=mu, scale=sigma)
    assert_allclose(density / norm.pdf(points, mu, sigma), [0, 1/mass, 1/mass, 1/mass, 0])


def test_two_step_lq_joint_mean_map_and_clipping():
    # x1 = u0, x2 = u1 - x1. Penalize u0, u1 - 2, and terminal error x2 - 2.
    dynamics = np.array([[1, 0], [0, 1], [-1, 1]])
    target = np.array([0, 2, 2])

    def cost(plan):
        return 0.5 * np.sum((dynamics @ plan - target)**2)

    assert_allclose(dynamics.T @ dynamics, PRECISION)
    assert_allclose(dynamics @ MEAN, target)
    clipped = np.clip(MEAN, *BOX.T)
    mode = lsq_linear(dynamics, target, bounds=tuple(BOX.T), tol=1e-13).x
    assert_allclose(clipped, [0, 1])
    assert_allclose(mode, [-0.5, 1], atol=1e-11, rtol=0)
    assert_allclose([cost(mode), cost(clipped)], [0.75, 1], atol=1e-11, rtol=0)

    law = box_quadrature(MEAN, COV, BOX)
    mass, *first_moments = conditional_integrals(MEAN, COV, BOX)
    assert_allclose(law.mean, np.array(first_moments) / mass, atol=1e-13, rtol=0)
    assert_allclose(law.mean, [-0.317858287538, 0.650809768791], atol=1e-12, rtol=0)
    assert mode[0] < law.mean[0] < clipped[0]


def test_joint_mass_entropy_and_free_energy_require_action_correlations():
    law = box_quadrature(MEAN, COV, BOX)
    log_mass = np.log(law.mass)
    assert_allclose(log_mass, np.log(conditional_integrals(MEAN, COV, BOX)[0]), atol=1e-13, rtol=0)
    assert_allclose([log_mass, law.entropy], [-2.630977166507, 0.434896326485], atol=1e-12, rtol=0)
    assert_allclose(law.covariance, [[0.211302782202, 0.017932749845],
                                     [0.017932749845, 0.091435664067]], atol=1e-12, rtol=0)

    delta = law.mean - MEAN
    energy = (np.trace(PRECISION @ law.covariance) + delta @ PRECISION @ delta) / 2
    gaussian_constant = np.log(2 * np.pi) + np.linalg.slogdet(COV)[1] / 2
    assert_allclose(energy + gaussian_constant - law.entropy, -log_mass, atol=1e-13, rtol=0)
    # A moment-matched Gaussian leaks mass through the upper wall and changes the entropy.
    projected = multivariate_normal(law.mean, law.covariance)
    assert norm.sf(BOX[1, 1], law.mean[1], np.sqrt(law.covariance[1, 1])) > 0.1
    assert projected.entropy() > law.entropy

    # Delete only the correlation while preserving every unbounded scalar marginal.
    independent = box_quadrature(MEAN, np.diag(COV.diagonal()), BOX)
    lo, hi = ((BOX - MEAN[:, None]) / np.sqrt(COV.diagonal())[:, None]).T
    assert_allclose(independent.mass, np.prod(ndtr(hi) - ndtr(lo)), atol=1e-13, rtol=0)
    assert_allclose(independent.mean[0], 0, atol=1e-13)
    assert independent.mass - law.mass > 0.01
    assert independent.mean[0] - law.mean[0] > 0.3


def test_joint_and_scalar_messages_keep_box_support_on_information_deletion():
    # Delete a Gaussian recipient from the bounded joint in natural coordinates.
    recipient_mean = np.array([-0.4, 0.2])
    recipient_precision = np.diag([4.0, 3.0])
    j = PRECISION - recipient_precision
    h = PRECISION @ MEAN - recipient_precision @ recipient_mean
    assert np.linalg.eigvalsh(j).max() < 0
    points = np.array([[0, 0], [-1, 1], [0.5, -0.2], [1, -1], [-1.01, 0], [0, 1.01]])
    inside = np.all((BOX[:, 0] <= points) & (points <= BOX[:, 1]), axis=1)
    message = inside * np.exp(points @ h - np.einsum("ni,ij,nj->n", points, j, points) / 2)
    recipient = multivariate_normal.pdf(points, recipient_mean, np.linalg.inv(recipient_precision))
    reconstructed = recipient * message
    cavity = multivariate_normal.pdf(points, MEAN, COV)
    assert_allclose(reconstructed / reconstructed[0], cavity / cavity[0] * [1, 1, 1, 1, 0, 0])

    # Deleting the *scalar marginal* leaves the other action's conditional interval probability.
    second_scale = np.sqrt(COV[1, 1] - COV[1, 0]**2 / COV[0, 0])
    for first, supported in ((-1.01, False), (-1., True), (0., True), (1., True), (1.01, False)):
        conditional_mean = MEAN[1] + COV[1, 0] / COV[0, 0] * (first - MEAN[0])
        lo, hi = (BOX[1] - conditional_mean) / second_scale
        scalar_message = (BOX[0, 0] <= first <= BOX[0, 1]) * (ndtr(hi) - ndtr(lo))
        joint_slice = quad_vec(lambda y: multivariate_normal.pdf([first, y], MEAN, COV), *BOX[1])[0]
        assert_allclose(norm.pdf(first, MEAN[0], np.sqrt(COV[0, 0])) * scalar_message,
                        joint_slice if supported else 0, atol=1e-13, rtol=0)


def test_gaussian_lift_matches_full_joint_log_normalizer_derivatives():
    state_mean = np.array([0.75, -0.25])
    state_cov = np.array([[1.8, 0.4], [0.4, 1.2]])
    state_action = np.array([[0.4, -0.15], [0.25, 0.55]])
    mean = np.r_[state_mean, MEAN]
    covariance = np.block([[state_cov, state_action], [state_action.T, COV]])
    gain = np.linalg.solve(COV, state_action.T).T
    residual = state_cov - gain @ state_action.T
    law = box_quadrature(MEAN, COV, BOX)
    cross = gain @ law.covariance
    lifted_mean = np.r_[state_mean + gain @ (law.mean - MEAN), law.mean]
    lifted_cov = np.block([[residual + cross @ gain.T, cross], [cross.T, law.covariance]])

    def log_partition(tilt):
        # Completing the square in all four coordinates leaves a shifted action marginal.
        shift = covariance @ tilt
        mass = conditional_integrals((mean + shift)[2:], COV, BOX)[0]
        return np.log(mass) + tilt @ (mean + shift / 2)

    step = 0.001
    offsets = step * np.eye(4)
    center = log_partition(np.zeros(4))
    for i, ei in enumerate(offsets):
        plus, minus = log_partition(ei), log_partition(-ei)
        assert_allclose((plus - minus) / (2 * step), lifted_mean[i], atol=1e-7, rtol=0)
        assert_allclose((plus + minus - 2 * center) / step**2, lifted_cov[i, i], atol=1e-7, rtol=0)
        for j, ej in enumerate(offsets[:i]):
            mixed = (log_partition(ei + ej) - log_partition(ei - ej)
                     - log_partition(-ei + ej) + log_partition(-ei - ej)) / (4 * step**2)
            assert_allclose(mixed, lifted_cov[i, j], atol=1e-7, rtol=0)

    delta = lifted_mean - mean
    energy = -multivariate_normal.logpdf(mean, mean, covariance) + (
        np.trace(np.linalg.solve(covariance, lifted_cov)) + delta @ np.linalg.solve(covariance, delta)
    ) / 2
    conditional_entropy = np.log(2 * np.pi * np.e) + np.linalg.slogdet(residual)[1] / 2
    assert_allclose(energy - law.entropy - conditional_entropy, -center, atol=1e-12, rtol=0)


def v1_blocks(structural_edges):
    """Classify the certified structural graph of the marginalized action cavity."""
    count, labels = connected_components(structural_edges, directed=False)
    blocks = [np.flatnonzero(labels == i) for i in range(count)]
    if any(len(block) > 2 for block in blocks):
        raise ValueError("connected action block exceeds width two")
    return blocks


def test_horizon_15_seven_certified_pairs_and_singleton():
    # A direct sum certifies independence. Alternate signs to retain both coupling directions.
    pair_covariances = [COV * [[1, (-1)**i], [(-1)**i, 1]] for i in range(7)]
    pieces = [*pair_covariances, np.array([[0.5]])]
    structure = block_diag(*(np.ones(piece.shape, dtype=bool) for piece in pieces))
    blocks = v1_blocks(structure)
    assert [len(block) for block in blocks] == [2] * 7 + [1]
    covariance = block_diag(*pieces)
    mean = np.r_[np.tile(MEAN, 7), -0.3]
    bounds = np.tile(BOX[0], (15, 1))
    laws = [box_quadrature(mean[b], covariance[np.ix_(b, b)], bounds[b]) for b in blocks]

    references = [conditional_integrals(MEAN, cov, BOX) for cov in pair_covariances]
    for law, raw in zip(laws[:2], references[:2], strict=True):
        assert_allclose(law.mean, raw[1:] / raw[0], atol=1e-13, rtol=0)
    assert_allclose([laws[0].mean[0], laws[1].mean[0]], [-0.317858287538, 0.317858287538],
                    atol=1e-12, rtol=0)
    lo, hi = (bounds[-1] - mean[-1]) / np.sqrt(covariance[-1, -1])
    product_mass = np.prod([raw[0] for raw in references]) * (ndtr(hi) - ndtr(lo))
    log_mass = sum(np.log(law.mass) for law in laws)
    assert_allclose(log_mass, np.log(product_mass), atol=1e-12, rtol=0)
    assert_allclose(log_mass, -18.632630109296, atol=1e-12, rtol=0)


def test_connected_dense_three_and_adjacent_horizon_15_exceed_v1_width():
    dense_three = np.full((3, 3), 0.2) + 0.8 * np.eye(3)
    links = np.diag(np.full(14, 0.0193), k=1)
    adjacent = np.eye(15) + links + links.T
    for covariance in (dense_three, adjacent):
        np.linalg.cholesky(covariance)
        # Every nonzero entry in these constructed covariances is a known coupling.
        with pytest.raises(ValueError, match="exceeds width two"):
            v1_blocks(covariance != 0)


def test_public_closed_region_matches_conditionally_integrated_pair_and_lift():
    import jax
    import jax.numpy as jnp
    from cffg.bounded_actions import BoundedGaussianPlan, BoundStatus

    plan = BoundedGaussianPlan((2,), (-1., -1.), (1., 1.))
    assert BoundedGaussianPlan.from_dict(plan.to_dict()) == plan
    with pytest.raises(ValueError):
        BoundedGaussianPlan.from_dict(dict(plan.to_dict(), kind="legacy"))
    with pytest.raises(ValueError):
        BoundedGaussianPlan((3,), (-1.,)*3, (1.,)*3)
    integral = conditional_integrals(MEAN, COV, BOX)
    ref = box_quadrature(MEAN, COV, BOX)
    gain = np.array([[.5, -.25]])
    def solve(m):
        return plan.solve(m, (COV,), 3., state_mean=[1.], state_gain=gain,
                          conditional_covariance=[[.8]])
    expected = np.r_[integral[1:] / integral[0], 1 + gain @ (ref.mean - MEAN)]
    cross = gain @ ref.covariance
    expected_cov = np.block([[ref.covariance, cross.T], [cross, .8 + cross @ gain.T]])
    for out in [solve(MEAN), jax.tree.map(lambda x: x[0],
                jax.jit(jax.vmap(solve))(jnp.asarray([MEAN, MEAN])))]:
        assert out.status == BoundStatus.SUCCESS
        assert_allclose(out.mean, expected)
        assert_allclose(out.covariance, expected_cov)
        assert_allclose(out.command, expected[0])
        assert_allclose(out.free_energy, 3 - np.log(integral[0]))
        assert_allclose(out.action_entropy, ref.entropy)
    assert_allclose(plan.log_factor([[0., 1.], [0., 1.01]]), [0., -np.inf])


def test_dense_joint_and_independent_blocks_keep_full_state_correlations():
    from cffg.bounded_actions import BoundedGaussianPlan

    pair = BoundedGaussianPlan((2,), (-1., -1.), (1., 1.))
    gain = np.array([[1., .25], [.5, -.5]])
    noise = np.array([[.3, .1], [.1, .4]])
    cross = gain @ COV
    joint = np.block([[COV, cross.T], [cross, noise + cross @ gain.T]])
    dense = pair.solve_joint(np.r_[MEAN, [2., 0.]], joint, 4.)
    structural = pair.solve(MEAN, (COV,), 4., state_mean=[2., 0.],
                            conditional_covariance=noise, state_gain=gain)
    assert dense.status == structural.status == 0
    assert_allclose(dense.covariance, structural.covariance)
    assert_allclose(dense.mean, structural.mean)
    mixed = BoundedGaussianPlan((2, 1), (-1.,)*3, (1.,)*3)
    with pytest.raises(ValueError, match="dense input"):
        mixed.solve_joint(np.zeros(3), np.eye(3), 0.)
    out = mixed.solve(np.r_[MEAN, 0.], (COV, [[1.]]), 0., state_mean=[0.],
                      state_gain=[[1., 0., 1.]], conditional_covariance=[[1.]])
    assert out.status == 0
    assert_allclose(out.covariance[:2, 2], 0.)
    assert out.covariance[3, 0] > 0 and out.covariance[3, 2] > 0
