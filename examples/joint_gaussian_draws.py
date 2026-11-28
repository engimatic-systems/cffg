"""One draw of parameters for a Gaussian model and a two-output regression."""

import numpy as np
from jax import random

from cffg import MNIW, NIW


def main():
    prior_key, regression_key = random.split(random.key(23))
    psi = np.array([[2., .5], [.5, 1.]])
    prior = NIW(np.zeros(2), kappa=3., psi=psi, nu=7.)
    observations = np.array([[1., .5], [.1, -.2]])
    posterior = prior.update(observations, observations[:, :, None] * observations[:, None, :])
    means, covariances = posterior.sample(prior_key, 4)
    print("First sampled location and covariance", means[0], covariances[0], sep="\n")
    print("Rollover covariance (inverse expected precision)", posterior.rollover()[1], sep="\n")
    print("Posterior mean covariance", posterior.mean_cov(), sep="\n")

    regression = MNIW(np.zeros((2, 3)), np.eye(3), psi, nu=7.)
    coefficients, noise_covariance = regression.sample(regression_key)
    print("Sampled [A b] and its noise covariance", coefficients, noise_covariance, sep="\n")


if __name__ == "__main__":
    main()
