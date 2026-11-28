"""Infer a pendulum length from angular velocity readings on a fixed candidate grid."""

import numpy as np

from cffg.finite_hypotheses import FiniteHypotheses


def main():
    lengths = np.array([[0.65], [1.0], [1.6]])
    bank = FiniteHypotheses(lengths, [[0.02**2]], prior=[1, 2, 1])
    # Each row is a measured starting angle and angular velocity for a short
    # trial. The likelihood uses independent noise on the final velocity.
    angle = np.array([0.7, -0.9, 0.4])
    velocity = np.array([0.1, -0.2, 0.5])
    dt, g = 0.05, 9.81
    means = velocity[:, None] - dt * g * np.sin(angle[:, None]) / lengths[:, 0]
    readings = velocity - dt * g * np.sin(angle) / 1.0 + np.array([0.01, -0.01, 0.])
    bank.update(means[:, :, None], readings[:, None])
    assert bank.weights.argmax() == 1
    print("Candidate lengths (m):", lengths.ravel().tolist())
    print("Posterior probabilities:", bank.weights.tolist())
    print("Log evidence for the three readings:", bank.log_evidence)


if __name__ == "__main__":
    main()
