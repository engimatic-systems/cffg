"""Track a static gain whose sign cannot be identified by the available sensor."""

import jax
import numpy as np

from cffg import FunctionSpec, MeanImplementation, register_mean_function
from cffg.static_parameter_particles import StaticParameterParticles


def gain_shape(input_dim, output_dim, configuration):
    if input_dim != 1 or output_dim != 1 or configuration:
        raise ValueError("the gain example uses scalar inputs and readings")
    return (1,)


def sensor_mean(applied, gain, configuration):
    return applied * gain[0] ** 2


def main():
    function = FunctionSpec("examples.squared_gain", "1.0.0")
    register_mean_function(MeanImplementation(function.impl_id, function.version, gain_shape, sensor_mean))
    learner = StaticParameterParticles(function, [[-1., 1.]], [[2., 2.]], [[.12**2]], input_dim=1,
                                       n_particles=1_024, key=jax.random.key(41), ess_fraction=.75)
    # Applied inputs can vary. A missing reading leaves the posterior and clock alone.
    for applied, reading in ((1., .54), (.8, .43), (1.2, None), (1.1, .59), (.9, .48)):
        update = learner.update([applied], None if reading is None else [reading])
        if reading is not None:
            print(f"ESS {update.ess:.0f}, log predictive density estimate {update.log_evidence_increment:+.3f}")

    gain = learner.particles[:, 0]
    weights = np.exp(learner.log_weights)
    # The likelihood is symmetric in gain. The two signs remain plausible.
    print(f"E[|gain|] ≈ {weights @ abs(gain):.3f}")
    print(f"P(gain > 0) ≈ {weights @ (gain > 0):.3f}")


if __name__ == "__main__":
    main()
