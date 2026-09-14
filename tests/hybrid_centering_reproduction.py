"""Executable D12 translation diagnostic, outside pytest discovery.

Run ``uv run python tests/hybrid_centering_reproduction.py``. Translating x
and the recurrence offset together represents the identical local problem.
The current expanded tilt reports OK but changes regime mass in the first
float32 case. The assertion deliberately fails until that defect is fixed;
this diagnostic is not a passing qualification or a relaxed tolerance.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np

from cffg.hybrid_natural import Natural, lift, normalize
from cffg.hybrid_separator import TransitionParameters, transition


def translation_gap(dtype, mean, variance):
    def tilted(center):
        parameters = TransitionParameters(jnp.ones((2, 1, 1), dtype),
                                          jnp.full((2, 1), -center, dtype), jnp.zeros((2, 1), dtype),
                                          jnp.zeros((2, 1, 1), dtype), jnp.zeros((2, 1), dtype),
                                          jnp.ones((2, 1, 1), dtype))
        constant = -np.log(2) - (np.log(2 * np.pi * variance) + center ** 2 / variance) / 2
        left = Natural(jnp.full(2, constant, dtype), jnp.full((2, 1), center / variance, dtype),
                       jnp.full((2, 1, 1), 1 / variance, dtype))
        right = Natural(jnp.zeros(2, dtype), jnp.zeros((2, 1), dtype), jnp.zeros((2, 1, 1), dtype))
        return normalize(transition(*parameters) + lift(left, right))

    reference, shifted = tilted(0), tilted(mean)
    return dict(dtype=np.dtype(dtype).name, mean=mean, variance=variance,
                status=int(shifted.status), centered_probability=np.asarray(reference.probability).tolist(),
                shifted_probability=np.asarray(shifted.probability).tolist(),
                maximum_probability_gap=float(abs(reference.probability - shifted.probability).max()),
                centered_mean=np.asarray(reference.mean[..., 0]).tolist(),
                shifted_mean_offset=np.asarray(shifted.mean[..., 0] - mean).tolist(),
                maximum_covariance_gap=float(abs(reference.covariance - shifted.covariance).max()))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    results = [translation_gap(*case) for case in
               ((np.float32, 100, 1e-4), (np.float32, 10000, 1), (np.float64, 1e8, 1))]
    print(json.dumps(results, indent=2))
    for result in results:
        budget = 512 * 4 * np.finfo(np.dtype(result["dtype"])).eps
        assert result["status"] == 0
        assert result["maximum_probability_gap"] <= budget, "translation changed label probabilities"
