"""Host action selection from categorical posterior probabilities."""

import numpy as np


def map_indices(probabilities):
    """Return exact posterior maximizers in action order."""
    q = np.asarray(probabilities)
    if q.ndim != 1 or q.size == 0 or q.dtype.kind not in "fiu":
        raise ValueError("action probabilities must be a nonempty real vector")
    if not np.isfinite(q).all() or np.any(q < 0) or not np.any(q > 0):
        raise ValueError("action probabilities must be finite, nonnegative and have positive mass")
    return np.flatnonzero(q == q.max())


def choose_map_action(probabilities, rng: np.random.Generator) -> int:
    """Choose uniformly among exact maxima, consuming RNG only on a tie."""
    winners = map_indices(probabilities)
    if winners.size == 1:
        return int(winners[0])
    return int(rng.choice(winners))
