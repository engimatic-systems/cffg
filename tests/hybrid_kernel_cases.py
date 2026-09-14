"""Small declared inputs and conversions for unbound D12 JAX kernel tests."""

import jax
import jax.numpy as jnp
import numpy as np

from cffg.hybrid_natural import Natural
from cffg.hybrid_separator import transition


def tolerance(dtype, terms=4):
    return 512 * terms * np.finfo(dtype).eps


def case(time=3, dtype=np.float64, *, independent=False):
    values = dict(weights=[0.35, -0.25], offsets=[-0.2, 0.3], c=[1, 1],
                  matrix=[0.65, 0.85], bias=[0.1, -0.15], noise=[0.8, 1.1],
                  site_constant=[-0.2, -0.4], site_linear=[0.2, -0.3],
                  site_precision=[0.9, 1.2])
    result = {key: np.tile(np.asarray(value, dtype), (time if key.startswith("site_") else time - 1, 1))
              for key, value in values.items()}
    if independent:
        result["weights"][:] = 0
        result["c"] = abs(result["offsets"])
        for key in ("matrix", "bias", "noise", "site_linear", "site_precision"):
            result[key][:, 1] = result[key][:, 0]
    return result


def stack(values):
    return jax.tree.map(lambda *x: jnp.stack([jnp.asarray(a) for a in x]), *values)


def unstack(values):
    return [jax.tree.map(lambda x: x[t], values) for t in range(values.constant.shape[0])]


def natural(value):
    return Natural(*(jnp.asarray(x) for x in value.arrays()), jnp.int32(0))


def build(args):
    arrays = {key: jnp.asarray(value) for key, value in args.items()}
    sites = Natural(arrays["site_constant"], arrays["site_linear"][..., None],
                    arrays["site_precision"][..., None, None],
                    jnp.zeros(arrays["site_constant"].shape[0], jnp.int32))
    factors = jax.vmap(transition)(arrays["weights"][..., None, None], arrays["offsets"][..., None],
                                   arrays["c"][..., None] ** 2, arrays["matrix"][..., None, None],
                                   arrays["bias"][..., None], arrays["noise"][..., None, None])
    return factors, sites


def messages(sites):
    def bank(selection):
        return Natural(jnp.where(jnp.isfinite(sites.constant[selection]), jnp.zeros_like(sites.constant[selection]), -jnp.inf),
                       jnp.zeros_like(sites.linear[selection]), jnp.zeros_like(sites.precision[selection]),
                       jnp.zeros(sites.constant.shape[0] - 1, jnp.int32))
    return bank(slice(None, -1)), bank(slice(1, None))


def weighted(belief):
    p = belief.probability
    return p, p[..., None] * belief.mean, p[..., None, None] * belief.second
