"""Released priors and appearance features for AXIOM identity and interaction mixtures."""

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from cffg.component_bank import ComponentBank
from cffg.component_bank_device import from_host
from cffg.experiments.axiom_evidence import release_view
from cffg.mixed_learning import MixedBank, MixedBatch, score


@dataclass(frozen=True)
class BankSettings:
    identities: int = 32
    motions: int = 500
    interactions_per_motion: int = 10
    color_precision: float = 1.0
    color_only: bool = False
    fixed_radius: bool = False

    def __post_init__(self):
        if any(
            type(n) is not int or n < 1
            for n in (self.identities, self.motions, self.interactions_per_motion)
        ):
            raise ValueError("all capacities must be positive integers")
        if self.motions < 4 or not math.isfinite(self.color_precision) or self.color_precision <= 0:
            raise ValueError("four seed motions and positive finite color precision required")


def _empty_bank(k, d, diagonal, categorical, dtype):
    if dtype not in (jnp.float32, jnp.float64) or jax.dtypes.canonicalize_dtype(dtype) != dtype:
        raise ValueError("requested learning precision unavailable")
    prior = MixedBank(
        jnp.zeros((k, d), dtype),
        jnp.full(k, 1e-4, dtype),
        jnp.broadcast_to(jnp.diag(jnp.asarray(diagonal, dtype)), (k, d, d)),
        jnp.full(k, 1 + 2 * d, dtype),
        jnp.full(k, 0.1, dtype),
        tuple(jnp.full((k, size), concentration, dtype) for size, concentration in categorical),
        jnp.zeros(k, bool),
    )
    return from_host(ComponentBank(prior))


def initial_banks(actions, *, settings=BankSettings(), dtype=jnp.float32):
    """Create empty fixed-capacity learning banks and their unchanging reset priors."""
    if type(actions) is not int or actions < 1:
        raise ValueError("at least one discrete action is required")
    i, t = settings.identities, settings.motions
    d = 5 if settings.fixed_radius else 7
    # Release rescales dimensions from index two, even in its color-only variant.
    features = 3 if settings.color_only else 5
    psi_i = [4.0, 4.0] + [4.0 / settings.color_precision] * (features - 2)
    identity = _empty_bank(i, features, psi_i, (), dtype)
    heads = ((i + 1, 1e-4), (i + 1, 1e-4), (2, 1e-4), (actions, 1e-4), (3, 1.0), (t, 1e-4))
    relation = _empty_bank(t * settings.interactions_per_motion, d, [75**-2] * d, heads, dtype)
    return identity, relation


def seeded_dormant(prior, key):
    """Reconstruct virgin parameters for an unmasked reduction query from its seed.

    Call with the mixture creation key. Persist that key alongside the model
    configuration when a replay needs the initialized, never-used rows.
    """
    k, d = prior.mu.shape
    dtype = prior.mu.dtype
    _, key = jax.random.split(key)
    key, normal = jax.random.split(key)
    mu = prior.mu + jax.random.normal(normal, (k, d), dtype)
    key, _ = jax.random.split(key)
    heads = []
    for head in prior.categories:
        key, _ = jax.random.split(key)
        key, label = jax.random.split(key)
        width = head.shape[1]
        period = min(k, width)
        row = jax.random.permutation(label, k)
        favored = (row[:, None] % period) == (jnp.arange(width)[None, :] % period)
        scale = jax.random.uniform(label, head.shape, dtype, minval=0.1, maxval=0.9)
        heads.append((head + 10 * favored) * scale)
    return prior._replace(mu=mu, categories=tuple(heads))


def identity_records(scene, *, color_only=False):
    """Use the previous descriptor's width, height and 100-times normalized RGB."""
    if scene.ndim != 2 or scene.shape[1] != 11:
        raise ValueError("expected one eleven-coordinate scene row per slot")
    features = scene[:, -5:].at[:, 2:].multiply(100)
    if color_only:
        features = features[:, 2:]
    n, d = features.shape
    return MixedBatch(features, jnp.zeros((n, d, d), scene.dtype), jnp.ones(n, bool), ())


def identity_labels(bank, records):
    """Read first-maximum identity per slot, including the all-unused uniform case."""
    empty = ~jnp.any(bank.active)
    model = release_view(bank)
    supported = model._replace(active=model.active.at[0].set(model.active[0] | empty))
    _, weights, proper = score(supported, records)
    labels = jnp.where(empty, 0, weights.argmax(axis=1)).astype(jnp.int32)
    return labels, proper
