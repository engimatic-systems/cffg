"""Online conjugate learning for a bank of Gaussian and categorical components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg import mixed_scoring


class MixedBank(NamedTuple):
    """K full-covariance NIW priors and categorical heads sharing a selector.

    mu is [K,d], psi is [K,d,d], and kappa, nu and alpha are [K].
    Each entry in categories contains [K,C] Dirichlet concentrations for
    one categorical field. Inactive components retain proper parameters.
    """

    mu: jax.Array
    kappa: jax.Array
    psi: jax.Array
    nu: jax.Array
    alpha: jax.Array
    categories: tuple[jax.Array, ...]
    active: jax.Array

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a versioned, independent host snapshot of all learned fields."""
        if not bool(mixed_scoring.validate_bank(self)):
            raise ValueError("invalid mixed-learning snapshot parameters")
        fields = {key: np.array(getattr(self, key), copy=True)
                  for key in ("mu", "kappa", "psi", "nu", "alpha", "active")}
        fields.update({f"category_{i}": np.array(head, copy=True)
                       for i, head in enumerate(self.categories)})
        fields["category_cardinalities"] = np.array(
            [head.shape[1] for head in self.categories], dtype=np.int64
        )
        fields["version"] = np.array(2, dtype=np.int32)
        return fields

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray]) -> MixedBank:
        """Restore saved precision and the ordered categorical layout.

        Legacy v1 records have only consecutively numbered head arrays, so
        their format cannot establish whether a trailing head was lost.
        """
        version = np.asarray(state.get("version"))
        if version.shape != () or version.dtype != np.int32 or version.item() not in (1, 2):
            raise ValueError("unknown mixed-learning snapshot version")
        required = {"mu", "kappa", "psi", "nu", "alpha", "active"}
        metadata = {"version"}
        cardinalities = None
        if version.item() == 2:
            metadata.add("category_cardinalities")
            cardinalities = np.asarray(state.get("category_cardinalities"))
            if (cardinalities.dtype != np.int64 or cardinalities.ndim != 1
                    or np.any(cardinalities < 1)):
                raise ValueError("invalid or missing snapshot category cardinalities")
            count = len(cardinalities)
        else:
            count = len(set(state) - required - metadata)
        heads = tuple(f"category_{i}" for i in range(count))
        if set(state) != required | metadata | set(heads):
            raise ValueError("unexpected or missing mixed-learning snapshot field")
        fields = {}
        for key in required | set(heads):
            value = np.asarray(state[key])
            dtype = jax.dtypes.canonicalize_dtype(value.dtype)
            if value.dtype != dtype:
                raise ValueError(f"{key}: saved dtype {value.dtype} is disabled in JAX")
            fields[key] = jnp.array(value, copy=True)
        bank = cls(fields["mu"], fields["kappa"], fields["psi"], fields["nu"],
                   fields["alpha"], tuple(fields[key] for key in heads), fields["active"])
        if cardinalities is not None and any(
            head.shape[1:] != (width,) for head, width in zip(bank.categories, cardinalities, strict=True)
        ):
            raise ValueError("snapshot categorical head shape disagrees with its cardinality")
        if not bool(mixed_scoring.validate_bank(bank)):
            raise ValueError("invalid mixed-learning snapshot parameters")
        return bank


class MixedBatch(NamedTuple):
    """Conditional feature moments and known category indices for N records.

    mean [N,d] and covariance [N,d,d] describe the same continuous field.
    gaussian_observed [N] selects observed rows. A point has zero covariance.
    Each integer category array has shape [N]; -1 denotes a missing field.
    """

    mean: jax.Array
    covariance: jax.Array
    gaussian_observed: jax.Array
    categories: tuple[jax.Array, ...]


def score(bank: MixedBank, batch: MixedBatch, *, include_mixing: bool = False):
    """Return absolute observed log-likelihood scores, responsibilities and validity.

    Only the normalized responsibilities incorporate optional expected log
    mixing weights. Unsupported components receive score -inf and zero mass.
    """
    return mixed_scoring.score(bank, batch, include_mixing=include_mixing)


def update(bank: MixedBank, batch: MixedBatch, weights: jax.Array) -> tuple[MixedBank, jax.Array]:
    """Add a batch with fixed responsibilities; return (new bank, valid).

    Every observed field uses the same [N,K] probability matrix. A partially
    observed record counts once toward mixture frequencies, and a record with
    no observed field contributes nothing. Invalid numerical inputs or an
    unrepresentable posterior return the input bank with valid=False.
    """
    valid = mixed_scoring.validate_inputs(bank, batch)
    n = batch.mean.shape[0]
    k = len(bank.mu)
    if np.shape(weights) != (n, k) or getattr(weights, "dtype", None) != bank.mu.dtype:
        raise ValueError("weights must have shape [N,K] and the bank's floating dtype")
    bank, batch, weights = jax.tree.map(jnp.asarray, (bank, batch, weights))
    # A tree sum of K normalized probabilities can accumulate rounding error.
    tolerance = (8 + k.bit_length()) * jnp.finfo(weights.dtype).eps
    valid &= (jnp.all(jnp.isfinite(weights)) & jnp.all(weights >= 0)
              & jnp.all(jnp.where(bank.active[None, :], True, weights == 0))
              & jnp.all(jnp.abs(weights.sum(axis=1) - 1) <= tolerance))
    if n == 0:
        return bank, valid

    def accumulate():
        observed = batch.gaussian_observed
        gaussian_weights = jnp.where(observed[:, None], weights, 0)
        mass = gaussian_weights.sum(axis=0)
        kn = bank.kappa + mass
        mean = jnp.where(observed[:, None], batch.mean, 0)
        cov = jnp.where(observed[:, None, None], batch.covariance, 0)
        cov = 0.5 * cov + 0.5 * jnp.swapaxes(cov, -1, -2)
        # Center where the evidence is largest, before forming any square.
        # Keeping the offset in local coordinates also preserves scatter when
        # the final absolute posterior mean rounds to the prior.
        anchor = jnp.where((mass >= bank.kappa)[:, None],
                           mean[jnp.argmax(gaussian_weights, axis=0)], bank.mu)
        delta = jnp.where(gaussian_weights[:, :, None] > 0, mean[:, None, :] - anchor, 0)
        prior_delta = bank.mu - anchor
        precision = jax.lax.Precision.HIGHEST
        center = ((bank.kappa / kn)[:, None] * prior_delta
                  + jnp.einsum("nk,nkd->kd", gaussian_weights / kn, delta, precision=precision))
        residual = jnp.where(gaussian_weights[:, :, None] > 0, delta - center, 0)
        scaled = jnp.sqrt(gaussian_weights)[:, :, None] * residual
        prior_residual = jnp.sqrt(bank.kappa)[:, None] * (prior_delta - center)
        within = jnp.einsum("nk,nij->kij", gaussian_weights, cov, precision=precision)
        psi = (bank.psi + within + jnp.einsum("nki,nkj->kij", scaled, scaled, precision=precision)
               + prior_residual[:, :, None] * prior_residual[:, None, :])
        present = observed
        heads = []
        for index, alpha in zip(batch.categories, bank.categories, strict=True):
            index = mixed_scoring.category_indices(index)
            observed_category = index >= 0
            present = present | observed_category
            hot = jax.nn.one_hot(index, alpha.shape[1], dtype=weights.dtype)
            hot = jnp.where(observed_category[:, None], hot, 0)
            counts = jnp.matmul(weights.T, hot, precision=precision)
            heads.append(jnp.where(counts > 0, alpha + counts, alpha))
        seen_mass = jnp.where(present[:, None], weights, 0).sum(axis=0)
        changed = mass > 0
        candidate = MixedBank(
            jnp.where(changed[:, None], anchor + center, bank.mu),
            jnp.where(changed, kn, bank.kappa),
            jnp.where(changed[:, None, None], psi, bank.psi),
            jnp.where(changed, bank.nu + mass, bank.nu),
            jnp.where(seen_mass > 0, bank.alpha + seen_mass, bank.alpha),
            tuple(heads), bank.active,
        )
        proper = mixed_scoring.validate_bank(candidate)
        return jax.tree.map(lambda new, old: jnp.where(proper, new, old), candidate, bank), proper

    return jax.lax.cond(valid, accumulate, lambda: (bank, jnp.asarray(False)))
