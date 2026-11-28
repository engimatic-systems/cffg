"""In-memory and checkpoint representations of a mixed component bank."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg import mixed_scoring


class MixedCarry(NamedTuple):
    """Low words of the continuous parameters, mixture masses and categorical counts."""

    mu: jax.Array
    kappa: jax.Array
    psi: jax.Array
    nu: jax.Array
    alpha: jax.Array
    categories: tuple[jax.Array, ...]


class _BankFields(NamedTuple):
    mu: jax.Array
    kappa: jax.Array
    psi: jax.Array
    nu: jax.Array
    alpha: jax.Array
    categories: tuple[jax.Array, ...]
    active: jax.Array
    carry: MixedCarry


def _zeros(value):
    if isinstance(value, np.ndarray):
        return np.zeros_like(value)
    return jnp.zeros_like(value)


def _zero_carry(mu, kappa, psi, nu, alpha, categories):
    return MixedCarry(_zeros(mu), _zeros(kappa), _zeros(psi), _zeros(nu), _zeros(alpha),
                      tuple(_zeros(head) for head in categories))


class MixedBank(_BankFields):
    """K Gaussian components and categorical heads sharing assignment weights.

    mu is [K,d], psi is [K,d,d], and kappa, nu and alpha are [K]. Each
    categorical head contains [K,C] Dirichlet concentrations. Inactive slots
    retain proper parameters. The primary fields hold rounded parameter
    values. The carry retains smaller corrections through subsequent updates.
    Use as_rounded() to trade accumulation accuracy for speed on public
    score/update calls. See "Tune mixed-learning speed" in the README for
    timings and a count-accuracy example in both supported precisions.
    """

    __slots__ = ()

    def __new__(cls, mu, kappa, psi, nu, alpha, categories, active, carry=None):
        if carry is None:
            carry = _zero_carry(mu, kappa, psi, nu, alpha, categories)
        return super().__new__(cls, mu, kappa, psi, nu, alpha, categories, active, carry)

    def _replace(self, /, **fields):
        # Replacing a parameter supplies a new value independent of its old low word.
        if 'carry' not in fields:
            changed = {key: jax.tree.map(_zeros, value) for key, value in fields.items()
                       if key in MixedCarry._fields}
            if changed:
                fields['carry'] = self.carry._replace(**changed)
        return super()._replace(**fields)

    def state_dict(self, *, exact=False) -> dict[str, np.ndarray]:
        """Save rounded primary fields, optionally including exact continuation state.

        The default retains the v2 compact layout. Saving leaves the live learner
        intact. Loading a compact record starts from the saved values with zero
        carry. Exact snapshots (v3) also persist the residuals at this boundary.
        """
        return _snapshot(self, exact)

    def as_rounded(self) -> RoundedBank:
        """Return a one-word bank for faster score/update workloads.

        The new bank starts from the visible values and discards the rounding
        remainders of means, covariance scales, and counts. Small future
        increments can disappear. Call as_compensated() on the result to
        retain new remainders starting from zero.
        """
        return RoundedBank(self.mu, self.kappa, self.psi, self.nu, self.alpha,
                           self.categories, self.active)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray]) -> MixedBank:
        """Restore saved precision and require every field of the declared layout.

        Legacy v1 records can verify only the contiguous sequence of head keys.
        """
        v = np.asarray(state.get("version"))
        if v.shape != () or v.dtype != np.int32 or v.item() not in (1, 2, 3):
            raise ValueError("unknown mixed-learning snapshot version")
        version = int(v)
        metadata = {"version"}
        widths = None
        if version >= 2:
            metadata.add("category_cardinalities")
            widths = np.asarray(state.get("category_cardinalities"))
            if widths.ndim != 1 or widths.dtype != np.int64 or np.any(widths <= 0):
                raise ValueError("invalid or missing snapshot category cardinalities")
            nheads = len(widths)
        else:
            nheads = len(set(state) - _PRIMARY - metadata)
        heads = tuple(f"category_{i}" for i in range(nheads))
        values = _PRIMARY | set(heads)
        if version == 3:
            values |= {f"carry_{key}" for key in (*_CARRIED, *heads)}
        if set(state) != values | metadata:
            raise ValueError("unexpected or missing mixed-learning snapshot field")

        arrays = {}
        for key in values:
            val = np.asarray(state[key])
            if val.dtype != jax.dtypes.canonicalize_dtype(val.dtype):
                raise ValueError(f"{key}: saved dtype {val.dtype} is disabled in JAX")
            arrays[key] = jnp.array(val, copy=True)
        cats = tuple(arrays[h] for h in heads)
        if widths is not None and any(h.shape[1:] != (width,) for h, width in zip(cats, widths)):
            raise ValueError("snapshot categorical head shape disagrees with its cardinality")
        carry = None
        if version == 3:
            carry = MixedCarry(*(arrays[f"carry_{key}"] for key in _CARRIED),
                               tuple(arrays[f"carry_{h}"] for h in heads))
        bank = cls(arrays["mu"], arrays["kappa"], arrays["psi"], arrays["nu"],
                   arrays["alpha"], cats, arrays["active"], carry)
        if not bool(mixed_scoring.validate_bank(bank, for_checkpoint=True)):
            raise ValueError("invalid mixed-learning snapshot parameters")
        return bank



class RoundedBank(NamedTuple):
    """One-word posterior for faster standalone mixed learning.

    Each update rounds the posterior immediately. Check the error at the
    largest count and feature offset in your data. The README section
    "Tune mixed-learning speed" compares this choice with MixedBank.
    """

    mu: jax.Array
    kappa: jax.Array
    psi: jax.Array
    nu: jax.Array
    alpha: jax.Array
    categories: tuple[jax.Array, ...]
    active: jax.Array

    def as_compensated(self) -> MixedBank:
        """Begin compensated learning from the current values with zero residuals."""
        return MixedBank(*self)

    def state_dict(self, *, exact=False) -> dict[str, np.ndarray]:
        """Save all seven primary fields in v2. This is exact for a rounded learner."""
        return _snapshot(self, False)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray]) -> RoundedBank:
        """Validate v1/v2/v3, then select rounded arithmetic from the saved primary fields."""
        return MixedBank.from_state_dict(state).as_rounded()


def _snapshot(bank, exact):
    if not bool(mixed_scoring.validate_bank(bank, for_checkpoint=True)):
        raise ValueError("invalid mixed-learning snapshot parameters")
    fields = {key: np.array(getattr(bank, key), copy=True) for key in _PRIMARY}
    for i, head in enumerate(bank.categories):
        fields[f"category_{i}"] = np.array(head, copy=True)
    fields["category_cardinalities"] = np.asarray([h.shape[1] for h in bank.categories],
                                                 dtype=np.int64)
    fields["version"] = np.int32(3 if exact else 2)
    if exact:
        for key in _CARRIED:
            fields[f"carry_{key}"] = np.array(getattr(bank.carry, key), copy=True)
        for i, low in enumerate(bank.carry.categories):
            fields[f"carry_category_{i}"] = np.array(low, copy=True)
    return fields


_PRIMARY = {"mu", "kappa", "psi", "nu", "alpha", "active"}
_CARRIED = ("mu", "kappa", "psi", "nu", "alpha")
